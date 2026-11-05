"""
pipeline.py — Orquestrador do componente I1 (Arquitetura 6c).
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict, deque
from typing import Any, Callable

from .config import AppConfig
from .contracts import Frame, P2ExecutionError, new_message_id, utc_now_iso_ms
from .inference import P2InferenceClient
from .ingestion import FrameSource, preprocess_frame_to_bgr
from .postprocessing import apply_nms, filter_by_confidence, finalize_detections
from .publishing import EventPublisher, build_envelope, compute_e2e_latency_ms

logger = logging.getLogger("ods.i1.observability")
ENDED_SESSION_CACHE_SIZE = 4096
TELEMETRY_HISTORY_SIZE = 4096


def classify_frame_error(stage: str, err: Exception) -> str:
    """
    Código de erro registrado na observabilidade (B2) para um quadro descartado (T7).
    Erros de P2 mantêm o código formal; os demais recebem um código interno de I1 conforme o estágio.
    """
    if isinstance(err, P2ExecutionError):
        return err.code
    if isinstance(err, ValueError) and stage == "preprocess":
        return "INVALID_FRAME"
    if isinstance(err, ValueError) and stage == "publish":
        return "CONTRACT_VIOLATION"
    return "INTERNAL_ERROR"


class DetectorPipeline:
    """Orquestrador único do componente I1 ('Um artefato por aplicação, um componente só')."""

    def __init__(
        self,
        config: AppConfig,
        source: FrameSource,
        publisher: EventPublisher,
        inference_client: P2InferenceClient | None = None,
        clock: Callable[[], str] = utc_now_iso_ms,
        message_id_factory: Callable[[], str] = new_message_id,
    ) -> None:
        self.config = config
        self.source = source
        self.publisher = publisher
        self.inference_client = inference_client or P2InferenceClient()
        # Injetáveis para que o replay de P6 seja testável de forma determinística (T6 / T17)
        self.clock = clock
        self.message_id_factory = message_id_factory

        # Sessão ativa vem sempre dos metadados ods.p4.frame do quadro (P4 ao vivo, P6 em replay)
        self.active_session_id: str | None = None
        self.ended_session_ids: OrderedDict[str, None] = OrderedDict()
        self.sequence: int = 0
        self.last_frame_index: int | None = None
        self.effective_artifact_version: str = config.artifact_version
        self.telemetry_history: deque[dict[str, Any]] = deque(maxlen=TELEMETRY_HISTORY_SIZE)
        self.telemetry_count = 0
        self.total_infer_latency_ms = 0.0
        self.error_counts: dict[str, int] = {}

    def _log_observability(self, level: int, record: dict[str, Any]) -> None:
        """Registro estruturado (JSON em uma linha) para a observabilidade de B2."""
        logger.log(level, "[B2-OBS] %s", json.dumps(record, ensure_ascii=False))

    def _record_frame_drop(
        self, frame: Frame, sequence: int | None, stage: str, err: Exception, code: str | None = None
    ) -> None:
        code = code or classify_frame_error(stage, err)
        self.error_counts[code] = self.error_counts.get(code, 0) + 1
        self._log_observability(
            logging.ERROR,
            {
                "event": "frame_dropped",
                "error_code": code,
                "stage": stage,
                "source_id": frame.source_id,
                "session_id": frame.session_id,
                "sequence": sequence,
                "frame_index": frame.frame_index,
                "artifact_version": self.effective_artifact_version,
                "message": str(err),
            },
        )

    def initialize(self) -> None:
        """Carrega o motor via P2.load() e obtém engine_version via P2.info() (T11)."""
        info = self.inference_client.load_engine(
            engine_ref=self.config.engine_ref,
            options={
                "artifact_id": self.config.artifact_id,
                "artifact_version": self.config.artifact_version,
                "num_classes": len(self.config.classes),
            },
        )
        # Especificação 6c (Seção 2.2): inserir o engine_version retornado por info() no artifact_version do envelope
        self.effective_artifact_version = info.engine_version

    def _switch_session(self, frame: Frame) -> None:
        """
        RNF 4 / T6: sequence é contado por producer + session_id a partir de 0 (Especificação 1.0, seção 3).
        P4 e P6 geram session_id novo em todo START, SEEK, LOOP, RESTART e RECONNECT (ods.session.started),
        e o session_id viaja em cada quadro: a troca de sessão é detectada no próprio quadro e só ela zera sequence.
        """
        previous = self.active_session_id
        if previous is not None:
            self.ended_session_ids[previous] = None
            self.ended_session_ids.move_to_end(previous)
            if len(self.ended_session_ids) > ENDED_SESSION_CACHE_SIZE:
                self.ended_session_ids.popitem(last=False)
        self.active_session_id = frame.session_id
        self.sequence = 0
        self.last_frame_index = None
        self._log_observability(
            logging.INFO,
            {
                "event": "session_changed",
                "source_id": frame.source_id,
                "session_id": frame.session_id,
                "previous_session_id": previous,
                "frame_index": frame.frame_index,
                "captured_at": frame.captured_at,
            },
        )

    def process_single_frame(self, frame: Frame) -> dict[str, Any] | None:
        if frame.session_id in self.ended_session_ids:
            # Quadro atrasado de sessão já encerrada: publicá-lo reabriria a sessão e repetiria (session_id, sequence)
            self._record_frame_drop(
                frame, None, "session", ValueError("quadro de sessão encerrada"), code="STALE_SESSION"
            )
            return None
        if frame.session_id != self.active_session_id:
            self._switch_session(frame)
        elif self.last_frame_index is not None and frame.frame_index <= self.last_frame_index:
            current_seq = self.sequence
            self.sequence += 1
            self._record_frame_drop(
                frame,
                current_seq,
                "session",
                ValueError(
                    f"frame_index não cresceu dentro da sessão {frame.session_id}: "
                    f"{frame.frame_index} <= {self.last_frame_index}"
                ),
                code="SESSION_PROTOCOL_VIOLATION",
            )
            return None

        current_seq = self.sequence
        self.sequence += 1  # Incrementa sempre para que descartes gerem lacuna implícita em sequence (RNF 5 / T7)
        self.last_frame_index = frame.frame_index

        # Diretriz RNF 5 / T7: qualquer falha em qualquer estágio descarta o quadro sem publicar detecção falsa,
        # registra o erro na observabilidade B2 e sinaliza a perda pela lacuna implícita em sequence.
        stage = "preprocess"
        try:
            bgr = preprocess_frame_to_bgr(frame)

            stage = "infer"
            raw_dets, hw_metrics = self.inference_client.run_inference(bgr)
            if hw_metrics.gpu_memory_mb > self.config.max_gpu_memory_mb:
                logger.warning(
                    f"[B2-OBS] Consumo de VRAM ({hw_metrics.gpu_memory_mb} MB) excedeu orçamento ({self.config.max_gpu_memory_mb} MB)"
                )

            stage = "postprocess"
            filtered = filter_by_confidence(raw_dets, self.config.conf_threshold)
            nms_dets = apply_nms(filtered, self.config.nms_iou_threshold)
            final_dets = finalize_detections(
                detections=nms_dets,
                class_map=self.config.class_map,
                image_width=frame.image_width,
                image_height=frame.image_height,
                enabled_classes=self.config.enabled_classes,
            )

            stage = "publish"
            envelope = build_envelope(
                frame=frame,
                sequence=current_seq,
                artifact_version=self.effective_artifact_version,
                detections=final_dets,
                published_at=self.clock(),
                message_id=self.message_id_factory(),
            )
            published_dict = self.publisher.publish(envelope)
        except Exception as err:
            self._record_frame_drop(frame, current_seq, stage, err)
            return None

        self.telemetry_history.append(
            {
                "sequence": current_seq,
                "frame_index": frame.frame_index,
                "infer_latency_ms": hw_metrics.latency_ms,
                "e2e_latency_ms": compute_e2e_latency_ms(envelope.captured_at, envelope.published_at),
                "gpu_memory_mb": hw_metrics.gpu_memory_mb,
                "num_detections": len(final_dets),
            }
        )
        self.telemetry_count += 1
        self.total_infer_latency_ms += hw_metrics.latency_ms
        return published_dict

    def run(self) -> None:
        try:
            self.initialize()
        except P2ExecutionError as err:
            # Falha de carregamento é fatal: nenhum quadro é consumido nem publicado (T7)
            self.error_counts[err.code] = self.error_counts.get(err.code, 0) + 1
            self._log_observability(
                logging.CRITICAL,
                {
                    "event": "engine_load_failed",
                    "error_code": err.code,
                    "engine_ref": self.config.engine_ref,
                    "artifact_version": self.config.artifact_version,
                    "message": str(err),
                },
            )
            raise
        try:
            for frame in self.source.frames():
                self.process_single_frame(frame)
        finally:
            self.inference_client.close()