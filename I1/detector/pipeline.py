"""
pipeline.py — Orquestrador do componente I1 (Arquitetura 6c).
"""

from __future__ import annotations

import logging
from typing import Any

from .config import AppConfig
from .contracts import Frame, P2ExecutionError
from . inference import P2InferenceClient
from .ingestion import FrameSource, preprocess_frame_to_bgr
from .postprocessing import apply_nms, filter_by_confidence, finalize_detections
from .publishing import EventPublisher, build_envelope, compute_e2e_latency_ms

logger = logging.getLogger("ods.i1.observability")


class DetectorPipeline:
    """Orquestrador único do componente I1 ('Um artefato por aplicação, um componente só')."""

    def __init__(
        self,
        config: AppConfig,
        source: FrameSource,
        publisher: EventPublisher,
        inference_client: P2InferenceClient | None = None,
    ) -> None:
        self.config = config
        self.source = source
        self.publisher = publisher
        self.inference_client = inference_client or P2InferenceClient()

        self.active_session_id: str = config.session_id
        self.sequence: int = 0
        self.effective_artifact_version: str = config.artifact_version
        self.telemetry_history: list[dict[str, Any]] = []

    def initialize(self) -> None:
        """Carrega o motor via P2.load() e obtém engine_version via P2.info() (T11)."""
        info = self.inference_client.load_engine(
            engine_ref=self.config.engine_ref,
            options={
                "artifact_id": self.config.artifact_id,
                "artifact_version": self.config.artifact_version,
            },
        )
        # Especificação 6c (Seção 2.2): inserir o engine_version retornado por info() no artifact_version do envelope
        self.effective_artifact_version = info.engine_version

    def handle_session_control_event(self, action: str, new_session_id: str | None = None) -> None:
        """
        RNF 4 / T6: Em eventos de START, SEEK, LOOP ou RESTART anunciados por ods.session.started,
        atualiza o session_id ativo e reinicia sequence em 0.
        """
        action_upper = action.upper()
        if action_upper in {"START", "SEEK", "LOOP", "RESTART"}:
            if new_session_id:
                self.active_session_id = new_session_id
            self.sequence = 0
            logger.info(f"[B2-OBS] Sessão reiniciada ({action_upper}): session_id={self.active_session_id}, sequence=0")

    def process_single_frame(self, frame: Frame) -> dict[str, Any] | None:
        # Verifica mudança automática de session_id vinda nos metadados de P4/P6 (RNF 4)
        if frame.session_id != self.active_session_id:
            self.handle_session_control_event("START", new_session_id=frame.session_id)

        current_seq = self.sequence
        self.sequence += 1  # Incrementa sempre para que descartes gerem lacuna implícita em sequence (RNF 5 / T7)

        try:
            bgr = preprocess_frame_to_bgr(frame)
            raw_dets, hw_metrics = self.inference_client.run_inference(bgr)
        except (ValueError, P2ExecutionError) as err:
            # Diretriz RNF 5 / T7: Não publicar detecção falsa; registrar falha na observabilidade B2
            # e sinalizar perda pela lacuna implícita em sequence.
            logger.error(
                f"[B2-OBS] Falha no quadro frame_index={frame.frame_index} (sequence={current_seq} suprimido): {err}"
            )
            return None

        if hw_metrics.gpu_memory_mb > self.config.max_gpu_memory_mb:
            logger.warning(
                f"[B2-OBS] Consumo de VRAM ({hw_metrics.gpu_memory_mb} MB) excedeu orçamento ({self.config.max_gpu_memory_mb} MB)"
            )

        filtered = filter_by_confidence(raw_dets, self.config.conf_threshold)
        nms_dets = apply_nms(filtered, self.config.nms_iou_threshold)
        final_dets = finalize_detections(
            detections=nms_dets,
            class_map=self.config.class_map,
            image_width=frame.image_width,
            image_height=frame.image_height,
            enabled_classes=self.config.enabled_classes,
        )

        envelope = build_envelope(
            frame=frame,
            sequence=current_seq,
            artifact_version=self.effective_artifact_version,
            detections=final_dets,
        )

        published_dict = self.publisher.publish(envelope)
        e2e_latency_ms = compute_e2e_latency_ms(envelope.captured_at, envelope.published_at)

        self.telemetry_history.append(
            {
                "sequence": current_seq,
                "frame_index": frame.frame_index,
                "infer_latency_ms": hw_metrics.latency_ms,
                "e2e_latency_ms": e2e_latency_ms,
                "gpu_memory_mb": hw_metrics.gpu_memory_mb,
                "num_detections": len(final_dets),
            }
        )
        return published_dict

    def run(self) -> None:
        self.initialize()
        try:
            for frame in self.source.frames():
                self.process_single_frame(frame)
        finally:
            self.inference_client.close()