"""
publishing.py — Validação de schema B1 acoplada, cálculo de latência e publicação EV-SINAL (T2, T12).
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
import json
from typing import Any, Callable, Protocol

from .contracts import Detection, DetectionPayload, EvSinalEnvelope, Frame, new_message_id, utc_now_iso_ms
from .schema_validation import schema_errors

DEFAULT_MESSAGE_HISTORY_LIMIT = 1000
DEFAULT_DEDUPLICATION_CACHE_SIZE = 65536


def validate_ods_i1_detections_schema(msg_dict: dict[str, Any]) -> None:
    """
    Validador JSON Schema incorporado ao publicador (RNF 3 / T2 / T12).
    Valida contra schemas/ods.i1.detections.v1.0.schema.json e rejeita qualquer mensagem fora do contrato,
    inclusive campos legados (camera_id, sequence_number, artifact_id) no envelope ou no payload.
    """
    errors = schema_errors(msg_dict)
    if errors:
        raise ValueError("Mensagem viola ods.i1.detections 1.0: " + "; ".join(errors))

    payload = msg_dict["payload"]
    width = payload["image_width"]
    height = payload["image_height"]
    for expected_id, detection in enumerate(payload["detections"]):
        if detection["detection_id"] != expected_id:
            raise ValueError("detection_id deve ser sequencial a partir de zero.")
        x_min, y_min, x_max, y_max = detection["bbox"]
        if x_max <= x_min or y_max <= y_min or x_max > width or y_max > height:
            raise ValueError("bbox deve ter área positiva e permanecer dentro dos limites da imagem.")


def compute_e2e_latency_ms(captured_at_iso: str, published_at_iso: str) -> float:
    """Calcula a latência ponta a ponta (published_at - captured_at) em milissegundos (RNF 5)."""
    cap_dt = datetime.fromisoformat(captured_at_iso.replace("Z", "+00:00"))
    pub_dt = datetime.fromisoformat(published_at_iso.replace("Z", "+00:00"))
    return round((pub_dt - cap_dt).total_seconds() * 1000.0, 3)


def build_envelope(
    frame: Frame,
    sequence: int,
    artifact_version: str,
    detections: list[Detection],
    published_at: str | None = None,
    message_id: str | None = None,
) -> EvSinalEnvelope:
    payload = DetectionPayload(
        frame_index=frame.frame_index,
        image_width=frame.image_width,
        image_height=frame.image_height,
        detections=detections,
    )
    return EvSinalEnvelope(
        captured_at=frame.captured_at,
        source_id=frame.source_id,
        session_id=frame.session_id,
        sequence=sequence,
        artifact_version=artifact_version,
        payload=payload,
        published_at=published_at or utc_now_iso_ms(),
        message_id=message_id or new_message_id(),
    )


class EventPublisher(Protocol):
    def publish(self, envelope: EvSinalEnvelope) -> dict[str, Any]:
        ...


class ValidatedStdoutPublisher:
    """Publicador para Degrau A e testes locais com validador B1 e deduplicação acoplados."""

    def __init__(
        self,
        verbose: bool = True,
        history_limit: int = DEFAULT_MESSAGE_HISTORY_LIMIT,
        deduplication_cache_size: int = DEFAULT_DEDUPLICATION_CACHE_SIZE,
        on_publish: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if history_limit < 0 or deduplication_cache_size <= 0:
            raise ValueError("history_limit deve ser >= 0 e deduplication_cache_size deve ser > 0.")
        self.verbose = verbose
        self.history_limit = history_limit
        self.deduplication_cache_size = deduplication_cache_size
        self.on_publish = on_publish
        self._seen_message_ids: OrderedDict[str, None] = OrderedDict()
        self.published_messages: list[dict[str, Any]] = []
        self.published_count = 0
        self.detection_count = 0

    def _record_published_message(self, msg_dict: dict[str, Any]) -> None:
        if self.on_publish is not None:
            self.on_publish(msg_dict)
        message_id = msg_dict["message_id"]
        self._seen_message_ids[message_id] = None
        self._seen_message_ids.move_to_end(message_id)
        if len(self._seen_message_ids) > self.deduplication_cache_size:
            self._seen_message_ids.popitem(last=False)

        self.published_count += 1
        self.detection_count += len(msg_dict["payload"]["detections"])
        if self.history_limit:
            self.published_messages.append(msg_dict)
            if len(self.published_messages) > self.history_limit:
                del self.published_messages[: len(self.published_messages) - self.history_limit]

    def publish(self, envelope: EvSinalEnvelope) -> dict[str, Any]:
        if envelope.message_id in self._seen_message_ids:
            raise ValueError(f"Mensagem duplicada bloqueada no publicador: {envelope.message_id}")

        msg_dict = envelope.to_dict()
        validate_ods_i1_detections_schema(msg_dict)

        if self.verbose:
            print(json.dumps(msg_dict, ensure_ascii=False))
        self._record_published_message(msg_dict)
        return msg_dict

class ConfluentKafkaPublisher:
    """Publicador Kafka parametrizado (RNF 3 / B3) com validação pré-envio."""

    def __init__(
        self,
        bootstrap_servers: str | None,
        topic: str = "ods.i1.detections",
        history_limit: int = DEFAULT_MESSAGE_HISTORY_LIMIT,
        deduplication_cache_size: int = DEFAULT_DEDUPLICATION_CACHE_SIZE,
        on_publish: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not bootstrap_servers:
            raise ValueError(
                "Broker Kafka não configurado: defina ODS_KAFKA_BROKER (ou broker_bootstrap_servers na build). "
                "Não há endereço padrão, para que testes e CI nunca publiquem no broker da nuvem por engano."
            )
        if history_limit < 0 or deduplication_cache_size <= 0:
            raise ValueError("history_limit deve ser >= 0 e deduplication_cache_size deve ser > 0.")
        from confluent_kafka import Producer
        self.topic = topic
        self.history_limit = history_limit
        self.deduplication_cache_size = deduplication_cache_size
        self.on_publish = on_publish
        self._seen_message_ids: OrderedDict[str, None] = OrderedDict()
        conf = {
            "bootstrap.servers": bootstrap_servers,
            "enable.idempotence": True,
            "message.timeout.ms": 5000,
            "socket.timeout.ms": 3000,
        }
        self.producer = Producer(conf)
        self.published_messages: list[dict[str, Any]] = []
        self.published_count = 0
        self.detection_count = 0
        self.delivery_errors = 0

    def _on_delivery(self, err: Any, _msg: Any) -> None:
        if err is not None:
            self.delivery_errors += 1

    def _record_published_message(self, msg_dict: dict[str, Any]) -> None:
        if self.on_publish is not None:
            self.on_publish(msg_dict)
        message_id = msg_dict["message_id"]
        self._seen_message_ids[message_id] = None
        self._seen_message_ids.move_to_end(message_id)
        if len(self._seen_message_ids) > self.deduplication_cache_size:
            self._seen_message_ids.popitem(last=False)

        self.published_count += 1
        self.detection_count += len(msg_dict["payload"]["detections"])
        if self.history_limit:
            self.published_messages.append(msg_dict)
            if len(self.published_messages) > self.history_limit:
                del self.published_messages[: len(self.published_messages) - self.history_limit]

    def publish(self, envelope: EvSinalEnvelope) -> dict[str, Any]:
        if envelope.message_id in self._seen_message_ids:
            raise ValueError(f"Duplicata bloqueada: {envelope.message_id}")

        msg_dict = envelope.to_dict()
        validate_ods_i1_detections_schema(msg_dict)

        payload_bytes = json.dumps(msg_dict, ensure_ascii=False).encode("utf-8")
        try:
            self.producer.produce(
                self.topic,
                key=envelope.source_id.encode("utf-8"),
                value=payload_bytes,
                on_delivery=self._on_delivery,
            )
            self.producer.poll(0)
        except BufferError:
            self.producer.flush(1.0)
            self.producer.produce(
                self.topic,
                key=envelope.source_id.encode("utf-8"),
                value=payload_bytes,
                on_delivery=self._on_delivery,
            )
        self._record_published_message(msg_dict)
        return msg_dict

    def flush(self, timeout: float = 3.0) -> int:
        remaining = self.producer.flush(timeout)
        return remaining + self.delivery_errors