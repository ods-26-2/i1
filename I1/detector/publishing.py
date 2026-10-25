"""
publishing.py — Validação de schema B1 acoplada, cálculo de latência e publicação EV-SINAL (T2, T12).
"""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any, Protocol

from .contracts import Detection, DetectionPayload, EvSinalEnvelope, Frame, utc_now_iso_ms


REQUIRED_ENVELOPE_FIELDS = {
    "message_type",
    "message_id",
    "schema",
    "schema_version",
    "producer",
    "published_at",
    "captured_at",
    "source_id",
    "session_id",
    "sequence",
    "artifact_version",
    "payload",
}

REQUIRED_PAYLOAD_FIELDS = {"frame_index", "image_width", "image_height", "detections"}


def validate_ods_i1_detections_schema(msg_dict: dict[str, Any]) -> None:
    """
    Validador JSON Schema incorporado ao publicador (RNF 3 / T2 / T12).
    Rejeita qualquer mensagem que viole o contrato ods.i1.detections 1.0 ou contenha campos legados.
    """
    missing_env = REQUIRED_ENVELOPE_FIELDS - set(msg_dict.keys())
    if missing_env:
        raise ValueError(f"Envelope inválido. Campos obrigatórios ausentes: {missing_env}")

    if msg_dict["schema"] != "ods.i1.detections" or msg_dict["schema_version"] != "1.0":
        raise ValueError("Schema ou versão incompatível com ods.i1.detections 1.0")

    if msg_dict["producer"] != "I1" or msg_dict["message_type"] != "event":
        raise ValueError("Producer deve ser 'I1' e message_type deve ser 'event'")

    payload = msg_dict["payload"]
    if not isinstance(payload, dict) or set(payload.keys()) != REQUIRED_PAYLOAD_FIELDS:
        raise ValueError(f"Payload deve conter exclusivamente {REQUIRED_PAYLOAD_FIELDS}, recebido: {list(payload.keys())}")

    for det in payload["detections"]:
        if set(det.keys()) != {"detection_id", "class", "confidence", "bbox"}:
            raise ValueError(f"Item de detecção inválido: {det}")
        if not isinstance(det["detection_id"], int) or det["detection_id"] < 0:
            raise ValueError("detection_id deve ser inteiro >= 0")
        if not isinstance(det["bbox"], list) or len(det["bbox"]) != 4 or not all(isinstance(v, int) for v in det["bbox"]):
            raise ValueError("bbox deve ser array de 4 inteiros [x_min, y_min, x_max, y_max]")


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
        published_at=utc_now_iso_ms(),
    )


class EventPublisher(Protocol):
    def publish(self, envelope: EvSinalEnvelope) -> dict[str, Any]:
        ...


class ValidatedStdoutPublisher:
    """Publicador para Degrau A e testes locais com validador B1 e deduplicação acoplados."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self._seen_message_ids: set[str] = set()
        self.published_messages: list[dict[str, Any]] = []

    def publish(self, envelope: EvSinalEnvelope) -> dict[str, Any]:
        if envelope.message_id in self._seen_message_ids:
            raise ValueError(f"Mensagem duplicada bloqueada no publicador: {envelope.message_id}")

        msg_dict = envelope.to_dict()
        validate_ods_i1_detections_schema(msg_dict)

        self._seen_message_ids.add(envelope.message_id)
        self.published_messages.append(msg_dict)

        if self.verbose:
            print(json.dumps(msg_dict, ensure_ascii=False))
        return msg_dict

# Adicione/substitua em I1/detector/publishing.py
class ConfluentKafkaPublisher:
    """Publicador Kafka parametrizado (RNF 3 / B3) com validação pré-envio."""

    def __init__(self, bootstrap_servers: str, topic: str = "ods.i1.detections") -> None:
        from confluent_kafka import Producer
        self.topic = topic
        self._seen_message_ids: set[str] = set()
        conf = {
            "bootstrap.servers": bootstrap_servers,
            "enable.idempotence": True,
            "message.timeout.ms": 5000,
            "socket.timeout.ms": 3000,
        }
        self.producer = Producer(conf)
        self.published_messages: list[dict[str, Any]] = []  

    def publish(self, envelope: EvSinalEnvelope) -> dict[str, Any]:
        if envelope.message_id in self._seen_message_ids:
            raise ValueError(f"Duplicata bloqueada: {envelope.message_id}")

        msg_dict = envelope.to_dict()
        validate_ods_i1_detections_schema(msg_dict)
        self._seen_message_ids.add(envelope.message_id)
        self.published_messages.append(msg_dict)

        payload_bytes = json.dumps(msg_dict, ensure_ascii=False).encode("utf-8")
        try:
            self.producer.produce(
                self.topic,
                key=envelope.source_id.encode("utf-8"),
                value=payload_bytes
            )
            self.producer.poll(0)
        except BufferError:
            self.producer.flush(1.0)
            self.producer.produce(
                self.topic,
                key=envelope.source_id.encode("utf-8"),
                value=payload_bytes
            )
        return msg_dict

    def flush(self, timeout: float = 3.0) -> int:
        return self.producer.flush(timeout)