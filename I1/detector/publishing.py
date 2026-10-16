"""
publishing.py — Módulo 4 do pipeline de I1.

build_event monta o Envelope padrão do barramento (schema, schema_version,
message_type, producer, producer_version, published_at) em torno do
DetectionPayload específico de I1. Publisher é a interface mínima para entregar
esse envelope; hoje só existe MockPublisher.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Protocol

from .contracts import AppConfig, Detection, DetectionPayload, Envelope, Frame

# Identidade do componente — nome fixo, versão do código de I1 (não confundir
# com artifact_version, que é a versão do modelo carregado).
COMPONENT_NAME = "I1"
COMPONENT_VERSION = "0.1.0"

# Valor combinado com o time de contratos para este componente.
SCHEMA_NAME = "ods.inferencia.deteccao_frame"
SCHEMA_VERSION = "1.0"


def build_event(
    frame: Frame,
    detections: List[Detection],
    config: AppConfig,
) -> Envelope:
    """Monta o envelope completo a partir do quadro, das detecções tratadas e da configuração da aplicação."""

    payload = DetectionPayload(
        camera_id=frame.camera_id,
        session_id=config.session_id,
        sequence_number=frame.sequence_number,
        captured_at=frame.captured_at,
        artifact_id=config.artifact_id,
        artifact_version=config.artifact_version,
        detections=detections,
    )

    return Envelope(
        schema=SCHEMA_NAME,
        schema_version=SCHEMA_VERSION,
        message_type="event",
        producer=COMPONENT_NAME,
        producer_version=COMPONENT_VERSION,
        published_at=time.time(),
        payload=payload,
    )


def event_to_wire_format(envelope: Envelope) -> Dict[str, Any]:
    """Serializa o Envelope para o formato JSON exato exigido pelo barramento."""

    return {
        "schema": envelope.schema,
        "schema_version": envelope.schema_version,
        "message_type": envelope.message_type,
        "producer": envelope.producer,
        "producer_version": envelope.producer_version,
        "published_at": envelope.published_at,
        "payload": {
            "camera_id": envelope.payload.camera_id,
            "session_id": envelope.payload.session_id,
            "sequence_number": envelope.payload.sequence_number,
            "captured_at": envelope.payload.captured_at,
            "artifact_id": envelope.payload.artifact_id,
            "artifact_version": envelope.payload.artifact_version,
            "detections": [
                {
                    "bbox": detection.bbox,
                    "class": detection.class_name,
                    "confidence": detection.confidence,
                }
                for detection in envelope.payload.detections
            ],
        },
    }


class Publisher(Protocol):
    def publish(self, envelope: Envelope) -> None:
        ...


class MockPublisher:
    def publish(self, envelope: Envelope) -> None:
        wire_envelope = event_to_wire_format(envelope)
        print(json.dumps(wire_envelope, ensure_ascii=False))