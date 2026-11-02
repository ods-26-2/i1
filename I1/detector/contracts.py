"""
contracts.py — Contratos de dados do componente I1 (Especificação 1.0 - Arquitetura 6c).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
import uuid
import numpy as np

PixelFormat = Literal["NV12", "RGB", "BGR", "GRAY"]

P2ErrorCode = Literal[
    "ENGINE_NOT_FOUND",
    "INCOMPATIBLE_VERSION",
    "OUT_OF_MEMORY",
    "INVALID_INPUT_SHAPE",
    "RUNTIME_ERROR",
]


class P2ExecutionError(RuntimeError):
    """Exceção tipada para os códigos de erro formais da biblioteca P2 (T11)."""

    def __init__(self, code: P2ErrorCode, message: str = "") -> None:
        self.code: P2ErrorCode = code
        super().__init__(f"[{code}] {message}" if message else f"[{code}]")


def utc_now_iso_ms() -> str:
    """Formata timestamp atual em ISO 8601 UTC com milissegundos (ex: 2026-09-23T14:30:00.120Z)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass(frozen=True)
class Frame:
    """Representa um quadro bruto e seus metadados sob o contrato ods.p4.frame (P4/P6)."""
    image: np.ndarray
    captured_at: str
    source_id: str
    session_id: str
    frame_index: int
    image_width: int
    image_height: int
    pixel_format: PixelFormat = "BGR"
    fps: float = 30.0
    raw: bool = True
    corrupted: bool = False


@dataclass(frozen=True)
class RawDetection:
    """Saída bruta do motor de inferência antes do pós-processamento."""
    bbox: tuple[float, float, float, float]  # (x_min, y_min, x_max, y_max) na escala do quadro bruto
    class_id: int
    confidence: float


@dataclass(frozen=True)
class InferenceMetrics:
    """Métricas de custo de hardware retornadas a cada chamada de infer() em P2."""
    latency_ms: float
    gpu_memory_mb: float


@dataclass(frozen=True)
class EngineInfo:
    """Metadados retornados por info(handle) da biblioteca P2."""
    engine_version: str
    input_shape: list[int]
    output_shape: list[int]
    precision: str
    max_batch: int = 1


@dataclass(frozen=True)
class Detection:
    """Detecção pós-processada conforme especificação ods.i1.detections 1.0."""
    detection_id: int
    class_name: str
    confidence: float
    bbox: list[int]  # [x_min, y_min, x_max, y_max] em pixels inteiros do quadro bruto

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "class": self.class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [int(coord) for coord in self.bbox],
        }


@dataclass(frozen=True)
class DetectionPayload:
    """Payload enxuto de ods.i1.detections 1.0 (sem redundância de metadados do envelope)."""
    frame_index: int
    image_width: int
    image_height: int
    detections: list[Detection] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "image_width": int(self.image_width),
            "image_height": int(self.image_height),
            "detections": [d.to_dict() for d in self.detections],
        }


@dataclass(frozen=True)
class EvSinalEnvelope:
    """
    Perfil de envelope EV-SINAL (11 campos obrigatórios + payload) para ods.i1.detections 1.0.
    Omitidos target e correlation_id quando não aplicáveis.
    """
    captured_at: str
    source_id: str
    session_id: str
    sequence: int
    artifact_version: str
    payload: DetectionPayload
    message_type: str = "event"
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    schema: str = "ods.i1.detections"
    schema_version: str = "1.0"
    producer: str = "I1"
    published_at: str = field(default_factory=utc_now_iso_ms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_type": self.message_type,
            "message_id": self.message_id,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "producer": self.producer,
            "published_at": self.published_at,
            "captured_at": self.captured_at,
            "source_id": self.source_id,
            "session_id": self.session_id,
            "sequence": int(self.sequence),
            "artifact_version": self.artifact_version,
            "payload": self.payload.to_dict(),
        }