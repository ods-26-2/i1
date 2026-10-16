"""
contracts.py — vocabulário compartilhado entre os módulos do pipeline de detecção (I1).

Estas estruturas atravessam todo o fluxo (ingestion -> inference -> postprocessing ->
publishing). Se o schema do evento de saída mudar, é este arquivo que se edita — não
os módulos que o consomem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class Frame:
    """Um quadro de vídeo já com os metadados exigidos pelo contrato de entrada (P4 -> I1)."""

    image: Any  # array de pixels (ex.: numpy.ndarray no formato BGR do OpenCV)
    camera_id: str
    captured_at: float  # timestamp de captura, não de recebimento
    sequence_number: int
    calibration_applied: bool = False
    calibration_profile_version: Optional[str] = None

@dataclass
class Detection:
    """Uma detecção única dentro de um quadro, já tratada por postprocessing.py."""

    bbox: List[float]  # [x1, y1, x2, y2] em coordenadas de pixel do frame
    class_id: int
    confidence: float
    class_name: Optional[str] = None  # preenchido por postprocessing.map_classes

@dataclass
class DetectionPayload:
    """Payload específico de I1 — o que efetivamente entra em `payload` no envelope."""

    camera_id: str
    session_id: str
    sequence_number: int
    captured_at: float
    artifact_id: str
    artifact_version: str
    detections: List[Detection] = field(default_factory=list)

@dataclass
class Envelope:
    """
    Envelope padrão do barramento, igual para todos os componentes do sistema
    (definido pelo time de contratos/TRA-1). payload carrega o conteúdo
    específico do componente produtor.
    """

    schema: str
    schema_version: str
    message_type: str
    producer: str
    producer_version: str
    published_at: float
    payload: DetectionPayload

@dataclass
class AppConfig:
    """Configuração de uma aplicação específica, carregada de configs/*.yaml."""

    class_map: Dict[int, str]
    conf_threshold: float
    artifact_id: str
    artifact_version: str
    nms_iou_threshold: float = 0.45
    session_id: str = "local-dev"
    camera_id: str = "cam-0"
    source: Optional[str] = None  # caminho do vídeo/webcam, usado pelo MockFrameSource