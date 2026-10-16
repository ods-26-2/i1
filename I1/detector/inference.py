"""
inference.py — Módulo 2 do pipeline de I1.

Hoje é um stub: carrega os pesos genéricos do YOLOv8n localmente e roda inferência
direta com a biblioteca ultralytics. Quando P2 publicar a interface genérica de
carregamento/execução na Jetson, só o corpo de load_engine e run muda — a
assinatura, que é o que pipeline.py usa, fica igual.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from ultralytics import YOLO

from .contracts import Frame

# bbox [x1, y1, x2, y2], class_id, confidence — ainda sem NMS/filtro/nome de
# classe, isso é responsabilidade de postprocessing.py
RawDetection = Tuple[List[float], int, float]


@dataclass
class Engine:
    """Envelope simples em torno do modelo carregado, para não vazar o tipo da ultralytics."""

    model: YOLO
    artifact_id: str
    artifact_version: str


def load_engine(
    artifact_id: str,
    artifact_version: str,
    weights_path: str = "models/yolov8n.pt",
    # weights_path: str = "yolov8n.pt",
) -> Engine:
    """
    Carrega os pesos do modelo de detecção.

    Por enquanto sempre lê o arquivo local genérico (models/yolov8n.pt), ignorando
    artifact_id/artifact_version na prática. Quando o artefato real de B5 e a API
    de execução de P2 estiverem prontos, esta função passa a resolver o artefato
    por artifact_id/artifact_version e delegar o carregamento/execução a P2 — sem
    mudar a assinatura usada pelo resto do pipeline.
    """

    model = YOLO(weights_path)
    return Engine(model=model, artifact_id=artifact_id, artifact_version=artifact_version)


def run(engine: Engine, frame: Frame) -> List[RawDetection]:
    """Roda a inferência sobre um único quadro e devolve detecções cruas (sem pós-processamento)."""

    results = engine.model(frame.image, verbose=False)[0]

    raw_detections: List[RawDetection] = []
    for box in results.boxes:
        bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        raw_detections.append((bbox, class_id, confidence))

    return raw_detections