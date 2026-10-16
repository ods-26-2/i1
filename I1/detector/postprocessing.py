"""
postprocessing.py — Módulo 3 do pipeline de I1.

Três funções puras: recebem detecções cruas e devolvem detecções tratadas, sem
efeito colateral. Isso as torna fáceis de testar isoladamente, sem precisar rodar
um modelo de verdade — basta construir listas de entrada na mão em
tests/test_postprocessing.py.
"""

from __future__ import annotations

from typing import Dict, List

from .contracts import Detection
from .inference import RawDetection


def filter_by_confidence(raw_detections: List[RawDetection], conf_threshold: float) -> List[RawDetection]:
    """Descarta detecções com confiança abaixo do limiar definido em AppConfig."""

    return [d for d in raw_detections if d[2] >= conf_threshold]


def apply_nms(raw_detections: List[RawDetection], iou_threshold: float = 0.45) -> List[RawDetection]:
    by_class: Dict[int, List[RawDetection]] = {}
    for detection in raw_detections:
        by_class.setdefault(detection[1], []).append(detection)

    keep: List[RawDetection] = []
    for class_detections in by_class.values():
        ordered = sorted(class_detections, key=lambda d: d[2], reverse=True)
        while ordered:
            best = ordered.pop(0)
            keep.append(best)
            ordered = [d for d in ordered if _iou(best[0], d[0]) < iou_threshold]

    return keep


def map_classes(raw_detections: List[RawDetection], class_map: Dict[int, str]) -> List[Detection]:
    """Traduz o índice de classe numérico para o nome definido em AppConfig.class_map."""

    detections: List[Detection] = []
    for bbox, class_id, confidence in raw_detections:
        detections.append(
            Detection(
                bbox=bbox,
                class_id=class_id,
                confidence=confidence,
                class_name=class_map.get(class_id, f"classe_{class_id}"),
            )
        )
    return detections


def _iou(box_a: List[float], box_b: List[float]) -> float:
    """Intersection over Union entre duas caixas [x1, y1, x2, y2]."""

    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area == 0.0:
        return 0.0

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)

    return inter_area / (area_a + area_b - inter_area)