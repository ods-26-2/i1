"""
postprocessing.py — Pós-processamento definitivo de I1 (T10):
  1. Filtro de confiança configurável
  2. NMS agrupado por classe
  3. Mapeamento para o vocabulário de classes do artefato MODEL (+ filtro estático opcional)
  4. Conversão de bbox para inteiros [x_min, y_min, x_max, y_max] no quadro bruto e detection_id >= 0
"""

from __future__ import annotations

from collections import defaultdict
from .contracts import Detection, RawDetection


def filter_by_confidence(detections: list[RawDetection], threshold: float) -> list[RawDetection]:
    return [d for d in detections if d.confidence >= threshold]


def _compute_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0.0 else 0.0


def apply_nms(detections: list[RawDetection], iou_threshold: float) -> list[RawDetection]:
    """Aplica Non-Maximum Suppression isoladamente para cada class_id."""
    by_class: dict[int, list[RawDetection]] = defaultdict(list)
    for det in detections:
        by_class[det.class_id].append(det)

    kept: list[RawDetection] = []
    for _, class_dets in by_class.items():
        sorted_dets = sorted(class_dets, key=lambda d: d.confidence, reverse=True)
        while sorted_dets:
            best = sorted_dets.pop(0)
            kept.append(best)
            sorted_dets = [
                d for d in sorted_dets if _compute_iou(best.bbox, d.bbox) < iou_threshold
            ]

    # Ordena por confiança decrescente para atribuição determinística de detection_id
    kept.sort(key=lambda d: d.confidence, reverse=True)
    return kept


def finalize_detections(
    detections: list[RawDetection],
    class_map: dict[int, str],
    image_width: int,
    image_height: int,
    enabled_classes: list[str] | None = None,
) -> list[Detection]:
    """
    Traduz class_id para string oficial, converte bbox para inteiros [x_min, y_min, x_max, y_max]
    dentro dos limites do quadro bruto e atribui detection_id sequencial (0, 1, 2...).
    """
    allowed = set(enabled_classes) if enabled_classes else None
    final_list: list[Detection] = []
    det_id = 0

    for raw in detections:
        if raw.class_id not in class_map:
            continue
        cls_name = class_map[raw.class_id]
        if allowed is not None and cls_name not in allowed:
            continue

        x_min = max(0, min(image_width, int(round(raw.bbox[0]))))
        y_min = max(0, min(image_height, int(round(raw.bbox[1]))))
        x_max = max(0, min(image_width, int(round(raw.bbox[2]))))
        y_max = max(0, min(image_height, int(round(raw.bbox[3]))))

        if x_max <= x_min or y_max <= y_min:
            continue

        final_list.append(
            Detection(
                detection_id=det_id,
                class_name=cls_name,
                confidence=raw.confidence,
                bbox=[x_min, y_min, x_max, y_max],
            )
        )
        det_id += 1

    return final_list