"""
config.py — carrega AppConfig de um arquivo YAML de configuração de aplicação.

Formato esperado (ver configs/exemplo.yaml):

class_map:
  0: pessoa
conf_threshold: 0.5
artifact_id: exemplo-app
artifact_version: "0.1.0"
nms_iou_threshold: 0.45
session_id: local-dev
camera_id: cam-0
source: null  # caminho do vídeo, ou omitido/null para usar a webcam (índice 0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from .contracts import AppConfig


def load_config(path: Union[str, Path]) -> AppConfig:
    """Lê um YAML de configuração de aplicação e devolve um AppConfig já validado."""

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    class_map = {int(k): v for k, v in raw.get("class_map", {}).items()}

    return AppConfig(
        class_map=class_map,
        conf_threshold=float(raw["conf_threshold"]),
        artifact_id=raw["artifact_id"],
        artifact_version=str(raw["artifact_version"]),
        nms_iou_threshold=float(raw.get("nms_iou_threshold", 0.45)),
        session_id=raw.get("session_id", "local-dev"),
        camera_id=raw.get("camera_id", "cam-0"),
        source=raw.get("source"),
    )