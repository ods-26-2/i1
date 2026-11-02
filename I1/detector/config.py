"""
config.py — Configuração da build da aplicação para o componente I1 (P3 / AppConfig).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import yaml


@dataclass(frozen=True)
class AppConfig:
    artifact_id: str
    artifact_version: str
    engine_ref: str  # Caminho para o .onnx/.engine ou URI do pacote B5
    classes: list[str]
    conf_threshold: float = 0.50
    nms_iou_threshold: float = 0.45
    enabled_classes: list[str] | None = None  # Filtro estático opcional de classes na build
    source_type: str = "mock"                 # "p4" ou "p6" ou "mock"
    source_uri: str | int = 0
    source_id: str = "cam0"
    session_id: str = "00000000-0000-4000-8000-000000000001"
    pixel_format: str = "BGR"
    broker_bootstrap_servers: str = field(
        default_factory=lambda: os.getenv("ODS_KAFKA_BROKER", "34.66.235.91:9093")
    )
    topic: str = "ods.i1.detections"
    max_gpu_memory_mb: float = 2048.0         # Orçamento máximo alocado por SIS-1 (RNF 1)

    @property
    def class_map(self) -> dict[int, str]:
        return {idx: name for idx, name in enumerate(self.classes)}

    @classmethod
    def from_yaml(cls, path: str | Path) -> AppConfig:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        # Se apontar para um metadata.json do pacote B5, carrega o array 'classes' oficial automaticamente
        classes = raw.get("classes")
        metadata_path = raw.get("b5_metadata_path")
        if not classes and metadata_path:
            meta_file = Path(metadata_path)
            if meta_file.exists():
                with open(meta_file, "r", encoding="utf-8") as mf:
                    meta_json = json.load(mf)
                classes = meta_json.get("classes", [])

        if isinstance(classes, dict):
            classes = [str(classes[k]) for k in sorted(classes.keys(), key=int)]
        elif not isinstance(classes, list) or not classes:
            raise ValueError("AppConfig requer uma lista 'classes' ou um 'b5_metadata_path' válido.")

        return cls(
            artifact_id=str(raw["artifact_id"]),
            artifact_version=str(raw["artifact_version"]),
            engine_ref=str(raw.get("engine_ref", f"I1/artifacts/{raw['artifact_id']}/{raw['artifact_version']}/{raw['artifact_id']}.onnx")),
            classes=[str(c) for c in classes],
            conf_threshold=float(raw.get("conf_threshold", 0.50)),
            nms_iou_threshold=float(raw.get("nms_iou_threshold", 0.45)),
            enabled_classes=raw.get("enabled_classes"),
            source_type=str(raw.get("source_type", "mock")),
            source_uri=raw.get("source_uri", 0),
            source_id=str(raw.get("source_id", "cam0")),
            session_id=str(raw.get("session_id", "00000000-0000-4000-8000-000000000001")),
            pixel_format=str(raw.get("pixel_format", "BGR")),
            broker_bootstrap_servers=str(
                raw.get("broker_bootstrap_servers", os.getenv("ODS_KAFKA_BROKER", "34.66.235.91:9093"))
            ),
            topic=str(raw.get("topic", "ods.i1.detections")),
            max_gpu_memory_mb=float(raw.get("max_gpu_memory_mb", 2048.0)),
        )