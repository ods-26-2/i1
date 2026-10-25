"""
inference.py — Integração de I1 com a API de biblioteca fechada de P2 (T11).
Assinatura oficial v6c:
  - load(engine_ref, options?) -> handle
  - infer(handle, input) -> output, {latency_ms, gpu_memory_mb}
  - info(handle) -> {engine_version, input_shape, output_shape, precision, max_batch}
  - release(handle)
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Protocol
import numpy as np
import torch

from .contracts import EngineInfo, InferenceMetrics, P2ExecutionError, RawDetection


class P2LibraryProtocol(Protocol):
    """Contrato estrito da biblioteca incorporada P2 (SIS-1) na Arquitetura 6c."""
    def load(self, engine_ref: str, options: dict[str, Any] | None = None) -> Any: ...
    def infer(self, handle: Any, input_tensor: np.ndarray) -> tuple[list[RawDetection], dict[str, float]]: ...
    def info(self, handle: Any) -> dict[str, Any]: ...
    def release(self, handle: Any) -> None: ...


class UltralyticsP2Adapter:
    """
    Implementação compatível com a assinatura fechada de P2 para execução do Degrau A (sem Jetson)
    usando diretamente o pacote ONNX/PT validado por B5. Quando SIS-1 entregar o módulo nativo P2,
    basta injetá-lo no lugar deste adaptador sem alterar uma linha de I1.
    """

    def load(self, engine_ref: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        options = options or {}
        path = Path(engine_ref)
        if not path.exists():
            raise P2ExecutionError("ENGINE_NOT_FOUND", f"Artefato não encontrado em: {engine_ref}")

        expected_version = options.get("artifact_version", "1.0.0")
        if options.get("reject_version"):
            raise P2ExecutionError("INCOMPATIBLE_VERSION", f"Versão incompatível: {expected_version}")

        try:
            from ultralytics import YOLO
            model = YOLO(str(path), task="detect")
            # Warm-up no load() para inicializar a sessão ONNX Runtime fora do infer()
            dummy_img = np.zeros((int(options.get("imgsz", 640)), int(options.get("imgsz", 640)), 3), dtype=np.uint8)
            model.predict(source=dummy_img, verbose=False, conf=0.01)
        except MemoryError as e:
            raise P2ExecutionError("OUT_OF_MEMORY", str(e)) from e
        except Exception as e:
            raise P2ExecutionError("RUNTIME_ERROR", f"Falha ao carregar modelo: {e}") from e

        imgsz = int(options.get("imgsz", 640))
        artifact_id = options.get("artifact_id", path.stem)
        engine_version = f"{artifact_id}-{expected_version}"

        return {
            "model": model,
            "engine_version": engine_version,
            "input_shape": [1, 3, imgsz, imgsz],
            "output_shape": [1, 84, 8400],
            "precision": str(options.get("precision", "FP32")),
            "max_batch": 1,
            "released": False,
        }

    def infer(self, handle: dict[str, Any], input_bgr: np.ndarray) -> tuple[list[RawDetection], dict[str, float]]:
        if handle is None or handle.get("released", True):
            raise P2ExecutionError("RUNTIME_ERROR", "Handle inválido ou já liberado.")

        if not isinstance(input_bgr, np.ndarray) or input_bgr.ndim != 3 or input_bgr.shape[2] != 3:
            raise P2ExecutionError(
                "INVALID_INPUT_SHAPE",
                f"Esperado tensor BGR HxWx3, recebido shape={getattr(input_bgr, 'shape', None)}",
            )

        t0 = time.perf_counter()
        try:
            # Importante: conf baixo e sem filtro de classe aqui, pois o pós-processamento é dever de I1
            results = handle["model"].predict(source=input_bgr, verbose=False, conf=0.01, iou=0.99)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise P2ExecutionError("OUT_OF_MEMORY", str(e)) from e
            raise P2ExecutionError("RUNTIME_ERROR", str(e)) from e
        except Exception as e:
            raise P2ExecutionError("RUNTIME_ERROR", str(e)) from e

        latency_ms = (time.perf_counter() - t0) * 1000.0
        gpu_mem_mb = (
            float(torch.cuda.memory_allocated(0) / (1024 * 1024))
            if torch.cuda.is_available()
            else 0.0
        )

        raw_detections: list[RawDetection] = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy()

            for box, conf, cls_id in zip(xyxy, confs, cls_ids):
                raw_detections.append(
                    RawDetection(
                        bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                        class_id=int(cls_id),
                        confidence=float(conf),
                    )
                )

        return raw_detections, {"latency_ms": round(latency_ms, 3), "gpu_memory_mb": round(gpu_mem_mb, 2)}

    def info(self, handle: dict[str, Any]) -> dict[str, Any]:
        if handle is None or handle.get("released", True):
            raise P2ExecutionError("RUNTIME_ERROR", "Handle inválido ao consultar info().")
        return {
            "engine_version": handle["engine_version"],
            "input_shape": handle["input_shape"],
            "output_shape": handle["output_shape"],
            "precision": handle["precision"],
            "max_batch": handle["max_batch"],
        }

    def release(self, handle: dict[str, Any]) -> None:
        if handle is not None:
            handle["model"] = None
            handle["released"] = True
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class P2InferenceClient:
    """Cliente gerenciador da interface P2 dentro do componente I1."""

    def __init__(self, p2_backend: P2LibraryProtocol | None = None) -> None:
        self.backend: P2LibraryProtocol = p2_backend or UltralyticsP2Adapter()
        self.handle: Any = None
        self.engine_info: EngineInfo | None = None

    def load_engine(self, engine_ref: str, options: dict[str, Any] | None = None) -> EngineInfo:
        self.handle = self.backend.load(engine_ref, options)
        raw_info = self.backend.info(self.handle)
        self.engine_info = EngineInfo(
            engine_version=str(raw_info["engine_version"]),
            input_shape=list(raw_info["input_shape"]),
            output_shape=list(raw_info["output_shape"]),
            precision=str(raw_info["precision"]),
            max_batch=int(raw_info.get("max_batch", 1)),
        )
        return self.engine_info

    def run_inference(self, bgr_image: np.ndarray) -> tuple[list[RawDetection], InferenceMetrics]:
        if self.handle is None:
            raise P2ExecutionError("RUNTIME_ERROR", "Engine não carregado antes de infer().")
        detections, stats = self.backend.infer(self.handle, bgr_image)
        metrics = InferenceMetrics(
            latency_ms=float(stats.get("latency_ms", 0.0)),
            gpu_memory_mb=float(stats.get("gpu_memory_mb", 0.0)),
        )
        return detections, metrics

    def close(self) -> None:
        if self.handle is not None:
            self.backend.release(self.handle)
            self.handle = None