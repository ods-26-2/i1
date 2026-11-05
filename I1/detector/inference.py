"""
inference.py — Integração de I1 com a API de biblioteca fechada do P2 (T11 / Contrato P2 1.1).
Suporta:
  1. NativeP2Backend      : Biblioteca oficial 'p2' (SIS-1) na Jetson (TensorRT) ou CPU (ONNX Runtime).
  2. UltralyticsP2Adapter : Adaptador local ONNX Runtime / Ultralytics para Degrau A sem p2 instalado.
  3. MockP2Backend        : Backend determinístico para testes unitários e de contrato (T11 / T15).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Protocol

import cv2
import numpy as np

from .contracts import (
    P2EngineInfo,
    P2ExecutionError,
    P2InferenceMetrics,
    RawDetection,
)

# Códigos de erro formais da API fechada P2 (Seção 3 do Contrato P2 1.1)
VALID_P2_ERROR_CODES = frozenset({
    "ENGINE_NOT_FOUND",
    "INCOMPATIBLE_VERSION",
    "OUT_OF_MEMORY",
    "INVALID_INPUT_SHAPE",
    "RUNTIME_ERROR",
})

PAD_VALUE = 114
PIXEL_SCALE = 255.0
BOX_ROWS = 4


@dataclass(frozen=True)
class _Letterboxed:
    tensor: np.ndarray
    scale: float
    pad_x: float
    pad_y: float


def _letterbox_bgr(image_bgr: np.ndarray, size: int = 640) -> _Letterboxed:
    """Pré-processamento letterbox BGR -> Tensor float32 [1, 3, size, size] exigido pelo Contrato P2 1.1."""
    height, width = image_bgr.shape[:2]
    if height <= 0 or width <= 0 or size <= 0:
        raise P2ExecutionError("INVALID_INPUT_SHAPE", f"Dimensões inválidas: {image_bgr.shape}, size={size}")
    scale = min(size / height, size / width)
    new_width, new_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = (size - new_width) / 2.0, (size - new_height) / 2.0
    top, left = round(pad_y - 0.1), round(pad_x - 0.1)
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[top : top + new_height, left : left + new_width] = resized.astype(np.uint8)
    rgb = canvas[:, :, ::-1]
    chw = rgb.transpose(2, 0, 1)[np.newaxis]
    tensor = np.ascontiguousarray(chw, dtype=np.float32) / np.float32(PIXEL_SCALE)
    return _Letterboxed(tensor=tensor, scale=scale, pad_x=pad_x, pad_y=pad_y)


def _decode_yolo_raw_output(
    output: np.ndarray,
    boxed: _Letterboxed,
    image_size: tuple[int, int],
    min_score: float = 0.01,
    expected_num_classes: int | None = None,
) -> list[RawDetection]:
    """
    Decodifica a saída bruta [1, 4 + num_classes, N] do P2 em RawDetections nas coordenadas originais.
    A filtragem final por conf_threshold e o NMS por classe ocorrem em postprocessing.py do I1.
    """
    if (
        output.ndim != 3
        or output.shape[0] != 1
        or output.shape[1] <= BOX_ROWS
        or output.shape[2] <= output.shape[1]
        or (
            expected_num_classes is not None
            and output.shape[1] != BOX_ROWS + expected_num_classes
        )
    ):
        raise P2ExecutionError(
            "RUNTIME_ERROR",
            f"Saída P2 deve ter shape [1, 4 + num_classes, N], recebido: {output.shape}",
        )
    if not np.isfinite(output).all():
        raise P2ExecutionError("RUNTIME_ERROR", "Saída P2 contém NaN ou infinito.")

    predictions = output[0].T  # [N, 4 + num_classes]
    class_scores = predictions[:, BOX_ROWS:]
    if np.any(class_scores < 0.0) or np.any(class_scores > 1.0):
        raise P2ExecutionError("RUNTIME_ERROR", "Escores de classe P2 devem estar no intervalo [0, 1].")
    classes = class_scores.argmax(axis=1).astype(np.int64)
    scores = class_scores.max(axis=1).astype(np.float32)

    keep = scores >= min_score
    if not np.any(keep):
        return []

    centers = predictions[keep, :BOX_ROWS]
    scores = scores[keep]
    classes = classes[keep]

    half = centers[:, 2:4] / np.float32(2.0)
    boxes = np.concatenate([centers[:, :2] - half, centers[:, :2] + half], axis=1)

    height, width = image_size
    offset = np.array([boxed.pad_x, boxed.pad_y, boxed.pad_x, boxed.pad_y], dtype=np.float32)
    restored = (boxes - offset) / np.float32(boxed.scale)
    limits = np.array([width, height, width, height], dtype=np.float32)
    clipped = np.clip(restored, 0.0, limits).astype(np.float32)

    return [
        RawDetection(
            class_id=int(classes[i]),
            confidence=float(scores[i]),
            bbox=(
                float(clipped[i, 0]),
                float(clipped[i, 1]),
                float(clipped[i, 2]),
                float(clipped[i, 3]),
            ),
        )
        for i in range(len(scores))
    ]


class P2LibraryProtocol(Protocol):
    """Assinatura fechada do runtime P2 na Arquitetura 6c."""

    def load(self, engine_ref: str, options: dict[str, Any] | None = None) -> Any: ...
    def infer(self, handle: Any, input_tensor: np.ndarray) -> tuple[list[RawDetection], P2InferenceMetrics]: ...
    def info(self, handle: Any) -> P2EngineInfo: ...
    def release(self, handle: Any) -> None: ...


class NativeP2Backend:
    """
    Conector oficial para a biblioteca 'p2' (https://github.com/ods-26-2/p2) na Jetson ou CPU.
    """

    def __init__(self, precision: str | None = None) -> None:
        import p2  # Lazy import da biblioteca do SIS-1

        self._p2 = p2
        self._precision = precision
        self._input_name: str = "images"
        self._output_name: str = "output0"
        self._imgsz: int = 640
        self._num_classes: int | None = None
        self._declared_output_shape: tuple[int, ...] | None = None

    def _translate_p2_error(self, err: Exception) -> P2ExecutionError:
        code = getattr(err, "code", None)
        code_str = str(code) if code is not None else "RUNTIME_ERROR"
        if code_str not in VALID_P2_ERROR_CODES:
            code_str = "RUNTIME_ERROR"
        return P2ExecutionError(code_str, str(err))  # type: ignore[arg-type]

    def load(self, engine_ref: str, options: dict[str, Any] | None = None) -> Any:
        # Se engine_ref vier como caminho de ficheiro local, extrai "<artifact_id>-<artifact_version>"
        clean_ref = engine_ref
        if "/" in engine_ref or "\\" in engine_ref or engine_ref.endswith(".onnx"):
            opts = options or {}
            art_id = opts.get("artifact_id")
            art_ver = opts.get("artifact_version")
            if art_id and art_ver:
                clean_ref = f"{art_id}-{art_ver}"
            else:
                clean_ref = Path(engine_ref).stem

        p2_opts: dict[str, Any] | None = None
        req_prec = self._precision or (options or {}).get("precision")
        if req_prec in ("FP16", "FP32"):
            p2_opts = {"precision": req_prec}

        try:
            handle = self._p2.load(clean_ref, p2_opts)
            raw_info = self._p2.info(handle)
            ((in_name, in_shape),) = raw_info["input_shape"].items()
            ((out_name, out_shape),) = raw_info["output_shape"].items()
            self._input_name = in_name
            self._output_name = out_name
            self._declared_output_shape = tuple(int(dim) for dim in out_shape)
            configured_classes = (options or {}).get("num_classes")
            self._num_classes = int(configured_classes) if configured_classes is not None else None
            if len(in_shape) >= 3 and in_shape[2] > 0:
                self._imgsz = int(in_shape[2])
            return handle
        except Exception as err:
            raise self._translate_p2_error(err) from err

    def infer(self, handle: Any, input_tensor: np.ndarray) -> tuple[list[RawDetection], P2InferenceMetrics]:
        if input_tensor is None or input_tensor.ndim != 3 or input_tensor.shape[2] != 3:
            raise P2ExecutionError(
                "INVALID_INPUT_SHAPE",
                f"Esperado tensor BGR 3 canais (H, W, 3), recebido: { getattr(input_tensor, 'shape', None) }",
            )
        h, w = input_tensor.shape[:2]
        boxed = _letterbox_bgr(input_tensor, size=self._imgsz)
        try:
            outputs, stats = self._p2.infer(handle, {self._input_name: boxed.tensor})
            raw_out = np.asarray(outputs[self._output_name], dtype=np.float32)
        except Exception as err:
            raise self._translate_p2_error(err) from err

        if self._declared_output_shape is not None and (
            len(self._declared_output_shape) != raw_out.ndim
            or any(
                expected > 0 and expected != actual
                for expected, actual in zip(self._declared_output_shape, raw_out.shape)
            )
        ):
            raise P2ExecutionError(
                "RUNTIME_ERROR",
                f"Shape da saída diverge de P2.info(): esperado {self._declared_output_shape}, recebido {raw_out.shape}",
            )

        dets = _decode_yolo_raw_output(
            raw_out,
            boxed,
            (h, w),
            min_score=0.01,
            expected_num_classes=self._num_classes,
        )
        try:
            metrics = P2InferenceMetrics(
                latency_ms=float(stats["latency_ms"]),
                gpu_memory_mb=float(stats["gpu_memory_mb"]),
            )
        except (KeyError, TypeError, ValueError) as err:
            raise P2ExecutionError("RUNTIME_ERROR", f"Métricas P2 inválidas: {err}") from err
        if not np.isfinite([metrics.latency_ms, metrics.gpu_memory_mb]).all():
            raise P2ExecutionError("RUNTIME_ERROR", "Métricas P2 contêm NaN ou infinito.")
        if metrics.latency_ms < 0.0 or metrics.gpu_memory_mb < 0.0:
            raise P2ExecutionError("RUNTIME_ERROR", "Métricas P2 não podem ser negativas.")
        return dets, metrics

    def info(self, handle: Any) -> P2EngineInfo:
        raw = self._p2.info(handle)
        first_in = next(iter(raw["input_shape"].values()), [1, 3, 640, 640])
        first_out = next(iter(raw["output_shape"].values()), [1, 7, 8400])
        return P2EngineInfo(
            engine_version=str(raw["engine_version"]),
            input_shape=list(first_in),
            output_shape=list(first_out),
            precision=str(raw["precision"]),
            max_batch=int(raw["max_batch"]),
        )

    def release(self, handle: Any) -> None:
        self._p2.release(handle)


class MockP2Backend:
    """Backend simulado do P2 para validar o contrato e os 5 códigos de erro em testes (T11 / T15)."""

    def __init__(
        self,
        simulated_detections: list[RawDetection] | None = None,
        simulate_error_on_load: str | None = None,
        simulate_error_on_infer: str | None = None,
        precision: str = "FP16",
    ) -> None:
        self.simulated_detections = simulated_detections or []
        self.simulate_error_on_load = simulate_error_on_load
        self.simulate_error_on_infer = simulate_error_on_infer
        self.precision = precision
        self._active_handles: dict[int, dict[str, Any]] = {}
        self._counter = 0

    def load(self, engine_ref: str, options: dict[str, Any] | None = None) -> int:
        if self.simulate_error_on_load:
            raise P2ExecutionError(self.simulate_error_on_load, f"Erro simulado no load: {engine_ref}")  # type: ignore[arg-type]
        if not engine_ref or "missing" in engine_ref.lower():
            raise P2ExecutionError("ENGINE_NOT_FOUND", f"Artefato não encontrado: {engine_ref}")

        opts = options or {}
        artifact_id = opts.get("artifact_id", "app-epi")
        artifact_version = opts.get("artifact_version", "1.0.0")
        self._counter += 1
        handle = self._counter
        self._active_handles[handle] = {
            "engine_ref": engine_ref,
            "engine_version": f"{artifact_id}-{artifact_version}-{self.precision.lower()}",
            "released": False,
        }
        return handle

    def infer(self, handle: int, input_tensor: np.ndarray) -> tuple[list[RawDetection], P2InferenceMetrics]:
        if handle not in self._active_handles or self._active_handles[handle]["released"]:
            raise P2ExecutionError("RUNTIME_ERROR", "Handle inválido ou já liberado via release().")
        if self.simulate_error_on_infer:
            raise P2ExecutionError(self.simulate_error_on_infer, "Erro simulado durante infer().")  # type: ignore[arg-type]
        if input_tensor is None or input_tensor.ndim != 3 or input_tensor.shape[2] != 3:
            raise P2ExecutionError(
                "INVALID_INPUT_SHAPE",
                f"Shape inválido para inferência P2: esperado (H, W, 3), recebido {getattr(input_tensor, 'shape', None)}",
            )
        return list(self.simulated_detections), P2InferenceMetrics(latency_ms=4.2, gpu_memory_mb=312.5)

    def info(self, handle: int) -> P2EngineInfo:
        if handle not in self._active_handles:
            raise P2ExecutionError("RUNTIME_ERROR", "Handle inexistente para chamada info().")
        meta = self._active_handles[handle]
        return P2EngineInfo(
            engine_version=meta["engine_version"],
            input_shape=[1, 3, 640, 640],
            output_shape=[1, 84, 8400],
            precision=self.precision,
            max_batch=1,
        )

    def release(self, handle: int) -> None:
        if handle in self._active_handles:
            self._active_handles[handle]["released"] = True


class UltralyticsP2Adapter:
    """Adaptador real de inferência ONNX/PyTorch encapsulado sob o protocolo estrito de P2 (Degrau A - T16)."""

    def __init__(self, imgsz: int = 640) -> None:
        self.imgsz = imgsz
        self._models: dict[int, dict[str, Any]] = {}
        self._counter = 0

    def load(self, engine_ref: str, options: dict[str, Any] | None = None) -> int:
        path = Path(engine_ref)
        if not path.exists():
            raise P2ExecutionError("ENGINE_NOT_FOUND", f"Arquivo de modelo ausente: {engine_ref}")

        try:
            from ultralytics import YOLO  # Lazy import para manter a CI leve

            model = YOLO(str(path), task="detect")
        except Exception as e:
            raise P2ExecutionError("INCOMPATIBLE_VERSION", f"Falha ao carregar modelo '{engine_ref}': {e}") from e

        opts = options or {}
        art_id = opts.get("artifact_id", path.stem)
        art_ver = opts.get("artifact_version", "1.0.0")

        self._counter += 1
        handle = self._counter
        self._models[handle] = {
            "model": model,
            "engine_version": f"{art_id}-{art_ver}",
            "precision": opts.get("precision", "FP32"),
        }
        return handle

    def infer(self, handle: int, input_tensor: np.ndarray) -> tuple[list[RawDetection], P2InferenceMetrics]:
        if handle not in self._models:
            raise P2ExecutionError("RUNTIME_ERROR", "Handle P2 não inicializado.")
        if input_tensor is None or input_tensor.ndim != 3 or input_tensor.shape[2] != 3:
            raise P2ExecutionError("INVALID_INPUT_SHAPE", "Tensor de entrada deve ser BGR (H, W, 3).")

        model = self._models[handle]["model"]
        t0 = time.perf_counter()
        try:
            results = model.predict(source=input_tensor, imgsz=self.imgsz, conf=0.01, verbose=False)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise P2ExecutionError("OUT_OF_MEMORY", str(e)) from e
            raise P2ExecutionError("RUNTIME_ERROR", str(e)) from e
        except Exception as e:
            raise P2ExecutionError("RUNTIME_ERROR", str(e)) from e

        latency_ms = (time.perf_counter() - t0) * 1000.0

        gpu_mem_mb = 256.0
        try:
            import torch  # Lazy import opcional

            if torch.cuda.is_available():
                gpu_mem_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except ImportError:
            pass

        raw_dets: list[RawDetection] = []
        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                raw_dets.append(
                    RawDetection(
                        class_id=int(clss[i]),
                        confidence=float(confs[i]),
                        bbox=(float(xyxy[i][0]), float(xyxy[i][1]), float(xyxy[i][2]), float(xyxy[i][3])),
                    )
                )

        return raw_dets, P2InferenceMetrics(latency_ms=round(latency_ms, 3), gpu_memory_mb=round(gpu_mem_mb, 2))

    def info(self, handle: int) -> P2EngineInfo:
        if handle not in self._models:
            raise P2ExecutionError("RUNTIME_ERROR", "Handle inválido em info().")
        meta = self._models[handle]
        return P2EngineInfo(
            engine_version=meta["engine_version"],
            input_shape=[1, 3, self.imgsz, self.imgsz],
            output_shape=[1, 84, 8400],
            precision=meta["precision"],
            max_batch=1,
        )

    def release(self, handle: int) -> None:
        self._models.pop(handle, None)


class P2InferenceClient:
    """Cliente de inferência consumido pelo pipeline de I1 (T11)."""

    def __init__(self, p2_backend: P2LibraryProtocol | None = None) -> None:
        self.backend: P2LibraryProtocol = p2_backend or MockP2Backend()
        self.handle: Any = None
        self.engine_info: P2EngineInfo | None = None

    def load_engine(self, engine_ref: str, options: dict[str, Any] | None = None) -> P2EngineInfo:
        handle: Any = None
        try:
            handle = self.backend.load(engine_ref, options)
            info = self.backend.info(handle)
            if isinstance(info, dict):
                info = P2EngineInfo(
                    engine_version=str(info["engine_version"]),
                    input_shape=list(info["input_shape"]),
                    output_shape=list(info["output_shape"]),
                    precision=str(info["precision"]),
                    max_batch=int(info["max_batch"]),
                )
        except Exception as err:
            if handle is not None:
                try:
                    self.backend.release(handle)
                except Exception as release_error:
                    release_message = f"Falha ao liberar handle P2 após falha de load/info: {release_error}"
                    if isinstance(err, P2ExecutionError):
                        err.args = (f"{err}; {release_message}",)
                    else:
                        err = RuntimeError(f"{err}; {release_message}")
            self.handle = None
            self.engine_info = None
            if isinstance(err, P2ExecutionError):
                raise
            raise P2ExecutionError("RUNTIME_ERROR", f"Falha durante load/info do P2: {err}") from err

        self.handle = handle
        self.engine_info = info
        return info

    def run_inference(self, bgr_image: np.ndarray) -> tuple[list[RawDetection], P2InferenceMetrics]:
        if self.handle is None:
            raise P2ExecutionError("RUNTIME_ERROR", "Engine não carregado antes de run_inference().")
        try:
            detections, metrics = self.backend.infer(self.handle, bgr_image)
            if isinstance(metrics, dict):
                metrics = P2InferenceMetrics(
                    latency_ms=float(metrics["latency_ms"]),
                    gpu_memory_mb=float(metrics["gpu_memory_mb"]),
                )
        except P2ExecutionError:
            raise
        except Exception as err:
            raise P2ExecutionError("RUNTIME_ERROR", f"Falha durante inferência P2: {err}") from err
        return detections, metrics

    def close(self) -> None:
        if self.handle is not None:
            self.backend.release(self.handle)
            self.handle = None