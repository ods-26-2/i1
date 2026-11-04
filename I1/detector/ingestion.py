"""
ingestion.py — Ingestão de quadros (P4/P6 sob contrato ods.p4.frame) e pré-processamento dinâmico (T10).
"""

from __future__ import annotations

from typing import Iterator, Protocol
import cv2
import numpy as np

from .contracts import Frame, PixelFormat, utc_now_iso_ms


class FrameSource(Protocol):
    """Protocolo para comutação transparente entre P4 (ao vivo) e P6 (replay) via configuração (T17)."""
    def frames(self) -> Iterator[Frame]:
        ...


def preprocess_frame_to_bgr(frame: Frame) -> np.ndarray:
    """
    Pré-processamento definitivo de I1 (T10):
    Lê dinamicamente pixel_format, image_width e image_height de cada quadro recebido
    e converte para matriz BGR 3 canais esperada pelo modelo, preservando dimensões originais.
    """
    img = frame.image
    if img is None or img.size == 0 or frame.corrupted:
        raise ValueError("Quadro corrompido ou buffer vazio recebido da fonte.")

    fmt: PixelFormat = frame.pixel_format
    h, w = frame.image_height, frame.image_width

    if fmt == "BGR":
        bgr = img
    elif fmt == "RGB":
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif fmt == "GRAY":
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif fmt == "NV12":
        # No formato NV12 planar, o buffer tem altura (H * 3 // 2) e largura W
        expected_nv12_h = (h * 3) // 2
        if img.shape[0] != expected_nv12_h or img.shape[1] != w:
            img = img.reshape((expected_nv12_h, w))
        bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
    else:
        raise ValueError(f"pixel_format não suportado: {fmt}")

    if bgr.shape[0] != h or bgr.shape[1] != w:
        raise ValueError(
            f"Dimensões convertidas ({bgr.shape[1]}x{bgr.shape[0]}) divergem do metadado ({w}x{h})."
        )
    return bgr


def unreadable_frame(source_id: str, session_id: str, frame_index: int) -> Frame:
    """
    Quadro marcado como corrompido para um arquivo que não pôde ser decodificado.
    Em vez de sumir em silêncio, segue para o pipeline, que o descarta com registro em B2 e lacuna em sequence (T7).
    """
    return Frame(
        image=np.empty((0, 0, 3), dtype=np.uint8),
        captured_at=utc_now_iso_ms(),
        source_id=source_id,
        session_id=session_id,
        frame_index=frame_index,
        image_width=0,
        image_height=0,
        corrupted=True,
    )


class OpenCVFrameSource:
    """Fonte de vídeo para Degrau A / testes locais simulando o contrato ods.p4.frame (P4/P6)."""

    def __init__(
        self,
        uri: str | int = 0,
        source_id: str = "cam0",
        session_id: str = "9c2a1f3e-4d5b-4c6a-9e8f-0a1b2c3d4e5f",
        simulate_pixel_format: PixelFormat = "BGR",
        max_frames: int | None = None,
    ) -> None:
        self.uri = uri
        self.source_id = source_id
        self.session_id = session_id
        self.simulate_pixel_format: PixelFormat = simulate_pixel_format
        self.max_frames = max_frames

    def _convert_from_bgr(self, bgr: np.ndarray) -> np.ndarray:
        if self.simulate_pixel_format == "BGR":
            return bgr
        if self.simulate_pixel_format == "RGB":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.simulate_pixel_format == "GRAY":
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.simulate_pixel_format == "NV12":
            yuv_i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
            h, w = bgr.shape[:2]
            y = yuv_i420[:h, :]
            u = yuv_i420[h : h + h // 4, :].reshape(-1)
            v = yuv_i420[h + h // 4 :, :].reshape(-1)
            uv = np.empty((u.size + v.size,), dtype=u.dtype)
            uv[0::2] = u
            uv[1::2] = v
            return np.vstack([y, uv.reshape((h // 2, w))])
        raise ValueError(f"Formato inválido: {self.simulate_pixel_format}")

    def frames(self) -> Iterator[Frame]:
        from pathlib import Path

        # Suporte a diretório de imagens (ex.: dataset de teste do B5 no Degrau A)
        if isinstance(self.uri, str) and Path(self.uri).is_dir():
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            img_files = sorted(
                p for p in Path(self.uri).iterdir() if p.suffix.lower() in valid_exts
            )
            if not img_files:
                raise RuntimeError(f"Nenhuma imagem encontrada no diretório: {self.uri}")

            for frame_idx, img_path in enumerate(img_files):
                if self.max_frames is not None and frame_idx >= self.max_frames:
                    break
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    yield unreadable_frame(self.source_id, self.session_id, frame_idx)
                    continue
                h, w = bgr.shape[:2]
                raw_buf = self._convert_from_bgr(bgr)
                yield Frame(
                    image=raw_buf,
                    captured_at=utc_now_iso_ms(),
                    source_id=self.source_id,
                    session_id=self.session_id,
                    frame_index=frame_idx,
                    image_width=w,
                    image_height=h,
                    pixel_format=self.simulate_pixel_format,
                    fps=30.0,
                    raw=True,
                )
            return

        cap = cv2.VideoCapture(self.uri)
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {self.uri}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_idx = 0

        try:
            while True:
                if self.max_frames is not None and frame_idx >= self.max_frames:
                    break
                ok, bgr = cap.read()
                if not ok:
                    break

                h, w = bgr.shape[:2]
                raw_buf = self._convert_from_bgr(bgr)

                yield Frame(
                    image=raw_buf,
                    captured_at=utc_now_iso_ms(),
                    source_id=self.source_id,
                    session_id=self.session_id,
                    frame_index=frame_idx,
                    image_width=w,
                    image_height=h,
                    pixel_format=self.simulate_pixel_format,
                    fps=fps,
                    raw=True,
                )
                frame_idx += 1
        finally:
            cap.release()