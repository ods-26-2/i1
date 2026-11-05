"""
ingestion.py — Ingestão de quadros (P4/P6 sob contrato ods.p4.frame) e pré-processamento dinâmico (T10).
"""

from __future__ import annotations

import json
import struct
import time
from datetime import datetime, timezone
from typing import Any, Iterator, Protocol
import cv2
import numpy as np

from .contracts import Frame, PixelFormat, utc_now_iso_ms


class FrameSource(Protocol):
    """Protocolo para comutação transparente entre P4 (ao vivo) e P6 (replay) via configuração (T17)."""
    def frames(self) -> Iterator[Frame]:
        ...


def parse_p4r1_packet(packet: bytes) -> tuple[dict[str, Any], bytes]:
    """Valida o envelope binário publicado pelo SHMPublisher do P4 e extrai os bytes do frame."""
    if len(packet) < 8 or packet[:4] != b"P4R1":
        raise ValueError("Buffer P4 inválido: magic P4R1 ausente ou cabeçalho incompleto.")

    header_size = struct.unpack("!I", packet[4:8])[0]
    payload_start = 8 + header_size
    if header_size == 0 or payload_start > len(packet):
        raise ValueError("Buffer P4 inválido: tamanho do cabeçalho fora dos limites.")

    try:
        metadata = json.loads(packet[8:payload_start].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise ValueError(f"Buffer P4 inválido: JSON do cabeçalho malformado: {err}") from err
    if not isinstance(metadata, dict):
        raise ValueError("Buffer P4 inválido: cabeçalho JSON deve ser um objeto.")

    required = (
        "source_id",
        "session_id",
        "frame_id",
        "timestamp",
        "width",
        "height",
        "format",
        "raw",
        "data_size",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Buffer P4 inválido: campos ausentes no cabeçalho: {', '.join(missing)}.")

    if not isinstance(metadata["source_id"], str) or not metadata["source_id"]:
        raise ValueError("Buffer P4 inválido: source_id deve ser texto não vazio.")
    if not isinstance(metadata["session_id"], str) or not metadata["session_id"]:
        raise ValueError("Buffer P4 inválido: session_id deve ser texto não vazio.")
    if type(metadata["frame_id"]) is not int or metadata["frame_id"] < 0:
        raise ValueError("Buffer P4 inválido: frame_id deve ser inteiro não negativo.")
    if metadata["timestamp"] is not None and (
        type(metadata["timestamp"]) is not int or metadata["timestamp"] < 0
    ):
        raise ValueError("Buffer P4 inválido: timestamp deve ser inteiro não negativo ou null.")
    for key in ("width", "height", "data_size"):
        if type(metadata[key]) is not int or metadata[key] <= 0:
            raise ValueError(f"Buffer P4 inválido: {key} deve ser inteiro positivo.")
    if not isinstance(metadata["format"], str) or metadata["format"] not in {
        "RG10",
        "BGR",
        "RGB",
        "GRAY",
        "NV12",
    }:
        raise ValueError(f"Buffer P4 inválido: formato não suportado: {metadata['format']!r}.")
    if type(metadata["raw"]) is not bool:
        raise ValueError("Buffer P4 inválido: raw deve ser booleano.")

    image_data = packet[payload_start:]
    if len(image_data) != metadata["data_size"]:
        raise ValueError(
            "Buffer P4 inválido: data_size declarado "
            f"({metadata['data_size']}) difere dos bytes recebidos ({len(image_data)})."
        )
    return metadata, image_data


def _image_array_from_p4_data(
    image_data: bytes, width: int, height: int, pixel_format: PixelFormat
) -> np.ndarray:
    """Constrói a matriz correspondente ao formato e dimensões anunciados no cabeçalho P4R1."""
    if pixel_format == "RG10":
        expected_size = width * height * 2
        if len(image_data) != expected_size:
            raise ValueError(
                f"Frame RG10 incompatível: esperados {expected_size} bytes, recebidos {len(image_data)}."
            )
        return np.frombuffer(image_data, dtype="<u2").reshape(height, width)
    if pixel_format in ("BGR", "RGB"):
        expected_size = width * height * 3
        if len(image_data) != expected_size:
            raise ValueError(
                f"Frame {pixel_format} incompatível: esperados {expected_size} bytes, recebidos {len(image_data)}."
            )
        return np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 3)
    if pixel_format == "GRAY":
        expected_size = width * height
        if len(image_data) != expected_size:
            raise ValueError(
                f"Frame GRAY incompatível: esperados {expected_size} bytes, recebidos {len(image_data)}."
            )
        return np.frombuffer(image_data, dtype=np.uint8).reshape(height, width)
    if pixel_format == "NV12":
        if width % 2 or height % 2:
            raise ValueError("Frame NV12 exige dimensões pares.")
        expected_size = width * height * 3 // 2
        if len(image_data) != expected_size:
            raise ValueError(
                f"Frame NV12 incompatível: esperados {expected_size} bytes, recebidos {len(image_data)}."
            )
        return np.frombuffer(image_data, dtype=np.uint8).reshape(height * 3 // 2, width)
    raise ValueError(f"pixel_format não suportado pelo protocolo P4: {pixel_format}")


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
        expected_nv12_h = (h * 3) // 2
        if img.shape[0] != expected_nv12_h or img.shape[1] != w:
            img = img.reshape((expected_nv12_h, w))
        bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
    elif fmt == "RG10":
        if img.ndim != 2 or img.shape != (h, w) or img.dtype != np.uint16:
            raise ValueError(f"Frame RG10 deve ser uint16 com shape ({h}, {w}), recebido {img.dtype} {img.shape}.")
        # RG10 do P4 é Bayer RGGB de 10 bits em words de 16 bits, como anunciado em P4.config e storage-bits=16.
        bayer10 = img & np.uint16(0x03FF)
        bgr16 = cv2.cvtColor(bayer10, cv2.COLOR_BayerRG2BGR)
        bgr = (bgr16 >> 2).astype(np.uint8)
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

        if isinstance(self.uri, str) and Path(self.uri).is_dir():
            valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            img_files = sorted(p for p in Path(self.uri).iterdir() if p.suffix.lower() in valid_exts)
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


class P4GStreamerShmSource:
    """
    Leitor do protocolo SHM P4R1: cada buffer carrega seu cabeçalho binário e os bytes do frame.
    """

    def __init__(
        self,
        shm_socket_path: str = "/tmp/p4_imx519_shm",
        max_frames: int | None = None,
    ) -> None:
        self.shm_socket_path = shm_socket_path
        self.max_frames = max_frames
        self._last_timestamp_ns: int | None = None
        self._last_frame_fps = 30.0

    @staticmethod
    def _monotonic_ns_to_iso_ms(timestamp_ns: int | None) -> str:
        if timestamp_ns is None:
            return utc_now_iso_ms()
        # P4 currently sends the V4L2 monotonic timestamp; map it onto this host's UTC clock.
        utc_timestamp_ns = timestamp_ns + time.time_ns() - time.monotonic_ns()
        dt = datetime.fromtimestamp(utc_timestamp_ns / 1e9, tz=timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

    def _build_gstreamer_pipeline(self) -> str:
        return (
            f'shmsrc socket-path="{self.shm_socket_path}" is-live=true do-timestamp=true ! '
            "application/x-ods-frame,version=1 ! "
            "appsink name=i1_sink sync=false max-buffers=1 drop=true"
        )

    def frames(self) -> Iterator[Frame]:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except (ImportError, ValueError) as err:
            raise RuntimeError(
                "P4GStreamerShmSource requer PyGObject e os bindings GStreamer 1.0."
            ) from err

        Gst.init(None)
        gst_pipeline = self._build_gstreamer_pipeline()
        try:
            pipeline = Gst.parse_launch(gst_pipeline)
        except Exception as err:
            raise RuntimeError(f"Pipeline GStreamer P4 inválida: {err}") from err
        sink = pipeline.get_by_name("i1_sink")
        if sink is None:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError("Pipeline GStreamer P4 não criou o appsink 'i1_sink'.")
        bus = pipeline.get_bus()
        result = pipeline.set_state(Gst.State.PLAYING)
        if result == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            raise RuntimeError(f"Não foi possível iniciar a pipeline P4: {gst_pipeline}")

        frame_idx = 0
        try:
            while True:
                if self.max_frames is not None and frame_idx >= self.max_frames:
                    break
                sample = sink.emit("try-pull-sample", Gst.SECOND)
                if sample is None:
                    message = bus.timed_pop_filtered(
                        0, Gst.MessageType.ERROR | Gst.MessageType.EOS
                    )
                    if message is not None and message.type == Gst.MessageType.ERROR:
                        error, debug = message.parse_error()
                        raise RuntimeError(f"Erro na ingestão GStreamer P4: {error}; {debug}")
                    if message is not None and message.type == Gst.MessageType.EOS:
                        break
                    continue

                buffer = sample.get_buffer()
                mapped_ok, mapped = buffer.map(Gst.MapFlags.READ)
                if not mapped_ok:
                    raise RuntimeError("Não foi possível mapear o buffer GStreamer P4 para leitura.")
                try:
                    packet = bytes(mapped.data)
                finally:
                    buffer.unmap(mapped)

                metadata, image_data = parse_p4r1_packet(packet)
                width, height = metadata["width"], metadata["height"]
                pixel_format: PixelFormat = metadata["format"]
                image = _image_array_from_p4_data(image_data, width, height, pixel_format)
                timestamp_ns = metadata["timestamp"]
                if timestamp_ns is not None and self._last_timestamp_ns is not None:
                    interval_ns = timestamp_ns - self._last_timestamp_ns
                    if interval_ns > 0:
                        self._last_frame_fps = 1_000_000_000 / interval_ns
                self._last_timestamp_ns = timestamp_ns
                yield Frame(
                    image=image,
                    captured_at=self._monotonic_ns_to_iso_ms(timestamp_ns),
                    source_id=metadata["source_id"],
                    session_id=metadata["session_id"],
                    frame_index=metadata["frame_id"],
                    image_width=width,
                    image_height=height,
                    pixel_format=pixel_format,
                    fps=self._last_frame_fps,
                    raw=metadata["raw"],
                )
                frame_idx += 1
        finally:
            pipeline.set_state(Gst.State.NULL)
