"""
ingestion.py — Módulo 1 do pipeline de I1.

Define a interface FrameSource e uma implementação que lê de vídeo local/webcam.
Quando o stream real de P4 estiver pronto, adiciona-se uma segunda implementação
(P4StreamSource) respeitando a mesma interface — o resto do pipeline não percebe
a troca, porque só conhece next_frame() -> Frame.
"""

from __future__ import annotations

import time
from typing import Optional, Protocol

import cv2

from .contracts import AppConfig, Frame


class FrameSource(Protocol):
    """Interface mínima que qualquer fonte de quadros precisa implementar."""

    def next_frame(self) -> Optional[Frame]:
        """Retorna o próximo Frame disponível, ou None quando a fonte se esgota."""
        ...

    def close(self) -> None:
        """Libera recursos da fonte (arquivo, câmera, conexão)."""
        ...


class MockFrameSource:
    """
    Lê de um arquivo de vídeo local ou de uma webcam (config.source=None -> índice 0)
    via OpenCV.

    Preenche os campos do contrato de entrada com valores plausíveis enquanto o
    stream real de P4 (com calibração e timestamp de origem confiáveis) não está
    disponível. calibration_applied fica sempre False aqui, de propósito: não há
    perfil de I3 sendo aplicado nesta fonte mock.
    """

    def __init__(self, config: AppConfig):
        source = config.source if config.source is not None else 0
        self._capture = cv2.VideoCapture(source)
        if not self._capture.isOpened():
            raise RuntimeError(f"Não foi possível abrir a fonte de vídeo: {source!r}")

        self._camera_id = config.camera_id
        self._sequence_number = 0

    def next_frame(self) -> Optional[Frame]:
        ok, image = self._capture.read()
        if not ok:
            return None

        frame = Frame(
            image=image,
            camera_id=self._camera_id,
            captured_at=time.time(),
            sequence_number=self._sequence_number,
            calibration_applied=False,
            calibration_profile_version=None,
        )
        self._sequence_number += 1
        return frame

    def close(self) -> None:
        self._capture.release()