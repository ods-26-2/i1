"""
pipeline.py — orquestra os 4 módulos do detector (I1): ingestion -> inference ->
postprocessing -> publishing.

É o único arquivo que conhece a ordem do fluxo. Os módulos individuais não sabem
uns dos outros — só contracts.py, que define o vocabulário comum entre eles.
"""

from __future__ import annotations

from .contracts import AppConfig, Frame
from .inference import Engine, load_engine, run as run_inference
from .ingestion import FrameSource, MockFrameSource
from .postprocessing import apply_nms, filter_by_confidence, map_classes
from .publishing import MockPublisher, Publisher, build_event


class Detector:
    """
    Componente I1: recebe uma fonte de quadros, um engine já carregado e um
    publicador, e executa o ciclo ingestão -> inferência -> pós-processamento ->
    publicação para cada quadro disponível.
    """

    def __init__(
        self,
        config: AppConfig,
        frame_source: FrameSource,
        engine: Engine,
        publisher: Publisher,
    ):
        self._config = config
        self._frame_source = frame_source
        self._engine = engine
        self._publisher = publisher

    def run_forever(self) -> None:
        """Processa quadros continuamente até a fonte se esgotar (Ctrl+C para interromper antes)."""

        try:
            while True:
                frame = self._frame_source.next_frame()
                if frame is None:
                    break
                self.process_frame(frame)
        finally:
            self._frame_source.close()

    def process_frame(self, frame: Frame) -> None:
        """Executa o pipeline completo sobre um único quadro já capturado."""

        raw_detections = run_inference(self._engine, frame)
        raw_detections = filter_by_confidence(raw_detections, self._config.conf_threshold)
        raw_detections = apply_nms(raw_detections, self._config.nms_iou_threshold)
        detections = map_classes(raw_detections, self._config.class_map)

        event = build_event(frame, detections, self._config)
        self._publisher.publish(event)


def build_default_detector(config: AppConfig) -> Detector:
    """
    Monta um Detector com as implementações mock/stub padrão: MockFrameSource,
    engine carregado localmente e MockPublisher. Ponto único de troca quando as
    implementações reais (P4StreamSource, API de P2, publisher via B3) estiverem
    prontas.
    """

    frame_source = MockFrameSource(config)
    engine = load_engine(config.artifact_id, config.artifact_version)
    publisher = MockPublisher()

    return Detector(config=config, frame_source=frame_source, engine=engine, publisher=publisher)