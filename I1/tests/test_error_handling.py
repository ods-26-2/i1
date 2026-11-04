"""
test_error_handling.py — Representação de erro do I1 (T7 / RNF 5).
Proposta do squad: quadro com falha não publica detecção, o erro é registrado na observabilidade (B2)
com código e a perda aparece como lacuna em sequence. Falha de carregamento do modelo é fatal.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Any, Iterator

import cv2
import numpy as np

I1_ROOT = Path(__file__).resolve().parent.parent
if str(I1_ROOT) not in sys.path:
    sys.path.insert(0, str(I1_ROOT))

from detector.config import AppConfig
from detector.contracts import EvSinalEnvelope, Frame, P2ExecutionError, RawDetection
from detector.inference import P2InferenceClient
from detector.ingestion import OpenCVFrameSource
from detector.pipeline import DetectorPipeline
from detector.publishing import ValidatedStdoutPublisher

SESSION_ID = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
OBS_LOGGER = "ods.i1.observability"


def make_frame(frame_index: int, corrupted: bool = False) -> Frame:
    return Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        captured_at="2026-10-18T12:00:00.100Z",
        source_id="cam0",
        session_id=SESSION_ID,
        frame_index=frame_index,
        image_width=640,
        image_height=480,
        corrupted=corrupted,
    )


class ListSource:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames
        self.consumed = 0

    def frames(self) -> Iterator[Frame]:
        for frame in self._frames:
            self.consumed += 1
            yield frame


class ScriptedP2Backend:
    """Backend P2 falso: falha no load com load_error, ou falha a chamada infer de número fail_on_call."""

    def __init__(
        self,
        load_error: str | None = None,
        fail_on_call: int | None = None,
        infer_error: str = "RUNTIME_ERROR",
        detections: list[RawDetection] | None = None,
    ) -> None:
        self.load_error = load_error
        self.fail_on_call = fail_on_call
        self.infer_error = infer_error
        self.detections = detections if detections is not None else [
            RawDetection(bbox=(10.0, 10.0, 100.0, 100.0), class_id=0, confidence=0.9)
        ]
        self.calls = 0

    def load(self, engine_ref: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.load_error:
            raise P2ExecutionError(self.load_error, "falha simulada no load")  # type: ignore[arg-type]
        return {"released": False}

    def infer(self, handle: dict[str, Any], input_bgr: np.ndarray) -> tuple[list[RawDetection], dict[str, float]]:
        call = self.calls
        self.calls += 1
        if call == self.fail_on_call:
            raise P2ExecutionError(self.infer_error, "falha simulada no infer")  # type: ignore[arg-type]
        return list(self.detections), {"latency_ms": 5.0, "gpu_memory_mb": 100.0}

    def info(self, handle: dict[str, Any]) -> dict[str, Any]:
        return {
            "engine_version": "app-test-1.0.0",
            "input_shape": [1, 3, 640, 640],
            "output_shape": [1, 5, 8400],
            "precision": "FP32",
            "max_batch": 1,
        }

    def release(self, handle: dict[str, Any]) -> None:
        handle["released"] = True


class FailingPublisher(ValidatedStdoutPublisher):
    """Publicador que lança a exceção configurada na chamada de número fail_on_call."""

    def __init__(self, fail_on_call: int, error: Exception) -> None:
        super().__init__(verbose=False)
        self.fail_on_call = fail_on_call
        self.error = error
        self.calls = 0

    def publish(self, envelope: EvSinalEnvelope) -> dict[str, Any]:
        call = self.calls
        self.calls += 1
        if call == self.fail_on_call:
            raise self.error
        return super().publish(envelope)


def obs_records(log_output: list[str]) -> list[dict[str, Any]]:
    return [json.loads(line.split("[B2-OBS] ", 1)[1]) for line in log_output if "[B2-OBS] {" in line]


class TestErrorHandling(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = AppConfig(
            artifact_id="app-test",
            artifact_version="1.0.0",
            engine_ref="app-test.onnx",
            classes=["objeto"],
            session_id=SESSION_ID,
        )

    def _run(self, frames: list[Frame], backend: ScriptedP2Backend, publisher: ValidatedStdoutPublisher | None = None):
        publisher = publisher or ValidatedStdoutPublisher(verbose=False)
        pipe = DetectorPipeline(self.cfg, ListSource(frames), publisher, P2InferenceClient(backend))
        with self.assertLogs(OBS_LOGGER, level="ERROR") as logs:
            pipe.run()
        return pipe, publisher, obs_records(logs.output)

    def test_load_failure_is_logged_and_fatal(self) -> None:
        for code in ("ENGINE_NOT_FOUND", "INCOMPATIBLE_VERSION", "OUT_OF_MEMORY"):
            with self.subTest(code):
                source = ListSource([make_frame(0)])
                publisher = ValidatedStdoutPublisher(verbose=False)
                pipe = DetectorPipeline(self.cfg, source, publisher, P2InferenceClient(ScriptedP2Backend(load_error=code)))
                with self.assertLogs(OBS_LOGGER, level="CRITICAL") as logs, self.assertRaises(P2ExecutionError):
                    pipe.run()

                (record,) = obs_records(logs.output)
                self.assertEqual(record["event"], "engine_load_failed")
                self.assertEqual(record["error_code"], code)
                self.assertEqual(source.consumed, 0)
                self.assertEqual(publisher.published_messages, [])
                self.assertEqual(pipe.error_counts, {code: 1})

    def test_infer_errors_drop_frame_and_leave_sequence_gap(self) -> None:
        for code in ("INVALID_INPUT_SHAPE", "RUNTIME_ERROR", "OUT_OF_MEMORY"):
            with self.subTest(code):
                backend = ScriptedP2Backend(fail_on_call=1, infer_error=code)
                pipe, publisher, records = self._run([make_frame(0), make_frame(1), make_frame(2)], backend)

                self.assertEqual([m["sequence"] for m in publisher.published_messages], [0, 2])
                self.assertEqual(pipe.error_counts, {code: 1})
                (record,) = records
                self.assertEqual(
                    {k: record[k] for k in ("event", "error_code", "stage", "sequence", "frame_index", "session_id")},
                    {
                        "event": "frame_dropped",
                        "error_code": code,
                        "stage": "infer",
                        "sequence": 1,
                        "frame_index": 1,
                        "session_id": SESSION_ID,
                    },
                )

    def test_corrupted_frame_is_dropped_as_invalid_frame(self) -> None:
        pipe, publisher, records = self._run(
            [make_frame(0), make_frame(1, corrupted=True), make_frame(2)], ScriptedP2Backend()
        )
        self.assertEqual([m["sequence"] for m in publisher.published_messages], [0, 2])
        self.assertEqual(pipe.error_counts, {"INVALID_FRAME": 1})
        self.assertEqual((records[0]["stage"], records[0]["sequence"]), ("preprocess", 1))

    def test_publish_failures_drop_frame_and_pipeline_continues(self) -> None:
        cases = {
            "CONTRACT_VIOLATION": ValueError("mensagem rejeitada pelo validador"),
            "INTERNAL_ERROR": BufferError("fila do produtor cheia"),
        }
        for code, error in cases.items():
            with self.subTest(code):
                publisher = FailingPublisher(fail_on_call=1, error=error)
                pipe, publisher, records = self._run(
                    [make_frame(0), make_frame(1), make_frame(2)], ScriptedP2Backend(), publisher
                )
                self.assertEqual([m["sequence"] for m in publisher.published_messages], [0, 2])
                self.assertEqual(pipe.error_counts, {code: 1})
                self.assertEqual(records[0]["stage"], "publish")

    def test_unexpected_postprocess_error_is_internal_error(self) -> None:
        broken = [RawDetection(bbox=(None, None, None, None), class_id=0, confidence=0.9)]  # type: ignore[arg-type]
        pipe, publisher, records = self._run([make_frame(0)], ScriptedP2Backend(detections=broken))
        self.assertEqual(publisher.published_messages, [])
        self.assertEqual(pipe.error_counts, {"INTERNAL_ERROR": 1})
        self.assertEqual(records[0]["stage"], "postprocess")

    def test_unreadable_image_becomes_sequence_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            cv2.imwrite(str(folder / "a.png"), np.zeros((48, 64, 3), dtype=np.uint8))
            (folder / "b.jpg").write_bytes(b"nao e uma imagem")
            cv2.imwrite(str(folder / "c.png"), np.zeros((48, 64, 3), dtype=np.uint8))

            frames = list(OpenCVFrameSource(uri=str(folder), session_id=SESSION_ID).frames())
            self.assertEqual([(f.frame_index, f.corrupted) for f in frames], [(0, False), (1, True), (2, False)])

            pipe, publisher, records = self._run(frames, ScriptedP2Backend(detections=[]))
            self.assertEqual([m["sequence"] for m in publisher.published_messages], [0, 2])
            self.assertEqual(pipe.error_counts, {"INVALID_FRAME": 1})
            self.assertEqual(records[0]["frame_index"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
