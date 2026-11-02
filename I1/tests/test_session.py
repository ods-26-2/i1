"""
test_session.py — Regra de sessão e sequence do I1 (T6 / RNF 4).
Especificação 1.0: sequence conta por producer + session_id a partir de 0; P4/P6 geram session_id novo em todo
START, SEEK, LOOP, RESTART e RECONNECT, e o session_id chega em cada quadro (ods.p4.frame).
Os cenários simulam o replay de P6: depois de SEEK ou LOOP, frame_index volta a 0 e captured_at volta no tempo.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import random
import sys
import unittest
import uuid
from typing import Any, Iterator

import numpy as np

I1_ROOT = Path(__file__).resolve().parent.parent
if str(I1_ROOT) not in sys.path:
    sys.path.insert(0, str(I1_ROOT))

from detector.config import AppConfig
from detector.contracts import Frame, RawDetection
from detector.inference import P2InferenceClient
from detector.pipeline import DetectorPipeline
from detector.publishing import ValidatedStdoutPublisher
from detector.schema_validation import schema_errors

SESSION_A = "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d"
SESSION_B = "2b3c4d5e-6f70-4a8b-9cad-1e2f3a4b5c6d"
SESSION_C = "3c4d5e6f-7081-4b9c-8dbe-2f3a4b5c6d7e"
RECORDING_START = datetime(2026, 9, 23, 14, 0, 0, tzinfo=timezone.utc)
OBS_LOGGER = "ods.i1.observability"


def captured_at(frame_index: int, offset_s: float = 0.0) -> str:
    """Instante original de captura na gravação (30 fps), preservado por P6 no replay."""
    moment = RECORDING_START + timedelta(seconds=offset_s + frame_index / 30.0)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def make_frame(session_id: str, frame_index: int, offset_s: float = 0.0, corrupted: bool = False) -> Frame:
    return Frame(
        image=np.zeros((480, 640, 3), dtype=np.uint8),
        captured_at=captured_at(frame_index, offset_s),
        source_id="cam0",
        session_id=session_id,
        frame_index=frame_index,
        image_width=640,
        image_height=480,
        corrupted=corrupted,
    )


def replay_scenario() -> list[Frame]:
    """START na sessão A em 120 s da gravação; SEEK de volta para 60 s (sessão B); LOOP ao início (sessão C)."""
    return (
        [make_frame(SESSION_A, i, offset_s=120.0) for i in range(4)]
        + [make_frame(SESSION_B, i, offset_s=60.0) for i in range(3)]
        + [make_frame(SESSION_C, i) for i in range(2)]
    )


class ListSource:
    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def frames(self) -> Iterator[Frame]:
        yield from self._frames


class FixedP2Backend:
    """Backend P2 determinístico: mesma detecção em todo quadro."""

    def load(self, engine_ref: str, options: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"released": False}

    def infer(self, handle: dict[str, Any], input_bgr: np.ndarray) -> tuple[list[RawDetection], dict[str, float]]:
        return [RawDetection(bbox=(10.0, 20.0, 110.0, 220.0), class_id=0, confidence=0.9)], {
            "latency_ms": 5.0,
            "gpu_memory_mb": 100.0,
        }

    def info(self, handle: dict[str, Any]) -> dict[str, Any]:
        return {
            "engine_version": "app-test-1.0.0-fp32",
            "input_shape": [1, 3, 640, 640],
            "output_shape": [1, 5, 8400],
            "precision": "FP32",
            "max_batch": 1,
        }

    def release(self, handle: dict[str, Any]) -> None:
        handle["released"] = True


def seeded_message_ids(seed: int = 0):
    """Gera UUIDs v4 válidos em sequência reproduzível."""
    rng = random.Random(seed)
    return lambda: str(uuid.UUID(int=rng.getrandbits(128), version=4))


class TestSessionAndSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = AppConfig(
            artifact_id="app-test",
            artifact_version="1.0.0",
            engine_ref="app-test.onnx",
            classes=["objeto"],
        )

    def _run(self, frames: list[Frame], **pipeline_kwargs: Any) -> tuple[DetectorPipeline, list[dict[str, Any]], list[dict[str, Any]]]:
        publisher = ValidatedStdoutPublisher(verbose=False)
        pipe = DetectorPipeline(self.cfg, ListSource(frames), publisher, P2InferenceClient(FixedP2Backend()), **pipeline_kwargs)
        with self.assertLogs(OBS_LOGGER, level="INFO") as logs:
            pipe.run()
        records = [json.loads(line.split("[B2-OBS] ", 1)[1]) for line in logs.output if "[B2-OBS] {" in line]
        return pipe, publisher.published_messages, records

    def test_sequence_restarts_only_when_session_id_changes(self) -> None:
        frames = replay_scenario()
        _, messages, records = self._run(frames)

        self.assertEqual(
            [(m["session_id"], m["sequence"]) for m in messages],
            [(SESSION_A, 0), (SESSION_A, 1), (SESSION_A, 2), (SESSION_A, 3),
             (SESSION_B, 0), (SESSION_B, 1), (SESSION_B, 2),
             (SESSION_C, 0), (SESSION_C, 1)],
        )
        # frame_index e captured_at são copiados do quadro (P4/P6), inclusive quando o tempo de captura volta
        self.assertEqual([m["payload"]["frame_index"] for m in messages], [f.frame_index for f in frames])
        self.assertEqual([m["captured_at"] for m in messages], [f.captured_at for f in frames])

        changes = [r for r in records if r["event"] == "session_changed"]
        self.assertEqual(
            [(c["previous_session_id"], c["session_id"]) for c in changes],
            [(None, SESSION_A), (SESSION_A, SESSION_B), (SESSION_B, SESSION_C)],
        )
        for message in messages:
            self.assertEqual(schema_errors(message), [])

    def test_long_session_never_resets(self) -> None:
        _, messages, _ = self._run([make_frame(SESSION_A, i) for i in range(50)])
        self.assertEqual([m["sequence"] for m in messages], list(range(50)))

    def test_late_frame_from_ended_session_is_dropped(self) -> None:
        frames = [
            make_frame(SESSION_A, 0),
            make_frame(SESSION_A, 1),
            make_frame(SESSION_B, 0),
            make_frame(SESSION_A, 2),  # chega atrasado, depois do SEEK
            make_frame(SESSION_B, 1),
        ]
        pipe, messages, records = self._run(frames)

        pairs = [(m["session_id"], m["sequence"]) for m in messages]
        self.assertEqual(pairs, [(SESSION_A, 0), (SESSION_A, 1), (SESSION_B, 0), (SESSION_B, 1)])
        self.assertEqual(len(pairs), len(set(pairs)))
        self.assertEqual(pipe.error_counts, {"STALE_SESSION": 1})
        (dropped,) = [r for r in records if r["event"] == "frame_dropped"]
        self.assertEqual((dropped["session_id"], dropped["frame_index"], dropped["sequence"]), (SESSION_A, 2, None))

    def test_sequence_gap_stays_inside_its_session(self) -> None:
        frames = [
            make_frame(SESSION_A, 0),
            make_frame(SESSION_A, 1, corrupted=True),
            make_frame(SESSION_A, 2),
            make_frame(SESSION_B, 0),
            make_frame(SESSION_B, 1),
        ]
        _, messages, _ = self._run(frames)
        self.assertEqual(
            [(m["session_id"], m["sequence"]) for m in messages],
            [(SESSION_A, 0), (SESSION_A, 2), (SESSION_B, 0), (SESSION_B, 1)],
        )

    def test_replay_is_deterministic_with_injected_clock_and_ids(self) -> None:
        runs = [
            self._run(
                replay_scenario(),
                clock=lambda: "2026-11-02T12:00:00.500Z",
                message_id_factory=seeded_message_ids(seed=7),
            )[1]
            for _ in range(2)
        ]
        self.assertEqual(runs[0], runs[1])

    def test_replay_matches_ignoring_message_id_and_published_at(self) -> None:
        def stripped(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [{k: v for k, v in m.items() if k not in ("message_id", "published_at")} for m in messages]

        first = self._run(replay_scenario())[1]
        second = self._run(replay_scenario())[1]
        self.assertEqual(stripped(first), stripped(second))
        self.assertTrue({m["message_id"] for m in first}.isdisjoint(m["message_id"] for m in second))


if __name__ == "__main__":
    unittest.main(verbosity=2)
