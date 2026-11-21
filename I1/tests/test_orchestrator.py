"""
test_orchestrator.py — Testes do orquestrador FIFO multi-aplicação do I1 (T4 / Jetson).
Cobre mapeamento equipe->artefato->vídeo, parsing de comandos Kafka, fila FIFO,
garantia de release() antes do próximo load(), HEARTBEAT/TTL, STOP e preempção force=true.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Any

import cv2
import numpy as np

I1_ROOT = Path(__file__).resolve().parent.parent
if str(I1_ROOT) not in sys.path:
    sys.path.insert(0, str(I1_ROOT))

from detector.config import AppConfig
from detector.contracts import EngineInfo, InferenceMetrics, RawDetection
from detector.ingestion import OpenCVFrameSource, P4GStreamerShmSource
from detector.orchestrator import (
    ARTIFACTS_ROOT,
    ControlCommand,
    FifoInferenceOrchestrator,
    QueuedJob,
    build_app_config_for_profile,
    parse_control_command,
    resolve_app_profile,
    resolve_squad_video_source,
)

HAS_JSONSCHEMA = importlib.util.find_spec("jsonschema") is not None


class FakeCommandSource:
    """Entrega lotes pré-programados de comandos a cada chamada de poll()."""

    def __init__(self, batches: list[list[ControlCommand]] | None = None) -> None:
        self.batches = list(batches or [])

    def push(self, *commands: ControlCommand) -> None:
        self.batches.append(list(commands))

    def poll(self, timeout_s: float = 0.0) -> list[ControlCommand]:
        if self.batches:
            return self.batches.pop(0)
        return []


class FakeClock:
    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += seconds
        return self.now


class RecordingPublisher:
    def __init__(self) -> None:
        self.published_messages: list[dict[str, Any]] = []
        self.flush_calls = 0

    def publish(self, envelope: Any) -> dict[str, Any]:
        msg = envelope.to_dict() if hasattr(envelope, "to_dict") else dict(envelope)
        self.published_messages.append(msg)
        return msg

    def flush(self, timeout: float = 3.0) -> int:
        self.flush_calls += 1
        return 0


class LifecycleTrackingP2Backend:
    """Garante e registra que apenas 1 engine fica carregada por vez (release antes de load)."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.active_engine: str | None = None
        self._handles: dict[int, str] = {}
        self._counter = 0

    def load(self, engine_ref: str, options: dict[str, Any] | None = None) -> int:
        if self.active_engine is not None:
            raise AssertionError(
                f"Violação de exclusividade P2: tentou load({engine_ref}) com {self.active_engine} ativo"
            )
        self.active_engine = engine_ref
        self.events.append(("load", engine_ref))
        self._counter += 1
        self._handles[self._counter] = engine_ref
        return self._counter

    def info(self, handle: int) -> EngineInfo:
        engine_ref = self._handles[handle]
        size = 1280 if "sports" in engine_ref else 640
        return EngineInfo(
            engine_version=f"{engine_ref}-fp16",
            input_shape=[1, 3, size, size],
            output_shape=[1, 8, 8400],
            precision="FP16",
            max_batch=1,
        )

    def infer(self, handle: int, input_tensor: np.ndarray) -> tuple[list[RawDetection], InferenceMetrics]:
        if self._handles.get(handle) != self.active_engine:
            raise AssertionError("Handle inválido ou já liberado durante infer().")
        return [RawDetection(bbox=(10.0, 10.0, 50.0, 50.0), class_id=0, confidence=0.95)], InferenceMetrics(
            latency_ms=4.5,
            gpu_memory_mb=180.0,
        )

    def release(self, handle: int) -> None:
        engine_ref = self._handles.pop(handle, None)
        if engine_ref is not None:
            self.events.append(("release", engine_ref))
            if self.active_engine == engine_ref:
                self.active_engine = None


class TestOrchestratorRoutingAndCommands(unittest.TestCase):
    def test_all_squads_and_aliases_resolve_to_valid_artifacts(self) -> None:
        expected = [
            ("INV1", "A4", "app-sports-1.2.0", None),
            ("inv-1", "A4", "app-sports-1.2.0", None),
            ("A4", "A4", "app-sports-1.2.0", None),
            ("VER1", "A1", "app-industry-1.0.0", None),
            ("VER3", "A5", "app-epi-1.0.0", None),
            ("VER5", "A8", "app-plates-1.0.0", None),
            ("VER6", "A3", "app-epi-1.0.0", ("pessoa",)),
        ]
        for alias, exp_app, exp_engine, exp_classes in expected:
            with self.subTest(alias=alias):
                profile = resolve_app_profile(alias)
                self.assertEqual(profile.app_id, exp_app)
                self.assertEqual(profile.engine_ref, exp_engine)
                self.assertEqual(profile.enabled_classes, exp_classes)
                self.assertTrue(profile.metadata_path(ARTIFACTS_ROOT).is_file())
                self.assertTrue(profile.onnx_path(ARTIFACTS_ROOT).is_file())

                cfg, imgsz = build_app_config_for_profile(
                    profile,
                    backend_mode="p2",
                    source_id=profile.squad,
                    session_id="9c2a1f3e-4d5b-4c6a-9e8f-0a1b2c3d4e5f",
                )
                self.assertEqual(cfg.engine_ref, exp_engine)
                self.assertEqual(imgsz, 1280 if profile.squad == "INV1" else 640)

    def test_parse_control_command_simple_and_cmd_envelope(self) -> None:
        cmd1 = parse_control_command('{"action": "start", "squad": "INV1"}')
        self.assertEqual(cmd1.action, "START")
        self.assertEqual(cmd1.profile.squad, "INV1")
        self.assertEqual(cmd1.ttl_seconds, 60.0)
        self.assertFalse(cmd1.force)

        cmd2 = parse_control_command(
            json.dumps(
                {
                    "schema": "ods.i1.command",
                    "message_type": "command",
                    "producer": "VER5",
                    "target": "I1",
                    "payload": {
                        "action": "START",
                        "ttl_seconds": 30,
                        "max_frames": 150,
                        "force": True,
                    },
                }
            )
        )
        self.assertEqual(cmd2.action, "START")
        self.assertEqual(cmd2.profile.squad, "VER5")
        self.assertEqual(cmd2.ttl_seconds, 30.0)
        self.assertEqual(cmd2.max_frames, 150)
        self.assertTrue(cmd2.force)

        cmd3 = parse_control_command({"action": "HEARTBEAT", "app_id": "A1", "ttl_seconds": 45})
        self.assertEqual(cmd3.action, "HEARTBEAT")
        self.assertEqual(cmd3.profile.squad, "VER1")

        for bad in (
            "{}",
            '{"action": "PAUSE", "squad": "INV1"}',
            '{"action": "START"}',
            '{"action": "START", "squad": "DESCONHECIDA"}',
            '{"action": "START", "squad": "INV1", "session_id": "not-a-uuid"}',
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_control_command(bad)

    def test_resolve_squad_video_source_finds_mp4_and_shm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            videos_dir = Path(tmp)
            inv1_mp4 = videos_dir / "INV1.mp4"
            ver5_mp4 = videos_dir / "ver-5.mp4"
            ver3_dir = videos_dir / "VER3"
            ver3_dir.mkdir()
            ver3_mp4 = ver3_dir / "epi_sample.mp4"
            for p in (inv1_mp4, ver5_mp4, ver3_mp4):
                p.write_bytes(b"")

            src_inv1 = resolve_squad_video_source(resolve_app_profile("INV1"), videos_dir=videos_dir)
            self.assertIsInstance(src_inv1, OpenCVFrameSource)
            self.assertEqual(src_inv1.uri, str(inv1_mp4))
            self.assertEqual(src_inv1.source_id, "INV1")

            src_ver5 = resolve_squad_video_source(resolve_app_profile("VER5"), videos_dir=videos_dir)
            self.assertIsInstance(src_ver5, OpenCVFrameSource)
            self.assertEqual(src_ver5.uri, str(ver5_mp4))

            src_ver3 = resolve_squad_video_source(resolve_app_profile("VER3"), videos_dir=videos_dir)
            self.assertIsInstance(src_ver3, OpenCVFrameSource)
            self.assertEqual(src_ver3.uri, str(ver3_mp4))

            shm_src = resolve_squad_video_source(
                resolve_app_profile("VER5"),
                videos_dir=videos_dir,
                explicit_source="video",
            )
            self.assertIsInstance(shm_src, P4GStreamerShmSource)
            self.assertEqual(shm_src.shm_socket_path, "/tmp/p4_file_shm")


class TestFifoOrchestratorScheduling(unittest.TestCase):
    def test_fifo_queue_deduplicates_and_runs_sequentially_releasing_before_next_load(self) -> None:
        clock = FakeClock()
        cmds = FakeCommandSource()
        pub = RecordingPublisher()
        events: list[tuple[str, str]] = []
        active_model: list[str] = []

        def fake_runner(job: QueuedJob, should_abort: Any) -> int:
            assert not active_model, f"Outro modelo ainda estava carregado: {active_model}"
            active_model.append(job.profile.engine_ref)
            events.append(("load", job.profile.engine_ref))
            frames = 0
            try:
                # Durante o 1o frame do INV1, VER5 pede START duas vezes e VER1 pede START uma vez
                if job.profile.squad == "INV1":
                    cmds.push(
                        parse_control_command({"action": "START", "squad": "VER5"}),
                        parse_control_command({"action": "START", "squad": "VER5"}),
                        parse_control_command({"action": "START", "squad": "VER1"}),
                    )
                for _ in range(3):
                    if should_abort():
                        break
                    frames += 1
            finally:
                active_model.pop()
                events.append(("release", job.profile.engine_ref))
            return frames

        orch = FifoInferenceOrchestrator(
            command_source=cmds,
            publisher=pub,
            clock=clock,
            job_runner=fake_runner,
        )
        cmds.push(parse_control_command({"action": "START", "squad": "INV1"}))

        # Executa os 3 jobs na ordem FIFO: INV1 -> VER5 -> VER1
        for _ in range(3):
            summary = orch.step(wait_when_idle=False)
            self.assertIsNotNone(summary)

        self.assertIsNone(orch.step(wait_when_idle=False))
        self.assertEqual(
            [(s.squad, s.engine_ref, s.frames_published) for s in orch.completed_jobs],
            [
                ("INV1", "app-sports-1.2.0", 3),
                ("VER5", "app-plates-1.0.0", 3),
                ("VER1", "app-industry-1.0.0", 3),
            ],
        )
        self.assertEqual(
            events,
            [
                ("load", "app-sports-1.2.0"),
                ("release", "app-sports-1.2.0"),
                ("load", "app-plates-1.0.0"),
                ("release", "app-plates-1.0.0"),
                ("load", "app-industry-1.0.0"),
                ("release", "app-industry-1.0.0"),
            ],
        )
        self.assertEqual(pub.flush_calls, 3)

    def test_heartbeat_ttl_expiry_stop_and_force_preemption(self) -> None:
        clock = FakeClock(start=0.0)
        cmds = FakeCommandSource()
        pub = RecordingPublisher()

        def ttl_runner(job: QueuedJob, should_abort: Any) -> int:
            frames = 0
            # Frame 1: recebe HEARTBEAT renovando TTL=+10s no t=8s
            if job.profile.squad == "INV1":
                clock.advance(8.0)
                cmds.push(parse_control_command({"action": "HEARTBEAT", "squad": "INV1", "ttl_seconds": 10.0}))
                if not should_abort():
                    frames += 1
                # Avança para t=12s (ainda válido até t=18s por causa do heartbeat!)
                clock.advance(4.0)
                if not should_abort():
                    frames += 1
                # Avança para t=19s -> TTL expira!
                clock.advance(7.0)
                should_abort()
                return frames

            # VER5 será interrompido por STOP; VER3 será preemptado por force=true de VER1
            if job.profile.squad == "VER5":
                if not should_abort():
                    frames += 1
                cmds.push(parse_control_command({"action": "STOP", "squad": "VER5"}))
                should_abort()
                return frames

            if job.profile.squad == "VER3":
                if not should_abort():
                    frames += 1
                cmds.push(parse_control_command({"action": "START", "squad": "VER1", "force": True}))
                should_abort()
                return frames

            return 1

        orch = FifoInferenceOrchestrator(
            command_source=cmds,
            publisher=pub,
            clock=clock,
            job_runner=ttl_runner,
        )
        cmds.push(
            parse_control_command({"action": "START", "squad": "INV1", "ttl_seconds": 10.0}),
            parse_control_command({"action": "START", "squad": "VER5"}),
            parse_control_command({"action": "START", "squad": "VER3"}),
        )

        for _ in range(4):
            orch.step(wait_when_idle=False)

        self.assertEqual(
            [(s.squad, s.reason, s.frames_published) for s in orch.completed_jobs],
            [
                ("INV1", "ttl_expired", 2),
                ("VER5", "stopped_by_command", 1),
                ("VER3", "preempted_by:VER1", 1),
                ("VER1", "source_completed", 1),
            ],
        )


@unittest.skipUnless(HAS_JSONSCHEMA, "requer jsonschema instalado para validar envelope EV-SINAL")
class TestFifoOrchestratorPipelineIntegration(unittest.TestCase):
    def test_end_to_end_fifo_with_mp4_videos_and_real_pipeline(self) -> None:
        from detector.inference import P2InferenceClient
        from detector.publishing import ValidatedStdoutPublisher

        backend = LifecycleTrackingP2Backend()
        cmds = FakeCommandSource()
        pub = ValidatedStdoutPublisher(verbose=False)

        with tempfile.TemporaryDirectory() as tmp:
            videos_dir = Path(tmp)
            for name in ("INV1.mp4", "VER5.mp4"):
                path = videos_dir / name
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
                self.assertTrue(writer.isOpened())
                for _ in range(2):
                    writer.write(np.zeros((240, 320, 3), dtype=np.uint8))
                writer.release()

            cmds.push(
                parse_control_command({"action": "START", "squad": "INV1"}),
                parse_control_command({"action": "START", "squad": "VER5"}),
            )
            orch = FifoInferenceOrchestrator(
                command_source=cmds,
                publisher=pub,
                videos_dir=videos_dir,
                backend_mode="p2",
                client_factory=lambda _cfg, _imgsz: P2InferenceClient(p2_backend=backend),
            )
            orch.step(wait_when_idle=False)
            orch.step(wait_when_idle=False)

        self.assertEqual(
            backend.events,
            [
                ("load", "app-sports-1.2.0"),
                ("release", "app-sports-1.2.0"),
                ("load", "app-plates-1.0.0"),
                ("release", "app-plates-1.0.0"),
            ],
        )
        self.assertEqual(len(pub.published_messages), 4)
        self.assertEqual(
            [(m["source_id"], m["artifact_version"], m["sequence"]) for m in pub.published_messages],
            [
                ("INV1", "app-sports-1.2.0-fp16", 0),
                ("INV1", "app-sports-1.2.0-fp16", 1),
                ("VER5", "app-plates-1.0.0-fp16", 0),
                ("VER5", "app-plates-1.0.0-fp16", 1),
            ],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
