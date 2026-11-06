from __future__ import annotations

import json
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any
import unittest
from unittest import mock

import cv2
import numpy as np

I1_ROOT = Path(__file__).resolve().parent.parent
if str(I1_ROOT) not in sys.path:
    sys.path.insert(0, str(I1_ROOT))

from detector.config import AppConfig
from detector.contracts import Frame, P2ExecutionError
from detector.inference import (
    NativeP2Backend,
    P2InferenceClient,
    _decode_yolo_raw_output,
    _letterbox_bgr,
)
from detector.ingestion import (
    OpenCVFrameSource,
    P4GStreamerShmSource,
    _image_array_from_p4_data,
    parse_p4r1_packet,
    preprocess_frame_to_bgr,
)
from detector.pipeline import DetectorPipeline
from detector.publishing import ValidatedStdoutPublisher
from run_detector import DirectoryImageSource, resolve_frame_source


def build_p4r1_packet(
    image_data: bytes,
    *,
    pixel_format: str = "RG10",
    width: int = 4,
    height: int = 4,
) -> bytes:
    metadata = {
        "source_id": "imx519",
        "session_id": "1a2b3c4d-5e6f-4a7b-8c9d-0e1f2a3b4c5d",
        "frame_id": 7,
        "timestamp": 1_000_000_000,
        "width": width,
        "height": height,
        "format": pixel_format,
        "raw": True,
        "data_size": len(image_data),
    }
    encoded_metadata = json.dumps(metadata).encode("utf-8")
    return b"P4R1" + struct.pack("!I", len(encoded_metadata)) + encoded_metadata + image_data


class TestP4Ingestion(unittest.TestCase):
    def test_p4r1_header_supplies_frame_metadata_and_payload(self) -> None:
        image_data = np.zeros((4, 4), dtype="<u2").tobytes()
        metadata, payload = parse_p4r1_packet(build_p4r1_packet(image_data))

        self.assertEqual(metadata["source_id"], "imx519")
        self.assertEqual(metadata["frame_id"], 7)
        self.assertEqual(metadata["format"], "RG10")
        self.assertEqual(payload, image_data)

    def test_p4r1_rejects_bad_magic_and_payload_size(self) -> None:
        valid = build_p4r1_packet(bytes(4 * 4 * 2))
        with self.assertRaisesRegex(ValueError, "P4R1"):
            parse_p4r1_packet(b"BAD!" + valid[4:])

        with self.assertRaisesRegex(ValueError, "data_size"):
            parse_p4r1_packet(valid[:-1])

    def test_rg10_packet_converts_to_bgr_for_inference(self) -> None:
        raw_data = np.full((4, 4), 512, dtype="<u2").tobytes()
        metadata, payload = parse_p4r1_packet(build_p4r1_packet(raw_data))
        image = _image_array_from_p4_data(
            payload, metadata["width"], metadata["height"], metadata["format"]
        )
        frame = Frame(
            image=image,
            captured_at="2026-11-05T00:00:00.000Z",
            source_id=metadata["source_id"],
            session_id=metadata["session_id"],
            frame_index=metadata["frame_id"],
            image_width=metadata["width"],
            image_height=metadata["height"],
            pixel_format=metadata["format"],
        )

        converted = preprocess_frame_to_bgr(frame)

        self.assertEqual(converted.shape, (4, 4, 3))
        self.assertEqual(converted.dtype, np.uint8)

    def test_bgr8_packet_from_p4_video_decodes_to_bgr(self) -> None:
        bgr_data = np.full((4, 4, 3), 128, dtype=np.uint8).tobytes()
        metadata, payload = parse_p4r1_packet(
            build_p4r1_packet(bgr_data, pixel_format="BGR8", width=4, height=4)
        )
        self.assertEqual(metadata["format"], "BGR8")
        image = _image_array_from_p4_data(
            payload, metadata["width"], metadata["height"], metadata["format"]
        )
        frame = Frame(
            image=image,
            captured_at="2026-11-05T00:00:00.000Z",
            source_id="file_video",
            session_id=metadata["session_id"],
            frame_index=metadata["frame_id"],
            image_width=metadata["width"],
            image_height=metadata["height"],
            pixel_format=metadata["format"],
        )
        converted = preprocess_frame_to_bgr(frame)
        self.assertEqual(converted.shape, (4, 4, 3))
        self.assertEqual(converted.dtype, np.uint8)
        np.testing.assert_array_equal(converted, image)

    def test_rg10_rejects_unexpected_buffer_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "Frame RG10 incompatível"):
            _image_array_from_p4_data(bytes(10), 4, 4, "RG10")

    def test_missing_capture_timestamp_uses_valid_receive_time(self) -> None:
        captured_at = P4GStreamerShmSource._monotonic_ns_to_iso_ms(None)

        self.assertRegex(captured_at, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")


class TestP2VideoAndSourceResolution(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = AppConfig(
            artifact_id="app-plates",
            artifact_version="1.0.0",
            engine_ref="app-plates-1.0.0",
            classes=["placa", "veiculo"],
            conf_threshold=0.50,
            source_id="cam-video",
        )

    def test_resolve_frame_source_distinguishes_shm_video_and_directory(self) -> None:
        shm_src = resolve_frame_source("shm", "/tmp/p4_imx519_shm", self.cfg, 10)
        self.assertIsInstance(shm_src, P4GStreamerShmSource)
        self.assertEqual(shm_src.shm_socket_path, "/tmp/p4_imx519_shm")

        imx477_src = resolve_frame_source("imx477", "/tmp/p4_imx519_shm", self.cfg, 10)
        self.assertIsInstance(imx477_src, P4GStreamerShmSource)
        self.assertEqual(imx477_src.shm_socket_path, "/tmp/p4_imx477_shm")

        p4_file_src = resolve_frame_source("p4_video", "/tmp/p4_imx519_shm", self.cfg, 10)
        self.assertIsInstance(p4_file_src, P4GStreamerShmSource)
        self.assertEqual(p4_file_src.shm_socket_path, "/tmp/p4_file_shm")

        sock_src = resolve_frame_source("/tmp/p4_imx519_shm", "/tmp/p4_imx519_shm", self.cfg, 10)
        self.assertIsInstance(sock_src, P4GStreamerShmSource)

        tmp_video_src = resolve_frame_source("/tmp/video_teste.mp4", "/tmp/p4_imx519_shm", self.cfg, 10)
        self.assertIsInstance(tmp_video_src, OpenCVFrameSource)
        self.assertEqual(tmp_video_src.uri, "/tmp/video_teste.mp4")

        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            dir_src = resolve_frame_source(tmp_dir, "/tmp/p4_imx519_shm", self.cfg, 10)
            self.assertIsInstance(dir_src, DirectoryImageSource)

    def test_native_p2_backend_with_mp4_video_pipeline(self) -> None:
        released_handles: list[int] = []

        class FakeP2Module:
            @staticmethod
            def load(engine_ref: str, options: dict[str, Any] | None = None) -> int:
                assert engine_ref == "app-plates-1.0.0"
                return 99

            @staticmethod
            def info(handle: int) -> dict[str, Any]:
                assert handle == 99
                return {
                    "engine_version": "app-plates-1.0.0-fp16",
                    "input_shape": {"images": [1, 3, 640, 640]},
                    "output_shape": {"output0": [1, 6, 8400]},
                    "precision": "FP16",
                    "max_batch": 1,
                }

            @staticmethod
            def infer(handle: int, inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
                assert handle == 99
                tensor = inputs["images"]
                assert tensor.shape == (1, 3, 640, 640)
                assert tensor.dtype == np.float32
                out = np.zeros((1, 6, 8400), dtype=np.float32)
                # Caixa centrada em (320, 320) com largura 160 e altura 80 no espaço letterbox 640x640
                out[0, 0, 0] = 320.0
                out[0, 1, 0] = 320.0
                out[0, 2, 0] = 160.0
                out[0, 3, 0] = 80.0
                out[0, 4, 0] = 0.91  # classe 0: placa
                out[0, 5, 0] = 0.12  # classe 1: veiculo
                return {"output0": out}, {"latency_ms": 3.75, "gpu_memory_mb": 256.0}

            @staticmethod
            def release(handle: int) -> None:
                released_handles.append(handle)

        with tempfile.TemporaryDirectory() as tmp_dir:
            video_path = Path(tmp_dir) / "input_sample.mp4"
            width, height = 320, 240
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                15.0,
                (width, height),
            )
            self.assertTrue(writer.isOpened(), "OpenCV VideoWriter deveria abrir para criar MP4 de teste")
            for idx in range(4):
                frame_bgr = np.full((height, width, 3), idx * 30, dtype=np.uint8)
                writer.write(frame_bgr)
            writer.release()

            with mock.patch.dict(sys.modules, {"p2": FakeP2Module}):
                backend = NativeP2Backend(precision="FP16")
                source = resolve_frame_source(str(video_path), "/tmp/p4_imx519_shm", self.cfg, max_frames=4)
                self.assertIsInstance(source, OpenCVFrameSource)
                pub = ValidatedStdoutPublisher(verbose=False)
                pipe = DetectorPipeline(self.cfg, source, pub, P2InferenceClient(backend))
                pipe.run()

        self.assertEqual(released_handles, [99])
        self.assertEqual(len(pub.published_messages), 4)
        self.assertEqual([m["sequence"] for m in pub.published_messages], [0, 1, 2, 3])
        self.assertEqual([m["payload"]["frame_index"] for m in pub.published_messages], [0, 1, 2, 3])
        for msg in pub.published_messages:
            self.assertEqual(msg["artifact_version"], "app-plates-1.0.0-fp16")
            self.assertEqual(msg["payload"]["image_width"], width)
            self.assertEqual(msg["payload"]["image_height"], height)
            dets = msg["payload"]["detections"]
            self.assertEqual(len(dets), 1)
            self.assertEqual(dets[0]["class"], "placa")
            self.assertAlmostEqual(dets[0]["confidence"], 0.91, places=4)
            # Centro (320, 320) em letterbox 640x640 corresponde ao centro (160, 120) da imagem 320x240 (scale=2.0, pad_y=80)
            self.assertEqual(dets[0]["bbox"], [120, 100, 200, 140])

    def test_letterbox_and_decode_reject_invalid_shapes(self) -> None:
        with self.assertRaises(P2ExecutionError) as ctx_in:
            _letterbox_bgr(np.zeros((0, 10, 3), dtype=np.uint8))
        self.assertEqual(ctx_in.exception.code, "INVALID_INPUT_SHAPE")

        boxed = _letterbox_bgr(np.zeros((240, 320, 3), dtype=np.uint8), size=640)
        with self.assertRaises(P2ExecutionError) as ctx_out:
            _decode_yolo_raw_output(
                np.zeros((1, 5, 8400), dtype=np.float32),
                boxed,
                (240, 320),
                expected_num_classes=2,
            )
        self.assertEqual(ctx_out.exception.code, "RUNTIME_ERROR")


if __name__ == "__main__":
    unittest.main()
