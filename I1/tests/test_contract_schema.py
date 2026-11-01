"""
test_contract_schema.py — Contrato ods.i1.detections 1.0 validado a partir do JSON Schema oficial (T2 / T15).
Não requer GPU, torch nem pesos: cobre o schema, o validador do publicador e a serialização real de build_envelope.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from typing import Any, Callable

import numpy as np

I1_ROOT = Path(__file__).resolve().parent.parent
if str(I1_ROOT) not in sys.path:
    sys.path.insert(0, str(I1_ROOT))

from detector.contracts import Detection, Frame
from detector.publishing import ValidatedStdoutPublisher, build_envelope, validate_ods_i1_detections_schema
from detector.schema_validation import schema_errors

# Exemplo de ods.i1.detections da Especificação de Interfaces 1.0 (seção 6), com o envelope EV-SINAL completo
SPEC_EXAMPLE: dict[str, Any] = {
    "message_type": "event",
    "message_id": "5f1e2c7a-9b1d-4e0a-8c6f-2d3a4b5c6d7e",
    "schema": "ods.i1.detections",
    "schema_version": "1.0",
    "producer": "I1",
    "published_at": "2026-09-23T14:30:00.120Z",
    "captured_at": "2026-09-23T14:29:59.870Z",
    "source_id": "cam0",
    "session_id": "9c2a1f3e-4d5b-4c6a-9e8f-0a1b2c3d4e5f",
    "sequence": 48213,
    "artifact_version": "yolo-a5-1.4.0",
    "payload": {
        "frame_index": 48213,
        "image_width": 1920,
        "image_height": 1080,
        "detections": [
            {"detection_id": 0, "class": "person", "confidence": 0.93, "bbox": [412, 180, 598, 702]},
            {"detection_id": 1, "class": "helmet", "confidence": 0.81, "bbox": [455, 182, 548, 251]},
        ],
    },
}

ENVELOPE_REQUIRED = [
    "message_type", "message_id", "schema", "schema_version", "producer", "published_at",
    "captured_at", "source_id", "session_id", "sequence", "artifact_version", "payload",
]
PAYLOAD_REQUIRED = ["frame_index", "image_width", "image_height", "detections"]


def _det(msg: dict[str, Any]) -> dict[str, Any]:
    return msg["payload"]["detections"][0]


VALID_MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    "exemplo da especificação": lambda m: None,
    "quadro vazio": lambda m: m["payload"].__setitem__("detections", []),
    "correlation_id opcional": lambda m: m.__setitem__("correlation_id", "c3d4e5f6-7a8b-4c9d-8e0f-1a2b3c4d5e6f"),
    "confidence nos limites": lambda m: _det(m).__setitem__("confidence", 1.0),
    "sequence zero": lambda m: m.__setitem__("sequence", 0),
}

INVALID_MUTATIONS: dict[str, Callable[[dict[str, Any]], None]] = {
    # Campos legados do rascunho da Sprint 2 (ods.inferencia.deteccao_frame)
    "schema legado": lambda m: m.__setitem__("schema", "ods.inferencia.deteccao_frame"),
    "camera_id no envelope": lambda m: m.__setitem__("camera_id", "cam0"),
    "sequence_number no envelope": lambda m: m.__setitem__("sequence_number", 0),
    "artifact_id no envelope": lambda m: m.__setitem__("artifact_id", "app-epi"),
    "camera_id no payload": lambda m: m["payload"].__setitem__("camera_id", "cam0"),
    "session_id repetido no payload": lambda m: m["payload"].__setitem__("session_id", m["session_id"]),
    # Envelope
    "target em evento": lambda m: m.__setitem__("target", "I2"),
    "message_type command": lambda m: m.__setitem__("message_type", "command"),
    "producer diferente de I1": lambda m: m.__setitem__("producer", "I2"),
    "schema_version 2.0": lambda m: m.__setitem__("schema_version", "2.0"),
    "message_id não UUID": lambda m: m.__setitem__("message_id", "abc"),
    "message_id UUID v1": lambda m: m.__setitem__("message_id", "5f1e2c7a-9b1d-1e0a-8c6f-2d3a4b5c6d7e"),
    "session_id não UUID": lambda m: m.__setitem__("session_id", "sess-A"),
    "correlation_id não UUID": lambda m: m.__setitem__("correlation_id", "abc"),
    "captured_at sem milissegundos": lambda m: m.__setitem__("captured_at", "2026-09-23T14:29:59Z"),
    "published_at fora de UTC": lambda m: m.__setitem__("published_at", "2026-09-23T11:30:00.120-03:00"),
    "sequence negativo": lambda m: m.__setitem__("sequence", -1),
    "sequence booleano": lambda m: m.__setitem__("sequence", True),
    "sequence string": lambda m: m.__setitem__("sequence", "1"),
    "source_id vazio": lambda m: m.__setitem__("source_id", ""),
    "artifact_version vazio": lambda m: m.__setitem__("artifact_version", ""),
    # Payload
    "image_width zero": lambda m: m["payload"].__setitem__("image_width", 0),
    "frame_index negativo": lambda m: m["payload"].__setitem__("frame_index", -1),
    "detections nulo": lambda m: m["payload"].__setitem__("detections", None),
    # Itens de detecção
    "detection_id negativo": lambda m: _det(m).__setitem__("detection_id", -1),
    "detection_id booleano": lambda m: _det(m).__setitem__("detection_id", True),
    "class vazia": lambda m: _det(m).__setitem__("class", ""),
    "confidence acima de 1": lambda m: _det(m).__setitem__("confidence", 1.5),
    "confidence negativa": lambda m: _det(m).__setitem__("confidence", -0.1),
    "bbox float": lambda m: _det(m).__setitem__("bbox", [412.5, 180.0, 598.0, 702.0]),
    "bbox com 3 valores": lambda m: _det(m).__setitem__("bbox", [412, 180, 598]),
    "bbox negativa": lambda m: _det(m).__setitem__("bbox", [-1, 180, 598, 702]),
    "campo extra na detecção": lambda m: _det(m).__setitem__("track_id", "t-0391"),
}


def _mutated(mutation: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    msg = copy.deepcopy(SPEC_EXAMPLE)
    mutation(msg)
    return msg


class TestContractSchema(unittest.TestCase):
    def test_valid_messages_pass(self) -> None:
        for name, mutation in VALID_MUTATIONS.items():
            with self.subTest(name):
                msg = _mutated(mutation)
                self.assertEqual(schema_errors(msg), [])
                validate_ods_i1_detections_schema(msg)

    def test_invalid_messages_are_rejected(self) -> None:
        for name, mutation in INVALID_MUTATIONS.items():
            with self.subTest(name):
                msg = _mutated(mutation)
                self.assertNotEqual(schema_errors(msg), [])
                with self.assertRaises(ValueError):
                    validate_ods_i1_detections_schema(msg)

    def test_missing_required_fields_are_rejected(self) -> None:
        for field_name in ENVELOPE_REQUIRED:
            with self.subTest(envelope=field_name):
                msg = copy.deepcopy(SPEC_EXAMPLE)
                del msg[field_name]
                with self.assertRaises(ValueError):
                    validate_ods_i1_detections_schema(msg)
        for field_name in PAYLOAD_REQUIRED:
            with self.subTest(payload=field_name):
                msg = copy.deepcopy(SPEC_EXAMPLE)
                del msg["payload"][field_name]
                with self.assertRaises(ValueError):
                    validate_ods_i1_detections_schema(msg)


class TestPublisherOutputMatchesSchema(unittest.TestCase):
    """A serialização real de build_envelope/EvSinalEnvelope deve sair conforme o schema."""

    def _frame(self, session_id: str) -> Frame:
        return Frame(
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
            captured_at="2026-10-18T12:00:00.100Z",
            source_id="cam0",
            session_id=session_id,
            frame_index=42,
            image_width=1280,
            image_height=720,
        )

    def test_built_envelopes_are_valid(self) -> None:
        frame = self._frame("9c2a1f3e-4d5b-4c6a-9e8f-0a1b2c3d4e5f")
        detections = [
            Detection(detection_id=0, class_name="placa_1L_BR", confidence=0.98765, bbox=[10, 20, 110, 60]),
            Detection(detection_id=1, class_name="placa_2L_ME", confidence=0.5, bbox=[200, 300, 280, 360]),
        ]
        for dets in ([], detections):
            with self.subTest(num_detections=len(dets)):
                msg = build_envelope(frame, sequence=0, artifact_version="app-plates-1.0.0", detections=dets).to_dict()
                self.assertEqual(schema_errors(msg), [])

    def test_publisher_blocks_invalid_session_id(self) -> None:
        envelope = build_envelope(self._frame("sess-A"), sequence=0, artifact_version="app-plates-1.0.0", detections=[])
        publisher = ValidatedStdoutPublisher(verbose=False)
        with self.assertRaises(ValueError):
            publisher.publish(envelope)
        self.assertEqual(publisher.published_messages, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
