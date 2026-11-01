"""
test_contract_schema.py — Contrato ods.i1.detections 1.0 validado a partir do JSON Schema oficial (T2 / T15).
Os casos vêm de schemas/examples/{valid,invalid}/*.json, os mesmos entregues a B1 (TRA-1) e à CI de B4 (TRA-2).
Não requer GPU, torch nem pesos: cobre o schema, o validador do publicador e a serialização real de build_envelope.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest
from typing import Any

import numpy as np

I1_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = I1_ROOT.parent
if str(I1_ROOT) not in sys.path:
    sys.path.insert(0, str(I1_ROOT))

from detector.contracts import Detection, Frame
from detector.publishing import ValidatedStdoutPublisher, build_envelope, validate_ods_i1_detections_schema
from detector.schema_validation import schema_errors

EXAMPLES_DIR = REPO_ROOT / "schemas" / "examples"
SPEC_EXAMPLE_FILE = EXAMPLES_DIR / "valid" / "01_exemplo_da_especificacao.json"

ENVELOPE_REQUIRED = [
    "message_type", "message_id", "schema", "schema_version", "producer", "published_at",
    "captured_at", "source_id", "session_id", "sequence", "artifact_version", "payload",
]
PAYLOAD_REQUIRED = ["frame_index", "image_width", "image_height", "detections"]


def load_examples(kind: str) -> dict[str, dict[str, Any]]:
    files = sorted((EXAMPLES_DIR / kind).glob("*.json"))
    return {p.name: json.loads(p.read_text(encoding="utf-8")) for p in files}


class TestContractSchema(unittest.TestCase):
    def test_valid_examples_pass(self) -> None:
        examples = load_examples("valid")
        self.assertTrue(examples, "Nenhum exemplo válido encontrado")
        for name, msg in examples.items():
            with self.subTest(name):
                self.assertEqual(schema_errors(msg), [])
                validate_ods_i1_detections_schema(msg)

    def test_invalid_examples_are_rejected(self) -> None:
        examples = load_examples("invalid")
        self.assertTrue(examples, "Nenhum exemplo inválido encontrado")
        for name, msg in examples.items():
            with self.subTest(name):
                self.assertNotEqual(schema_errors(msg), [])
                with self.assertRaises(ValueError):
                    validate_ods_i1_detections_schema(msg)

    def test_missing_required_fields_are_rejected(self) -> None:
        base = json.loads(SPEC_EXAMPLE_FILE.read_text(encoding="utf-8"))
        for field_name in ENVELOPE_REQUIRED:
            with self.subTest(envelope=field_name):
                msg = copy.deepcopy(base)
                del msg[field_name]
                with self.assertRaises(ValueError):
                    validate_ods_i1_detections_schema(msg)
        for field_name in PAYLOAD_REQUIRED:
            with self.subTest(payload=field_name):
                msg = copy.deepcopy(base)
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
