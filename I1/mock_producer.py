"""
mock_producer.py — Produtor falso (mock) oficial de I1 para a CI de B4 e os consumidores (T2 / T15).
Não requer GPU nem pesos YOLO. Os payloads vêm de schemas/examples/valid/, os mesmos exemplos que o validador
aceita: mock e validador nascem do mesmo JSON Schema. O envelope de cada mensagem (message_id, tempos,
sessão, sequence, frame_index) é gerado aqui, e toda mensagem passa pelo validador do publicador real.

Uso na CI ou por outras equipes:
  python I1/mock_producer.py --frames 5 --mode stdout
  ODS_KAFKA_BROKER=<host:porta> python I1/mock_producer.py --frames 10 --mode kafka
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.config import BROKER_ENV_VAR
from detector.contracts import Detection, DetectionPayload, EvSinalEnvelope, utc_now_iso_ms
from detector.publishing import ConfluentKafkaPublisher, ValidatedStdoutPublisher

VALID_EXAMPLES_DIR = PROJECT_ROOT.parent / "schemas" / "examples" / "valid"
DEFAULT_SESSION_ID = "9c2a1f3e-4d5b-4c6a-9e8f-0a1b2c3d4e5f"


def load_example_payloads(examples_dir: Path = VALID_EXAMPLES_DIR) -> list[dict[str, Any]]:
    """Payloads dos exemplos válidos, em ordem de arquivo (inclui o quadro vazio)."""
    files = sorted(examples_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Nenhum exemplo válido em {examples_dir}")
    return [json.loads(p.read_text(encoding="utf-8"))["payload"] for p in files]


def payload_from_example(example: dict[str, Any], frame_index: int) -> DetectionPayload:
    return DetectionPayload(
        frame_index=frame_index,
        image_width=example["image_width"],
        image_height=example["image_height"],
        detections=[
            Detection(detection_id=d["detection_id"], class_name=d["class"], confidence=d["confidence"], bbox=d["bbox"])
            for d in example["detections"]
        ],
    )


def generate_mock_envelopes(
    count: int,
    session_id: str = DEFAULT_SESSION_ID,
    source_id: str = "cam0",
    artifact_version: str = "mock-1.0.0",
) -> Iterator[EvSinalEnvelope]:
    """Uma mensagem por quadro, percorrendo os exemplos válidos em ciclo; sequence e frame_index a partir de 0."""
    payloads = load_example_payloads()
    for seq in range(count):
        captured_at = utc_now_iso_ms()
        yield EvSinalEnvelope(
            captured_at=captured_at,
            source_id=source_id,
            session_id=session_id,
            sequence=seq,
            artifact_version=artifact_version,
            payload=payload_from_example(payloads[seq % len(payloads)], frame_index=seq),
            published_at=utc_now_iso_ms(),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Produtor Mock Oficial de I1 (ods.i1.detections 1.0)")
    parser.add_argument("--frames", type=int, default=5, help="Quantidade de eventos a emitir")
    parser.add_argument("--mode", choices=["stdout", "kafka"], default="stdout")
    parser.add_argument(
        "--broker",
        type=str,
        default=os.getenv(BROKER_ENV_VAR),
        help=f"host:porta do broker no modo kafka (padrão: variável {BROKER_ENV_VAR}; sem ela, o modo kafka falha)",
    )
    parser.add_argument("--topic", type=str, default="ods.i1.detections")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--session-id", type=str, default=DEFAULT_SESSION_ID, help="UUID v4 da sessão simulada")
    parser.add_argument("--source-id", type=str, default="cam0")
    parser.add_argument("--artifact-version", type=str, default="mock-1.0.0")
    args = parser.parse_args()

    publisher = (
        ConfluentKafkaPublisher(bootstrap_servers=args.broker, topic=args.topic)
        if args.mode == "kafka"
        else ValidatedStdoutPublisher(verbose=True)
    )

    envelopes = generate_mock_envelopes(args.frames, args.session_id, args.source_id, args.artifact_version)
    for seq, env in enumerate(envelopes):
        publisher.publish(env)
        if args.fps > 0 and seq < args.frames - 1:
            time.sleep(1.0 / args.fps)

    if hasattr(publisher, "flush"):
        remaining = publisher.flush()
        if remaining > 0:
            print(
                f"[I1 - Mock Kafka] Falha ao entregar {remaining} mensagem(ns) ao broker '{args.broker}'.",
                file=sys.stderr,
            )
            sys.exit(3)


if __name__ == "__main__":
    main()
