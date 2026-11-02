"""
mock_producer.py — Produtor falso (mock) oficial e validador de contrato de I1 para a CI de B4 (T2 / T15).
Não requer GPU nem pesos YOLO. Emite mensagens válidas sob o schema ods.i1.detections 1.0 (EV-SINAL).

Uso na CI ou por outras equipas:
  python I1/mock_producer.py --frames 5 --mode stdout
  python I1/mock_producer.py --frames 10 --mode kafka --broker 34.66.235.91:9093
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.contracts import Detection, DetectionPayload, EvSinalEnvelope, utc_now_iso_ms
from detector.publishing import ConfluentKafkaPublisher, ValidatedStdoutPublisher


def generate_mock_envelope(seq: int, session_id: str, source_id: str, empty: bool = False) -> EvSinalEnvelope:
    dets = [] if empty else [
        Detection(detection_id=0, class_name="pessoa", confidence=0.94, bbox=[412, 180, 598, 702]),
        Detection(detection_id=1, class_name="capacete", confidence=0.96, bbox=[450, 182, 540, 255]),
    ]
    payload = DetectionPayload(
        frame_index=seq,
        image_width=1280,
        image_height=720,
        detections=dets,
    )
    cap_time = utc_now_iso_ms()
    return EvSinalEnvelope(
        captured_at=cap_time,
        source_id=source_id,
        session_id=session_id,
        sequence=seq,
        artifact_version="app-epi-1.0.0-mock",
        payload=payload,
        published_at=utc_now_iso_ms(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Produtor Mock Oficial de I1 (ods.i1.detections 1.0)")
    parser.add_argument("--frames", type=int, default=5, help="Quantidade de eventos a emitir")
    parser.add_argument("--mode", choices=["stdout", "kafka"], default="stdout")
    parser.add_argument("--broker", type=str, default="34.66.235.91:9093")
    parser.add_argument("--topic", type=str, default="ods.i1.detections")
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    publisher = (
        ConfluentKafkaPublisher(bootstrap_servers=args.broker, topic=args.topic)
        if args.mode == "kafka"
        else ValidatedStdoutPublisher(verbose=True)
    )

    session_id = "9c2a1f3e-4d5b-4c6a-9e8f-0a1b2c3d4e5f"
    for seq in range(args.frames):
        # A cada 3 quadros emite um quadro vazio ("detections": []) para testar a regra obrigatória nos consumidores
        is_empty = (seq % 3 == 2)
        env = generate_mock_envelope(seq=seq, session_id=session_id, source_id="cam0", empty=is_empty)
        publisher.publish(env)
        if args.fps > 0 and seq < args.frames - 1:
            time.sleep(1.0 / args.fps)

    if hasattr(publisher, "flush"):
        publisher.flush()


if __name__ == "__main__":
    main()