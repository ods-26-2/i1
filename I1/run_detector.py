"""
run_detector.py — Ponto de entrada (CLI) para executar o componente I1 sobre:
  - Webcam ao vivo (--source 0)
  - Arquivo de vídeo (--source caminho/video.mp4)
  - Pasta de imagens de teste (--source I1/artifacts/app-plates/1.0.0/fixtures)

Salva os eventos validados em JSONL e exibe resumo de telemetria (Degrau A - T16).
"""

from __future__ import annotations
from detector.publishing import ConfluentKafkaPublisher, ValidatedStdoutPublisher

import argparse
import json
from pathlib import Path
import sys
from typing import Iterator
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent if PROJECT_ROOT.name == "I1" else PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.config import AppConfig
from detector.contracts import Frame, utc_now_iso_ms
from detector.inference import P2InferenceClient, UltralyticsP2Adapter
from detector.ingestion import OpenCVFrameSource
from detector.pipeline import DetectorPipeline
from detector.publishing import ValidatedStdoutPublisher


class DirectoryImageSource:
    """Lê uma pasta de imagens (.jpg, .png) como se fosse um stream de quadros ods.p4.frame."""

    def __init__(self, folder: Path, source_id: str, session_id: str, max_frames: int | None = None) -> None:
        self.folder = folder
        self.source_id = source_id
        self.session_id = session_id
        self.max_frames = max_frames

    def frames(self) -> Iterator[Frame]:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        files = sorted([p for p in self.folder.iterdir() if p.suffix.lower() in exts])
        if self.max_frames is not None:
            files = files[: self.max_frames]

        for idx, img_path in enumerate(files):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            yield Frame(
                image=bgr,
                captured_at=utc_now_iso_ms(),
                source_id=self.source_id,
                session_id=self.session_id,
                frame_index=idx,
                image_width=w,
                image_height=h,
                pixel_format="BGR",
                fps=30.0,
                raw=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o pipeline completo do Detector I1 (Degrau A)")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="0 para webcam, caminho de um vídeo .mp4, ou pasta de imagens (ex: I1/artifacts/app-plates/1.0.0/fixtures)",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/app-plates/1.0.0/app-plates.onnx"),
        help="Caminho para o artefato .onnx gerado por B5",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/app-plates/1.0.0/metadata.json"),
        help="Caminho para o metadata.json gerado por B5",
    )
    parser.add_argument("--max-frames", type=int, default=20, help="Limite de quadros para processar")
    parser.add_argument("--conf", type=float, default=0.50, help="Limiar de confiança")
    parser.add_argument("--output-jsonl", type=str, default="degrau_a_events.jsonl", help="Arquivo de saída dos eventos")

    parser.add_argument(
        "--mode",
        choices=["stdout", "kafka"],
        default="stdout",
        help="Destino de publicação: 'stdout' (terminal local) ou 'kafka' (broker B3)",
    )

    args = parser.parse_args()

    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg = AppConfig(
        artifact_id=meta["artifact_id"],
        artifact_version=meta["artifact_version"],
        engine_ref=args.onnx,
        classes=meta["classes"],
        conf_threshold=args.conf,
    )

    src_path = Path(args.source)
    if src_path.is_dir():
        source = DirectoryImageSource(src_path, cfg.source_id, cfg.session_id, max_frames=args.max_frames)
    else:
        uri: int | str = int(args.source) if args.source.isdigit() else args.source
        source = OpenCVFrameSource(uri=uri, source_id=cfg.source_id, session_id=cfg.session_id, max_frames=args.max_frames)

    if args.mode == "kafka":
        publisher = ConfluentKafkaPublisher(
            bootstrap_servers=cfg.broker_bootstrap_servers,
            topic=cfg.topic,
        )
    else:
        publisher = ValidatedStdoutPublisher(verbose=True)

    client = P2InferenceClient(p2_backend=UltralyticsP2Adapter())
    pipeline = DetectorPipeline(cfg, source, publisher, client)

    print(f"[I1] Iniciando execução com artefato {cfg.artifact_id} v{cfg.artifact_version} sobre '{args.source}'...\n")
    pipeline.run()  

    if hasattr(publisher, "flush"):
        remaining = publisher.flush()
        print(f"[I1 - Kafka] Flush concluído (mensagens pendentes na fila: {remaining})")

    out_file = Path(args.output_jsonl)
    with open(out_file, "w", encoding="utf-8") as f:
        for msg in publisher.published_messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    total_frames = len(publisher.published_messages)
    total_dets = sum(len(m["payload"]["detections"]) for m in publisher.published_messages)
    avg_lat = (
        sum(t["infer_latency_ms"] for t in pipeline.telemetry_history) / len(pipeline.telemetry_history)
        if pipeline.telemetry_history
        else 0.0
    )

    print(f"\n[I1 - Resumo Degrau A]")
    print(f"  - Quadros processados e validados (B1): {total_frames}")
    print(f"  - Total de objetos detectados:          {total_dets}")
    print(f"  - Latência média de inferência:         {avg_lat:.2f} ms")
    print(f"  - Eventos salvos em:                    {out_file.resolve()}")


if __name__ == "__main__":
    main()  