"""
run_detector.py — Ponto de entrada (CLI) para executar o componente I1:
  - Imagens de teste / datasets (--source caminho/pasta)
  - Vídeo ou Webcam (--source 0 ou video.mp4)
  - Memória compartilhada IPC/GStreamer da Jetson com P4 (--source shm)

Suporta publicação em stdout ou Kafka, e comutação automática entre
a biblioteca oficial 'p2' (SIS-1) e o executor local ONNX Runtime/Ultralytics.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Iterator
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent if PROJECT_ROOT.name == "I1" else PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.config import AppConfig
from detector.contracts import Frame, P2ExecutionError, RawDetection, utc_now_iso_ms
from detector.inference import (
    MockP2Backend,
    NativeP2Backend,
    P2InferenceClient,
    UltralyticsP2Adapter,
)
from detector.ingestion import (
    OpenCVFrameSource,
    P4GStreamerShmSource,
    unreadable_frame,
)
from detector.pipeline import DetectorPipeline
from detector.publishing import ConfluentKafkaPublisher, ValidatedStdoutPublisher

MEDIA_EXTENSIONS = frozenset({
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg", ".ts",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
})


class DirectoryImageSource:
    """Lê uma pasta de imagens (.jpg, .png) simulando um fluxo de quadros ods.p4.frame."""

    def __init__(
        self, folder: Path, source_id: str, session_id: str, max_frames: int | None = None
    ) -> None:
        self.folder = folder
        self.source_id = source_id
        self.session_id = session_id
        self.max_frames = max_frames

    def frames(self) -> Iterator[Frame]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted([p for p in self.folder.iterdir() if p.suffix.lower() in exts])
        if self.max_frames is not None:
            files = files[: self.max_frames]

        for idx, img_path in enumerate(files):
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                yield unreadable_frame(self.source_id, self.session_id, idx)
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


P4_SOCKET_ALIASES: dict[str, str] = {
    "imx519": "/tmp/p4_imx519_shm",
    "imx477": "/tmp/p4_imx477_shm",
    "video": "/tmp/p4_file_shm",
    "file": "/tmp/p4_file_shm",
    "p4_video": "/tmp/p4_file_shm",
    "p4_file": "/tmp/p4_file_shm",
    "p4_file_shm": "/tmp/p4_file_shm",
}


def resolve_frame_source(
    source_arg: str,
    shm_socket: str,
    cfg: AppConfig,
    max_frames: int | None,
) -> P4GStreamerShmSource | DirectoryImageSource | OpenCVFrameSource:
    """Seleciona a fonte de quadros (SHM do P4, diretório de imagens ou vídeo/webcam OpenCV)."""
    if source_arg in P4_SOCKET_ALIASES:
        return P4GStreamerShmSource(
            shm_socket_path=P4_SOCKET_ALIASES[source_arg],
            max_frames=max_frames,
        )

    src_path = Path(source_arg)
    is_media_file = src_path.suffix.lower() in MEDIA_EXTENSIONS or (
        src_path.exists() and src_path.is_file() and not src_path.is_socket()
    )
    if source_arg == "shm" or (
        source_arg.startswith("/tmp/") and not src_path.is_dir() and not is_media_file
    ):
        socket_path = shm_socket if source_arg == "shm" else source_arg
        return P4GStreamerShmSource(
            shm_socket_path=socket_path,
            max_frames=max_frames,
        )
    if src_path.is_dir():
        return DirectoryImageSource(
            src_path, cfg.source_id, cfg.session_id, max_frames=max_frames
        )
    uri: int | str = int(source_arg) if source_arg.isdigit() else source_arg
    return OpenCVFrameSource(
        uri=uri,
        source_id=cfg.source_id,
        session_id=cfg.session_id,
        max_frames=max_frames,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o pipeline completo do Detector I1 (Degraus A, B e C)")

    # 1. Entrada de vídeo / frames
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Caminho de pasta de imagens, vídeo .mp4, 0 para webcam, ou 'shm' para socket IPC do P4 na Jetson",
    )
    parser.add_argument(
        "--shm-socket",
        type=str,
        default="/tmp/p4_imx519_shm",
        help="Socket SHM do P4 (P4R1; padrão da câmera IMX519 do P4)",
    )
    parser.add_argument("--max-frames", type=int, default=20, help="Limite de quadros para processar")
    parser.add_argument(
        "--source-id",
        type=str,
        default="cam0",
        help="Identificador da fonte para vídeo, webcam ou pasta de imagens (SHM usa o source_id do P4)",
    )

    # 2. Definição do Modelo / Metadados
    parser.add_argument(
        "--onnx",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/app-plates/1.0.0/app-plates.onnx"),
        help="Caminho do arquivo .onnx (usado no backend ultralytics/ONNX Runtime)",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/app-plates/1.0.0/metadata.json"),
        help="Caminho para o metadata.json gerado por B5",
    )
    parser.add_argument(
        "--engine-ref",
        type=str,
        default=None,
        help="Identificador do engine para a biblioteca p2 (ex: app-plates-1.0.0 ou app-epi-1.0.0)",
    )

    # 3. Motor de Inferência (P2 vs Ultralytics vs Mock)
    parser.add_argument(
        "--backend",
        choices=["auto", "p2", "ultralytics", "mock"],
        default="auto",
        help="Backend de inferência: 'p2' (biblioteca oficial SIS-1), 'ultralytics' (local), 'mock' (simulado), ou 'auto'",
    )
    parser.add_argument("--conf", type=float, default=0.50, help="Limiar de confiança para detecções")

    # 4. Publicação e Saída
    parser.add_argument(
        "--mode",
        choices=["stdout", "kafka"],
        default="stdout",
        help="Destino de publicação: 'stdout' (terminal local) ou 'kafka' (broker B3)",
    )
    parser.add_argument(
        "--broker",
        type=str,
        default=os.getenv(BROKER_ENV_VAR, "172.18.4.56:9092"),
        help=f"host:porta do broker no modo kafka (padrão: 172.18.4.56:9092 na rede Dcomp ou variável {BROKER_ENV_VAR})",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default="degrau_a_events.jsonl",
        help="Arquivo onde salvar o histórico de eventos emitidos",
    )

    args = parser.parse_args()

    # Logs da observabilidade [B2-OBS] vão para o stderr para não corromper o stdout de eventos
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    with open(args.metadata, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Resolve o identificador do modelo
    engine_ref = args.engine_ref or f"{meta['artifact_id']}-{meta['artifact_version']}"
    if args.backend == "ultralytics":
        engine_ref = args.onnx

    cfg = AppConfig(
        artifact_id=meta["artifact_id"],
        artifact_version=meta["artifact_version"],
        engine_ref=engine_ref,
        classes=meta["classes"],
        conf_threshold=args.conf,
        source_id=args.source_id,
    )

    # --- Seleção da Fonte de Quadros (Source) ---
    source = resolve_frame_source(args.source, args.shm_socket, cfg, args.max_frames)

    out_file = Path(args.output_jsonl)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    jsonl_stream = out_file.open("w", encoding="utf-8", buffering=1)

    def persist_message(message: dict[str, Any]) -> None:
        jsonl_stream.write(json.dumps(message, ensure_ascii=False) + "\n")

    # --- Seleção do Publicador (Publisher) ---
    broker_server = args.broker or cfg.broker_bootstrap_servers or "172.18.4.56:9092"
    if args.mode == "kafka":
        publisher = ConfluentKafkaPublisher(
            bootstrap_servers=broker_server,
            topic=cfg.topic,
            history_limit=0,
            on_publish=persist_message,
        )
    else:
        publisher = ValidatedStdoutPublisher(
            verbose=True,
            history_limit=0,
            on_publish=persist_message,
        )

    # --- Seleção do Backend de Inferência (P2 Client) ---
    model_imgsz = int(meta.get("input_shape", [1, 3, 640, 640])[2])
    if args.backend == "p2":
        backend = NativeP2Backend()
    elif args.backend == "ultralytics":
        backend = UltralyticsP2Adapter(imgsz=model_imgsz)
    elif args.backend == "mock":
        backend = MockP2Backend(
            simulated_detections=[
                RawDetection(class_id=0, confidence=0.92, bbox=(20.0, 20.0, 120.0, 80.0)),
            ]
        )
    else:
        # Modo "auto": tenta a biblioteca oficial 'p2' (Jetson/com TensorRT); se não houver, usa Ultralytics
        try:
            import p2  # noqa: F401
            backend = NativeP2Backend()
        except ImportError:
            backend = UltralyticsP2Adapter(imgsz=model_imgsz)
            cfg = AppConfig(
                artifact_id=meta["artifact_id"],
                artifact_version=meta["artifact_version"],
                engine_ref=args.onnx,
                classes=meta["classes"],
                conf_threshold=args.conf,
                source_id=args.source_id,
            )

    client = P2InferenceClient(p2_backend=backend)
    pipeline = DetectorPipeline(cfg, source, publisher, client)

    print(f"[I1] Iniciando execução com artefato {cfg.artifact_id} v{cfg.artifact_version} sobre '{args.source}'...\n")
    try:
        pipeline.run()
    except P2ExecutionError as err:
        print(f"[I1] Falha ao carregar o modelo ({err.code}); nenhum quadro foi processado.", file=sys.stderr)
        sys.exit(2)
    finally:
        jsonl_stream.close()

    if hasattr(publisher, "flush"):
        remaining = publisher.flush()
        print(f"[I1 - Kafka] Flush concluído (mensagens pendentes na fila: {remaining})")
        if remaining > 0:
            print(
                f"[I1 - Kafka] Falha ao entregar {remaining} mensagem(ns) ao broker '{broker_server}'.",
                file=sys.stderr,
            )
            sys.exit(3)

    total_frames = publisher.published_count
    total_dets = publisher.detection_count
    avg_lat = (
        pipeline.total_infer_latency_ms / pipeline.telemetry_count
        if pipeline.telemetry_count
        else 0.0
    )

    print(f"\n[I1 - Resumo Degrau A]")
    print(f"  - Quadros processados e validados (B1): {total_frames}")
    print(f"  - Total de objetos detectados:          {total_dets}")
    print(f"  - Latência média de inferência:         {avg_lat:.2f} ms")
    print(f"  - Quadros descartados por código:       {pipeline.error_counts or 'nenhum'}")
    print(f"  - Eventos salvos em:                    {out_file.resolve()}")


if __name__ == "__main__":
    main()