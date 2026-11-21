"""
run_orchestrator.py — Daemon orquestrador FIFO do componente I1 na Jetson.

Escuta comandos de controle no tópico Kafka (padrão: 'ods.i1.command'), carrega sob demanda
o artefato correspondente no P2 (1 modelo por vez na GPU), consome o vídeo da equipe em
'/home/sys2/SIS-2/videos' (ou SHM do P4) e publica detecções validadas em 'ods.i1.detections'.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detector.config import BROKER_ENV_VAR
from detector.orchestrator import (
    ControlCommand,
    DEFAULT_CLOUD_BROKER,
    DEFAULT_COMMAND_TOPIC,
    DEFAULT_DETECTIONS_TOPIC,
    DEFAULT_TTL_SECONDS,
    DEFAULT_VIDEOS_DIR,
    FifoInferenceOrchestrator,
    parse_control_command,
)
from detector.publishing import ConfluentKafkaPublisher, ValidatedStdoutPublisher

logger = logging.getLogger("ods.i1.orchestrator.cli")


class KafkaCommandSource:
    """Consumidor Kafka não bloqueante para o tópico de controle 'ods.i1.command'."""

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str = DEFAULT_COMMAND_TOPIC,
        group_id: str = "ods-i1-orchestrator",
        auto_offset_reset: str = "latest",
        default_ttl: float = DEFAULT_TTL_SECONDS,
    ) -> None:
        from confluent_kafka import Consumer

        self.topic = topic
        self.default_ttl = default_ttl
        self.consumer = Consumer(
            {
                "bootstrap.servers": bootstrap_servers,
                "group.id": group_id,
                "auto.offset.reset": auto_offset_reset,
                "enable.auto.commit": True,
            }
        )
        self.consumer.subscribe([topic])

    def poll(self, timeout_s: float = 0.0, max_batch: int = 16) -> list[ControlCommand]:
        commands: list[ControlCommand] = []
        for idx in range(max_batch):
            msg = self.consumer.poll(timeout_s if idx == 0 else 0.0)
            if msg is None:
                break
            err = msg.error()
            if err is not None:
                if err.name() != "_PARTITION_EOF":
                    logger.warning("[I1-ORCH] Erro no consumidor Kafka: %s", err)
                continue
            raw_val = msg.value()
            if not raw_val:
                continue
            try:
                cmd = parse_control_command(raw_val, default_ttl=self.default_ttl)
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                logger.warning("[I1-ORCH] Comando ignorado em '%s': %s", self.topic, exc)
                continue
            commands.append(cmd)
        return commands

    def close(self) -> None:
        self.consumer.close()


def send_single_command(broker: str, topic: str, raw_command: str) -> None:
    """Valida e envia um único comando JSON para o tópico de controle e aguarda o flush."""
    cmd = parse_control_command(raw_command)
    from confluent_kafka import Producer

    producer = Producer(
        {
            "bootstrap.servers": broker,
            "message.timeout.ms": 5000,
            "socket.timeout.ms": 3000,
        }
    )
    delivery_errors: list[str] = []

    def _on_delivery(err: Any, _msg: Any) -> None:
        if err is not None:
            delivery_errors.append(str(err))

    payload = raw_command.encode("utf-8")
    producer.produce(
        topic,
        key=cmd.profile.squad.encode("utf-8"),
        value=payload,
        on_delivery=_on_delivery,
    )
    remaining = producer.flush(5.0)
    if remaining > 0 or delivery_errors:
        raise RuntimeError(
            f"Falha ao entregar comando ao broker '{broker}' (pendentes={remaining}, erros={delivery_errors})."
        )
    print(
        f"[I1-ORCH] Comando {cmd.action} para '{cmd.profile.squad}' ({cmd.profile.engine_ref}) "
        f"entregue em '{topic}' no broker '{broker}'."
    )


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw and raw.strip() else default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw and raw.strip() else default


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orquestrador FIFO multi-aplicação do I1 na Jetson (escuta ods.i1.command)"
    )
    parser.add_argument(
        "--broker",
        type=str,
        default=os.getenv(BROKER_ENV_VAR, DEFAULT_CLOUD_BROKER),
        help=f"host:porta do broker Kafka (padrão: {DEFAULT_CLOUD_BROKER} ou variável {BROKER_ENV_VAR})",
    )
    parser.add_argument(
        "--command-topic",
        type=str,
        default=os.getenv("I1_COMMAND_TOPIC", DEFAULT_COMMAND_TOPIC),
        help=f"Tópico Kafka de comandos de entrada (padrão: {DEFAULT_COMMAND_TOPIC})",
    )
    parser.add_argument(
        "--detections-topic",
        type=str,
        default=os.getenv("I1_DETECTIONS_TOPIC", DEFAULT_DETECTIONS_TOPIC),
        help=f"Tópico Kafka de saída de detecções (padrão: {DEFAULT_DETECTIONS_TOPIC})",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path(os.getenv("I1_VIDEOS_DIR", str(DEFAULT_VIDEOS_DIR))),
        help=f"Pasta com os vídeos nomeados por equipe (padrão: {DEFAULT_VIDEOS_DIR})",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "p2", "ultralytics", "mock"],
        default=os.getenv("I1_BACKEND", "p2"),
        help="Backend de inferência ('p2' na Jetson GPU, 'ultralytics', 'mock' ou 'auto')",
    )
    parser.add_argument(
        "--mode",
        choices=["kafka", "stdout"],
        default=os.getenv("I1_MODE", "kafka"),
        help="Destino das detecções publicadas ('kafka' ou 'stdout')",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=_env_float("I1_CONF", 0.50),
        help="Limiar mínimo de confiança para detecções",
    )
    parser.add_argument(
        "--default-ttl",
        type=float,
        default=_env_float("I1_DEFAULT_TTL", DEFAULT_TTL_SECONDS),
        help="Tempo máximo padrão (segundos) de uma sessão ativa sem heartbeat/STOP",
    )
    parser.add_argument(
        "--max-frames-per-job",
        type=int,
        default=_env_int("I1_MAX_FRAMES", 0),
        help="Limite de quadros por requisição (0 = processa até EOF, STOP ou TTL)",
    )
    parser.add_argument(
        "--default-source",
        type=str,
        default=os.getenv("I1_SOURCE") or None,
        help="Sobrescreve a fonte padrão (ex: 'video' para /tmp/p4_file_shm ou 'imx519')",
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default=os.getenv("I1_CONSUMER_GROUP", "ods-i1-orchestrator"),
        help="Consumer group Kafka do orquestrador",
    )
    parser.add_argument(
        "--offset-reset",
        choices=["latest", "earliest"],
        default=os.getenv("I1_OFFSET_RESET", "latest"),
        help="Posição inicial ao iniciar sem offset salvo",
    )
    parser.add_argument(
        "--allow-fixtures-fallback",
        action="store_true",
        help="Se não achar vídeo da equipe em --videos-dir, usa fixtures locais do artefato quando existirem",
    )
    parser.add_argument(
        "--send-command",
        type=str,
        default=None,
        help="Utilitário: publica um comando JSON em --command-topic e encerra",
    )

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.send_command:
        try:
            send_single_command(args.broker, args.command_topic, args.send_command)
        except Exception as err:
            print(f"[I1-ORCH] Erro ao enviar comando: {err}", file=sys.stderr)
            sys.exit(2)
        return

    command_source = KafkaCommandSource(
        bootstrap_servers=args.broker,
        topic=args.command_topic,
        group_id=args.group_id,
        auto_offset_reset=args.offset_reset,
        default_ttl=args.default_ttl,
    )

    publisher = (
        ConfluentKafkaPublisher(
            bootstrap_servers=args.broker,
            topic=args.detections_topic,
            history_limit=0,
        )
        if args.mode == "kafka"
        else ValidatedStdoutPublisher(verbose=True, history_limit=0)
    )

    orchestrator = FifoInferenceOrchestrator(
        command_source=command_source,
        publisher=publisher,
        videos_dir=args.videos_dir,
        backend_mode=args.backend,
        conf_threshold=args.conf,
        default_max_frames=args.max_frames_per_job if args.max_frames_per_job > 0 else None,
        default_source=args.default_source,
        allow_fixtures_fallback=args.allow_fixtures_fallback,
    )

    def _handle_signal(signum: int, _frame: Any) -> None:
        logger.info("[I1-ORCH] Sinal %s recebido; encerrando após liberar modelo ativo...", signum)
        orchestrator.request_shutdown()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(
        f"[I1-ORCH] Aguardando comandos em '{args.command_topic}' (broker='{args.broker}', "
        f"backend='{args.backend}', mode='{args.mode}', videos_dir='{args.videos_dir}')..."
    )
    try:
        orchestrator.run_forever()
    finally:
        command_source.close()
        if hasattr(publisher, "flush"):
            publisher.flush()


if __name__ == "__main__":
    main()
