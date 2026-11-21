"""
orchestrator.py — Orquestrador FIFO multi-aplicação do componente I1 para inferência sob demanda na Jetson.

Recebe comandos via Kafka (tópico 'ods.i1.command'), mapeia a equipe/aplicação solicitante para
o respectivo pacote MODEL (I1/artifacts/) e vídeo de teste (/home/sys2/SIS-2/videos/<SQUAD>.mp4
ou socket SHM do P4), garante execução exclusiva de 1 modelo por vez no P2 (release antes de load)
e gerencia concorrência em fila FIFO com suporte a STOP, HEARTBEAT, TTL e preempção com force=true.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Callable, Protocol
import uuid

from .config import AppConfig
from .contracts import P2ExecutionError, RawDetection, new_message_id
from .inference import (
    MockP2Backend,
    NativeP2Backend,
    P2InferenceClient,
    P2LibraryProtocol,
    UltralyticsP2Adapter,
)
from .ingestion import FrameSource, OpenCVFrameSource, P4GStreamerShmSource

if TYPE_CHECKING:
    from .publishing import EventPublisher

logger = logging.getLogger("ods.i1.orchestrator")

DEFAULT_COMMAND_TOPIC = "ods.i1.command"
DEFAULT_DETECTIONS_TOPIC = "ods.i1.detections"
DEFAULT_CLOUD_BROKER = "34.66.235.91:9093"
DEFAULT_VIDEOS_DIR = Path("/home/sys2/SIS-2/videos")
DEFAULT_TTL_SECONDS = 60.0
ARTIFACTS_ROOT = Path(__file__).resolve().parents[1] / "artifacts"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
P4_SOCKET_ALIASES: dict[str, str] = {
    "imx519": "/tmp/p4_imx519_shm",
    "imx477": "/tmp/p4_imx477_shm",
    "video": "/tmp/p4_file_shm",
    "file": "/tmp/p4_file_shm",
    "p4_video": "/tmp/p4_file_shm",
    "p4_file": "/tmp/p4_file_shm",
    "p4_file_shm": "/tmp/p4_file_shm",
}


@dataclass(frozen=True)
class AppProfile:
    """Mapeia uma equipe/aplicação para seu artefato MODEL e fonte de vídeo."""

    squad: str
    app_id: str
    artifact_id: str
    artifact_version: str
    aliases: tuple[str, ...]
    video_candidates: tuple[str, ...]
    enabled_classes: tuple[str, ...] | None = None

    @property
    def engine_ref(self) -> str:
        return f"{self.artifact_id}-{self.artifact_version}"

    def artifact_dir(self, artifacts_root: Path = ARTIFACTS_ROOT) -> Path:
        return artifacts_root / self.artifact_id / self.artifact_version

    def metadata_path(self, artifacts_root: Path = ARTIFACTS_ROOT) -> Path:
        return self.artifact_dir(artifacts_root) / "metadata.json"

    def onnx_path(self, artifacts_root: Path = ARTIFACTS_ROOT) -> Path:
        return self.artifact_dir(artifacts_root) / f"{self.artifact_id}.onnx"


APP_PROFILES: tuple[AppProfile, ...] = (
    AppProfile(
        squad="INV1",
        app_id="A4",
        artifact_id="app-sports",
        artifact_version="1.2.0",
        aliases=("inv1", "inv-1", "a4", "app-sports", "esporte", "sports"),
        video_candidates=("INV1", "INV-1", "inv1", "inv-1", "A4", "esporte", "sports"),
    ),
    AppProfile(
        squad="VER1",
        app_id="A1",
        artifact_id="app-industry",
        artifact_version="1.0.0",
        aliases=("ver1", "ver-1", "a1", "app-industry", "industria", "industry"),
        video_candidates=("VER1", "VER-1", "ver1", "ver-1", "A1", "industria", "industry"),
    ),
    AppProfile(
        squad="VER3",
        app_id="A5",
        artifact_id="app-epi",
        artifact_version="1.0.0",
        aliases=("ver3", "ver-3", "a5", "app-epi", "epi"),
        video_candidates=("VER3", "VER-3", "ver3", "ver-3", "A5", "epi"),
    ),
    AppProfile(
        squad="VER5",
        app_id="A8",
        artifact_id="app-plates",
        artifact_version="1.0.0",
        aliases=("ver5", "ver-5", "a8", "app-plates", "placas", "plates", "estacionamento"),
        video_candidates=("VER5", "VER-5", "ver5", "ver-5", "A8", "plates", "placas"),
    ),
    AppProfile(
        squad="VER6",
        app_id="A3",
        artifact_id="app-epi",
        artifact_version="1.0.0",
        aliases=("ver6", "ver-6", "a3", "monitoramento", "monitoring"),
        video_candidates=("VER6", "VER-6", "ver6", "ver-6", "A3", "monitoramento"),
        enabled_classes=("pessoa",),
    ),
)


def _normalize_key(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def resolve_app_profile(identifier: str) -> AppProfile:
    """Resolve squad ('INV1'), app_id ('A4') ou artifact_id ('app-sports') para um AppProfile."""
    norm = _normalize_key(identifier)
    compact = norm.replace("-", "")
    if not norm:
        raise ValueError("Identificador de aplicação/equipe vazio no comando.")
    for profile in APP_PROFILES:
        for alias in profile.aliases:
            if norm == alias or compact == alias.replace("-", ""):
                return profile
    valid = ", ".join(f"{p.squad}/{p.app_id}/{p.artifact_id}" for p in APP_PROFILES)
    raise ValueError(f"Aplicação/equipe desconhecida: {identifier!r}. Opções: {valid}")


@dataclass(frozen=True)
class ControlCommand:
    """Comando normalizado recebido no tópico Kafka de controle (ods.i1.command)."""

    action: str  # START | STOP | HEARTBEAT
    profile: AppProfile
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    max_frames: int | None = None
    source: str | None = None
    source_id: str | None = None
    session_id: str | None = None
    force: bool = False


def parse_control_command(
    raw_message: bytes | str | dict[str, Any],
    default_ttl: float = DEFAULT_TTL_SECONDS,
) -> ControlCommand:
    """
    Interpreta JSON simples ({"action": "START", "squad": "INV1"}) ou envelope CMD 6c
    ({"schema": "ods.i1.command", "producer": "INV1", "payload": {...}}).
    """
    if isinstance(raw_message, bytes):
        raw_message = raw_message.decode("utf-8")
    data = json.loads(raw_message) if isinstance(raw_message, str) else raw_message
    if not isinstance(data, dict):
        raise ValueError("Comando deve ser um objeto JSON.")

    payload = data["payload"] if isinstance(data.get("payload"), dict) else data

    raw_action = payload.get("action", data.get("action"))
    if not isinstance(raw_action, str) or not raw_action.strip():
        raise ValueError("Campo 'action' obrigatório (START, STOP ou HEARTBEAT).")
    action = raw_action.strip().upper()
    if action not in {"START", "STOP", "HEARTBEAT"}:
        raise ValueError(f"Ação inválida: {action!r}. Esperado START, STOP ou HEARTBEAT.")

    identifier: str | None = None
    for source_dict, keys in (
        (payload, ("squad", "app_id", "app", "team", "artifact_id", "target", "source_id")),
        (data, ("squad", "app_id", "app", "team", "artifact_id", "producer")),
    ):
        for key in keys:
            val = source_dict.get(key)
            if isinstance(val, str) and val.strip() and val.strip().upper() != "I1":
                identifier = val.strip()
                break
        if identifier is not None:
            break
    if identifier is None:
        raise ValueError("Comando precisa informar 'squad', 'app_id' ou 'artifact_id'.")

    profile = resolve_app_profile(identifier)

    raw_ttl = payload.get("ttl_seconds", default_ttl)
    if isinstance(raw_ttl, bool) or not isinstance(raw_ttl, (int, float)) or float(raw_ttl) <= 0:
        raise ValueError(f"ttl_seconds deve ser > 0, recebido: {raw_ttl!r}.")

    raw_max_frames = payload.get("max_frames")
    max_frames: int | None = None
    if raw_max_frames is not None:
        if isinstance(raw_max_frames, bool) or not isinstance(raw_max_frames, int) or raw_max_frames <= 0:
            raise ValueError(f"max_frames deve ser inteiro positivo, recebido: {raw_max_frames!r}.")
        max_frames = raw_max_frames

    raw_source = payload.get("source")
    if raw_source is not None and (not isinstance(raw_source, str) or not raw_source.strip()):
        raise ValueError("Campo 'source' deve ser texto não vazio.")

    raw_source_id = payload.get("source_id")
    if raw_source_id is not None and (not isinstance(raw_source_id, str) or not raw_source_id.strip()):
        raise ValueError("Campo 'source_id' deve ser texto não vazio.")

    raw_session_id = payload.get("session_id")
    if raw_session_id is not None:
        if not isinstance(raw_session_id, str):
            raise ValueError("session_id deve ser UUID v4.")
        try:
            parsed_uuid = uuid.UUID(raw_session_id)
        except ValueError as err:
            raise ValueError(f"session_id inválido (esperado UUID v4): {raw_session_id!r}") from err
        if parsed_uuid.version != 4:
            raise ValueError(f"session_id deve ser UUID v4: {raw_session_id!r}")

    return ControlCommand(
        action=action,
        profile=profile,
        ttl_seconds=float(raw_ttl),
        max_frames=max_frames,
        source=raw_source.strip() if isinstance(raw_source, str) else None,
        source_id=raw_source_id.strip() if isinstance(raw_source_id, str) else None,
        session_id=raw_session_id,
        force=bool(payload.get("force", False)),
    )


def _find_video_in_dir(profile: AppProfile, videos_dir: Path) -> Path | None:
    if not videos_dir.exists() or not videos_dir.is_dir():
        return None

    for candidate in profile.video_candidates:
        for ext in VIDEO_EXTENSIONS:
            direct = videos_dir / f"{candidate}{ext}"
            if direct.is_file():
                return direct

    entries = sorted(videos_dir.iterdir(), key=lambda p: p.name.lower())
    candidate_norms = {c.lower().replace("-", "") for c in profile.video_candidates}

    for entry in entries:
        if entry.is_file() and entry.suffix.lower() in VIDEO_EXTENSIONS:
            stem_norm = entry.stem.lower().replace("-", "").replace("_", "")
            if stem_norm in candidate_norms or any(stem_norm.startswith(c) for c in candidate_norms):
                return entry

    for entry in entries:
        if entry.is_dir():
            dir_norm = entry.name.lower().replace("-", "").replace("_", "")
            if dir_norm in candidate_norms:
                media = sorted(
                    p for p in entry.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
                )
                return media[0] if media else entry
    return None


def resolve_squad_video_source(
    profile: AppProfile,
    videos_dir: Path = DEFAULT_VIDEOS_DIR,
    explicit_source: str | None = None,
    source_id: str | None = None,
    session_id: str | None = None,
    max_frames: int | None = None,
    artifacts_root: Path = ARTIFACTS_ROOT,
    allow_fixtures_fallback: bool = False,
) -> FrameSource:
    """Seleciona socket SHM do P4, arquivo em /home/sys2/SIS-2/videos ou fallback de fixtures."""
    effective_source_id = source_id or profile.squad
    effective_session_id = session_id or new_message_id()

    if explicit_source:
        lowered = explicit_source.lower()
        if lowered in P4_SOCKET_ALIASES:
            return P4GStreamerShmSource(
                shm_socket_path=P4_SOCKET_ALIASES[lowered],
                max_frames=max_frames,
            )
        if lowered == "shm" or explicit_source.startswith("/tmp/p4_"):
            socket_path = "/tmp/p4_file_shm" if lowered == "shm" else explicit_source
            return P4GStreamerShmSource(shm_socket_path=socket_path, max_frames=max_frames)
        explicit_path = Path(explicit_source)
        if not explicit_path.exists():
            raise FileNotFoundError(f"Fonte de vídeo explícita não encontrada: {explicit_source}")
        return OpenCVFrameSource(
            uri=str(explicit_path),
            source_id=effective_source_id,
            session_id=effective_session_id,
            max_frames=max_frames,
        )

    matched = _find_video_in_dir(profile, videos_dir)
    if matched is not None:
        return OpenCVFrameSource(
            uri=str(matched),
            source_id=effective_source_id,
            session_id=effective_session_id,
            max_frames=max_frames,
        )

    fixtures_dir = profile.artifact_dir(artifacts_root) / "fixtures"
    if allow_fixtures_fallback and fixtures_dir.is_dir():
        logger.warning(
            "[I1-ORCH] Vídeo de '%s' não encontrado em %s; usando fixtures locais (%s).",
            profile.squad,
            videos_dir,
            fixtures_dir,
        )
        return OpenCVFrameSource(
            uri=str(fixtures_dir),
            source_id=effective_source_id,
            session_id=effective_session_id,
            max_frames=max_frames,
        )

    raise FileNotFoundError(
        f"Nenhum vídeo encontrado para '{profile.squad}' ({profile.app_id}) em '{videos_dir}'."
    )


def build_app_config_for_profile(
    profile: AppProfile,
    backend_mode: str,
    source_id: str,
    session_id: str,
    conf_threshold: float = 0.50,
    artifacts_root: Path = ARTIFACTS_ROOT,
) -> tuple[AppConfig, int]:
    """Carrega metadata.json do artefato e monta AppConfig + input_size."""
    meta_path = profile.metadata_path(artifacts_root)
    if not meta_path.is_file():
        raise FileNotFoundError(f"metadata.json ausente para {profile.engine_ref}: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    input_shape = meta.get("input_shape", [1, 3, 640, 640])
    if isinstance(input_shape, dict):
        first_shape = next(iter(input_shape.values()), [1, 3, 640, 640])
        imgsz = int(first_shape[2]) if len(first_shape) >= 3 and int(first_shape[2]) > 0 else 640
    else:
        imgsz = int(input_shape[2]) if len(input_shape) >= 3 and int(input_shape[2]) > 0 else 640

    onnx_file = meta.get("onnx_file", f"{profile.artifact_id}.onnx")
    onnx_path = meta_path.parent / onnx_file
    use_onnx_ref = backend_mode == "ultralytics"
    if backend_mode == "auto":
        try:
            import p2  # noqa: F401
        except ImportError:
            use_onnx_ref = True

    engine_ref = str(onnx_path) if use_onnx_ref else f"{meta['artifact_id']}-{meta['artifact_version']}"
    cfg = AppConfig(
        artifact_id=meta["artifact_id"],
        artifact_version=meta["artifact_version"],
        engine_ref=engine_ref,
        classes=[str(c) for c in meta["classes"]],
        conf_threshold=conf_threshold,
        enabled_classes=list(profile.enabled_classes) if profile.enabled_classes else None,
        source_id=source_id,
        session_id=session_id,
    )
    return cfg, imgsz


def build_p2_client(backend_mode: str, imgsz: int) -> P2InferenceClient:
    """Instancia P2InferenceClient conforme backend selecionado ('p2', 'ultralytics', 'mock', 'auto')."""
    backend: P2LibraryProtocol
    if backend_mode == "p2":
        backend = NativeP2Backend()
    elif backend_mode == "ultralytics":
        backend = UltralyticsP2Adapter(imgsz=imgsz)
    elif backend_mode == "mock":
        backend = MockP2Backend(
            simulated_detections=[
                RawDetection(class_id=0, confidence=0.92, bbox=(20.0, 20.0, 120.0, 80.0)),
            ]
        )
    elif backend_mode == "auto":
        try:
            import p2  # noqa: F401
            backend = NativeP2Backend()
        except ImportError:
            backend = UltralyticsP2Adapter(imgsz=imgsz)
    else:
        raise ValueError(f"Backend desconhecido: {backend_mode!r}")
    return P2InferenceClient(p2_backend=backend)


class CommandSource(Protocol):
    """Fonte não bloqueante de comandos de orquestração."""

    def poll(self, timeout_s: float = 0.0) -> list[ControlCommand]:
        ...


@dataclass
class QueuedJob:
    """Requisição de inferência na fila FIFO do I1."""

    profile: AppProfile
    ttl_seconds: float
    max_frames: int | None
    source: str | None
    source_id: str
    session_id: str
    enqueued_at: float


@dataclass
class JobRunSummary:
    """Resultado de uma execução de inferência de uma equipe."""

    squad: str
    engine_ref: str
    reason: str
    frames_published: int
    error: str | None = None


class FifoInferenceOrchestrator:
    """
    Orquestra solicitações de múltiplas aplicações na Jetson em ordem FIFO, garantindo
    que apenas um modelo por vez esteja carregado no runtime P2 (GPU).
    """

    def __init__(
        self,
        command_source: CommandSource,
        publisher: EventPublisher,
        videos_dir: Path = DEFAULT_VIDEOS_DIR,
        backend_mode: str = "p2",
        conf_threshold: float = 0.50,
        default_max_frames: int | None = None,
        default_source: str | None = None,
        artifacts_root: Path = ARTIFACTS_ROOT,
        allow_fixtures_fallback: bool = False,
        poll_interval_s: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
        client_factory: Callable[[AppConfig, int], P2InferenceClient] | None = None,
        source_factory: Callable[[QueuedJob], FrameSource] | None = None,
        job_runner: Callable[[QueuedJob, Callable[[], bool]], int] | None = None,
    ) -> None:
        self.command_source = command_source
        self.publisher = publisher
        self.videos_dir = videos_dir
        self.backend_mode = backend_mode
        self.conf_threshold = conf_threshold
        self.default_max_frames = default_max_frames
        self.default_source = default_source
        self.artifacts_root = artifacts_root
        self.allow_fixtures_fallback = allow_fixtures_fallback
        self.poll_interval_s = poll_interval_s
        self.clock = clock
        self._client_factory = client_factory
        self._source_factory = source_factory
        self._job_runner = job_runner

        self._queue: deque[QueuedJob] = deque()
        self.active_job: QueuedJob | None = None
        self._active_expires_at: float = 0.0
        self._stop_active_reason: str | None = None
        self._shutdown_requested = False
        self.completed_jobs: list[JobRunSummary] = []

    def pending_squads(self) -> list[str]:
        return [job.profile.squad for job in self._queue]

    def request_shutdown(self) -> None:
        self._shutdown_requested = True

    def _make_job(self, cmd: ControlCommand) -> QueuedJob:
        return QueuedJob(
            profile=cmd.profile,
            ttl_seconds=cmd.ttl_seconds,
            max_frames=cmd.max_frames if cmd.max_frames is not None else self.default_max_frames,
            source=cmd.source or self.default_source,
            source_id=cmd.source_id or cmd.profile.squad,
            session_id=cmd.session_id or new_message_id(),
            enqueued_at=self.clock(),
        )

    def handle_command(self, cmd: ControlCommand) -> None:
        squad = cmd.profile.squad
        now = self.clock()

        if cmd.action == "STOP":
            if self.active_job is not None and self.active_job.profile.squad == squad:
                logger.info("[I1-ORCH] STOP recebido para equipe ativa '%s'.", squad)
                self._stop_active_reason = "stopped_by_command"
                return
            before = len(self._queue)
            self._queue = deque(job for job in self._queue if job.profile.squad != squad)
            if len(self._queue) < before:
                logger.info("[I1-ORCH] Removida equipe '%s' da fila FIFO por comando STOP.", squad)
            return

        if cmd.action == "HEARTBEAT":
            if self.active_job is not None and self.active_job.profile.squad == squad:
                self._active_expires_at = now + cmd.ttl_seconds
                logger.debug("[I1-ORCH] Lease renovado para '%s' (+%.1fs).", squad, cmd.ttl_seconds)
            return

        # action == "START"
        if self.active_job is not None and self.active_job.profile.squad == squad:
            self._active_expires_at = now + cmd.ttl_seconds
            logger.info("[I1-ORCH] START repetido para equipe ativa '%s'; lease renovado (+%.1fs).", squad, cmd.ttl_seconds)
            return

        new_job = self._make_job(cmd)

        if cmd.force and self.active_job is not None:
            logger.warning(
                "[I1-ORCH] Preempção imediata (force=true): '%s' interrompe '%s'.",
                squad,
                self.active_job.profile.squad,
            )
            self._queue = deque(job for job in self._queue if job.profile.squad != squad)
            self._queue.appendleft(new_job)
            self._stop_active_reason = f"preempted_by:{squad}"
            return

        for idx, existing in enumerate(self._queue):
            if existing.profile.squad == squad:
                self._queue[idx] = new_job
                logger.info("[I1-ORCH] Equipe '%s' já está na fila FIFO (posição %d); parâmetros atualizados.", squad, idx + 1)
                return

        self._queue.append(new_job)
        logger.info(
            "[I1-ORCH] Equipe '%s' (%s) enfileirada na posição %d.",
            squad,
            cmd.profile.engine_ref,
            len(self._queue),
        )

    def _poll_commands(self, timeout_s: float = 0.0) -> None:
        for cmd in self.command_source.poll(timeout_s=timeout_s):
            self.handle_command(cmd)

    def _should_abort_active(self) -> bool:
        self._poll_commands(timeout_s=0.0)
        if self._shutdown_requested:
            self._stop_active_reason = "shutdown"
            return True
        if self._stop_active_reason is not None:
            return True
        if self.clock() >= self._active_expires_at:
            self._stop_active_reason = "ttl_expired"
            return True
        return False

    def _run_pipeline_job(self, job: QueuedJob) -> int:
        from .pipeline import DetectorPipeline

        cfg, imgsz = build_app_config_for_profile(
            profile=job.profile,
            backend_mode=self.backend_mode,
            source_id=job.source_id,
            session_id=job.session_id,
            conf_threshold=self.conf_threshold,
            artifacts_root=self.artifacts_root,
        )
        source = (
            self._source_factory(job)
            if self._source_factory is not None
            else resolve_squad_video_source(
                profile=job.profile,
                videos_dir=self.videos_dir,
                explicit_source=job.source,
                source_id=job.source_id,
                session_id=job.session_id,
                max_frames=job.max_frames,
                artifacts_root=self.artifacts_root,
                allow_fixtures_fallback=self.allow_fixtures_fallback,
            )
        )
        client = (
            self._client_factory(cfg, imgsz)
            if self._client_factory is not None
            else build_p2_client(self.backend_mode, imgsz)
        )
        pipeline = DetectorPipeline(cfg, source, self.publisher, client)
        pipeline.initialize()

        published = 0
        try:
            for frame in source.frames():
                if self._should_abort_active():
                    break
                if pipeline.process_single_frame(frame) is not None:
                    published += 1
                if self._should_abort_active():
                    break
                if job.max_frames is not None and published >= job.max_frames:
                    self._stop_active_reason = "max_frames_reached"
                    break
        finally:
            pipeline.inference_client.close()
        return published

    def _execute_job(self, job: QueuedJob) -> JobRunSummary:
        self.active_job = job
        self._active_expires_at = self.clock() + job.ttl_seconds
        self._stop_active_reason = None

        logger.info(
            "[I1-ORCH] Carregando '%s' (%s) para equipe '%s' (TTL=%.1fs).",
            job.profile.engine_ref,
            job.profile.app_id,
            job.profile.squad,
            job.ttl_seconds,
        )
        frames_published = 0
        error_msg: str | None = None
        try:
            if self._job_runner is not None:
                frames_published = self._job_runner(job, self._should_abort_active)
            else:
                frames_published = self._run_pipeline_job(job)
        except (FileNotFoundError, RuntimeError, P2ExecutionError, ValueError) as err:
            error_msg = str(err)
            self._stop_active_reason = "error"
            logger.error("[I1-ORCH] Erro ao executar '%s': %s", job.profile.squad, err)
        finally:
            if hasattr(self.publisher, "flush"):
                pending = self.publisher.flush()
                if pending > 0:
                    logger.warning(
                        "[I1-ORCH] %d mensagem(ns) pendente(s) no flush após encerrar '%s'.",
                        pending,
                        job.profile.squad,
                    )
            reason = self._stop_active_reason or "source_completed"
            self.active_job = None
            self._stop_active_reason = None

        summary = JobRunSummary(
            squad=job.profile.squad,
            engine_ref=job.profile.engine_ref,
            reason=reason,
            frames_published=frames_published,
            error=error_msg,
        )
        self.completed_jobs.append(summary)
        logger.info(
            "[I1-ORCH] Sessão '%s' encerrada (%s) | quadros publicados: %d | fila restante: %s",
            summary.squad,
            summary.reason,
            summary.frames_published,
            self.pending_squads() or "vazia",
        )
        return summary

    def step(self, wait_when_idle: bool = True) -> JobRunSummary | None:
        """Processa comandos pendentes e, se houver job na fila FIFO, executa o próximo."""
        self._poll_commands(timeout_s=0.0)
        if not self._queue:
            if wait_when_idle and not self._shutdown_requested:
                self._poll_commands(timeout_s=self.poll_interval_s)
            if not self._queue:
                return None
        next_job = self._queue.popleft()
        return self._execute_job(next_job)

    def run_forever(self, max_jobs: int | None = None) -> list[JobRunSummary]:
        """Loop principal do orquestrador até interrupção ou max_jobs executados."""
        while not self._shutdown_requested:
            if max_jobs is not None and len(self.completed_jobs) >= max_jobs:
                break
            self.step(wait_when_idle=True)
        return list(self.completed_jobs)
