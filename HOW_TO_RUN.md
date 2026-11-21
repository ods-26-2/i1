# Guia Rápido — Componente I1 (MOD-1)

**Disciplina:** ODS 2026/2 (UFSJ) · **Arquitetura:** 6c / Interfaces 1.0
**Papel do I1:** Consome quadros de vídeo (SHM do P4, MP4 ou imagens), executa detecção de objetos (via **P2**, **ONNX/CPU** ou **Mock**) e publica eventos `ods.i1.detections` validados (B1) no terminal ou no **Kafka**.

> **Nota sobre `--build`:** Todos os serviços montam `./I1`, `./schemas` e `/tmp` do host. Use `--build` só na **1ª vez** (ou se alterar o `Dockerfile` / `Dockerfile.jetson`).

---

## 1. No Computador (Sem GPU / Sem Jetson)

### Testes automatizados (CI de B4 e validação local)
```bash
docker compose run --build --rm i1-ci-tests
```

### Produtor Mock (para consumidores `I2`, `I8`, `I9`, `S1` e testes sem câmera)
```bash
# Imprime 10 eventos válidos no terminal
docker compose run --rm i1-mock-stdout

# Publica 10 eventos no Kafka oficial da rede Dcomp (172.18.4.56:9093)
docker compose run --rm i1-mock-kafka

# Outro broker ou outra quantidade de eventos
ODS_KAFKA_BROKER="host:porta" docker compose run --rm i1-mock-kafka \
  python3 I1/mock_producer.py --frames 20 --mode kafka
```

### Detector real com ONNX na CPU (`app-plates`)
```bash
# Imagens de exemplo (stdout)
docker compose run --rm i1-degrau-a

# Vídeo .mp4 colocado em /tmp (salva o log JSONL também em /tmp)
I1_SOURCE=/tmp/video.mp4 I1_MAX_FRAMES=100 I1_OUTPUT_JSONL=/tmp/events.jsonl \
  docker compose run --rm i1-degrau-a

# Pipeline completo em modo mock (sem modelo nem GPU)
I1_BACKEND=mock I1_MAX_FRAMES=5 docker compose run --rm i1-degrau-a
```

---

## 2. Na Jetson com GPU (TensorRT + P2)

> **Mock vs. GPU:**
> - `I1_BACKEND=mock` e `i1-mock-*` rodam **apenas na CPU** (retornam caixas fixas ou JSONs de exemplo para testar o pipeline e o Kafka).
> - Para usar a **GPU da Jetson (TensorRT FP16)**, é obrigatório ter o pacote `p2[trt]`, compilar o `.onnx` com `p2-build` e rodar com `I1_BACKEND=p2` + `P2_BACKEND=trt`.

Escolha **uma** das duas formas abaixo para rodar com GPU na Jetson:

### Opção A — Direto na `.venv` da Jetson (Mais rápido, recomendado pelo `guia_rapido.md` do P2)

Como o TensorRT vem do JetPack 6.2.1 no sistema, a `.venv` precisa enxergar os pacotes do sistema (`--system-site-packages`):

```bash
# 1. Instalar (uma única vez)
/usr/bin/python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-jetson.txt

# 2. Compilar o modelo .onnx para engine TensorRT (gera FP16 e FP32)
export P2_ENGINE_DIR=/opt/ods/engines          # ou ~/p2-engines
p2-build I1/artifacts/app-plates/1.0.0/
p2-list                                        # confirma que a engine está pronta

# 3. Rodar o I1 usando a GPU (TensorRT FP16)
export P2_BACKEND=trt
python3 I1/run_detector.py \
  --backend p2 \
  --metadata I1/artifacts/app-plates/1.0.0/metadata.json \
  --source I1/artifacts/app-plates/1.0.0/fixtures \
  --max-frames 20
```

*(Para consumir a câmera/vídeo do P4 na `.venv`, troque `--source` por `imx519`, `imx477`, `video` ou `/tmp/video.mp4` e adicione `--mode kafka`.)*

### Opção B — Via Docker com GPU (`Dockerfile.jetson`)

O serviço `i1-jetson` usa o [`Dockerfile.jetson`](Dockerfile.jetson) (`nvcr.io/nvidia/l4t-jetpack:r36.4.0` + `runtime: nvidia` + `p2[trt,cpu]` + `P2_BACKEND=trt`).

```bash
# 1. Compilar a engine TensorRT para /opt/ods/engines no host (uma vez por modelo)
sudo mkdir -p /opt/ods/engines
docker compose --profile jetson run --build --rm i1-p2-build

# 2. Executar o I1 na GPU (fixtures no stdout)
docker compose --profile jetson run --rm i1-jetson

# Exemplo com vídeo .mp4 em /tmp publicando no Kafka da Jetson
I1_SOURCE=/tmp/video.mp4 I1_MAX_FRAMES=100 I1_MODE=kafka \
  docker compose --profile jetson run --rm i1-jetson
```

---

## 3. Orquestrador Sob Demanda na Jetson (Multi-Aplicação + Fila FIFO via Kafka)

Quando as aplicações rodam na nuvem e apenas a inferência roda na Jetson, o `run_orchestrator.py` escuta o tópico **`ods.i1.command`**, carrega **1 modelo por vez na GPU (`p2.load`)**, lê o vídeo correspondente da equipe em **`/home/sys2/SIS-2/videos`** (ou SHM do P4) e publica os eventos em **`ods.i1.detections`**. Ao finalizar ou trocar de equipe, ele sempre chama `p2.release()` antes de carregar o próximo item da fila FIFO.

### Como iniciar o orquestrador na Jetson
```bash
# Via .venv da Jetson (GPU TensorRT + broker Google Cloud 34.66.235.91:9093)
export P2_ENGINE_DIR=/opt/ods/engines
export P2_BACKEND=trt
python3 I1/run_orchestrator.py --broker 34.66.235.91:9093 --videos-dir /home/sys2/SIS-2/videos

# Ou via Docker GPU:
docker compose --profile jetson run --rm i1-orchestrator
```

### Protocolo no tópico `ods.i1.command` (para as aplicações na nuvem)
Envie um JSON informando a ação e a equipe (`squad`), aplicação (`app_id`) ou modelo (`artifact_id`):

```json
{"action": "START", "squad": "INV1"}
```

- **Ações suportadas:**
  - `START`: entra na **fila FIFO** (ou inicia na hora se a GPU estiver livre; se a equipe já estiver rodando, renova o tempo dela).
  - `HEARTBEAT`: renova o tempo ativo (`ttl_seconds`, padrão `60s`) sem recarregar o modelo.
  - `STOP`: encerra a execução da equipe (ou remove da fila) e libera a GPU imediatamente para o próximo.
- **Campos opcionais:** `"ttl_seconds": 60`, `"max_frames": 300`, `"source_id": "cam_esporte"`, `"source": "video"`, `"force": true` (interrompe a equipe atual e assume a GPU na próxima iteração).

### Mapeamento automático de equipes, modelos e vídeos
| Equipe (`squad`) | Aplicação (`app_id`) | Artefato (`engine_ref`) | Vídeo em `/home/sys2/SIS-2/videos/` |
| :--- | :--- | :--- | :--- |
| `INV1` | `A4` (Esporte) | `app-sports-1.2.0` | `INV1.mp4` (ou pasta `INV1/`) |
| `VER1` | `A1` (Indústria) | `app-industry-1.0.0` | `VER1.mp4` |
| `VER3` | `A5` (EPI) | `app-epi-1.0.0` | `VER3.mp4` |
| `VER5` | `A8` (Placas) | `app-plates-1.0.0` | `VER5.mp4` |
| `VER6` | `A3` (Monitoramento) | `app-epi-1.0.0` *(classe `pessoa`)* | `VER6.mp4` |

> **Como cada aplicação filtra sua saída em `ods.i1.detections`:** verifique `source_id` (padrão igual ao nome da equipe, ex.: `"INV1"`) ou `artifact_version` (ex.: `"app-sports-1.2.0-fp16"`).

### Testar enviando um comando pelo terminal
```bash
python3 I1/run_orchestrator.py --broker 34.66.235.91:9093 --send-command '{"action": "START", "squad": "INV1"}'
```

---

## 4. Trocar de Aplicação Manualmente ("Um artefato por aplicação, um componente só")

O código do `I1` não muda entre aplicações. Basta apontar `--metadata` (e `--onnx` no modo CPU) para um dos pacotes já embarcados em `I1/artifacts/`:

| Aplicação | Pacote em `I1/artifacts/` | Classes | Entrada |
| :--- | :--- | :--- | :--- |
| Estacionamento / Placas (`A8`) | `app-plates/1.0.0` | `placa_1L_BR`, `placa_1L_ME`, `placa_2L_BR`, `placa_2L_ME` | `640×640` |
| EPI / Segurança (`A5`) | `app-epi/1.0.0` | `pessoa`, `capacete`, `colete` | `640×640` |
| Indústria / Peças (`A1`) | `app-industry/1.0.0` | `peca` | `640×640` |
| Esporte / Bola (`A4`) | `app-sports/1.2.0` | `ball` | `1280×1280` |

**Exemplo rodando `app-epi` na Jetson (Docker GPU):**
```bash
# 1. Compilar a engine de EPI (uma vez)
P2_PACKAGE_DIR=/app/I1/artifacts/app-epi/1.0.0 \
  docker compose --profile jetson run --rm i1-p2-build

# 2. Rodar o I1 com EPI
I1_METADATA_PATH=/app/I1/artifacts/app-epi/1.0.0/metadata.json \
I1_SOURCE=/tmp/video_epi.mp4 \
I1_MODE=kafka \
  docker compose --profile jetson run --rm i1-jetson
```

**Para subir um modelo novo de B5:** coloque `<artifact_id>.onnx` + `metadata.json` em `I1/artifacts/<artifact_id>/<versao>/` (ou monte outra pasta do host via `I1_ARTIFACTS_DIR=/caminho/no/host`).

---

## 5. Integração P4 → I1 → Kafka (Ao Vivo ou Vídeo Simulado via SHM)

O `I1` aceita diretamente os apelidos de socket do P4: `imx519` (`/tmp/p4_imx519_shm`), `imx477` (`/tmp/p4_imx477_shm`) e `video` (`/tmp/p4_file_shm`).

1. **Terminal 1 (no repositório `p4`):**
   ```bash
   ./scripts/p4_demo.sh video    # ou imx519 / imx477
   ```
2. **Terminal 2 (na pasta `i1/`):**
   ```bash
   # Na Jetson com GPU (P2 TensorRT) publicando no Kafka (172.18.4.56:9093):
   I1_SOURCE=video I1_MODE=kafka docker compose --profile jetson run --rm i1-jetson

   # Ou no computador/CPU (ONNX Runtime) sem GPU:
   I1_SOURCE=video I1_MODE=kafka docker compose run --rm i1-degrau-a
   ```

> **Correlação de quadros:** Em `shm`, `payload.frame_index` preserva o `frame_id` do P4 e `sequence` conta os eventos emitidos por `session_id`. Quadros corrompidos são suprimidos com log `[B2-OBS]` e deixam uma lacuna implícita em `sequence`.

---

## 6. Referência Rápida

### Variáveis de Ambiente (`docker-compose.yaml`)

| Variável | Padrão (`i1-degrau-a` / `i1-jetson` / `i1-orchestrator`) | Descrição |
| :--- | :--- | :--- |
| `I1_BACKEND` | `ultralytics` / `p2` / `p2` | Motor de inferência (`p2`, `ultralytics`, `mock`, `auto`). |
| `P2_BACKEND` | *(auto)* / `trt` / `trt` | Força backend dentro do `p2` (`trt` = GPU TensorRT, `ort` = CPU ONNX Runtime). |
| `P2_ENGINE_DIR` | `./I1/artifacts` / `/opt/ods/engines` | Pasta com as engines compiladas pelo `p2-build`. |
| `P2_PACKAGE_DIR` | `/app/I1/artifacts/app-plates/1.0.0` | Pacote compilado pelo serviço `i1-p2-build`. |
| `I1_VIDEOS_DIR` | `/home/sys2/SIS-2/videos` | Pasta com vídeos nomeados por equipe usados pelo `i1-orchestrator`. |
| `I1_COMMAND_TOPIC` | `ods.i1.command` | Tópico Kafka de comandos do orquestrador. |
| `I1_DEFAULT_TTL` | `60.0` | Tempo limite (s) sem heartbeat/STOP por sessão no orquestrador. |
| `I1_SOURCE` | `.../app-plates/1.0.0/fixtures` | Pasta de imagens, vídeo `/tmp/*.mp4`, `imx519`, `imx477`, `video`, `shm` ou `0`. |
| `P4_SHM_SOCKET` | `/tmp/p4_imx519_shm` | Socket SHM usado quando `I1_SOURCE=shm`. |
| `I1_MODE` | `stdout` (ou `kafka` no orchestrator) | Destino dos eventos (`stdout` ou `kafka`). |
| `ODS_KAFKA_BROKER` | `172.18.4.56:9093` (nuvem: `34.66.235.91:9093`) | Endereço `host:porta` do broker Kafka. |
| `I1_MAX_FRAMES` | `10` / `20` | Limite de quadros por execução. |
| `I1_CONF` | `0.50` | Limiar mínimo de confiança. |
| `I1_SOURCE_ID` | `cam0` | Identificador da câmera/fonte (em `shm`, usa o do cabeçalho `P4R1`). |
| `I1_METADATA_PATH` | `.../app-plates/1.0.0/metadata.json` | `metadata.json` da aplicação. |
| `I1_ONNX_PATH` | `.../app-plates/1.0.0/app-plates.onnx` | `.onnx` da aplicação (usado com `ultralytics`). |
| `I1_OUTPUT_JSONL` | `/app/*_events.jsonl` | Caminho do arquivo de saída JSONL (use `/tmp/...` para salvar no host). |

### Contrato de Saída (`ods.i1.detections` v1.0 — Perfil EV-SINAL)
Schema formal em [`schemas/ods.i1.detections.v1.0.schema.json`](schemas/ods.i1.detections.v1.0.schema.json). Se o broker Kafka estiver inacessível no `flush()`, o processo encerra com `exit code 3`.

```json
{
  "message_type": "event",
  "message_id": "5f1e2c7a-9b1d-4e0a-8c6f-2d3a4b5c6d7e",
  "schema": "ods.i1.detections",
  "schema_version": "1.0",
  "producer": "I1",
  "published_at": "2026-10-21T14:30:00.120Z",
  "captured_at": "2026-10-21T14:29:59.870Z",
  "source_id": "cam0",
  "session_id": "9c2a1f3e-4d5b-4c6a-9e8f-0a1b2c3d4e5f",
  "sequence": 0,
  "artifact_version": "app-plates-1.0.0-fp16",
  "payload": {
    "frame_index": 0,
    "image_width": 1920,
    "image_height": 1080,
    "detections": [
      {
        "detection_id": 0,
        "class": "placa_1L_BR",
        "confidence": 0.9312,
        "bbox": [412, 180, 598, 702]
      }
    ]
  }
}
```
