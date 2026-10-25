# Squad MOD-1 — Componente I1 (Detecção por Quadro)

**Disciplina:** Oficina de Desenvolvimento de Software (ODS 2026/2 — UFSJ)  
**Especificação:** Interfaces 1.0 (Arquitetura 6c Oficial)  
**Integrantes:**
- Felipe Girardi Siqueira (Líder — Matrícula: 2023008120)
- Thiago Mateus de Almeida Dias Orsini (Matrícula: 2022004959)
- Eduardo Monteiro Costa Pires (Matrícula: 212050024)

---

## 1. Visão Geral e Arquitetura (Versão 6c)

Este repositório contém o **Componente I1 (Camada 2)** do squad **MOD-1**: detector de objetos por quadro projetado sob o princípio **"Um artefato por aplicação, um componente só"**. O código-fonte é único e agnóstico de domínio: a adaptação para diferentes aplicações ocorre exclusivamente pela substituição do artefato `MODEL` (`.onnx` + `metadata.json`).

Os artefatos são produzidos pela **Frente Transversal B5**, mantida no repositório [`ods-26-2/b5`](https://github.com/ods-26-2/b5). Uma cópia do pacote em uso fica em `I1/artifacts/`, para que o I1 possa ser construído e testado sem depender do repositório de B5 (o `hash_sha256` do `metadata.json` identifica a versão publicada).

### Estrutura de Diretórios

```text
.
├── I1/
│   ├── artifacts/
│   │   └── app-plates/1.0.0/        # Cópia do pacote MODEL publicado por B5 (repo b5)
│   │       ├── app-plates.onnx
│   │       ├── metadata.json
│   │       └── fixtures/            # Imagens de teste usadas no Degrau A e no T16
│   ├── detector/
│   │   ├── config.py                # AppConfig e leitura automática de metadados de B5
│   │   ├── contracts.py             # Envelope EV-SINAL, ods.p4.frame e exceções de P2
│   │   ├── ingestion.py             # Conversão dinâmica BGR, RGB, GRAY e NV12
│   │   ├── inference.py             # Cliente da API fechada P2 (load, infer, info, release)
│   │   ├── postprocessing.py        # Filtro de confiança, NMS por classe e bbox inteira
│   │   ├── publishing.py            # Validador B1 acoplado, deduplicação UUIDv4, Stdout/Kafka
│   │   └── pipeline.py              # Orquestrador, controle de sessão (B2) e telemetria
│   ├── tests/
│   │   └── test_i1_sprint3.py       # Suíte automatizada de auditoria e contrato (T15)
│   ├── mock_producer.py             # Gerador de eventos sintéticos para CI (B4) e consumidores
│   └── run_detector.py              # CLI principal para execução do pipeline I1
├── schemas/
│   └── ods.i1.detections.v1.0.schema.json # Contrato formal JSON Schema (Draft 2020-12)
├── docs/diagrams/
├── Dockerfile
├── docker-compose.yaml
├── .dockerignore
└── STATUS_SPRINT3.md                # Relatório de acompanhamento das 21 tarefas da Sprint 3
```

---

## 2. Guia para Outros Squads: Execução via Docker

Não é necessário configurar ambiente Python localmente. Todos os fluxos de teste, simulação e inferência estão containerizados no `docker-compose.yaml`.

### 2.1. Para a Equipe de Testes e CI (B4 / TRA-2 — Tarefa T15)
Executa a suíte de auditoria `I1/tests/test_i1_sprint3.py`, validando:
- Aderência estrita ao envelope **EV-SINAL** (`ods.i1.detections` v1.0) e rejeição de campos legados;
- Emissão de `"detections": []` em quadros vazios e bloqueio de duplicatas por `message_id` (UUID v4);
- Pré-processamento dinâmico (`BGR`, `RGB`, `GRAY`, `NV12`) e NMS agrupado por classe;
- Tratamento dos 5 códigos de erro formais da biblioteca `P2`;
- Reinício determinístico de `sequence = 0` em eventos `START`, `SEEK`, `LOOP` e `RESTART` (**T6**);
- Supressão de quadro corrompido com log `[B2-OBS]` e lacuna implícita em `sequence` (**T7**);
- Execução fim a fim do **Degrau A** com o modelo ONNX real (**T16**).

```bash
docker compose run --rm i1-ci-tests
```

### 2.2. Para Consumidores (I2, I8, I9, S1) e B4: Produtor Mock (`mock_producer.py`)
Emite eventos sintéticos validados pelo validador acoplado de B1:

- **Modo `stdout` (imprime 10 eventos no terminal sem uso de rede):**
  ```bash
  docker compose run --rm i1-mock-stdout
  ```

- **Modo `kafka` (publica eventos no tópico `ods.i1.detections` do broker oficial `34.66.235.91:9093`):**
  ```bash
  docker compose run --rm i1-mock-kafka
  ```

- **Customizando a quantidade de quadros ou o endereço do broker Kafka:**
  ```bash
  ODS_KAFKA_BROKER=34.66.235.91:9093 docker compose run --rm i1-mock-kafka \
    python3 I1/mock_producer.py --frames 20 --mode kafka
  ```

### 2.3. Execução do Detector I1 com Modelos ONNX Reais (Degrau A — T16 e T20)
Comprova experimentalmente a propriedade **"Um artefato por aplicação, um componente só"**:

- **Aplicação 2 — Estacionamento / Placas (`app-plates` v1.0.0 — Squad VER-5):** (padrão do serviço)
  ```bash
  docker compose run --rm i1-degrau-a
  ```

- **Outro artefato ou outra fonte de imagens** (mesmo código, trocando apenas `--onnx` e `--metadata`):
  ```bash
  docker compose run --rm i1-degrau-a \
    python3 I1/run_detector.py \
    --onnx /app/I1/artifacts/app-plates/1.0.0/app-plates.onnx \
    --metadata /app/I1/artifacts/app-plates/1.0.0/metadata.json \
    --source /app/I1/artifacts/app-plates/1.0.0/fixtures \
    --max-frames 10 \
    --conf 0.50 \
    --output-jsonl degrau_a_plates_events.jsonl
  ```

> O pacote da Aplicação 1 (`app-epi`) ainda não possui `.onnx` publicado; será adicionado em `I1/artifacts/app-epi/1.0.0/` quando B5 o liberar.

#### Parâmetros Suportados pelo `I1/run_detector.py`

| Parâmetro | Padrão | Descrição |
| :--- | :--- | :--- |
| `--onnx` | `I1/artifacts/app-plates/1.0.0/app-plates.onnx` | Caminho para o arquivo `.onnx` do pacote B5. |
| `--metadata` | `I1/artifacts/app-plates/1.0.0/metadata.json` | Caminho para o `metadata.json` (carrega `artifact_id`, `artifact_version` e `classes`). |
| `--source` | `"0"` | `0` para webcam, caminho de vídeo `.mp4` ou diretório de imagens de teste. |
| `--max-frames` | `20` | Limite de quadros a processar na sessão. |
| `--conf` | `0.50` | Limiar de confiança mínimo para manter detecções. |
| `--mode` | `"stdout"` | Destino de publicação: `stdout` (terminal) ou `kafka` (broker B3). |
| `--output-jsonl` | `"degrau_a_events.jsonl"` | Arquivo local onde os eventos validados são persistidos. |

---

## 3. Contrato de Saída (`ods.i1.detections` v1.0 — Perfil EV-SINAL)

O schema formal em `schemas/ods.i1.detections.v1.0.schema.json` valida a estrutura abaixo em 100% das publicações de **I1**:

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
  "artifact_version": "app-epi-1.0.0",
  "payload": {
    "frame_index": 0,
    "image_width": 1920,
    "image_height": 1080,
    "detections": [
      {
        "detection_id": 0,
        "class": "capacete",
        "confidence": 0.9312,
        "bbox": [412, 180, 598, 702]
      }
    ]
  }
}
```