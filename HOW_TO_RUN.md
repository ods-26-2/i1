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

Os artefatos são produzidos pela **Frente Transversal B5**, mantida no repositório [`ods-26-2/b5`](https://github.com/ods-26-2/b5). Uma cópia dos pacotes em uso fica em `I1/artifacts/`, para que o I1 possa ser construído e testado sem depender do repositório de B5 (o `hash_sha256` do `metadata.json` identifica a versão publicada).

### 1.1. Como o `I1` é Invocado e Como Atende às Diferentes Aplicações (`A1` a `A8`)

Conforme a **Arquitetura 6c Oficial**:
1. **Forma de invocação (Serviço Contínuo de Stream):** O `I1` **não é um servidor HTTP/RPC** nem uma biblioteca importada por outros componentes. Ele é executado como um **container daemon** orquestrado via `docker-compose` pelo componente **P3** (*Execução, build e armazenamento*, squad `SIS-1`), rodando `python3 I1/run_detector.py` em loop contínuo:
   - **Entrada:** consome quadros do socket de memória compartilhada (`SHM`) publicado por **P4** (`/tmp/p4_imx519_shm`, via volume `/tmp:/tmp` e `ipc: host`) em produção, ou vídeos/streams de **P6** em sessões de replay;
   - **Inferência:** invoca a biblioteca **P2** *in-process* (`p2.load`, `p2.info`, `p2.infer`, `p2.release`) para executar o modelo na GPU da Jetson via TensorRT (ou `ultralytics`/`mock` em desenvolvimento);
   - **Saída:** valida cada evento com o validador acoplado de **B1** e publica no tópico Kafka **`ods.i1.detections`**, que é consumido assincronamente por `I2`, `I7`, `I8`, `I9`, `S1` e `S2`.
2. **Múltiplas aplicações ("Um artefato por aplicação, um componente só"):** Na Arquitetura 6c, cada aplicação (`A1` Indústria, `A2` Acesso, `A3` Monitoramento, `A4` Esporte, `A5` EPI, `A8` Catraca inteligente) gera uma *build* e uma implantação `docker-compose` independentes. O código do `I1` nunca muda entre aplicações: basta parametrizar o container daquela aplicação com as variáveis `I1_METADATA_PATH` (e `I1_ONNX_PATH` quando em modo ONNX) apontando para o artefato correspondente.

### 1.2. Onde Ficam Salvos os Artefatos (Repositório vs. Jetson) e Como Adicionar Novos Modelos (ex: `app-epi`)

Todos os artefatos de diferentes aplicações e versões **podem e devem coexistir dentro da mesma pasta `I1/artifacts/`**, organizados no formato `I1/artifacts/<artifact_id>/<artifact_version>/`.

Existem **duas etapas** do artefato `MODEL` na arquitetura (`B5 -> P2 -> I1`):

1. **Pacote de Modelo do B5 (`.onnx` + `metadata.json`) — fica em `I1/artifacts/<artifact_id>/<artifact_version>/`:**
   - Contém o `metadata.json` (lido pelo `I1` em todos os modos para descobrir `artifact_id`, `artifact_version` e a lista ordenada de `classes`) e o `.onnx` agnóstico de hardware (usado pelo `I1` no modo local/CPU `ultralytics` e pelo `P2` para compilar a engine).
   - No container, essa pasta fica em `/app/I1/artifacts` (ou pode ser montada do host via variável `I1_ARTIFACTS_DIR`).
2. **Engine TensorRT compilada pelo P2 na Jetson (`.engine`) — fica em `/opt/ods/engines` no host da Jetson:**
   - Na placa Jetson, a ferramenta `p2-build` (do componente **P2**) compila o `.onnx` de B5 para o JetPack/TensorRT exato da placa e registra o `.engine` no diretório padrão do P2: **`/opt/ods/engines`** (controlado pela variável `P2_ENGINE_DIR=/opt/ods/engines`).
   - Quando o `I1` roda com `I1_BACKEND=p2`, ele lê o `metadata.json` (ex.: `artifact_id: "app-epi"`, `artifact_version: "1.0.0"`), monta o identificador `"app-epi-1.0.0"` e chama `p2.load("app-epi-1.0.0")`, que busca a engine correspondente em `/opt/ods/engines`.

#### Passo a passo: Subindo um novo artefato (exemplo: `app-epi` v1.0.0)

1. Crie a pasta da aplicação e versão dentro de `I1/artifacts/`:
   ```text
   I1/artifacts/
   ├── app-plates/
   │   └── 1.0.0/
   │       ├── app-plates.onnx
   │       ├── metadata.json
   │       └── fixtures/
   └── app-epi/
       └── 1.0.0/
           ├── app-epi.onnx
           ├── metadata.json
           └── fixtures/          # (Opcional) vídeos ou imagens de teste de EPI
   ```
2. Garanta que o `I1/artifacts/app-epi/1.0.0/metadata.json` contenha os campos do contrato B5 (`artifact_id`, `artifact_version`, `classes`, etc.):
   ```json
   {
     "artifact_id": "app-epi",
     "artifact_version": "1.0.0",
     "model_format": "onnx",
     "model_file": "app-epi.onnx",
     "input_shape": [1, 3, 640, 640],
     "opset_version": 17,
     "classes": ["pessoa", "capacete", "colete", "luva", "oculos"],
     "conf_threshold_default": 0.50,
     "iou_threshold_default": 0.45
   }
   ```
3. Execute o `I1` apontando para o novo artefato (sem precisar reconstruir a imagem Docker, pois `./I1:/app/I1:ro` já está montado):
   ```bash
   # Localmente (ONNX Runtime / Ultralytics):
   I1_ONNX_PATH=/app/I1/artifacts/app-epi/1.0.0/app-epi.onnx \
   I1_METADATA_PATH=/app/I1/artifacts/app-epi/1.0.0/metadata.json \
   I1_SOURCE=/tmp/video_epi.mp4 \
   I1_OUTPUT_JSONL=/tmp/epi_events.jsonl \
   docker compose run --rm i1-degrau-a

   # Na Jetson (via P2 TensorRT, com a engine app-epi-1.0.0 em /opt/ods/engines):
   P2_ENGINE_DIR=/opt/ods/engines \
   I1_BACKEND=p2 \
   I1_METADATA_PATH=/app/I1/artifacts/app-epi/1.0.0/metadata.json \
   I1_SOURCE=shm \
   ODS_KAFKA_BROKER="<broker-host:porta>" \
   I1_MODE=kafka \
   docker compose --profile jetson run --rm i1-jetson
   ```

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
│   │   ├── ingestion.py             # P4R1/SHM e conversão BGR, RGB, GRAY, NV12 e RG10
│   │   ├── inference.py             # Cliente da API fechada P2 (load, infer, info, release)
│   │   ├── postprocessing.py        # Filtro de confiança, NMS por classe e bbox inteira
│   │   ├── publishing.py            # Validador B1 acoplado, deduplicação UUIDv4, Stdout/Kafka
│   │   ├── schema_validation.py     # Validador gerado do JSON Schema oficial
│   │   └── pipeline.py              # Orquestrador, controle de sessão (B2) e telemetria
│   ├── tests/
│   │   ├── test_contract_schema.py  # Contrato: schema, exemplos, mock e publicador (sem GPU/torch)
│   │   ├── test_p4_ingestion.py     # Parser P4R1, formatos de buffer e timestamp ausente
│   │   ├── test_session.py          # Sessão e sequence em replay (T6)
│   │   ├── test_error_handling.py   # Representação de erro (T7)
│   │   └── test_i1_sprint3.py       # Suíte de auditoria da Sprint 3 (inclui Degrau A)
│   ├── mock_producer.py             # Produtor falso gerado dos exemplos do schema (B4 e consumidores)
│   └── run_detector.py              # CLI principal para execução do pipeline I1
├── schemas/
│   ├── ods.i1.detections.v1.0.schema.json # Contrato formal JSON Schema (Draft 2020-12)
│   └── examples/{valid,invalid}/    # Mensagens de exemplo para B1 e para a CI de B4
├── .github/workflows/i1-tests.yml   # CI a cada push: contrato (leve) e pipeline completo
├── requirements-contract.txt        # Dependências da suíte de contrato
├── requirements.txt                 # Dependências do pipeline completo (Degrau A)
├── docs/diagrams/
├── Dockerfile
├── docker-compose.yaml
├── .dockerignore
└── STATUS_SPRINT3.md                # Relatório de acompanhamento das 21 tarefas da Sprint 3
```

---

## 2. Guia para Outros Squads: Execução via Docker

Não é necessário configurar ambiente Python localmente. Todos os fluxos de teste, simulação e inferência estão containerizados no `docker-compose.yaml`.

> **Nota sobre `--build` e tempo de execução:**
> - Todos os serviços do `docker-compose.yaml` montam `./I1:/app/I1:ro` e `./schemas:/app/schemas:ro` diretamente do host, além de `/tmp:/tmp`.
> - Use `--build` apenas na **primeira execução** (ou se alterar o `Dockerfile`). Nas execuções seguintes, omita `--build` (`docker compose run --rm <serviço>`) para reutilizar a imagem local instantaneamente (~4 a 8 segundos).
> - O `Dockerfile` instala previamente `torch` e `torchvision` de CPU (`https://download.pytorch.org/whl/cpu`) antes do `ultralytics`, evitando o download desnecessário de ~3,5 GB de pacotes CUDA em máquinas de desenvolvimento/CI.

### 2.0. Resumo Rápido dos 4 Cenários de Aceitação

1. **Rodar localmente no computador de forma mockada:**
   ```bash
   # Opção A: Produtor mock oficial (emite eventos sintéticos a partir dos exemplos do JSON Schema)
   docker compose run --rm i1-mock-stdout

   # Opção B: Pipeline completo do detector (ingestão + pós-processamento + validação B1) com MockP2Backend
   I1_BACKEND=mock I1_MAX_FRAMES=5 docker compose run --rm i1-degrau-a

   # Opção C: Suíte automatizada completa (39 testes de contrato, sessão, P4R1, P2 + vídeo MP4 e Degrau A)
   docker compose run --rm i1-ci-tests
   ```

2. **Rodar na Jetson de forma mockada (sem exigir câmera CSI ativa nem engine TensorRT):**
   ```bash
   # Produtor mock oficial em ARM64
   docker compose run --rm i1-mock-stdout

   # Pipeline completo em modo mockado (na imagem padrão ou usando o perfil jetson)
   I1_BACKEND=mock I1_MAX_FRAMES=10 docker compose run --rm i1-degrau-a
   # Ou no perfil jetson (com runtime nvidia disponível):
   I1_BACKEND=mock I1_MAX_FRAMES=10 docker compose --profile jetson run --rm i1-jetson
   ```

3. **Rodar usando a inferência de P2 e um vídeo `.mp4` como entrada:**
   ```bash
   # Copie ou aponte o vídeo em /tmp (já montado por padrão em /tmp:/tmp no Compose)
   # No computador (usando p2[cpu] instalado na imagem) ou sem runtime nvidia:
   P2_ENGINE_DIR=/caminho/no/host/para/engines \
   I1_BACKEND=p2 \
   I1_SOURCE=/tmp/video.mp4 \
   I1_MAX_FRAMES=100 \
   docker compose run --rm i1-degrau-a

   # Na Jetson (usando o perfil jetson e runtime nvidia):
   P2_ENGINE_DIR=/caminho/no/host/para/engines \
   I1_BACKEND=p2 \
   I1_SOURCE=/tmp/video.mp4 \
   I1_MAX_FRAMES=100 \
   docker compose --profile jetson run --rm i1-jetson
   ```

4. **Funcionar com o Broker Kafka Oficial da Jetson (`172.18.4.56:9092` na rede Dcomp):**
   ```bash
   # Com o produtor mock (serviço dedicado i1-mock-kafka, já configurado por padrão com 172.18.4.56:9092):
   docker compose run --rm i1-mock-kafka

   # Com o pipeline completo do detector (publicando detecções reais no tópico ods.i1.detections):
   I1_MODE=kafka \
   I1_BACKEND=p2 \
   I1_SOURCE=imx519 \
   docker compose --profile jetson run --rm i1-jetson

   # Para apontar para outro broker externo:
   ODS_KAFKA_BROKER="<outro-broker:porta>" I1_MODE=kafka docker compose run --rm i1-mock-kafka
   ```
   > **Configuração do Kafka:** O broker oficial na Jetson (`172.18.4.56:9092`) está configurado como padrão no `docker-compose.yaml` e nos executáveis de CLI (`run_detector.py` e `mock_producer.py`) para testes imediatos na rede interna do Dcomp. Para apontar para outro broker, basta exportar a variável `ODS_KAFKA_BROKER="host:porta"`. Se o broker estiver inacessível e houver mensagens não entregues ao final da execução, o processo encerra com código de erro (`exit code 3`).

---

### 2.1. Para a Equipe de Testes e CI (B4 / TRA-2 — Tarefa T15)
A mesma suíte roda automaticamente a cada push no GitHub Actions (`.github/workflows/i1-tests.yml`), em dois jobs:
`contract` (só `requirements-contract.txt`, sem GPU nem torch) e `pipeline` (todas as suítes, com o ONNX real na CPU).
Nenhum job publica no Kafka. Localmente, via Docker, todas as 39 suítes de `I1/tests/` validam:
- Aderência estrita ao envelope **EV-SINAL** (`ods.i1.detections` v1.0), gerado do JSON Schema, contra os exemplos de `schemas/examples/` e rejeição de campos legados;
- Produtor falso (`mock_producer.py`) gerado dos mesmos exemplos válidos e saída real do publicador (`ValidatedStdoutPublisher` e `ConfluentKafkaPublisher`) conforme o schema;
- Emissão de `"detections": []` em quadros vazios e bloqueio de duplicatas por `message_id` (UUID v4);
- Pré-processamento dinâmico (`BGR`, `RGB`, `GRAY`, `NV12`, `RG10`), parser binário `P4R1` e NMS agrupado por classe;
- Integração com `NativeP2Backend` (letterbox 640x640, decodificação de saída bruta `[1, 4 + num_classes, N]`, validação de shapes contra `p2.info()`) consumindo vídeo `.mp4` via `OpenCVFrameSource`;
- Roteamento automático de `--source` (`resolve_frame_source`) distinguindo sockets SHM (`/tmp/p4_*_shm` ou `shm`), vídeos em `/tmp/*.mp4` e diretórios de imagens;
- Tratamento dos 5 códigos de erro formais da biblioteca `P2`;
- Reinício de `sequence = 0` somente na troca de `session_id` (P4/P6 geram sessão nova em `START`, `SEEK`, `LOOP`, `RESTART` e `RECONNECT`) e preservação de `payload.frame_index` diante de saltos upstream em `I1/tests/test_session.py` (**T6**);
- Supressão de quadro corrompido com log `[B2-OBS]` e lacuna implícita em `sequence` (**T7**);
- Execução fim a fim do **Degrau A** com o modelo ONNX real (**T16**).

```bash
docker compose run --rm i1-ci-tests
```

### 2.2. Para Consumidores (I2, I8, I9, S1) e B4: Produtor Mock (`mock_producer.py`)
Emite eventos sintéticos validados pelo validador acoplado de B1:

- **Modo `stdout` (imprime 10 eventos no terminal sem uso de rede):**
  ```bash
  docker compose run --build --rm i1-mock-stdout
  ```

- **Modo `kafka` (publica no tópico `ods.i1.detections`):** requer um broker acessível pelo container. O endereço
  vem de `ODS_KAFKA_BROKER` (ou `--broker`); não há broker local no Compose nem endereço padrão. O produtor
  configura `bootstrap.servers`, mas não configura credenciais/autenticação Kafka. Use um broker compatível com
  essa configuração:
  ```bash
  ODS_KAFKA_BROKER="<broker-host:porta>" docker compose run --build --rm i1-mock-kafka
  ```

- **Customizando a quantidade de quadros:**
  ```bash
  ODS_KAFKA_BROKER="<broker-host:porta>" docker compose run --build --rm i1-mock-kafka \
    python3 I1/mock_producer.py --frames 20 --mode kafka
  ```

### 2.3. Execução do Detector I1 com Modelos Reais, Vídeos e P2 (Degrau A — T16 e T20)
Comprova experimentalmente a propriedade **"Um artefato por aplicação, um componente só"**:

- **Aplicação 2 — Estacionamento / Placas (`app-plates` v1.0.0 — Squad VER-5):** (padrão do serviço)
  ```bash
  docker compose run --rm i1-degrau-a
  ```
  Por padrão, esse serviço usa o backend `ultralytics`/ONNX Runtime e processa as fixtures embarcadas no modo `stdout`. Graças à parametrização via variáveis de ambiente (`I1_BACKEND`, `I1_SOURCE`, `I1_MODE`, `P2_ENGINE_DIR`, `ODS_KAFKA_BROKER`), ele também pode rodar com `p2` ou `mock`, ler vídeos `.mp4` ou o socket SHM do P4, e publicar no Kafka sem precisar sobrescrever o comando do Compose.

- **Testando com um vídeo `.mp4` (via `/tmp` já montado ou volume extra):**
  Como `/tmp:/tmp` já está montado no `docker-compose.yaml` e o `resolve_frame_source()` diferencia automaticamente arquivos de mídia (`.mp4`, `.avi`, `.mov`, `.mkv`, etc.) de sockets SHM (`/tmp/p4_*_shm`), basta colocar ou apontar o vídeo em `/tmp`:
  ```bash
  # Usando o backend P2 e um vídeo em /tmp:
  P2_ENGINE_DIR=/caminho/no/host/para/engines \
  I1_BACKEND=p2 \
  I1_SOURCE=/tmp/camera-01.mp4 \
  I1_MAX_FRAMES=300 \
  I1_OUTPUT_JSONL=/tmp/camera-01-events.jsonl \
  docker compose run --rm i1-degrau-a

  # Ou usando o backend Ultralytics/ONNX local:
  I1_BACKEND=ultralytics \
  I1_SOURCE=/tmp/camera-01.mp4 \
  I1_MAX_FRAMES=300 \
  I1_OUTPUT_JSONL=/tmp/camera-01-events.jsonl \
  docker compose run --rm i1-degrau-a
  ```
  *(Se o vídeo estiver fora de `/tmp`, você também pode montar outra pasta com `--volume /caminho/no/host/videos:/videos:ro` e passar `I1_SOURCE=/videos/camera-01.mp4`).*

O caminho de vídeo usa a resolução efetivamente decodificada em cada quadro e o pré-processamento faz letterbox/redimensionamento para o tamanho de entrada do modelo (`640x640`), restaurando as coordenadas das bounding boxes para as dimensões originais do vídeo. Isso testa um vídeo de cada vez; `cam0` é o `source_id` padrão e pode ser alterado com `--source-id` ou `I1_SOURCE_ID`. Um arquivo de sensor RAW sem codec/container decodificável pelo OpenCV não deve ser passado como arquivo de vídeo comum (para RAW `RG10` da IMX519 via P4, use `I1_SOURCE=shm`).

Para selecionar um pacote B5 diferente no serviço Compose, aponte `I1_ARTIFACTS_DIR` para o diretório de artefatos do host (montado em `/app/I1/artifacts`):

```bash
I1_ARTIFACTS_DIR=/caminho/no/host/B5/artifacts \
I1_ONNX_PATH=/app/I1/artifacts/app-epi/1.0.0/app-epi.onnx \
I1_METADATA_PATH=/app/I1/artifacts/app-epi/1.0.0/metadata.json \
I1_SOURCE=/tmp/epi.mp4 I1_SOURCE_ID=cam-epi I1_MAX_FRAMES=300 \
docker compose run --rm i1-degrau-a
```

#### Estado da execução na Jetson e do fluxo P4/Kafka

**Ainda não é possível clonar e executar o fluxo de produção Jetson/P4 apenas com o Compose deste repositório.**
O serviço `i1-jetson` é um template de execução P2 e foi colocado no perfil `jetson` para distingui-lo dos testes
mock; ele não é iniciado por um `docker compose up` sem perfil. Os bloqueios atuais são:

- A imagem do `Dockerfile` instala a release `p2-0.2.0a1` com extra CPU, não TensorRT. A API dessa release é
  compatível com as chamadas do adaptador I1, mas `runtime: nvidia` não converte `p2[cpu]` em `p2[trt]`.
  A documentação atual do P2 indica JetPack 6.2.1 / L4T 36.4.7 / TensorRT 10.3.0 / CUDA 12.6 e requer compilar
  o engine para a plataforma-alvo com `p2-build`.
- O serviço precisa do engine `app-plates-1.0.0` disponível para P2 no diretório do host `${P2_ENGINE_DIR:-/opt/ods/engines}`.
  O ONNX em `I1/artifacts/` não é carregado pelo backend P2. Também é necessário Docker com runtime NVIDIA
  configurado para a Jetson.
- O código atual de P4 (`ods-26-2/p4`, branch `main`) não envia um `video/x-raw` BGR convencional. O produtor
  publica buffers com caps GStreamer `application/x-ods-frame` e um cabeçalho binário `P4R1` que inclui
  `source_id`, `session_id`, `frame_id`, `timestamp`, dimensões, formato, `raw` e tamanho dos dados, seguido
  dos bytes do frame. O evento Kafka `ods.p4.frame` usa `frame_id`, `timestamp`, `width`, `height` e `format`.
- O I1 foi adaptado para consumir as caps `application/x-ods-frame`, validar/decodificar o `P4R1` e usar os
  metadados do próprio buffer, sem tentar parear buffers e mensagens Kafka por ordem de chegada. A configuração
  de referência da IMX519 usa `RG10` e `raw=true`; o I1 trata `RG10` como Bayer RGGB de 10 bits em words de
  16 bits, converte para BGR e rejeita tamanhos inesperados.
- O leitor de SHM usa PyGObject/GStreamer do sistema. O Docker fixa Python 3.11 sobre Debian Bookworm para que
  o módulo `python3-gi` do Debian tenha ABI compatível com o Python da aplicação; execução local fora do Docker
  precisa disponibilizar esses bindings e os plugins GStreamer, incluindo `shmsrc` e `appsink`.
- A representação de `RG10` como RGGB/10-bit alinhado nos 16 bits é inferida do formato selecionado no código
  V4L2 do P4 e de `storage-bits=16` no publisher. Ainda precisa ser confirmada com um frame real da câmera e
  comparada com a imagem/saída conhecida do P4 antes de validar precisão de inferência.
- O cabeçalho SHM P4R1 pode trazer `timestamp: null` quando o driver não fornece timestamp de captura confiável.
  Nesse caso, o I1 usa o horário de recepção como `captured_at`; quando o timestamp do driver está presente,
  converte o relógio monotônico do host para UTC. A precisão temporal precisa ser medida no hardware.
- O P4 disponibiliza três fontes no SHM: câmeras físicas `imx519` (`/tmp/p4_imx519_shm`, RG10) e `imx477` (`/tmp/p4_imx477_shm`, RG10), e vídeo MP4 simulado (`/tmp/p4_file_shm`, BGR8). O `I1` aceita os aliases diretos (`I1_SOURCE=imx519`, `I1_SOURCE=imx477`, `I1_SOURCE=video` ou `I1_SOURCE=p4_video`) ou caminho explícito de socket (`P4_SHM_SOCKET`).
- O broker Kafka oficial na Jetson foi configurado em **`172.18.4.56:9092`** (acessível internamente na rede do Dcomp) e já está definido como valor padrão no `docker-compose.yaml` para todos os serviços com Kafka.
- O script `./scripts/p4_demo.sh` do repositório P4 (`ods-26-2/p4`) já configura as variáveis de ambiente necessárias e publica no socket correto:
  - `./scripts/p4_demo.sh imx519`
  - `./scripts/p4_demo.sh imx477`
  - `./scripts/p4_demo.sh video [caminho.mp4]`
  - `./scripts/p4_demo.sh kafka [caminho.mp4]` (publica no broker `172.18.4.56:9092`)

#### Execução Integrada P4 → I1 na Jetson (com P2 ou Degrau A)

1. Em um terminal, inicie a fonte no P4:
   ```bash
   # Exemplo com a câmera IMX519:
   ./scripts/p4_demo.sh imx519

   # Ou exemplo com vídeo simulado:
   ./scripts/p4_demo.sh video
   ```

2. Em outro terminal, na pasta `i1/`, suba o detector consumindo o socket correspondente:
   ```bash
   # Consumindo IMX519 via P2 com publicação no Kafka da Jetson:
   I1_SOURCE=imx519 I1_MODE=kafka docker compose --profile jetson run --rm i1-jetson

   # Ou consumindo vídeo simulado do P4 (BGR8 via SHM) com ONNX/CPU:
   I1_SOURCE=video I1_MODE=kafka docker compose run --rm i1-degrau-a
   ```

Esse teste exercita o buffer SHM real, o cabeçalho P4R1, a conversão RG10 e a inferência ONNX em CPU; não valida
TensorRT/P2, publicação Kafka nem desempenho em tempo real. O P4 atual não espera o consumidor SHM para iniciar
o envio, portanto o I1 pode perder os primeiros frames se iniciar depois do P4. Selecione a câmera por
`P4_CAMERA` no P4 e passe o socket correspondente em `P4_SHM_SOCKET` no I1.

Na Jetson, é possível tentar os testes de contrato e do produtor mock (não usam GPU nem Kafka em `stdout`),
desde que o build da imagem seja compatível com ARM64:

```bash
docker compose run --build --rm i1-ci-tests
docker compose run --build --rm i1-mock-stdout
```

O mock valida mensagens de saída de I1, mas não simula inferência P2, captura P4 ou integração Kafka de ponta a
ponta. Para testar apenas a publicação Kafka do mock, configure explicitamente um broker acessível:

```bash
ODS_KAFKA_BROKER="<broker-host:porta>" \
  docker compose run --build --rm i1-mock-kafka
```

O serviço P2 só deve ser executado depois de preparar uma imagem JetPack/P2 compatível e instalar/montar o engine
correspondente. Para testar o engine com fixtures (padrão do serviço), a partir do diretório do Compose:

```bash
P2_ENGINE_DIR="/caminho/no/host/para/engines" \
  docker compose --profile jetson run --build --rm i1-jetson
```

Esse comando ainda depende do runtime NVIDIA e do engine registrado no P2 (ou `I1_BACKEND=mock` para testar o pipeline de forma mockada na Jetson). Para publicar a saída no Kafka, passe
também `ODS_KAFKA_BROKER="<broker-host:porta>"` e `I1_MODE=kafka`. A ingestão shm
agora entende o protocolo P4R1. Para usar o socket da IMX519 em vez das fixtures, configure o P4 em modo completo,
inicie-o com o mesmo `/tmp` visível ao container e rode:

```bash
P2_ENGINE_DIR="/caminho/no/host/para/engines" \
I1_SOURCE=shm \
P4_SHM_SOCKET=/tmp/p4_imx519_shm \
docker compose --profile jetson run --build --rm i1-jetson
```

Isso ainda precisa ser validado na Jetson com a câmera real. O I1 usa metadados `P4R1` do SHM para cada quadro;
Kafka é usado para publicar a saída de I1 e não para sincronizar a entrada P4. A conversão RAW precisa ser
conferida contra os bytes/formato efetivamente entregues pelo driver da câmera.

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
| `--backend` | `"auto"` | Backend de inferência: `p2` (biblioteca SIS-1), `ultralytics` (ONNX local), `mock` (simulado) ou `auto`. |
| `--onnx` | `I1/artifacts/app-plates/1.0.0/app-plates.onnx` | Caminho para o arquivo `.onnx` do pacote B5. |
| `--metadata` | `I1/artifacts/app-plates/1.0.0/metadata.json` | Caminho para o `metadata.json` (carrega `artifact_id`, `artifact_version` e `classes`). |
| `--source` | `"0"` | `0` para webcam, caminho de vídeo `.mp4`, diretório de imagens de teste ou `shm` para socket P4. |
| `--shm-socket` | `"/tmp/p4_imx519_shm"` | Caminho do socket SHM do P4 quando `--source shm`. |
| `--max-frames` | `20` | Limite de quadros a processar na sessão. |
| `--source-id` | `"cam0"` | Identificador da fonte local; em `shm`, o I1 usa o `source_id` do cabeçalho P4R1. |
| `--conf` | `0.50` | Limiar de confiança mínimo para manter detecções. |
| `--mode` | `"stdout"` | Destino de publicação: `stdout` (terminal) ou `kafka` (broker B3, exige `ODS_KAFKA_BROKER`). |
| `--output-jsonl` | `"degrau_a_events.jsonl"` | Arquivo local onde os eventos validados são persistidos. |

#### Variáveis de Ambiente Suportadas no `docker-compose.yaml` (`i1-degrau-a` e `i1-jetson`)

| Variável | Padrão (`i1-degrau-a` / `i1-jetson`) | Descrição |
| :--- | :--- | :--- |
| `I1_BACKEND` | `ultralytics` / `p2` | Seleciona o motor de inferência (`ultralytics`, `p2`, `mock`, `auto`). |
| `I1_SOURCE` | `/app/I1/artifacts/app-plates/1.0.0/fixtures` | Fonte de entrada: pasta de imagens, vídeo (ex.: `/tmp/video.mp4`), `shm` ou `0`. |
| `I1_MODE` | `stdout` | Destino de publicação (`stdout` ou `kafka`). |
| `ODS_KAFKA_BROKER` | `""` (vazio) | `host:porta` do broker Kafka quando `I1_MODE=kafka` (ou no serviço `i1-mock-kafka`). |
| `P2_ENGINE_DIR` | `./I1/artifacts` / `/opt/ods/engines` | Diretório no host montado em `/opt/ods/engines` dentro do container para a biblioteca `p2`. |
| `P4_SHM_SOCKET` | `/tmp/p4_imx519_shm` | Caminho do socket GStreamer SHM quando `I1_SOURCE=shm`. |
| `I1_MAX_FRAMES` | `10` / `20` | Quantidade máxima de quadros a processar na execução. |
| `I1_SOURCE_ID` | `cam0` | `source_id` publicado para vídeos/pastas (no modo `shm` prevalece o `source_id` do cabeçalho `P4R1`). |
| `I1_CONF` | `0.50` | Limiar de confiança mínimo para manter detecções. |
| `I1_OUTPUT_JSONL` | `/app/degrau_a_plates_events.jsonl` | Caminho onde salvar o arquivo `.jsonl` (use `/tmp/...` para persistir direto no host). |
| `I1_ARTIFACTS_DIR` | `./I1/artifacts` | Diretório de artefatos no host montado em `/app/I1/artifacts`. |
| `I1_ONNX_PATH` | `/app/I1/artifacts/app-plates/1.0.0/app-plates.onnx` | Caminho do arquivo `.onnx` dentro do container. |
| `I1_METADATA_PATH` | `/app/I1/artifacts/app-plates/1.0.0/metadata.json` | Caminho do `metadata.json` dentro do container. |

No fluxo P4, `payload.frame_index` preserva o `frame_id` original do buffer. `sequence` conta os eventos que o I1
processa/publica dentro de cada `session_id`; portanto, se o consumidor SHM descartar frames, é esperado haver
saltos em `frame_index`, sem exigir que `sequence` seja igual ao índice da câmera. Use
`(source_id, session_id, payload.frame_index)` para relacionar uma detecção ao frame de origem. `sequence` serve
para ordenar a saída I1 e identificar lacunas causadas por falhas dentro do processamento I1.

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