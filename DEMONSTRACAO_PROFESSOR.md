# Roteiro de Demonstração e Validação Integrada — Componente I1 (MOD-1)

**Disciplina:** Oficina de Desenvolvimento de Software (ODS 2026/2 — UFSJ)  
**Professor:** Alvaro  
**Componente:** I1 — Detecção de Objetos por Quadro (Camada 2)  
**Squad MOD-1:** Felipe Girardi Siqueira (Líder), Thiago Mateus de Almeida Dias Orsini, Eduardo Monteiro Costa Pires  
**Arquitetura:** Interfaces 1.0 (Versão 6c Oficial)

---

## 1. Visão Executiva e Princípio Fundamental

> **"Um artefato por aplicação, um componente só."**  
> O código do **I1** é único, genérico e agnóstico de domínio. A adaptação para diferentes aplicações (**A1** Indústria, **A5** EPI, **A8** Estacionamento) ocorre **exclusivamente pela parametrização do artefato MODEL** (`metadata.json` + `.onnx` ou `.engine`), sem alterar nenhuma linha de código ou lógica de inferência.

### O Papel do I1 na Arquitetura 6c
O I1 atua como o **motor central de percepção visual da Jetson**:
1. **Entrada de Vídeo:** Consome quadros em alta taxa via **Memória Compartilhada (SHM)** disponibilizados pelo componente **P4** (`application/x-ods-frame`, protocolo binário `P4R1`).
2. **Inferência de Alta Performance:** Executa a inferência via biblioteca **P2** utilizando os núcleos de GPU/TensorRT da NVIDIA Jetson Orin (ou CPU via ONNX Runtime para testes/desenvolvimento).
3. **Publicação Contratual:** Cada quadro processado é envelopado no formato oficial **EV-SINAL** (`ods.i1.detections` v1.0), validado contra o JSON Schema formal de **B1** e publicado no **Apache Kafka** da Jetson (`172.18.4.56:9093`), alimentando os consumidores de rastreamento (**I2**), zonas (**I8**), ocorrências (**I9**) e persistência (**S1** / **S2**).

---

## 2. Topologia de Integração na Jetson

```text
[ Squad SIS-2 (P4 - Fonte de Vídeo) ]
  ├── Câmeras CSI (IMX519 / IMX477) -> 1920x1080 RG10 RAW
  └── Vídeo Simulado (MP4)            -> Resolução nativa BGR8
               │
               ▼  (Pacotes P4R1 via GStreamer shmsink)
[ Memória Compartilhada (IPC /tmp) ]
  ├── /tmp/p4_imx519_shm
  ├── /tmp/p4_imx477_shm
  └── /tmp/p4_file_shm
               │
               ▼  (P4GStreamerShmSource + Conversão BGR)
[ Squad MOD-1 (Componente I1) ]
  ├── Ingestão & Decodificação P4R1 (suporta RG10 e BGR8)
  ├── Pré-processamento Dinâmico (Letterbox 640x640)
  ├── Inferência In-Process GPU via P2 TensorRT (/opt/ods/engines)
  ├── Pós-processamento (NMS intra-classe + Bounding Boxes inteiras)
  └── Validador B1 Integrado (JSON Schema v1.0 + UUIDv4)
               │
               ▼  (Publicação Kafka: tópico 'ods.i1.detections')
[ Broker Kafka da Jetson / Rede Dcomp: 172.18.4.56:9093 ]
               │
               ▼  (Assinatura assíncrona do tópico)
[ Consumidores Downstream ]
  ├── I2 (Rastreamento Multi-Objeto)
  ├── I8 (Violação de Zonas)
  ├── I9 (Detector de Ocorrências)
  └── S1 / S2 (Persistência e Histórico)
```

---

## 3. Roteiro Passo a Passo para Apresentação Prática

Para apresentar amanhã, execute os cenários abaixo em ordem. Cada cenário comprova um nível de maturidade do componente.

### Cenário 1: Auditoria Automatizada da Suíte de Testes (100% OK)
> **Objetivo:** Mostrar ao professor que o componente possui bateria de testes rigorosa validando contrato, sessão, P4R1, P2 e Degrau A.

```bash
docker compose run --rm i1-ci-tests
```

* **Tempo de Execução:** ~10 segundos.
* **O que observar na saída:**
  * **40 testes executados e aprovados (`OK`)**.
  * Validação do envelope `ods.i1.detections` (11 campos obrigatórios e rejeição de campos legados).
  * Decodificação dos buffers binários `P4R1` nos formatos `RG10` (câmeras CSI) e `BGR8` (vídeo MP4 do P4).
  * Controle de sessão (`session_id`) e preservação de lacunas em quadros descartados (`sequence`).

---

### Cenário 2: Validação Direta do Broker Kafka da Jetson (`172.18.4.56:9093`)
> **Objetivo:** Demonstrar conectividade e publicação imediata no broker oficial configurado pela manhã.

```bash
docker compose run --rm i1-mock-kafka
```

* **O que acontece:**
  1. O serviço `i1-mock-kafka` conecta-se a `172.18.4.56:9093` no tópico `ods.i1.detections`.
  2. Publica 10 mensagens sintéticas oficiais validadas pelo JSON Schema de B1.
  3. Realiza o `flush()` síncrono confirmando que 100% das mensagens foram entregues (pendências: 0).
* **Para inspecionar os eventos chegando no Kafka:**
  No terminal do P4 (ou consumidor da Jetson):
  ```bash
  ./scripts/p4_kafka_ler.sh
  ```

---

### Cenário 3: Demonstração Integrada P4 → I1 (Modo Câmera Simulada / Vídeo MP4)
> **Objetivo:** Demonstrar o pipeline completo ponta a ponta sem depender de iluminação ou alinhamento de câmera física.

1. **Terminal 1 — Iniciar o P4 com vídeo:**
   ```bash
   cd ~/SIS-2/p4
   ./scripts/p4_demo.sh video
   ```
   *O P4 começará a publicar os quadros em `/tmp/p4_file_shm` com formato BGR8.*

2. **Terminal 2 — Iniciar o I1 consumindo o P4 e publicando no Kafka:**
   ```bash
   cd ~/caminho/i1
   I1_SOURCE=video I1_MODE=kafka I1_MAX_FRAMES=100 docker compose run --rm i1-degrau-a
   ```

* **O que observar:**
  * O `I1` detecta automaticamente o socket `/tmp/p4_file_shm`.
  * Lê o cabeçalho `P4R1`, converte o formato `BGR8`, aplica letterbox e executa a inferência.
  * Valida cada detecção e publica no broker `172.18.4.56:9093`.
  * Ao atingir os 100 quadros, exibe o resumo de telemetria com latência média e contagem de objetos.

---

### Cenário 4: Demonstração com Câmera Física na Jetson (IMX519 / IMX477 + GPU TensorRT P2)
> **Objetivo:** Demonstrar o funcionamento em hardware real com GPU (TensorRT FP16 via `p2[trt]`, diferente do modo `I1_BACKEND=mock` que roda apenas em CPU).

1. **Pré-requisito (uma única vez na Jetson — compilar o `.onnx` em `.engine`):**
   - **Via Docker GPU (`Dockerfile.jetson`):**
     ```bash
     sudo mkdir -p /opt/ods/engines
     docker compose --profile jetson run --build --rm i1-p2-build
     ```
   - **Ou via `.venv` no host (conforme `guia_rapido.md` do P2):**
     ```bash
     /usr/bin/python3 -m venv --system-site-packages .venv
     source .venv/bin/activate
     pip install -r requirements-jetson.txt
     export P2_ENGINE_DIR=/opt/ods/engines
     p2-build I1/artifacts/app-plates/1.0.0/
     p2-list
     ```

2. **Terminal 1 — Iniciar a câmera no P4:**
   ```bash
   cd ~/SIS-2/p4
   ./scripts/p4_demo.sh imx519
   ```

3. **Terminal 2 — Iniciar o I1 com aceleração GPU P2/TensorRT (`P2_BACKEND=trt`):**
   ```bash
   cd ~/caminho/i1
   # Via Docker GPU (Dockerfile.jetson):
   I1_SOURCE=imx519 I1_MODE=kafka docker compose --profile jetson run --rm i1-jetson

   # Ou via .venv no host:
   P2_ENGINE_DIR=/opt/ods/engines P2_BACKEND=trt \
     python3 I1/run_detector.py --backend p2 --metadata I1/artifacts/app-plates/1.0.0/metadata.json --source imx519 --mode kafka
   ```

* **Destaques técnicos para ressaltar:**
  * O P4 transmite em `RG10` (RAW Bayer de 10 bits empacotado em 16 bits).
  * O I1 faz a conversão direta de Bayer para BGR usando otimizações vetorizadas antes de entregar à GPU.
  * A inferência roda em TensorRT FP16 consumindo a engine compilada em `/opt/ods/engines` (enquanto `I1_BACKEND=mock` serve apenas para simular o pipeline em CPU sem exigir GPU/TensorRT).

---

## 4. Tabela de Compatibilidade e Entregáveis

| Critério de Aceitação | Estado | Evidência Técnica |
| :--- | :---: | :--- |
| **Roda localmente de forma mockada** | ✅ Validado | `docker compose run --rm i1-mock-stdout` e `I1_BACKEND=mock docker compose run --rm i1-degrau-a` |
| **Roda na Jetson de forma mockada** | ✅ Validado | Suporte a ARM64, `MockP2Backend` simulando tensores sem exigir GPU |
| **Inferência P2 + Vídeo como entrada** | ✅ Validado | Suporte a MP4 nativo via OpenCV e leitor SHM de arquivos do P4 |
| **Integração com Kafka Oficial** | ✅ Validado | Default `172.18.4.56:9093` no Compose e CLI com monitoramento de flush |
| **Integração com P4 (IMX519 / IMX477 / Vídeo)** | ✅ Validado | Protocolo `P4R1` com suporte a `RG10` e `BGR8` |
| **Contrato EV-SINAL v1.0 (B1)** | ✅ Validado | Validador acoplado em 100% das mensagens, deduplicação por UUIDv4 |

---

## 5. Perguntas Frequentes do Professor e Como Responder

### P: "Por que o broker Kafka não ficou fixo no código Python?"
> **Resposta:**  
> *"Por diretriz arquitetural da especificação (RNF 3 e Tarefa T15), o código Python não deve ter brokers hardcoded para impedir que execuções de testes automatizados ou pipelines de CI disparem conexões não autorizadas na rede. No entanto, no nível de implantação da Jetson (`docker-compose.yaml` e defaults da CLI), definimos `172.18.4.56:9093` como padrão. Assim, na Jetson funciona 'out-of-the-box' sem configurações manuais, mantendo a flexibilidade de apontar para outro broker via variável `ODS_KAFKA_BROKER`."*

### P: "Como o I1 se adapta para outras aplicações (ex: EPI, Vôlei, Indústria)?"
> **Resposta:**  
> *"Seguimos à risca o princípio 'Um artefato por aplicação, um componente só'. O código do I1 não muda. O container recebe apenas as variáveis de ambiente `I1_METADATA_PATH` e `I1_ONNX_PATH` (ou o identificador da engine TensorRT compilada em `/opt/ods/engines`). O detector lê o `metadata.json`, extrai a lista de classes do domínio e a resolução de entrada, adaptando-se em tempo de execução."*

### P: "O que acontece se o P4 descartar quadros ou houver oscilação de taxa?"
> **Resposta:**  
> *"O I1 desacopla o `frame_index` da câmera do `sequence` do publicador (Tarefa T6). O `frame_index` preserva a identidade temporal do quadro gerado pelo P4, enquanto o `sequence` mede estritamente os eventos emitidos pelo I1 na sessão. Se um quadro vier corrompido, o I1 suprime o envio, registra a ocorrência no log de observabilidade B2 (`[B2-OBS]`) e deixa uma lacuna deliberada no `sequence`, permitindo que os componentes downstream detectem a falha pontual."*

### P: "E se o broker Kafka cair ou ficar lento?"
> **Resposta:**  
> *"O publicador do I1 opera com fila assíncrona de alto throughput e callback de confirmação de entrega (`_on_delivery`). Ao final da sessão, é executado um `flush()` obrigatório. Se houver mensagens não entregues ou erro de timeout no broker, o processo é encerrado com código de erro `exit code 3`, evitando que falhas passem despercebidas."*

