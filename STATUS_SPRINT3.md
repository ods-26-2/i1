# Relatório de Status da Sprint 3 — Squad MOD-1 (B5 e I1)

**Disciplina:** ODS 2026/2 (UFSJ) — Especificação de Interfaces 1.0 (Arquitetura 6c)  
**Data de Atualização:** 21 de Outubro de 2026  
**Integrantes:** Felipe Girardi Siqueira (Líder), Thiago Mateus de Almeida Dias Orsini, Eduardo Monteiro Costa Pires

---

## 1. Visão Geral do Progresso (21 Tarefas da Sprint 3)

| Categoria de Status | Quantidade | Tarefas (IDs no Jira) |
| :--- | :---: | :--- |
| **100% Concluídas (Código + Validação Fechada)** | **12** | `T1`, `T5`, `T8`, `T9`, `T10`, `T11`, `T12`, `T13`, `T15`, `T16`, `T19`, `T20` |
| **Implementadas no Código (Falta apenas "OK" de outro squad)** | **3** | `T2`, `T6`, `T7` |
| **Aguardando Retorno Externo / Execução na Jetson (Degraus B e C)** | **6** | `T3`, `T4`, `T14`, `T17`, `T18`, `T21` |

---

## 2. Tarefas Concluídas e Evidências para o Jira

| ID | Tarefa | Responsável | O que foi feito e onde está a evidência no repositório |
| :--- | :--- | :--- | :--- |
| **T1** | Fechar status da Sprint 2 e priorizar backlog | Felipe | Auditoria das 19 tarefas da S2 (12 concluídas e 7 absorvidas na S3) documentada na Seção 1 e Tabela 1 do planejamento da Sprint 3. |
| **T5** | Fechar pacote MODEL e anunciá-lo no registro | Thiago | Modelo piloto `app-epi` v1.0.0 (`pessoa`, `capacete`, `colete`) treinado (`mAP@0.5 = 0.936`), exportado para ONNX e empacotado com os 5 JSONs oficiais em `B5/artifacts/app-epi/1.0.0/`. |
| **T8** | Alinhar contrato com os consumidores (I2, I8, I9, S1) | Felipe | Payload real do contrato `ods.i1.detections` 1.0 enviado e confirmado com os grupos da lista fechada (INV-1, VER-1, VER-3 e VER-6). |
| **T9** | Definir a segunda aplicação | Thiago / Felipe | Definida a aplicação de Estacionamento/Placas junto ao squad **VER-5** (dataset balanceado de 20.000 imagens com 4 classes: placas Mercosul e antigas de carro e moto). |
| **T10** | Implementar pré e pós-processamento definitivos de I1 | Felipe | Conversão dinâmica (`BGR`, `RGB`, `GRAY`, `NV12`) em `I1/detector/ingestion.py`; filtro de confiança, NMS por classe, `detection_id >= 0` e `bbox` inteira em `I1/detector/postprocessing.py`. |
| **T11** | Integrar I1 com a API de biblioteca do P2 | Felipe | Protocolo fechado (`load`, `infer`, `info`, `release`), propagação de `engine_version` para `artifact_version` e tratamento dos 5 códigos de erro formais em `I1/detector/inference.py`. |
| **T12** | Publicar no barramento com envelope EV-SINAL | Felipe | `ValidatedStdoutPublisher` e `ConfluentKafkaPublisher` com validador B1 pré-envio, deduplicação por `message_id` (UUID v4) e cálculo de latência ponta a ponta em `I1/detector/publishing.py`. |
| **T13** | Elaborar relatório de avaliação e perda de acurácia | Thiago / Felipe | Ferramental `B5/tools/evaluate_quantization.py` executado sobre os 2 artefatos (`app-epi` e `app-plates`), gerando tabelas comparativas de acurácia (`mAP@0.5`) e desempenho (`ms`, `FPS`, `VRAM`, `MB`) em `FP32`, `FP16` e `INT8`. |
| **T15** | Entregar suíte de contrato do I1 à CI de B4 | Eduardo / Felipe | Infraestrutura Docker (`Dockerfile`, `docker-compose.yaml`), schema formal (`schemas/ods.i1.detections.v1.0.schema.json`), `I1/mock_producer.py` e suíte `I1/tests/test_i1_sprint3.py` prontos no repositório. |
| **T16** | Integração Degrau A (sem hardware) | Felipe | Pipeline completo de I1 executado fora da Jetson via `I1/run_detector.py` e serviço `i1-degrau-a`, validando 100% dos eventos emitidos e salvando telemetria em `.jsonl`. |
| **T19** | Dataset e treino da segunda aplicação | Thiago / Felipe | Treinamento do modelo `app-plates` v1.0.0 (`yolov8s`) sobre as 20k imagens concluído; `.onnx`, laudo de quantização e os 5 JSONs gerados em `B5/artifacts/app-plates/1.0.0/`. |
| **T20** | Validar "um artefato por aplicação, um componente só" | Felipe | Comprovada a execução das duas aplicações (`app-epi` e `app-plates`) pelo mesmo código/commit de `I1/run_detector.py`, alterando apenas `--onnx` e `--metadata`. |

---

## 3. Tarefas Implementadas Aguardando Homologação Rápida (Ação Imediata)

Estas 3 tarefas **já estão 100% programadas e testadas no código**, precisando apenas que a nossa equipe envie os artefatos/logs abaixo para os squads transversais darem o "de acordo" formal:

### 3.1. [T2] Conciliar e Validar Contrato `ods.i1.detections` 1.0 com B1
- **Responsável MOD-1:** Eduardo
- **Contato Externo:** João Victor de Carvalho (Squad TRA-1 / Frente B1)
- **O que fazer para fechar:** Enviar o arquivo `schemas/ods.i1.detections.v1.0.schema.json` e o log do **Teste 01** de `python3 I1/tests/test_i1_sprint3.py` para o João Victor confirmar a conformidade com o catálogo de B1.

### 3.2. [T6] Homologar Regra de Sessão e `sequence` com TRA-2
- **Responsável MOD-1:** Eduardo
- **Contato Externo:** Vítor José Batista Oliveira (Squad TRA-2 / Frentes B2 e P6)
- **Como comprovar:** No código (`I1/detector/pipeline.py`, método `handle_session_control_event`), sempre que o `session_id` muda ou chega um comando `START`, `SEEK`, `LOOP` ou `RESTART`, o componente I1 atualiza a sessão ativa, zera o contador (`sequence = 0`) e emite o log `[B2-OBS] Sessão reiniciada`.
- **O que fazer para fechar:** Rodar `python3 I1/tests/test_i1_sprint3.py`, copiar a saída do **Teste 04 (`test_04_t6_t7_sessao_seek_loop_e_salto_quadro_corrompido`)** e enviar ao Vítor José demonstrando o reset determinístico para `sequence = 0`.

### 3.3. [T7] Homologar Representação de Erro e Lacuna de `sequence` com TRA-1
- **Responsável MOD-1:** Eduardo
- **Contato Externo:** João Victor de Carvalho (Squad TRA-1 / Frente B1)
- **Como comprovar:** Em `I1/detector/pipeline.py`, o contador `sequence` é incrementado a cada quadro recebido. Se o quadro estiver corrompido ou ocorrer erro em P2:
  1. Nenhuma detecção falsa é publicada (`return None`);
  2. O erro é registrado no log de observabilidade: `[B2-OBS] Falha no quadro frame_index=1 (sequence=1 suprimido)`;
  3. O quadro seguinte sai com `sequence = 2`, gerando a lacuna implícita `0 -> 2` que sinaliza o descarte aos consumidores.
- **O que fazer para fechar:** Enviar o mesmo log do **Teste 04** de `test_i1_sprint3.py` ao João Victor (TRA-1).

---

## 4. Tarefas Dependentes de Outros Squads e Hardware (Degraus B e C)

| ID | Tarefa | Responsável | Dependência Externa / Próximo Passo |
| :--- | :--- | :--- | :--- |
| **T3** | Confirmar formato dos quadros com SIS-2 (P4) | Felipe | Alinhado com Arthur (SIS-2); entregarão a definição do consumo em 1 semana. O módulo `ingestion.py` já converte `BGR`, `RGB`, `GRAY` e `NV12` dinamicamente. |
| **T4** | Confirmar pacote e runtime com SIS-1 (P2/P3) | Felipe | E-mail enviado ao João Vitor Simão (SIS-1); aguardando confirmação do orçamento de RAM/VRAM e versão de TensorRT na Jetson Orin Nano. |
| **T14** | Publicar pacote para o runtime P2 e registrar latência na placa | Eduardo / Felipe | Enviar os pacotes `app-epi` e `app-plates` + laudos `quantization_report.md` (T13) ao SIS-1. Após compilação na Jetson, preencher `latency_ms` e `gpu_memory_mb` no `run.json`. |
| **T17** | Integração Degrau B (Jetson + P2 + Replay P6) | Felipe / Eduardo | Executar `I1` na Jetson Orin Nano usando o engine compilado por P2 e consumindo o stream de replay de P6 (TRA-2). |
| **T18** | Integração Degrau C (Ao Vivo na Jetson com P4 e Kafka) | Felipe | Executar `I1` na Jetson recebendo stream ao vivo de P4 (SIS-2) e publicando no broker Kafka (`--mode kafka`) para consumo por I2, I8, I9 e S1. |
| **T21** | Relatório final de fechamento da Sprint 3 | Felipe, Thiago e Eduardo | Consolidar as tabelas já prontas do Degrau A (`app-epi` e `app-plates`) com as medições colhidas na Jetson nos Degraus B e C. |