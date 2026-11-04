# Squad MOD-1 — Componente I1 (Detecção por Quadro)
**Disciplina:** Oficina de Desenvolvimento de Software (ODS 2026/2 — UFSJ)  
**Especificação:** Interfaces 1.0 (Arquitetura 6c Oficial)  
**Integrantes:** Felipe Girardi Siqueira (Líder), Thiago Mateus de Almeida Dias Orsini, Eduardo Monteiro Costa Pires

Os artefatos de modelo consumidos por este componente são produzidos pela Frente B5, no repositório
[`ods-26-2/b5`](https://github.com/ods-26-2/b5).

Instruções completas de execução (Docker, mock producer, Degrau A e contrato de saída) em [`HOW_TO_RUN.md`](HOW_TO_RUN.md).

---

## 1. Estrutura do Repositório

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
│   │   ├── schema_validation.py     # Validador gerado do JSON Schema oficial
│   │   └── pipeline.py              # Orquestrador, controle de sessão (B2) e telemetria
│   ├── tests/
│   │   ├── test_contract_schema.py  # Contrato: schema, exemplos, mock e publicador (sem GPU/torch)
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
├── HOW_TO_RUN.md                    # Instruções sobre como executar o componente
└── STATUS_SPRINT3.md                # Relatório de acompanhamento das 21 tarefas da Sprint 3
```
