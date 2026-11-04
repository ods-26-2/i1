# Exemplos de `ods.i1.detections` 1.0

Mensagens completas (envelope EV-SINAL + payload), uma por arquivo, para o validador de B1 (TRA-1)
e para a suíte de contrato da CI de B4 (TRA-2).

- `valid/`: todas devem passar em `../ods.i1.detections.v1.0.schema.json`.
  `01_exemplo_da_especificacao.json` é o exemplo da seção 6 da Especificação de Interfaces 1.0,
  com os campos de envelope que lá são omitidos.
- `invalid/`: cada arquivo é o exemplo da especificação com **uma única** violação, descrita no nome
  do arquivo (campos legados da Sprint 2, `target` em evento, UUIDs, timestamps, faixas numéricas, bbox).

Validação local (sem GPU):

```bash
python3 I1/tests/test_contract_schema.py
```
