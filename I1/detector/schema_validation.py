"""
schema_validation.py — Validador de ods.i1.detections gerado do JSON Schema oficial (B1 / T2 / T15).
O arquivo schemas/ods.i1.detections.v1.0.schema.json é a única fonte do contrato: publicador, mock e testes
validam contra ele, sem regras duplicadas em código.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schemas" / "ods.i1.detections.v1.0.schema.json"


@lru_cache(maxsize=1)
def get_validator() -> Draft202012Validator:
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def schema_errors(msg_dict: dict[str, Any]) -> list[str]:
    """Lista as violações do contrato em ordem determinística (caminho do campo + mensagem)."""
    errors = sorted(get_validator().iter_errors(msg_dict), key=lambda e: list(map(str, e.absolute_path)))
    return [f"{'/'.join(map(str, e.absolute_path)) or '<raiz>'}: {e.message}" for e in errors]
