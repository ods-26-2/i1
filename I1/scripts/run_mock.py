"""
scripts/run_mock.py — entrypoint para rodar o pipeline de I1 de ponta a ponta,
com fonte mock (vídeo local ou webcam) e engine genérico (YOLOv8n pré-treinado).

Uso:
    python -m scripts.run_mock --config configs/exemplo.yaml

Cada evento processado é impresso como JSON no stdout, no formato do schema
negociado com B1.
"""

from __future__ import annotations

import argparse

from detector.config import load_config
from detector.pipeline import build_default_detector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Caminho para o YAML de configuração da aplicação")
    args = parser.parse_args()

    config = load_config(args.config)
    detector = build_default_detector(config)

    print(f"Rodando I1 com artifact_id={config.artifact_id} sobre a fonte {config.source or 'webcam (0)'}...")
    detector.run_forever()


if __name__ == "__main__":
    main()