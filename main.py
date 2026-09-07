"""BermGuard AI - punto de entrada CLI.

Uso:
    python main.py --input /app/test --output /app/output --method 1
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("bermguard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BermGuard AI - inspeccion de pretiles y monitoreo de maquinaria")
    parser.add_argument("--input", type=Path, required=True,
                        help="Carpeta con los videos a procesar")
    parser.add_argument("--output", type=Path, required=True,
                        help="Carpeta donde escribir los artefactos")
    parser.add_argument("--method", type=str, default="1",
                        choices=["1", "2", "all"],
                        help="Metodo de deteccion (SUP-27)")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        logger.error("La carpeta de entrada no existe: %s", args.input)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    from src.pipeline import run_pipeline
    return run_pipeline(args.input, args.output, args.method, args.config)


if __name__ == "__main__":
    sys.exit(main())