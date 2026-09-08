"""Descarga y cachea todos los pesos durante el build de la imagen.

Se ejecuta con red disponible en tiempo de construccion para garantizar que
el contenedor no requiera conectividad en runtime (SUP-24).

Caso critico detectado: YOLO-World descarga el encoder de texto CLIP la
primera vez que se invoca set_classes(). Esa descarga ocurre despues de
cargar los pesos del detector, por lo que copiar unicamente el .pt no basta:
hay que forzar la inicializacion del vocabulario durante el build.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

WEIGHTS_DIR = Path("weights")


def prepare_yolo_seg(config: dict) -> None:
    """Descarga los pesos del metodo 1."""
    from ultralytics import YOLO

    target = Path(config["weights"]["yolo_seg"])
    name = target.name
    print(f"[1/2] Preparando {name}")
    YOLO(name)
    if Path(name).exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(name, target)
    print(f"      -> {target}")


def prepare_yolo_world(config: dict) -> None:
    """Descarga los pesos del metodo 2 y cachea el encoder CLIP."""
    from ultralytics import YOLOWorld

    target = Path(config["weights"]["yolo_world"])
    name = target.name
    print(f"[2/2] Preparando {name}")
    model = YOLOWorld(name)
    if Path(name).exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(name, target)

    prompts = [p for lst in config["open_vocabulary"]["prompts"].values()
               for p in lst]
    print(f"      Cacheando CLIP con {len(prompts)} prompts")
    YOLOWorld(str(target)).set_classes(prompts)
    print(f"      -> {target} + encoder de texto cacheado")


def main() -> int:
    config_path = Path("configs/default.yaml")
    if not config_path.exists():
        print(f"ERROR: no se encuentra {config_path}", file=sys.stderr)
        return 1

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    prepare_yolo_seg(config)
    prepare_yolo_world(config)

    print("\nPesos listos. Contenido de weights/:")
    for item in sorted(WEIGHTS_DIR.rglob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / 1e6
            print(f"  {item.relative_to(WEIGHTS_DIR)}  ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())