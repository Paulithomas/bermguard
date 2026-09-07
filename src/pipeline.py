"""Orquestador del pipeline (SUP-25, SUP-26).

Version de esqueleto: escribe artefactos placeholder para validar la CLI, el
Dockerfile y la estructura de salida antes de integrar los modelos.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")


def probe_readable(video_path: Path) -> str | None:
    """Verifica que el archivo sea un video legible.

    Returns:
        Resolucion como "AnchoxAlto", o None si el archivo no es legible.
    """
    cap = cv2.VideoCapture(str(video_path))
    readable = cap.isOpened() and cap.read()[0]
    resolution = None
    if readable:
        resolution = (f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                      f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    cap.release()
    return resolution


def process_video(video_path: Path, output_dir: Path, method: str) -> None:
    """Genera los cuatro artefactos requeridos para un video.

    Los artefactos se escriben siempre que el video sea legible, incluso ante
    deteccion nula (SUP-26). Un archivo ilegible levanta excepcion y el
    orquestador la registra sin abortar el lote (SUP-25).

    Raises:
        ValueError: si el archivo no es un video legible.
    """
    resolution = probe_readable(video_path)
    if resolution is None:
        raise ValueError(f"Archivo ilegible o no es un video: {video_path.name}")

    output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(video_path, output_dir / f"{video_path.stem}_osd.mp4")
    (output_dir / "berm_height.png").touch()
    (output_dir / "vehicle_distribution.png").touch()

    metadata = {
        "video": video_path.name,
        "method": method,
        "assumptions_version": "1.1",
        "status": "skeleton",
        "berm_detected": False,
        "avg_fps": None,
        "avg_ms_per_frame": None,
        "proximity_alerts": 0,
        "resolution": resolution,
        "notes": "Esqueleto: artefactos placeholder, sin inferencia.",
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_pipeline(input_dir: Path, output_dir: Path,
                 method: str, config_path: Path) -> int:
    """Procesa todos los videos de la carpeta de entrada."""
    from src.utils.device import resolve_device

    logger.info("Dispositivo: %s", resolve_device())
    logger.info("Metodo: %s", method)

    videos = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in VIDEO_EXTENSIONS)

    if not videos:
        logger.warning("No se encontraron videos en %s", input_dir)
        return 0

    methods = ["1", "2"] if method == "all" else [method]
    failures = 0

    for video_path in videos:
        for m in methods:
            target = output_dir / video_path.stem
            if method == "all":
                target = target / f"method_{m}"
            try:
                process_video(video_path, target, m)
                logger.info("Procesado %s (metodo %s)", video_path.name, m)
            except Exception as exc:
                logger.warning("Omitido %s: %s", video_path.name, exc)
                failures += 1

    logger.info("Completado: %d videos, %d omitidos", len(videos), failures)
    return 0
