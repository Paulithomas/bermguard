"""Seleccion de metodo de deteccion segun el argumento CLI."""

from __future__ import annotations

from typing import Any

from src.detection.base import VehicleDetector


def build_detector(method: str, config: dict[str, Any]) -> VehicleDetector:
    """Instancia el detector correspondiente al metodo solicitado.

    Raises:
        ValueError: si el metodo no esta registrado.
    """
    if method == "1":
        from src.detection.yolo_seg import YoloSegDetector
        return YoloSegDetector(config)

    if method == "2":
        from src.detection.yolo_world import YoloWorldDetector
        return YoloWorldDetector(config)

    raise ValueError(f"Metodo no reconocido: {method}")