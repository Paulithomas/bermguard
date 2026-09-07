"""Tipos de dominio compartidos por los metodos de deteccion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class VehicleClass(str, Enum):
    """Clases de maquinaria de interes (SUP-06)."""
    CAEX = "caex"
    DOZER = "dozer"
    OTHER_HEAVY = "other_heavy"


class RiskLevel(str, Enum):
    """Nivel de riesgo por proximidad (SUP-04)."""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Detection:
    """Una deteccion de maquinaria en un frame.

    Attributes:
        bbox: (x1, y1, x2, y2) en pixeles de imagen.
        mask: Mascara binaria opcional, del tamano del frame.
        track_id: Identificador persistente asignado por el tracker.
    """
    vehicle_class: VehicleClass
    confidence: float
    bbox: tuple[float, float, float, float]
    mask: np.ndarray | None = None
    track_id: int | None = None
    risk_level: RiskLevel = RiskLevel.SAFE

    @property
    def ground_point(self) -> tuple[float, float]:
        """Punto de contacto con el suelo: centro inferior del bbox (SUP-05)."""
        x1, _, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    @property
    def height_px(self) -> float:
        """Altura aparente en pixeles, usada como referencia metrica (SUP-15)."""
        _, y1, _, y2 = self.bbox
        return y2 - y1


@dataclass
class FrameResult:
    """Resultado de inferencia sobre un frame."""
    frame_index: int
    detections: list[Detection] = field(default_factory=list)
    inference_ms: float = 0.0