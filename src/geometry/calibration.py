"""Calibracion metrica automatica por toma (SUP-14, SUP-15, SUP-16).

Estima la linea de horizonte y la altura de camara a partir de las detecciones
de maquinaria, sin intervencion manual. Implementa la cascada de tres niveles
definida en SUP-15 y declara siempre el origen de la escala en la salida.

Fundamento. Para dos objetos de igual altura real apoyados en el mismo plano,
la recta que une sus bases y la que une sus cimas se intersectan en el
horizonte. Con el horizonte conocido, la formula de metrologia de vista unica
despeja la altura de camara:

    h_camara = H_objeto * (y_base - y_horizonte) / (y_base - y_cima)

Validado en recon 07 (marcado manual, dispersion 2.5%) y sobre bboxes de
deteccion (dos CAEX a 232 px y 47 px de altura aparente arrojan 11.62 m y
11.63 m, diferencia de 1 cm).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from src.detection.types import Detection

logger = logging.getLogger(__name__)


class ScaleSource(str, Enum):
    """Origen de la escala metrica, declarado en metadata.json (SUP-15)."""
    VEHICLE_REFERENCE = "vehicle_reference"
    ASSUMED_CAMERA_HEIGHT = "assumed_camera_height"
    NONE = "none"


@dataclass
class Calibration:
    """Parametros metricos de una toma."""
    y_horizon: float
    camera_height_m: float
    scale_source: ScaleSource
    scale_confidence: float
    n_samples: int
    dispersion_pct: float

    @property
    def is_metric(self) -> bool:
        """Indica si la salida puede expresarse en metros (SUP-16)."""
        return self.scale_source is not ScaleSource.NONE

    def pixels_per_meter(self, y_image: float) -> float:
        """Escala local en px/m sobre el plano de suelo a la fila y_image."""
        if self.camera_height_m <= 0:
            return float("nan")
        return max(0.0, y_image - self.y_horizon) / self.camera_height_m


def _horizon_from_pair(det_a: Detection, det_b: Detection) -> float:
    """Interseccion de la recta de bases con la recta de cimas.

    Devuelve NaN si los vehiculos estan a profundidad similar, caso en que las
    rectas son casi paralelas y la interseccion es numericamente inestable.
    """
    xa1, ya_top, xa2, ya_base = det_a.bbox
    xb1, yb_top, xb2, yb_base = det_b.bbox
    xa = (xa1 + xa2) / 2.0
    xb = (xb1 + xb2) / 2.0

    if abs(ya_base - yb_base) < 20.0:
        return float("nan")

    a1, b1 = yb_base - ya_base, xa - xb
    c1 = a1 * xa + b1 * ya_base
    a2, b2 = yb_top - ya_top, xa - xb
    c2 = a2 * xa + b2 * ya_top

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return float("nan")
    return float((a1 * c2 - a2 * c1) / det)


def _camera_height(det: Detection, y_horizon: float,
                   object_height_m: float) -> float:
    """Altura de camara implicada por una deteccion de altura real conocida."""
    _, y_top, _, y_base = det.bbox
    denom = y_base - y_top
    if denom <= 1e-6 or y_base <= y_horizon:
        return float("nan")
    return object_height_m * (y_base - y_horizon) / denom


def calibrate_from_detections(
        samples: list[list[Detection]], frame_height: int,
        config: dict[str, Any]) -> Calibration:
    """Estima la calibracion de una toma a partir de multiples frames.

    Args:
        samples: Detecciones por frame, tipicamente de frames muestreados.
        frame_height: Alto del frame, usado para validar el horizonte.
        config: Configuracion con alturas de referencia y limites fisicos.

    Returns:
        Calibracion con el origen de escala declarado. Nunca lanza excepcion:
        ante datos insuficientes degrada al nivel siguiente de la cascada.
    """
    ref = config.get("reference", {})
    calib_cfg = config.get("calibration", {})
    caex_h = ref.get("caex_height_m", 7.4)
    fallback_h = calib_cfg.get("fallback_camera_height_m", 11.95)
    min_h = calib_cfg.get("min_camera_height_m", 3.0)
    max_h = calib_cfg.get("max_camera_height_m", 40.0)

    horizons: list[float] = []
    for detections in samples:
        if len(detections) < 2:
            continue
        ordered = sorted(detections, key=lambda d: d.bbox[3], reverse=True)
        y_h = _horizon_from_pair(ordered[0], ordered[-1])
        if not np.isnan(y_h) and 0 < y_h < frame_height:
            horizons.append(y_h)

    # Nivel 1: dos o mas vehiculos permiten estimar horizonte y altura.
    if horizons:
        y_horizon = float(np.median(horizons))
        heights = [
            _camera_height(d, y_horizon, caex_h)
            for detections in samples for d in detections]
        heights = [h for h in heights if not np.isnan(h) and min_h < h < max_h]

        if heights:
            arr = np.array(heights)
            mean_h = float(arr.mean())
            dispersion = float(arr.std() / mean_h * 100) if mean_h else 100.0
            confidence = max(0.0, min(1.0, 1.0 - dispersion / 50.0))
            logger.info("Calibracion nivel 1: horizonte=%.1f h_camara=%.2f m "
                        "dispersion=%.1f%% (n=%d)",
                        y_horizon, mean_h, dispersion, len(heights))
            return Calibration(y_horizon, mean_h,
                               ScaleSource.VEHICLE_REFERENCE,
                               confidence, len(heights), dispersion)

    # Nivel 2: un vehiculo, se asume altura de camara y se despeja el horizonte.
    single = [d for detections in samples for d in detections]
    if single:
        largest = max(single, key=lambda d: d.height_px)
        _, y_top, _, y_base = largest.bbox
        y_horizon = y_base - (y_base - y_top) * fallback_h / caex_h
        if 0 < y_horizon < frame_height:
            logger.info("Calibracion nivel 2: horizonte=%.1f, h_camara "
                        "asumida=%.2f m", y_horizon, fallback_h)
            return Calibration(y_horizon, fallback_h,
                               ScaleSource.ASSUMED_CAMERA_HEIGHT,
                               0.4, 1, float("nan"))

    # Nivel 3: sin referencia, la salida se expresa en unidades normalizadas.
    logger.warning("Calibracion nivel 3: sin referencia metrica en la toma")
    return Calibration(frame_height / 2.0, float("nan"),
                       ScaleSource.NONE, 0.0, 0, float("nan"))