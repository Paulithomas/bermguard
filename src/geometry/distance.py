"""Calculo de proximidad entre maquinaria (SUP-04, SUP-05).

La distancia se mide entre los puntos de contacto con el suelo, proyectados
al plano de rasante mediante la escala derivada de la calibracion.

Limitacion declarada (SUP-05): la medicion centro-a-centro sobreestima la
separacion real entre carrocerias. Un CAEX mide del orden de 15 m de largo,
por lo que la distancia entre superficies es menor. Se reporta como cota
superior conservadora.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

from src.detection.types import Detection, RiskLevel


@dataclass
class PairDistance:
    """Distancia medida entre dos equipos."""
    index_a: int
    index_b: int
    distance_m: float
    risk_level: RiskLevel


def pixels_per_meter(y_image: float, y_horizon: float,
                     camera_height_m: float) -> float:
    """Escala local en px/m sobre el plano de suelo, a la altura y_image.

    Deriva de la formula de vista unica (SUP-15): un objeto de 1 m apoyado en
    el suelo a la fila y_image ocupa (y_image - y_horizon) / h_camara pixeles.
    """
    if camera_height_m <= 0:
        return float("nan")
    return max(0.0, (y_image - y_horizon)) / camera_height_m


def ground_distance_m(det_a: Detection, det_b: Detection, y_horizon: float,
                      camera_height_m: float) -> float:
    """Distancia entre dos equipos sobre el plano de rasante, en metros.

    La escala se evalua en el punto medio de ambas bases para atenuar el
    efecto de la perspectiva entre vehiculos a distinta profundidad.
    """
    xa, ya = det_a.ground_point
    xb, yb = det_b.ground_point

    scale = pixels_per_meter((ya + yb) / 2.0, y_horizon, camera_height_m)
    if scale <= 1e-6 or math.isnan(scale):
        return float("nan")

    return math.hypot(xb - xa, yb - ya) / scale


def classify_risk(distance_m: float, config: dict[str, Any]) -> RiskLevel:
    """Asigna nivel de riesgo segun los umbrales configurados (SUP-04)."""
    prox = config.get("proximity", {})
    critical = prox.get("critical_m", 10.0)
    warning = prox.get("warning_m", 20.0)

    if math.isnan(distance_m):
        return RiskLevel.SAFE
    if distance_m < critical:
        return RiskLevel.CRITICAL
    if distance_m <= warning:
        return RiskLevel.WARNING
    return RiskLevel.SAFE


def compute_proximity(detections: list[Detection], y_horizon: float,
                      camera_height_m: float,
                      config: dict[str, Any]) -> list[PairDistance]:
    """Evalua todos los pares y propaga el riesgo maximo a cada deteccion.

    Cada equipo recibe el nivel de riesgo mas severo entre sus pares, que es
    el criterio conservador correcto para una alerta de seguridad.
    """
    pairs: list[PairDistance] = []
    severity = {RiskLevel.SAFE: 0, RiskLevel.WARNING: 1, RiskLevel.CRITICAL: 2}

    for det in detections:
        det.risk_level = RiskLevel.SAFE

    for i, j in itertools.combinations(range(len(detections)), 2):
        distance = ground_distance_m(
            detections[i], detections[j], y_horizon, camera_height_m)
        risk = classify_risk(distance, config)
        pairs.append(PairDistance(i, j, distance, risk))

        for k in (i, j):
            if severity[risk] > severity[detections[k].risk_level]:
                detections[k].risk_level = risk

    return pairs