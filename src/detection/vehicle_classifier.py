"""Discriminacion CAEX/dozer por consistencia fisica.

Motivacion. COCO no contiene una clase para maquinaria de oruga: tanto el
CAEX como el bulldozer se detectan como "truck" (verificado sobre el material
de muestra, ambos con conf 0.58-0.66). La relacion de aspecto tampoco
discrimina: un CAEX de perfil completo alcanza ratio 1.98, indistinguible del
1.94 de un dozer.

Estrategia. Bajo la formula de metrologia de vista unica (SUP-15), cada
deteccion implica una altura de camara distinta segun se asuma una u otra
clase. Se elige la hipotesis cuya altura implicada sea mas consistente con la
mediana de la escena.

Limitacion declarada. Es una mitigacion, no una solucion. La via correcta es
fine-tune sobre datos del dominio (SUP-12), pendiente de implementacion.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from src.detection.types import Detection, VehicleClass

logger = logging.getLogger(__name__)


def implied_camera_height(detection: Detection, object_height_m: float,
                          y_horizon: float) -> float:
    """Altura de camara implicada por una deteccion bajo una hipotesis de clase.

    h_camara = H_objeto * (y_base - y_horizonte) / (y_base - y_techo)
    """
    _, y_top, _, y_base = detection.bbox
    denom = y_base - y_top
    if denom <= 1e-6 or y_base <= y_horizon:
        return float("nan")
    return object_height_m * (y_base - y_horizon) / denom


def classify_by_consistency(detections: list[Detection], y_horizon: float,
                            config: dict[str, Any]) -> list[Detection]:
    """Reasigna la clase de cada deteccion por consistencia fisica.

    Requiere al menos dos detecciones para establecer una referencia de escena.
    Con una sola deteccion no hay informacion para discriminar y se conserva
    la clase original.
    """
    if len(detections) < 2:
        return detections

    ref = config.get("reference", {})
    h_caex = ref.get("caex_height_m", 7.4)
    h_dozer = ref.get("dozer_height_m", 4.5)

    caex_hypothesis = [implied_camera_height(d, h_caex, y_horizon)
                       for d in detections]
    valid = [h for h in caex_hypothesis if not np.isnan(h)]
    if not valid:
        return detections

    reference_h = float(np.median(valid))

    for detection, h_as_caex in zip(detections, caex_hypothesis):
        if np.isnan(h_as_caex):
            continue
        h_as_dozer = h_as_caex * (h_dozer / h_caex)

        if abs(h_as_dozer - reference_h) < abs(h_as_caex - reference_h):
            detection.vehicle_class = VehicleClass.DOZER
        else:
            detection.vehicle_class = VehicleClass.CAEX

    return detections