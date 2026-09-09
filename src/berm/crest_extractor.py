"""Extraccion de la cresta del pretil por textura direccional (SUP-01).

Fundamento empirico. Se evaluaron tres senales sobre el material de muestra:

1. Gradiente vertical (Sobel): descartado. Responde a sombras de maquinaria,
   huellas de neumatico y rocas del primer plano antes que a la cresta, que
   presenta bajo contraste al ser tierra sobre tierra.
2. Varianza local con umbral de Otsu: descartado. La umbralizacion global
   retiene todo lo texturado, incluidos vehiculos y ripio.
3. Varianza local + apertura morfologica con kernel horizontal alargado:
   adoptado. El pretil es una estructura extendida horizontalmente, mientras
   que los falsos positivos (vehiculos, rocas) son compactos. El kernel 81x3
   suprime lo compacto y realza lo extendido.

Limitacion abierta. La seleccion entre candidatas no es robusta: el perfil de
energia por fila presenta varios picos bien formados (talud de fondo, cresta,
ripio de primer plano) y el criterio de maxima energia no siempre selecciona
el pretil. Ver seccion de trabajo futuro en el README.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from src.detection.types import Detection

logger = logging.getLogger(__name__)


@dataclass
class BermProfile:
    """Perfil de la cresta detectada en un frame."""
    crest_y: np.ndarray          # fila de la cresta por columna, NaN si no hay
    base_y: np.ndarray           # fila de la base por columna
    coverage: float              # fraccion de columnas con deteccion valida
    detected: bool

    @property
    def height_px(self) -> np.ndarray:
        """Altura aparente en pixeles por columna."""
        return self.base_y - self.crest_y


def texture_map(gray: np.ndarray, window: int = 15) -> np.ndarray:
    """Desviacion estandar local: material suelto alto, suelo compactado bajo."""
    f = gray.astype(np.float32)
    mean = cv2.blur(f, (window, window))
    sq = cv2.blur(f * f, (window, window))
    var = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return cv2.normalize(var, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) # type: ignore[call-overload]


def horizontal_response(texture: np.ndarray, kernel_width: int) -> np.ndarray:
    """Realza estructuras horizontales continuas y suprime blobs compactos."""
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, 3))
    return cv2.morphologyEx(texture, cv2.MORPH_OPEN, kern)


def vehicle_exclusion_mask(detections: list[Detection],
                           shape: tuple[int, int]) -> np.ndarray:
    """Mascara que marca los pixeles ocupados por maquinaria.

    La cresta se interpola en los tramos ocluidos en lugar de seguir el
    contorno del vehiculo (SUP-01).
    """
    mask = np.zeros(shape, dtype=np.uint8)
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        pad = int((x2 - x1) * 0.05)
        cv2.rectangle(mask, (max(0, x1 - pad), max(0, y1 - pad)),
                      (min(shape[1], x2 + pad), min(shape[0], y2 + pad)),
                      255, -1)
    return mask


def operational_band(detections: list[Detection],
                     shape: tuple[int, int]) -> tuple[int, int]:
    """Banda vertical donde se busca el pretil.

    Anclada en las bases de la maquinaria, que definen el plano operativo
    (SUP-02). El pretil de contencion se ubica en el mismo plano o ligeramente
    por detras, nunca en el primer plano proximo a camara: el margen inferior
    es por tanto estrecho, para excluir las huellas de neumatico del suelo
    transitable, que presentan alta textura y compiten con la cresta.
    """
    h, _ = shape
    bases = [d.bbox[3] for d in detections]
    if not bases:
        return int(h * 0.40), int(h * 0.75)

    lo = max(0, int(min(bases) - h * 0.15))
    hi = min(h, int(max(bases) + h * 0.04))
    return lo, hi


def extract_crest(frame: np.ndarray, detections: list[Detection],
                  config: dict[str, Any]) -> BermProfile:
    """Extrae la cresta y la base del pretil en un frame."""
    berm_cfg = config.get("berm", {})
    min_coverage = berm_cfg.get("min_horizontal_coverage", 0.20)
    kernel_width = berm_cfg.get("kernel_width", 81)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    response = horizontal_response(texture_map(gray), kernel_width)
    response[vehicle_exclusion_mask(detections, (h, w)) > 0] = 0

    lo, hi = operational_band(detections, (h, w))
    response[:lo, :] = 0
    response[hi:, :] = 0

    crest = np.full(w, np.nan, dtype=np.float32)
    base = np.full(w, np.nan, dtype=np.float32)

    energy = response.sum(axis=1)
    if energy.max() <= 0:
        return BermProfile(crest, base, 0.0, False)

    peak_row = int(np.argmax(energy))
    search = int(h * 0.08)

    for x in range(w):
        col = response[max(0, peak_row - search):
                       min(h, peak_row + search), x]
        if col.size == 0 or col.max() < 20:
            continue
        crest[x] = max(0, peak_row - search) + int(np.argmax(col))
        below = response[int(crest[x]):hi, x]
        idx = np.where(below < 10)[0]
        base[x] = crest[x] + (idx[0] if idx.size else search)

    coverage = float(np.count_nonzero(~np.isnan(crest)) / w)
    detected = coverage >= min_coverage

    if not detected:
        logger.debug("Pretil no detectado: cobertura %.0f%% < %.0f%%",
                     coverage * 100, min_coverage * 100)


    crest = _smooth_series(crest)
    base = _smooth_series(base)

    return BermProfile(crest, base, coverage, detected)


def _smooth_series(series: np.ndarray, window: int = 91,
                   max_jump: float = 25.0) -> np.ndarray:
    """Filtro de mediana con rechazo de discontinuidades.

    Ademas del suavizado, descarta los puntos que se apartan mas de max_jump
    pixeles de la mediana local: corresponden a saltos entre candidatas
    distintas del perfil de energia, no a variacion real de la cresta.
    """
    out = series.copy()
    valid = ~np.isnan(series)
    if valid.sum() < window:
        return out

    half = window // 2
    for x in np.where(valid)[0]:
        lo, hi = max(0, x - half), min(len(series), x + half + 1)
        neighborhood = series[lo:hi]
        neighborhood = neighborhood[~np.isnan(neighborhood)]
        if not neighborhood.size:
            continue
        median = float(np.median(neighborhood))
        out[x] = np.nan if abs(series[x] - median) > max_jump else median

    return out