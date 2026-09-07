"""Renderizado del On-Screen Display sobre los frames procesados.

Los parametros geometricos se expresan relativos al ancho de imagen (SUP-28):
el material presenta resoluciones heterogeneas (1920x1080 y 1280x720) y un
grosor fijo en pixeles se veria fino en una y grueso en la otra.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from src.berm.crest_extractor import BermProfile
from src.detection.types import Detection, RiskLevel

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _hex_to_bgr(value: str) -> tuple[int, int, int]:
    """Convierte '#RRGGBB' a la tupla BGR que espera OpenCV."""
    v = value.lstrip("#")
    r, g, b = (int(v[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def risk_colors(config: dict[str, Any]) -> dict[RiskLevel, tuple[int, int, int]]:
    """Paleta del semaforo de proximidad (SUP-04)."""
    prox = config.get("proximity", {})
    return {
        RiskLevel.SAFE: _hex_to_bgr(prox.get("color_safe", "#10B981")),
        RiskLevel.WARNING: _hex_to_bgr(prox.get("color_warning", "#F59E0B")),
        RiskLevel.CRITICAL: _hex_to_bgr(prox.get("color_critical", "#EF4444")),
    }


def draw_detections(frame: np.ndarray, detections: list[Detection],
                    config: dict[str, Any]) -> np.ndarray:
    """Dibuja bboxes y etiquetas con el color del nivel de riesgo."""
    colors = risk_colors(config)
    h, w = frame.shape[:2]
    thickness = max(2, int(w * 0.0025))
    scale = w / 1280.0

    for det in detections:
        color = colors[det.risk_level]
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        tag = det.vehicle_class.value.upper()
        if det.track_id is not None:
            tag += f" #{det.track_id}"

        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.6 * scale, thickness)
        ty = max(th + 4, y1 - 6)
        cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 6, ty + 2),
                      color, -1)
        cv2.putText(frame, tag, (x1 + 3, ty - 2), FONT, 0.6 * scale,
                    (255, 255, 255), thickness)

    return frame


def draw_berm(frame: np.ndarray, profile: BermProfile,
              config: dict[str, Any]) -> np.ndarray:
    """Delinea cresta y base del pretil, y rellena el cuerpo entre ambas."""
    if not profile.detected:
        return frame

    h, w = frame.shape[:2]
    thickness = max(2, int(w * 0.002))

    body = np.zeros((h, w), dtype=np.uint8)
    for x in range(w):
        cy, by = profile.crest_y[x], profile.base_y[x]
        if np.isnan(cy) or np.isnan(by):
            continue
        cv2.line(body, (x, int(cy)), (x, int(by)), 255, 1)

    overlay = frame.copy()
    overlay[body > 0] = (60, 90, 200)
    frame = cv2.addWeighted(frame, 0.75, overlay, 0.25, 0)

    for series, color in ((profile.crest_y, (0, 0, 255)),
                          (profile.base_y, (0, 200, 255))):
        pts = [(x, int(series[x])) for x in range(w)
               if not np.isnan(series[x])]
        for i in range(1, len(pts)):
            if pts[i][0] - pts[i - 1][0] <= 3:
                cv2.line(frame, pts[i - 1], pts[i], color, thickness)

    return frame


def draw_hud(frame: np.ndarray, frame_index: int, fps: float,
             berm_height_m: float | None, n_alerts: int,
             scale_source: str, config: dict[str, Any]) -> np.ndarray:
    """Panel de informacion en la esquina superior izquierda."""
    h, w = frame.shape[:2]
    scale = w / 1280.0
    pad = int(w * 0.012)
    line_h = int(26 * scale)

    height_txt = (f"Pretil: {berm_height_m:.2f} m"
                  if berm_height_m is not None else "Pretil: no detectado")
    lines = [
        f"Frame {frame_index}  |  {fps:.1f} FPS",
        height_txt,
        f"Alertas proximidad: {n_alerts}",
        f"Escala: {scale_source}",
    ]

    box_w = int(w * 0.28)
    box_h = line_h * len(lines) + pad
    panel = frame.copy()
    cv2.rectangle(panel, (pad, pad), (pad + box_w, pad + box_h), (0, 0, 0), -1)
    frame = cv2.addWeighted(frame, 0.6, panel, 0.4, 0)

    for i, text in enumerate(lines):
        y = pad + line_h * (i + 1) - int(6 * scale)
        cv2.putText(frame, text, (pad + int(10 * scale), y), FONT,
                    0.55 * scale, (255, 255, 255), max(1, int(1.5 * scale)))

    return frame


def render_frame(frame: np.ndarray, detections: list[Detection],
                 profile: BermProfile, frame_index: int, fps: float,
                 berm_height_m: float | None, n_alerts: int,
                 scale_source: str, config: dict[str, Any]) -> np.ndarray:
    """Compone el OSD completo sobre un frame."""
    out = frame.copy()
    out = draw_berm(out, profile, config)
    out = draw_detections(out, detections, config)
    out = draw_hud(out, frame_index, fps, berm_height_m, n_alerts,
                   scale_source, config)
    return out