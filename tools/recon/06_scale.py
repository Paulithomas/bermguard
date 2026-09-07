"""Recon 06 - Calibracion manual de escala metrica.

Valida SUP-07, SUP-14 y SUP-17. Marcado interactivo de horizonte y vehiculo
de referencia para estimar h_camara y su dispersion.

Usa matplotlib en lugar de cv2.imshow: en macOS la ventana de OpenCV tiene
problemas de foco y no permite zoom preciso.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

CAEX_HEIGHT_M = 7.4  # SUP-07: clase 300 t

PROMPTS = [
    "linea A del suelo: punto CERCANO",
    "linea A del suelo: punto LEJANO",
    "linea B del suelo: punto CERCANO",
    "linea B del suelo: punto LEJANO",
    "CAEX: BASE (contacto con el suelo)",
    "CAEX: TECHO (punto mas alto)",
]


@dataclass
class FrameCalibration:
    """Calibracion derivada de un frame marcado manualmente."""
    video: str
    frame: int
    y_horizon: float
    y_base: float
    y_top: float
    h_camera_m: float
    valid: bool


def collect_points(frame: np.ndarray, title: str) -> list[tuple[float, float]]:
    """Recoge 6 clics sobre el frame. Zoom con la lupa de la barra."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{title}\n1) {PROMPTS[0]}", fontsize=11)
    points: list[tuple[float, float]] = []

    def on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None:
            return
        if fig.canvas.toolbar.mode:
            return
        points.append((float(event.xdata), float(event.ydata)))
        ax.plot(event.xdata, event.ydata, "r+", markersize=14, mew=2)
        ax.annotate(str(len(points)), (event.xdata, event.ydata),
                    color="yellow", fontsize=12, xytext=(6, 6),
                    textcoords="offset points")
        if len(points) < len(PROMPTS):
            ax.set_title(f"{title}\n{len(points) + 1}) {PROMPTS[len(points)]}",
                         fontsize=11)
        else:
            ax.set_title(f"{title}\nListo. Cierra la ventana.", fontsize=11)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.tight_layout()
    plt.show()
    return points


def vanishing_y(points: list[tuple[float, float]]) -> float:
    """Coordenada y del punto de fuga de dos lineas del suelo (SUP-14)."""
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = points[:4]

    a1, b1 = y2 - y1, x1 - x2
    c1 = a1 * x1 + b1 * y1
    a2, b2 = y4 - y3, x3 - x4
    c2 = a2 * x3 + b2 * y3

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return float("nan")
    return float((a1 * c2 - a2 * c1) / det)


def camera_height(y_horizon: float, y_base: float, y_top: float) -> float:
    """Despeja h_camara de la formula de metrologia de vista unica (SUP-15).

    H_objeto = h_camara * (y_base - y_top) / (y_base - y_horizon)
    """
    denom = y_base - y_top
    if abs(denom) < 1e-6:
        return float("nan")
    return float(CAEX_HEIGHT_M * (y_base - y_horizon) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibracion manual de escala")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    results: list[FrameCalibration] = []

    for f_idx in args.frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] frame {f_idx} ilegible")
            continue

        pts = collect_points(frame, f"{args.video.name} - frame {f_idx}")
        if len(pts) < 6:
            print(f"[WARN] frame {f_idx}: {len(pts)}/6 puntos, omitido")
            continue

        y_h = vanishing_y(pts)
        y_base, y_top = pts[4][1], pts[5][1]
        h_cam = camera_height(y_h, y_base, y_top)
        valid = bool(not np.isnan(h_cam) and 3.0 < h_cam < 40.0)

        results.append(FrameCalibration(
            args.video.name, f_idx, round(y_h, 1), round(y_base, 1),
            round(y_top, 1), round(h_cam, 2), valid))
        print(f"frame {f_idx}: y_horizonte={y_h:.1f}  h_camara={h_cam:.2f} m"
              f"{'' if valid else '  <- fuera de rango'}")

    cap.release()

    valid_h = [r.h_camera_m for r in results if r.valid]
    if valid_h:
        arr = np.array(valid_h)
        mean, std = float(arr.mean()), float(arr.std())
        dispersion = std / mean * 100 if mean else float("nan")
        print(f"\nh_camara: media={mean:.2f} m  desv={std:.2f} m  "
              f"dispersion={dispersion:.1f}%  (n={len(valid_h)})")
        if dispersion < 10:
            print("SUP-17: metros como unidad primaria, confianza moderada.")
        elif dispersion < 25:
            print("SUP-17: metros con banda de incertidumbre obligatoria.")
        else:
            print("SUP-17: unidad normalizada pasa a primaria.")
    else:
        mean = std = dispersion = float("nan")

    payload = {
        "caex_height_m": CAEX_HEIGHT_M,
        "frames": [asdict(r) for r in results],
        "h_camera_mean_m": round(mean, 2) if valid_h else None,
        "h_camera_std_m": round(std, 2) if valid_h else None,
        "dispersion_pct": round(dispersion, 1) if valid_h else None,
    }
    json_path = args.output / "scale.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Guardado en: {json_path}")


if __name__ == "__main__":
    main()