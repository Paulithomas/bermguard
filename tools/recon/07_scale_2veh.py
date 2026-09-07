"""Recon 07 - Calibracion de escala por dos vehiculos a distinta profundidad.

Alternativa a 06_scale.py, que dependia de lineas paralelas del suelo. En este
material los caminos son curvos y el punto de fuga resulta inestable.

Con dos vehiculos de altura conocida apoyados en el mismo plano, la linea que
une sus bases y la que une sus techos se intersectan en el horizonte.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

CAEX_HEIGHT_M = 7.4  # SUP-07

PROMPTS = [
    "CAEX CERCANO: BASE (donde la rueda toca el suelo)",
    "CAEX CERCANO: TECHO (borde superior de la tolva)",
    "CAEX LEJANO: BASE (donde la rueda toca el suelo)",
    "CAEX LEJANO: TECHO (borde superior de la tolva)",
]


@dataclass
class FrameCalibration:
    """Calibracion derivada de dos vehiculos en un frame."""
    video: str
    frame: int
    y_horizon: float
    h_camera_m: float
    px_near: float
    px_far: float
    valid: bool
    note: str


def collect_points(frame: np.ndarray, title: str) -> list[tuple[float, float]]:
    """Recoge 4 clics sobre el frame."""
    fig, ax = plt.subplots(figsize=(15, 9))
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ax.set_title(f"{title}\n1) {PROMPTS[0]}", fontsize=11)
    points: list[tuple[float, float]] = []

    def on_click(event) -> None:
        if event.inaxes != ax or event.xdata is None:
            return
        if fig.canvas.toolbar.mode:
            return
        points.append((float(event.xdata), float(event.ydata)))
        ax.plot(event.xdata, event.ydata, "r+", markersize=16, mew=2)
        ax.annotate(str(len(points)), (event.xdata, event.ydata),
                    color="yellow", fontsize=13, xytext=(8, 8),
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


def horizon_from_two_objects(pts: list[tuple[float, float]]) -> float:
    """Interseccion de la linea de bases con la linea de techos."""
    (xb1, yb1), (xt1, yt1), (xb2, yb2), (xt2, yt2) = pts

    a1, b1 = yb2 - yb1, xb1 - xb2
    c1 = a1 * xb1 + b1 * yb1
    a2, b2 = yt2 - yt1, xt1 - xt2
    c2 = a2 * xt1 + b2 * yt1

    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-6:
        return float("nan")
    return float((a1 * c2 - a2 * c1) / det)


def camera_height(y_horizon: float, y_base: float, y_top: float) -> float:
    """h_camara = H_objeto * (y_base - y_horizon) / (y_base - y_top)."""
    denom = y_base - y_top
    if abs(denom) < 1e-6:
        return float("nan")
    return float(CAEX_HEIGHT_M * (y_base - y_horizon) / denom)


def main() -> None:
    parser = argparse.ArgumentParser(description="Escala por dos vehiculos")
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
        pts = pts[:4]
        if len(pts) < 4:
            print(f"[WARN] frame {f_idx}: {len(pts)}/4 puntos, omitido")
            continue

        px_near = pts[0][1] - pts[1][1]
        px_far = pts[2][1] - pts[3][1]

        if px_near <= 0 or px_far <= 0:
            note = "base/techo invertidos"
            results.append(FrameCalibration(
                args.video.name, f_idx, float("nan"), float("nan"),
                round(px_near, 1), round(px_far, 1), False, note))
            print(f"frame {f_idx}: {note} (near={px_near:.0f} far={px_far:.0f})")
            continue

        y_h = horizon_from_two_objects(pts)
        h_cam = camera_height(y_h, pts[0][1], pts[1][1])

        h, _ = frame.shape[:2]
        if not (0 < y_h < h):
            note = f"horizonte fuera de imagen (y={y_h:.0f}, alto={h})"
            valid = False
        elif not (3.0 < h_cam < 40.0):
            note = f"h_camara fuera de rango fisico"
            valid = False
        else:
            note = "ok"
            valid = True

        results.append(FrameCalibration(
            args.video.name, f_idx, round(y_h, 1), round(h_cam, 2),
            round(px_near, 1), round(px_far, 1), valid, note))
        print(f"frame {f_idx}: y_horizonte={y_h:.1f}  h_camara={h_cam:.2f} m  "
              f"[{note}]")

    cap.release()

    valid_h = [r.h_camera_m for r in results if r.valid]
    if valid_h:
        arr = np.array(valid_h)
        mean, std = float(arr.mean()), float(arr.std())
        dispersion = std / mean * 100 if mean else float("nan")
        print(f"\nh_camara: media={mean:.2f} m  desv={std:.2f} m  "
              f"dispersion={dispersion:.1f}%  (n={len(valid_h)})")
        if dispersion < 10:
            print("SUP-17: metros como unidad primaria.")
        elif dispersion < 25:
            print("SUP-17: metros con banda de incertidumbre.")
        else:
            print("SUP-17: unidad normalizada pasa a primaria.")
    else:
        mean = std = dispersion = float("nan")
        print("\nNingun frame valido.")

    payload = {
        "method": "two_vehicles",
        "caex_height_m": CAEX_HEIGHT_M,
        "frames": [asdict(r) for r in results],
        "h_camera_mean_m": round(mean, 2) if valid_h else None,
        "h_camera_std_m": round(std, 2) if valid_h else None,
        "dispersion_pct": round(dispersion, 1) if valid_h else None,
    }
    json_path = args.output / "scale_2veh.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Guardado en: {json_path}")


if __name__ == "__main__":
    main()