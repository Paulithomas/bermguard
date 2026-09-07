"""Recon 05 - Medicion de deriva de camara dentro de cada toma.

Valida SUP-10. Estima homografias ORB+RANSAC sobre ventanas cortas y acumula
el desplazamiento del centro de imagen.

La medicion extremo-a-extremo resulto inviable: las tomas contienen
transiciones graduales no detectadas (ver SUP-09), por lo que el primer y
ultimo frame pueden pertenecer a escenas distintas y no comparten estructura.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

MIN_INLIERS = 30


@dataclass
class DriftMeasurement:
    """Deriva medida en una toma por acumulacion de ventanas."""
    video: str
    shot_id: int
    frame_start: int
    frame_end: int
    drift_total_px: float
    drift_total_pct: float
    drift_max_step_pct: float
    n_windows: int
    n_valid: int
    is_static: bool
    note: str


def background_mask(shape: tuple[int, int]) -> np.ndarray:
    """Mascara sobre la banda media: terreno y pretil, sin cielo ni maquinaria."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[h // 4 : (3 * h) // 4, :] = 255
    return mask


def estimate_step(frame_a: np.ndarray, frame_b: np.ndarray,
                  n_features: int = 3000) -> tuple[float, int]:
    """Desplazamiento del centro entre dos frames. Devuelve (px, inliers)."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    mask = background_mask(gray_a.shape)

    orb = cv2.ORB_create(nfeatures=n_features)
    kp_a, desc_a = orb.detectAndCompute(gray_a, mask)
    kp_b, desc_b = orb.detectAndCompute(gray_b, mask)

    if desc_a is None or desc_b is None or len(desc_a) < 20 or len(desc_b) < 20:
        return float("nan"), 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(desc_a, desc_b), key=lambda m: m.distance)

    if len(matches) < 12:
        return float("nan"), len(matches)

    src = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or inliers is None:
        return float("nan"), 0

    n_inliers = int(inliers.sum())
    h, w = gray_a.shape
    center = np.float32([[[w / 2, h / 2]]])
    moved = cv2.perspectiveTransform(center, H)
    return float(np.linalg.norm(moved[0][0] - center[0][0])), n_inliers


def measure_shot(video_path: Path, f_start: int, f_end: int,
                 window: int) -> DriftMeasurement:
    """Acumula deriva sobre ventanas consecutivas dentro de una toma."""
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    indices = list(range(f_start, f_end, window))
    steps_pct: list[float] = []
    total_px = 0.0
    n_valid = 0

    for i in range(len(indices) - 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[i])
        ret_a, frame_a = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, indices[i + 1])
        ret_b, frame_b = cap.read()

        if not (ret_a and ret_b):
            continue

        step_px, n_inliers = estimate_step(frame_a, frame_b)
        if np.isnan(step_px) or n_inliers < MIN_INLIERS:
            continue

        total_px += step_px
        steps_pct.append(step_px / width * 100)
        n_valid += 1

    cap.release()

    n_windows = max(0, len(indices) - 1)
    total_pct = total_px / width * 100 if width else float("nan")
    max_step = max(steps_pct) if steps_pct else float("nan")

    coverage = n_valid / n_windows if n_windows else 0.0
    if coverage < 0.5:
        note = f"cobertura {coverage:.0%}, baja confianza"
        is_static = False
    else:
        note = f"cobertura {coverage:.0%}"
        is_static = bool(max_step < 2.0)

    return DriftMeasurement(
        video=video_path.name, shot_id=0,
        frame_start=f_start, frame_end=f_end,
        drift_total_px=round(total_px, 2),
        drift_total_pct=round(total_pct, 2),
        drift_max_step_pct=round(max_step, 2) if steps_pct else float("nan"),
        n_windows=n_windows, n_valid=n_valid,
        is_static=is_static, note=note)


def main() -> None:
    parser = argparse.ArgumentParser(description="Medicion de deriva de camara")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    with open(args.shots, encoding="utf-8") as f:
        shots = json.load(f)

    args.output.mkdir(parents=True, exist_ok=True)
    results: list[DriftMeasurement] = []

    for shot in shots:
        m = measure_shot(args.input / shot["video"],
                         shot["frame_start"], shot["frame_end"], args.window)
        m.shot_id = shot["shot_id"]
        results.append(m)

    header = (f"{'video':<16}{'toma':>6}{'total %':>9}{'max paso %':>12}"
              f"{'ventanas':>10}{'validas':>9}{'fija':>7}  nota")
    print(header)
    print("-" * (len(header) + 12))
    for r in results:
        flag = "si" if r.is_static else "NO"
        print(f"{r.video:<16}{r.shot_id:>6}{r.drift_total_pct:>9.2f}"
              f"{r.drift_max_step_pct:>12.2f}{r.n_windows:>10}"
              f"{r.n_valid:>9}{flag:>7}  {r.note}")

    json_path = args.output / "drift.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    n_static = sum(1 for r in results if r.is_static)
    print(f"\nTomas con deriva por paso < 2%: {n_static}/{len(results)}")
    print(f"Guardado en: {json_path}")


if __name__ == "__main__":
    main()
