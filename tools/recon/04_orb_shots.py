"""Recon 04 - Deteccion de cortes por discontinuidad estructural (ORB)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def match_ratio_series(video_path: Path, n_features: int = 500) -> list[float]:
    """Serie temporal de la tasa de coincidencia ORB entre frames consecutivos."""
    cap = cv2.VideoCapture(str(video_path))
    orb = cv2.ORB_create(nfeatures=n_features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    ratios: list[float] = []
    prev_desc: np.ndarray | None = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, desc = orb.detectAndCompute(gray, None)

        if prev_desc is not None and desc is not None and len(desc) > 10:
            matches = matcher.match(prev_desc, desc)
            good = [m for m in matches if m.distance < 50]
            ratios.append(len(good) / max(len(prev_desc), len(desc)))
        else:
            ratios.append(1.0)

        prev_desc = desc

    cap.release()
    return ratios


def find_cuts(ratios: list[float], drop_factor: float = 0.45,
              min_gap: int = 15) -> list[int]:
    """Detecta caidas abruptas de la tasa de coincidencia."""
    arr = np.array(ratios)
    median = float(np.median(arr))
    candidates = np.where(arr < median * drop_factor)[0]

    cuts: list[int] = []
    for c in candidates:
        if not cuts or c - cuts[-1] >= min_gap:
            cuts.append(int(c))
    return cuts


def print_diagnostics(name: str, ratios: list[float], fps: float) -> None:
    """Estadisticos de la serie para calibrar el umbral."""
    arr = np.array(ratios)
    print(f"\n{name}")
    print(f"  mediana={np.median(arr):.3f}  min={arr.min():.3f}  "
          f"p5={np.percentile(arr, 5):.3f}  p10={np.percentile(arr, 10):.3f}")
    lowest = np.argsort(arr)[:6]
    detail = ", ".join(f"f{i}({i / fps:.1f}s)={arr[i]:.3f}"
                       for i in sorted(lowest))
    print(f"  6 valores mas bajos: {detail}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deteccion de cortes por ORB")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drop", type=float, default=0.45)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    results: dict[str, list[int]] = {}

    for video_path in sorted(args.input.glob("*.mp4")):
        cap = cv2.VideoCapture(str(video_path))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        ratios = match_ratio_series(video_path)
        print_diagnostics(video_path.name, ratios, fps)

        cuts = find_cuts(ratios, args.drop)
        results[video_path.name] = cuts

        times = ", ".join(f"f{c} ({c / fps:.1f}s)" for c in cuts)
        print(f"  cortes detectados: {len(cuts)} -> {times or 'ninguno'}")

    with open(args.output / "orb_cuts.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nGuardado en: {args.output / 'orb_cuts.json'}")


if __name__ == "__main__":
    main()
