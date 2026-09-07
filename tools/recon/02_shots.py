"""Recon 02 — Segmentación de tomas y clasificación lumínica.

Valida SUP-09 (un archivo puede contener múltiples tomas).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from scenedetect import detect, AdaptiveDetector


VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")


@dataclass
class Shot:
    """Una toma continua dentro de un archivo de video."""
    video: str
    shot_id: int
    frame_start: int
    frame_end: int
    duration_s: float
    mean_brightness: float
    lighting: str


def classify_lighting(brightness: float) -> str:
    """Clasificación provisional por brillo medio. Ver limitación en SUP-22."""
    if brightness < 60:
        return "noche"
    if brightness < 110:
        return "crepusculo"
    return "dia"


def mean_brightness(video_path: Path, frame_start: int, frame_end: int,
                    samples: int = 5) -> float:
    """Luminancia media de una toma, muestreando frames equiespaciados."""
    cap = cv2.VideoCapture(str(video_path))
    indices = np.linspace(frame_start, max(frame_start, frame_end - 1),
                          samples, dtype=int)
    values: list[float] = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            values.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))

    cap.release()
    return float(np.mean(values)) if values else 0.0


def save_boundary_frames(video_path: Path, shots: list[Shot],
                         output_dir: Path) -> None:
    """Guarda los frames adyacentes a cada corte para verificación visual."""
    if len(shots) < 2:
        return

    cap = cv2.VideoCapture(str(video_path))
    stem = video_path.stem

    for i in range(len(shots) - 1):
        for label, idx in (("antes", shots[i].frame_end - 1),
                           ("despues", shots[i + 1].frame_start)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                name = f"{stem}_corte{i + 1}_{label}_f{int(idx)}.png"
                cv2.imwrite(str(output_dir / name), frame)

    cap.release()


def detect_shots(video_path: Path, threshold: float) -> list[Shot]:
    """Segmenta un video en tomas usando detección adaptativa."""
    scenes = detect(str(video_path), AdaptiveDetector(adaptive_threshold=threshold))

    if not scenes:
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        brightness = mean_brightness(video_path, 0, total)
        return [Shot(video_path.name, 1, 0, total, total / fps,
                     round(brightness, 1), classify_lighting(brightness))]

    shots: list[Shot] = []
    for i, (start, end) in enumerate(scenes, start=1):
        f_start, f_end = start.frame_num, end.frame_num
        brightness = mean_brightness(video_path, f_start, f_end)
        shots.append(Shot(
            video=video_path.name,
            shot_id=i,
            frame_start=f_start,
            frame_end=f_end,
            duration_s=(end.seconds - start.seconds),
            mean_brightness=round(brightness, 1),
            lighting=classify_lighting(brightness),
        ))

    return shots


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentación de tomas")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1.5)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No existe la carpeta de entrada: {args.input}")

    frames_dir = args.output / "shot_boundaries"
    frames_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(p for p in args.input.iterdir()
                    if p.suffix.lower() in VIDEO_EXTENSIONS)

    all_shots: list[Shot] = []
    for video_path in videos:
        try:
            shots = detect_shots(video_path, args.threshold)
            save_boundary_frames(video_path, shots, frames_dir)
            all_shots.extend(shots)
        except Exception as exc:
            print(f"[WARN] Omitido {video_path.name}: {exc}")

    header = (f"{'video':<16}{'toma':>6}{'inicio':>8}{'fin':>7}"
              f"{'dur (s)':>9}{'brillo':>8}{'luz':>13}")
    print(header)
    print("-" * len(header))
    for s in all_shots:
        print(f"{s.video:<16}{s.shot_id:>6}{s.frame_start:>8}{s.frame_end:>7}"
              f"{s.duration_s:>9.2f}{s.mean_brightness:>8.1f}{s.lighting:>13}")

    json_path = args.output / "shots.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in all_shots], f, indent=2)

    print(f"\nTomas detectadas: {len(all_shots)} en {len(videos)} videos")
    print(f"Guardado en: {json_path}")


if __name__ == "__main__":
    main()