"""Recon 01 — Inventario técnico de los videos de muestra.

Valida SUP-08 (duración y presupuesto computacional) del registro de supuestos.
Contrasta el conteo de frames reportado por el contenedor contra un conteo real.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2


VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")


@dataclass
class VideoProbe:
    """Metadata técnica de un archivo de video."""
    filename: str
    fps: float
    width: int
    height: int
    frames_reported: int      # lo que dice el contenedor
    frames_actual: int        # lo que hay de verdad
    duration_s: float
    codec: str
    counts_match: bool


def decode_fourcc(fourcc_float: float) -> str:
    """Convierte el FOURCC de OpenCV (float) a su representación textual."""
    code = int(fourcc_float)
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def count_frames_actual(cap: cv2.VideoCapture) -> int:
    """Cuenta frames leyendo el video completo. Fuente de verdad."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return count


def probe_video(path: Path) -> VideoProbe:
    """Extrae la metadata técnica de un video.

    Raises:
        IOError: si el archivo no puede abrirse.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec = decode_fourcc(cap.get(cv2.CAP_PROP_FOURCC))

    frames_actual = count_frames_actual(cap)
    cap.release()

    duration_s = frames_actual / fps if fps > 0 else 0.0

    return VideoProbe(
        filename=path.name,
        fps=fps,
        width=width,
        height=height,
        frames_reported=frames_reported,
        frames_actual=frames_actual,
        duration_s=duration_s,
        codec=codec,
        counts_match=(frames_reported == frames_actual),
    )


def probe_directory(input_dir: Path) -> list[VideoProbe]:
    """Procesa todos los videos de un directorio.

    Los archivos ilegibles se registran y se omiten sin abortar el lote (SUP-25).
    """
    videos = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in VIDEO_EXTENSIONS
    )

    probes: list[VideoProbe] = []
    for video_path in videos:
        try:
            probes.append(probe_video(video_path))
        except Exception as exc:
            print(f"[WARN] Omitido {video_path.name}: {exc}")

    return probes


def print_table(probes: list[VideoProbe]) -> None:
    """Imprime los resultados en formato tabular legible."""
    header = (
        f"{'archivo':<16}{'fps':>7}{'resolución':>13}"
        f"{'reportado':>11}{'real':>8}{'dur (s)':>9}{'codec':>8}{'ok':>5}"
    )
    print(header)
    print("-" * len(header))

    for p in probes:
        resolution = f"{p.width}x{p.height}"
        flag = "sí" if p.counts_match else "NO"
        print(
            f"{p.filename:<16}{p.fps:>7.2f}{resolution:>13}"
            f"{p.frames_reported:>11}{p.frames_actual:>8}"
            f"{p.duration_s:>9.2f}{p.codec:>8}{flag:>5}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventario técnico de videos")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"No existe la carpeta de entrada: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    probes = probe_directory(args.input)
    print_table(probes)

    json_path = args.output / "probe.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in probes], f, indent=2)

    print(f"\nGuardado en: {json_path}")


if __name__ == "__main__":
    main()