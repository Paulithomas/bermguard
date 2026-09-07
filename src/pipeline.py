"""Orquestador del pipeline de inferencia (SUP-25, SUP-26, SUP-27)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from src.berm.crest_extractor import extract_crest
from src.detection.factory import build_detector
from src.detection.types import Detection
from src.geometry.calibration import Calibration, calibrate_from_detections
from src.geometry.distance import compute_proximity
from src.osd.renderer import render_frame

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv")
CALIBRATION_SAMPLES = 12


def load_config(path: Path) -> dict[str, Any]:
    """Carga la configuracion, con valores por defecto si el archivo falta."""
    if not path.exists():
        logger.warning("Config no encontrada en %s, se usan defaults", path)
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def sample_for_calibration(video_path: Path, detector: Any,
                           n_samples: int) -> tuple[list[list[Detection]], int]:
    """Muestrea frames equiespaciados para estimar la calibracion de la toma."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    samples: list[list[Detection]] = []
    for idx in np.linspace(0, max(0, total - 1), n_samples, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            samples.append(detector.detect(frame, int(idx)).detections)

    cap.release()
    return samples, height


def config_max_height(calibration: Calibration) -> float:
    """Cota superior de plausibilidad fisica para la altura del pretil.

    Deriva de SUP-01: una estructura mas alta que este umbral no es un pretil
    de contencion sino un talud de botadero o un banco del rajo.
    """
    return 8.0


def berm_height_m(profile: Any, calibration: Calibration) -> float | None:
    """Altura mediana del pretil en metros, o None si no es medible."""
    if not profile.detected or not calibration.is_metric:
        return None

    heights_px = profile.height_px
    rows = profile.crest_y
    valid = ~np.isnan(heights_px) & ~np.isnan(rows)
    if not valid.any():
        return None

    scales = np.array([calibration.pixels_per_meter(r) for r in rows[valid]])
    usable = scales > 1e-6
    if not usable.any():
        return None

    value = float(np.median(heights_px[valid][usable] / scales[usable]))

    max_h = config_max_height(calibration)
    return value if 0.3 < value < max_h else None


def process_video(video_path: Path, output_dir: Path, method: str,
                  config: dict[str, Any]) -> dict[str, Any]:
    """Procesa un video y escribe sus artefactos.

    Raises:
        ValueError: si el archivo no es un video legible.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not (cap.isOpened() and cap.read()[0]):
        cap.release()
        raise ValueError(f"Archivo ilegible o no es un video: {video_path.name}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_dir.mkdir(parents=True, exist_ok=True)

    detector = build_detector(method, config)
    detector.load()
    detector.set_tracking(False)
    detector.warmup()

    samples, _ = sample_for_calibration(video_path, detector,
                                        CALIBRATION_SAMPLES)
    calibration = calibrate_from_detections(samples, height, config)
    detector.set_tracking(True)
    detector.reset_tracker()

    writer = cv2.VideoWriter(
        str(output_dir / f"{video_path.stem}_osd.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    heights: list[float | None] = []
    positions: list[dict[str, Any]] = []
    frame_times: list[float] = []
    alerts_total = 0
    frame_index = 0



    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        result = detector.detect(frame, frame_index)
        pairs = compute_proximity(result.detections, calibration.y_horizon,
                                  calibration.camera_height_m, config)
        profile = extract_crest(frame, result.detections, config)
        h_m = berm_height_m(profile, calibration)
        frame_times.append((time.perf_counter() - start) * 1000.0)

        n_alerts = sum(1 for p in pairs if p.risk_level.value != "safe")
        alerts_total += n_alerts
        heights.append(h_m)

        for det in result.detections:
            gx, gy = det.ground_point
            positions.append({
                "frame": frame_index, "track_id": det.track_id,
                "class": det.vehicle_class.value, "x": gx, "y": gy,
                "risk": det.risk_level.value,
            })

        writer.write(render_frame(
            frame, result.detections, profile, frame_index, fps, h_m,
            n_alerts, calibration.scale_source.value, config))
        frame_index += 1

    cap.release()
    writer.release()

    from src.analytics.plots import plot_berm_height, plot_vehicle_map
    plot_berm_height(heights, fps, output_dir / "berm_height.png", calibration)
    plot_vehicle_map(positions, (width, height),
                     output_dir / "vehicle_distribution.png")

    valid_h = [h for h in heights if h is not None]
    metadata = {
        "video": video_path.name,
        "method": method,
        "detector": detector.name,
        "assumptions_version": "1.1",
        "resolution": f"{width}x{height}",
        "source_fps": round(fps, 2),
        "frames_processed": frame_index,
        "avg_fps": round(1000.0 / float(np.mean(frame_times)), 2) if frame_times else None,
        "avg_ms_per_frame": round(float(np.mean(frame_times)), 2) if frame_times else None,
        "p95_ms_per_frame": round(float(np.percentile(frame_times, 95)), 2) if frame_times else None,
        "proximity_alerts": alerts_total,
        "berm_detected": bool(valid_h),
        "berm_height_median_m": round(float(np.median(valid_h)), 2) if valid_h else None,
        "berm_height_p10_m": round(float(np.percentile(valid_h, 10)), 2) if valid_h else None,
        "berm_height_p90_m": round(float(np.percentile(valid_h, 90)), 2) if valid_h else None,
        "berm_coverage_frames": round(len(valid_h) / max(1, frame_index), 3),
        "calibration": {
            "scale_source": calibration.scale_source.value,
            "scale_confidence": round(calibration.scale_confidence, 2),
            "y_horizon": round(calibration.y_horizon, 1),
            "camera_height_m": (round(calibration.camera_height_m, 2)
                                if calibration.is_metric else None),
            "dispersion_pct": (round(calibration.dispersion_pct, 1)
                               if not np.isnan(calibration.dispersion_pct) else None),
        },
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def run_pipeline(input_dir: Path, output_dir: Path,
                 method: str, config_path: Path) -> int:
    """Procesa todos los videos de la carpeta de entrada."""
    from src.utils.device import resolve_device

    config = load_config(config_path)
    logger.info("Dispositivo: %s | Metodo: %s", resolve_device(), method)

    videos = sorted(p for p in input_dir.iterdir()
                    if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        logger.warning("No se encontraron videos en %s", input_dir)
        return 0

    methods = ["1", "2"] if method == "all" else [method]
    failures = 0

    for video_path in videos:
        for m in methods:
            target = output_dir / video_path.stem
            if method == "all":
                target = target / f"method_{m}"
            try:
                meta = process_video(video_path, target, m, config)
                logger.info("%s: %d frames, %s FPS, %d alertas, pretil=%s",
                            video_path.name, meta["frames_processed"],
                            meta["avg_fps"], meta["proximity_alerts"],
                            meta["berm_height_median_m"])
            except NotImplementedError as exc:
                logger.warning("Metodo %s no disponible: %s", m, exc)
            except Exception as exc:
                logger.warning("Omitido %s: %s", video_path.name, exc)
                failures += 1

    logger.info("Completado: %d videos, %d omitidos", len(videos), failures)
    return 0