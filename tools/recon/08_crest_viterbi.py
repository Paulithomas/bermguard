#!/usr/bin/env python3
"""Benchmark de consistencia temporal en la seleccion de la fila de cresta.

Ataca la limitacion declarada en SUP-01: `extract_crest` elige la fila del
pretil con `peak_row = argmax(energy)`, una decision independiente por frame.
Cuando el perfil de energia tiene varios picos bien formados (talud de fondo,
cresta, ripio de primer plano), el maximo salta entre estructuras.

La hipotesis contrastada es la registrada en la bitacora: el pretil es estatico
en coordenadas del mundo, de modo que su fila en la imagen solo puede moverse
por la deriva de camara medida (1-2 px/frame, SUP-10). Una seleccion que se
desplace mas que eso entre frames contiguos es un cambio de estructura.

Segmentacion por toma (SUP-09). Tanto la compensacion de deriva como el prior
de transicion solo son validos dentro de una escena. Cruzando un corte, la
cresta puede estar legitimamente en cualquier fila, y penalizar ese salto
introduce dos errores: contamina la deriva acumulada con homografias entre
escenas distintas (el fenomeno que la bitacora documenta al descartar la
primera medicion de deriva, seccion 6) y castiga una discontinuidad real.
Cada toma se procesa como secuencia independiente.

Los cortes se toman de la union de `shots.json` y `orb_cuts.json`, que es la
decision declarada en la bitacora. Esa union recupera 12 de 16 cortes reales:
las 4 transiciones graduales de 3 a 6 frames no son detectables por ningun
metodo basado en discontinuidad entre frames contiguos, y siguen contaminando
la toma que las contiene. El benchmark no las resuelve y no pretende hacerlo.

Metricas. `mad_stabilized_px` usa la desviacion absoluta mediana escalada por
1.4826, no la desviacion estandar: sobre una trayectoria con transiciones
residuales, sigma mide la separacion entre modos en vez de la estabilidad de
la seleccion. `mean_energy` acompana siempre a las metricas de estabilidad,
porque una trayectoria constante las optimiza de forma trivial sin relacion
alguna con el pretil.

Uso:
    python tools/recon/08_crest_viterbi.py --all --max-frames 120
    python tools/recon/08_crest_viterbi.py --video data/videos/video_02.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    from src.berm.crest_extractor import horizontal_response, texture_map
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        f"No se pudo importar el extractor de textura desde src/: {exc}\n"
        "Ejecutar desde la raiz del proyecto."
    ) from exc

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("crest_viterbi")

KERNEL_WIDTH = 81
BAND_LO_FRAC = 0.40
BAND_HI_FRAC = 0.75
MIN_INLIERS = 30
ORB_FEATURES = 2000
MAX_PLAUSIBLE_DRIFT_PX = 10.0    # SUP-10: 1-2 px/frame medidos
MIN_SHOT_FRAMES = 8
DEFAULT_LAMBDAS = (0.5, 2.0, 8.0)
MAD_SCALE = 1.4826               # consistencia con sigma bajo normalidad

RECON = ROOT / "data" / "recon"


def load_cut_frames(video_name: str) -> list[int]:
    """Union de los cortes de `shots.json` y `orb_cuts.json` para un video."""
    cuts: set[int] = set()

    shots_path = RECON / "shots.json"
    if shots_path.exists():
        with open(shots_path, encoding="utf-8") as f:
            for shot in json.load(f):
                if shot.get("video") == video_name:
                    start = int(shot.get("frame_start", 0))
                    if start > 0:
                        cuts.add(start)

    orb_path = RECON / "orb_cuts.json"
    if orb_path.exists():
        with open(orb_path, encoding="utf-8") as f:
            for frame in json.load(f).get(video_name, []):
                cuts.add(int(frame))

    return sorted(cuts)


def shot_ranges(n_frames: int, cuts: list[int]) -> list[tuple[int, int]]:
    """Convierte una lista de cortes en rangos [inicio, fin) de toma."""
    bounds = [0] + [c for c in cuts if 0 < c < n_frames] + [n_frames]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)
            if bounds[i + 1] - bounds[i] >= MIN_SHOT_FRAMES]


def row_energy_profiles(video_path: Path, max_frames: int | None
                        ) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int]]:
    """Perfiles de energia por fila y frames en gris, banda operativa aplicada.

    Replica la cadena de `extract_crest` hasta `energy = response.sum(axis=1)`,
    omitiendo la exclusion por deteccion de vehiculos: el benchmark corre sin
    detector para ser reproducible sin pesos, y la omision afecta por igual a
    los dos metodos comparados.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir {video_path}")

    profiles: list[np.ndarray] = []
    grays: list[np.ndarray] = []
    band: tuple[int, int] | None = None

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames is not None and len(profiles) >= max_frames):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if band is None:
            h = gray.shape[0]
            band = (int(h * BAND_LO_FRAC), int(h * BAND_HI_FRAC))
        lo, hi = band
        response = horizontal_response(texture_map(gray), KERNEL_WIDTH)
        profiles.append(response[lo:hi, :].sum(axis=1).astype(np.float64))
        grays.append(gray)

    cap.release()
    if not profiles:
        raise SystemExit(f"Sin frames legibles en {video_path}")
    return np.array(profiles), grays, band  # type: ignore[return-value]


def estimate_vertical_drift(grays: list[np.ndarray], band: tuple[int, int]
                            ) -> tuple[np.ndarray, int]:
    """Desplazamiento vertical con signo entre frames contiguos, en px.

    Homografia ORB+RANSAC restringida a la banda del terreno hacia abajo. La
    restriccion corrige el error registrado en la bitacora (seccion 7):
    enmascarar la parte superior deja solo cielo, que no rinde keypoints.

    Se descartan los pasos con menos de MIN_INLIERS correspondencias y los que
    devuelven un desplazamiento mayor que MAX_PLAUSIBLE_DRIFT_PX. Este segundo
    filtro aplica el mismo criterio que la bitacora uso para descartar la
    primera medicion de deriva: con pocas correspondencias validas RANSAC
    encuentra alguna transformacion que encaja sin que ello signifique nada.
    Un paso descartado se trata como deriva nula, no como dato inventado.
    """
    lo, _ = band
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    dy = np.zeros(len(grays), dtype=np.float64)
    rejected = 0

    for t in range(1, len(grays)):
        prev, curr = grays[t - 1][lo:, :], grays[t][lo:, :]
        kp_a, des_a = orb.detectAndCompute(prev, None)
        kp_b, des_b = orb.detectAndCompute(curr, None)
        if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
            rejected += 1
            continue

        matches = matcher.match(des_a, des_b)
        if len(matches) < MIN_INLIERS:
            rejected += 1
            continue

        src = np.float32([kp_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        matrix, mask = cv2.estimateAffinePartial2D(
            src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if matrix is None or mask is None or int(mask.sum()) < MIN_INLIERS:
            rejected += 1
            continue

        step = float(matrix[1, 2])
        if abs(step) > MAX_PLAUSIBLE_DRIFT_PX:
            rejected += 1
            continue
        dy[t] = step

    return dy, rejected


def normalized(energy: np.ndarray) -> np.ndarray:
    span = float(np.ptp(energy))
    if span <= 0:
        return np.zeros_like(energy)
    return (energy - float(energy.min())) / span


def solve_viterbi(profiles: np.ndarray, dy: np.ndarray,
                  lambda_smooth: float, tau_px: float) -> np.ndarray:
    """Camino optimo de filas de cresta sobre una toma.

    Emision: -energia normalizada a [0,1] por frame.
    Transicion: lambda * min((|i - (j + dy_t)| / tau)^2, 1), truncada para
    quedar acotada en [0, lambda] y ser comparable con la emision. Sin
    truncamiento un solo salto grande domina el camino completo y el termino
    de energia se vuelve irrelevante.

    El residuo se toma respecto de la deriva esperada, no respecto de la fila
    anterior: penalizar |y_t - y_{t-1}| castigaria al pretil real, que si se
    desplaza en la imagen, y favoreceria estructuras estaticas en pantalla.
    """
    n_frames, n_rows = profiles.shape
    if n_frames == 1:
        return np.array([int(np.argmax(profiles[0]))], dtype=np.int32)

    rows = np.arange(n_rows, dtype=np.float64)
    accum = -normalized(profiles[0])
    back = np.zeros((n_frames, n_rows), dtype=np.int32)

    for t in range(1, n_frames):
        residual = np.abs(rows[:, None] - (rows[None, :] + dy[t]))
        trans = lambda_smooth * np.minimum((residual / tau_px) ** 2, 1.0)
        total = accum[None, :] + trans
        back[t] = np.argmin(total, axis=1)
        accum = -normalized(profiles[t]) + np.min(total, axis=1)

    path = np.zeros(n_frames, dtype=np.int32)
    path[-1] = int(np.argmin(accum))
    for t in range(n_frames - 2, -1, -1):
        path[t] = back[t + 1, path[t + 1]]
    return path


def mad(values: np.ndarray) -> float:
    """Desviacion absoluta mediana escalada, robusta a transiciones residuales."""
    if values.size == 0:
        return float("nan")
    median = float(np.median(values))
    return float(MAD_SCALE * np.median(np.abs(values - median)))


def evaluate(path: np.ndarray, profiles: np.ndarray, dy: np.ndarray,
             tau_px: float) -> dict:
    """Metricas de estabilidad y de fidelidad de una trayectoria de una toma."""
    stabilized = path.astype(np.float64) - np.cumsum(dy)
    residual = np.abs(np.diff(path.astype(np.float64)) - dy[1:])
    jump_rate = float(np.mean(residual > tau_px) * 100.0) if residual.size else 0.0

    energies = [float(normalized(profiles[t])[row]) for t, row in enumerate(path)]

    return {
        "mad_stabilized_px": round(mad(stabilized), 2),
        "jump_rate_pct": round(jump_rate, 2),
        "mean_energy": round(float(np.mean(energies)), 4),
        "median_row": round(float(np.median(path)), 1),
        "n_frames": int(len(path)),
    }


def aggregate(per_shot: list[dict]) -> dict:
    """Agregado ponderado por numero de frames sobre las tomas de un video."""
    total = sum(s["n_frames"] for s in per_shot)
    if not total:
        return {"mad_stabilized_px": float("nan"), "jump_rate_pct": float("nan"),
                "mean_energy": float("nan"), "n_frames": 0, "n_shots": 0}

    def weighted(key: str) -> float:
        return sum(s[key] * s["n_frames"] for s in per_shot) / total

    return {
        "mad_stabilized_px": round(weighted("mad_stabilized_px"), 2),
        "jump_rate_pct": round(weighted("jump_rate_pct"), 2),
        "mean_energy": round(weighted("mean_energy"), 4),
        "n_frames": total,
        "n_shots": len(per_shot),
    }


def plot_comparison(paths: dict[str, np.ndarray], band: tuple[int, int],
                    cuts: list[int], n_frames: int, output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lo, _ = band
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=150)
    palette = {"argmax": ("#94A3B8", 1.0),
               "viterbi_l0.5": ("#FCA5A5", 1.4),
               "viterbi_l2": ("#EF4444", 1.8),
               "viterbi_l8": ("#7F1D1D", 1.8)}

    for cut in cuts:
        if 0 < cut < n_frames:
            ax.axvline(cut, color="#0EA5E9", ls="--", lw=1.0, alpha=0.7)

    for name, path in paths.items():
        color, lw = palette.get(name, ("#DC2626", 1.5))
        ax.plot(np.arange(len(path)), path + lo, color=color, lw=lw,
                label=name, alpha=0.9)

    ax.set_title("Seleccion de fila de cresta por toma (lineas azules: cortes)",
                 fontsize=12)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Fila de cresta (px, coordenadas de imagen)")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def run_video(video_path: Path, max_frames: int | None, tau_px: float,
              lambdas: tuple[float, ...], out_dir: Path) -> dict:
    logger.info("\n=== %s ===", video_path.name)
    profiles, grays, band = row_energy_profiles(video_path, max_frames)
    n_frames = len(profiles)

    cuts = load_cut_frames(video_path.name)
    ranges = shot_ranges(n_frames, cuts)
    logger.info("  %d frames, banda %d-%d px, %d toma(s), cortes en %s",
                n_frames, band[0], band[1], len(ranges),
                [c for c in cuts if c < n_frames] or "ninguno")

    names = ["argmax"] + [f"viterbi_l{lam:g}" for lam in lambdas]
    full_paths = {name: np.zeros(n_frames, dtype=np.int32) for name in names}
    per_shot: dict[str, list[dict]] = {name: [] for name in names}
    rejected_total = 0

    for start, end in ranges:
        sub_profiles = profiles[start:end]
        dy, rejected = estimate_vertical_drift(grays[start:end], band)
        rejected_total += rejected

        shot_paths = {
            "argmax": np.array([int(np.argmax(p)) for p in sub_profiles],
                               dtype=np.int32)
        }
        for lam in lambdas:
            shot_paths[f"viterbi_l{lam:g}"] = solve_viterbi(
                sub_profiles, dy, lam, tau_px)

        for name, path in shot_paths.items():
            full_paths[name][start:end] = path
            metrics = evaluate(path, sub_profiles, dy, tau_px)
            metrics["shot"] = f"{start}-{end}"
            per_shot[name].append(metrics)

    if rejected_total:
        logger.info("  deriva: %d pasos descartados por baja confianza",
                    rejected_total)

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(full_paths, band, cuts, n_frames,
                    out_dir / f"{video_path.stem}_crest_paths.png")

    logger.info("  %-16s %8s %10s %10s", "metodo", "MAD", "saltos%", "energia")
    summary = {}
    for name in names:
        agg = aggregate(per_shot[name])
        summary[name] = {"aggregate": agg, "per_shot": per_shot[name]}
        logger.info("  %-16s %8.2f %9.2f%% %10.4f", name,
                    agg["mad_stabilized_px"], agg["jump_rate_pct"],
                    agg["mean_energy"])

    return {
        "video": video_path.name,
        "frames": n_frames,
        "band_px": list(band),
        "cuts_used": [c for c in cuts if c < n_frames],
        "shots": [f"{a}-{b}" for a, b in ranges],
        "drift_steps_rejected": rejected_total,
        "tau_px": tau_px,
        "methods": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--tau-px", type=float, default=4.0)
    parser.add_argument("--out", type=Path, default=RECON / "crest_viterbi")
    args = parser.parse_args()

    if args.all:
        videos = sorted((ROOT / "data" / "videos").glob("*.mp4"))
    elif args.video:
        videos = [args.video]
    else:
        raise SystemExit("Indicar --video RUTA o --all")

    results = [run_video(v, args.max_frames, args.tau_px, DEFAULT_LAMBDAS,
                         args.out) for v in videos]

    report = args.out / "crest_viterbi.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    with open(report, "w", encoding="utf-8") as f:
        json.dump({"tau_px": args.tau_px, "lambdas": list(DEFAULT_LAMBDAS),
                   "max_plausible_drift_px": MAX_PLAUSIBLE_DRIFT_PX,
                   "results": results}, f, indent=2, ensure_ascii=False)
    logger.info("\nReporte escrito en %s", report)


if __name__ == "__main__":
    main()