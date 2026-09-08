"""Generacion de los graficos tecnicos de salida (Modulo 2, entregable 3).

Se exportan en PNG a 150 dpi. Ambos declaran explicitamente los tramos sin
dato en lugar de interpolarlos: un hueco en la serie es informacion operativa
(el pretil no fue detectable en ese tramo), no ruido a ocultar.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # backend sin ventana, requerido en contenedor
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

RISK_COLORS = {"safe": "#10B981", "warning": "#F59E0B", "critical": "#EF4444"}

MAX_TRACKS_IN_MATRIX = 10


def plot_berm_height(heights: list[float | None], fps: float,
                     output_path: Path, calibration: Any) -> None:
    """Curva temporal de la altura del pretil.

    Se grafica la serie cruda junto con la mediana movil. El suavizado opera
    como prior de rigidez sobre una escena que no la satisface (SUP-20), por
    lo que ambas curvas se exportan: la cruda es la evidencia, la filtrada la
    interpretacion.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=150)

    times = np.arange(len(heights)) / fps
    values = np.array([np.nan if h is None else h for h in heights],
                      dtype=float)
    valid = ~np.isnan(values)

    if valid.any():
        ax.plot(times, values, color="#94A3B8", lw=0.9, alpha=0.8,
                label="Altura cruda")

        window = max(5, int(fps))
        smooth = np.full_like(values, np.nan)
        for i in range(len(values)):
            lo, hi = max(0, i - window // 2), min(len(values), i + window // 2 + 1)
            chunk = values[lo:hi]
            chunk = chunk[~np.isnan(chunk)]
            if chunk.size:
                smooth[i] = np.median(chunk)
        ax.plot(times, smooth, color="#DC2626", lw=2.0,
                label=f"Mediana movil ({window} frames)")

        median = float(np.nanmedian(values))
        ax.axhline(median, color="#0EA5E9", ls="--", lw=1.2,
                   label=f"Mediana global: {median:.2f} m")

        # Tramos sin deteccion
        gaps = np.where(~valid)[0]
        for g in gaps:
            ax.axvspan(g / fps, (g + 1) / fps, color="#E2E8F0", alpha=0.5,
                       lw=0)

        coverage = valid.sum() / len(values)
        subtitle = (f"Cobertura {coverage:.0%}  |  escala: "
                    f"{calibration.scale_source.value}  |  confianza "
                    f"{calibration.scale_confidence:.2f}")
    else:
        ax.text(0.5, 0.5, "Pretil no detectado en la secuencia",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="#64748B")
        subtitle = "Sin mediciones validas"

    ax.set_title("Evolucion de la altura del pretil", fontsize=13, pad=14)
    ax.text(0.5, 1.02, subtitle, ha="center", transform=ax.transAxes,
            fontsize=9, color="#64748B")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Altura (m)")
    ax.grid(alpha=0.25, ls=":")
    if valid.any():
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Grafico de altura escrito en %s", output_path)


def _pair_distance(a: dict[str, Any], b: dict[str, Any],
                   calibration: Any, metric: bool) -> float | None:
    """Distancia entre dos puntos de contacto con el suelo.

    En modo metrico la escala se evalua en el punto medio de las coordenadas
    y de ambos vehiculos (SUP-31). La escala px/m depende de la profundidad en
    una vista en perspectiva, por lo que no existe un factor unico valido para
    un par: el punto medio es la aproximacion de primer orden y evita el sesgo
    de elegir arbitrariamente la escala de uno de los dos. La aproximacion se
    degrada cuando la separacion en profundidad es grande.
    """
    d_px = float(np.hypot(a["x"] - b["x"], a["y"] - b["y"]))
    if not metric:
        return d_px
    ppm = calibration.pixels_per_meter((a["y"] + b["y"]) / 2.0)
    if not ppm or ppm <= 0:
        return None
    return d_px / ppm


def _min_distance_matrix(positions: list[dict[str, Any]], calibration: Any,
                         metric: bool) -> tuple[np.ndarray, list[str]]:
    """Matriz simetrica de distancia minima registrada entre pares de tracks.

    Solo se comparan detecciones del mismo frame: dos vehiculos que ocuparon
    la misma posicion en instantes distintos no estuvieron proximos. Los
    tracks sin id estable se excluyen, ya que no pueden asociarse entre
    frames. Se conservan los tracks de mayor permanencia para mantener la
    matriz legible.
    """
    presence: dict[tuple[str, Any], int] = {}
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for p in positions:
        if p.get("track_id") is None:
            continue
        key = (p["class"], p["track_id"])
        presence[key] = presence.get(key, 0) + 1
        by_frame.setdefault(p["frame"], []).append(p)

    tracks = [k for k, _ in sorted(presence.items(), key=lambda kv: kv[1],
                                   reverse=True)[:MAX_TRACKS_IN_MATRIX]]
    index = {t: i for i, t in enumerate(tracks)}
    n = len(tracks)
    matrix = np.full((n, n), np.nan, dtype=float)

    for dets in by_frame.values():
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = dets[i], dets[j]
                ka = (a["class"], a["track_id"])
                kb = (b["class"], b["track_id"])
                if ka not in index or kb not in index or ka == kb:
                    continue
                d = _pair_distance(a, b, calibration, metric)
                if d is None:
                    continue
                ia, ib = index[ka], index[kb]
                if np.isnan(matrix[ia, ib]) or d < matrix[ia, ib]:
                    matrix[ia, ib] = d
                    matrix[ib, ia] = d

    labels = [f"{cls} #{tid}" for cls, tid in tracks]
    return matrix, labels


def _draw_distance_matrix(ax: Any, matrix: np.ndarray, labels: list[str],
                          unit: str, critical_m: float,
                          warning_m: float) -> None:
    """Renderiza la matriz con anotaciones y codigo de color por umbral."""
    n = len(labels)
    if n < 2 or np.all(np.isnan(matrix)):
        ax.text(0.5, 0.5, "Sin pares concurrentes", ha="center", va="center",
                transform=ax.transAxes, fontsize=11, color="#64748B")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Matriz de distancias minimas", fontsize=11)
        return

    display = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="#F1F5F9")
    vmax = float(np.nanmax(matrix))
    im = ax.imshow(display, cmap=cmap, vmin=0.0, vmax=max(vmax, warning_m))

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "-", ha="center", va="center", fontsize=8,
                        color="#94A3B8")
                continue
            value = matrix[i, j]
            if np.isnan(value):
                continue
            weight = "bold" if value < critical_m else "normal"
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    fontsize=7.5, color="#0F172A", fontweight=weight)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title(f"Matriz de distancias minimas ({unit})", fontsize=11)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.2)
    ax.tick_params(which="minor", length=0)

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    if unit == "m":
        cbar.ax.axhline(critical_m, color="#7F1D1D", lw=1.4)
        cbar.ax.axhline(warning_m, color="#78350F", lw=1.4, ls="--")
        cbar.set_label(f"critico {critical_m:.0f} m  |  alerta "
                       f"{warning_m:.0f} m", fontsize=7)


def plot_vehicle_map(positions: list[dict[str, Any]],
                     frame_size: tuple[int, int],
                     output_path: Path,
                     calibration: Any = None,
                     critical_m: float = 10.0,
                     warning_m: float = 20.0) -> None:
    """Distribucion espacial de maquinaria y matriz de distancias minimas.

    Panel izquierdo: puntos de contacto con el suelo (SUP-05) coloreados por
    nivel de riesgo. Panel central: distancia minima registrada entre cada par
    de equipos, en metros cuando la calibracion es metrica y en pixeles cuando
    degrado a nivel no metrico. Panel derecho: permanencia por track.
    """
    width, height = frame_size
    metric = calibration is not None and calibration.is_metric
    unit = "m" if metric else "px"

    fig, (ax_map, ax_mat, ax_bar) = plt.subplots(
        1, 3, figsize=(17, 5.2), dpi=150,
        gridspec_kw={"width_ratios": [2, 1.7, 1]})

    if positions:
        for risk, color in RISK_COLORS.items():
            pts = [p for p in positions if p["risk"] == risk]
            if pts:
                ax_map.scatter([p["x"] for p in pts], [p["y"] for p in pts],
                               s=14, c=color, alpha=0.65,
                               label=f"{risk} ({len(pts)})",
                               edgecolors="none")
        ax_map.legend(fontsize=8, loc="upper right", framealpha=0.9)

        matrix, labels = _min_distance_matrix(positions, calibration, metric)
        _draw_distance_matrix(ax_mat, matrix, labels, unit, critical_m,
                              warning_m)

        counts: dict[str, int] = {}
        for p in positions:
            key = f"{p['class']} #{p['track_id']}" if p["track_id"] else p["class"]
            counts[key] = counts.get(key, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
        ax_bar.barh([k for k, _ in top][::-1], [v for _, v in top][::-1],
                    color="#0EA5E9", height=0.6)
        ax_bar.set_xlabel("Frames con presencia")
        ax_bar.set_title("Permanencia por equipo", fontsize=11)
        ax_bar.grid(axis="x", alpha=0.25, ls=":")
        ax_bar.tick_params(axis="y", labelsize=7)
    else:
        for ax, msg in ((ax_map, "Sin detecciones"),
                        (ax_mat, "Sin detecciones"),
                        (ax_bar, "Sin datos")):
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#64748B")
            if ax is ax_mat:
                ax.set_xticks([])
                ax.set_yticks([])

    ax_map.set_xlim(0, width)
    ax_map.set_ylim(height, 0)          # origen arriba, como en la imagen
    ax_map.set_title("Distribucion espacial de maquinaria", fontsize=11)
    ax_map.set_xlabel("x (px)")
    ax_map.set_ylabel("y (px)")
    ax_map.grid(alpha=0.2, ls=":")
    ax_map.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.debug("Mapa de vehiculos escrito en %s", output_path)


def min_distance_summary(positions: list[dict[str, Any]],
                         calibration: Any) -> dict[str, Any]:
    """Resumen de proximidad para metadata.json.

    Devuelve la distancia minima global registrada entre cualquier par de
    equipos concurrentes y el par que la produjo.
    """
    metric = calibration is not None and calibration.is_metric
    matrix, labels = _min_distance_matrix(positions, calibration, metric)
    if not labels or np.all(np.isnan(matrix)):
        return {"min_pair_distance": None, "min_pair_distance_unit": None,
                "min_pair": None, "tracks_compared": len(labels)}

    flat = np.nanargmin(np.where(np.isnan(matrix), np.inf, matrix))
    i, j = divmod(int(flat), matrix.shape[1])
    return {
        "min_pair_distance": round(float(matrix[i, j]), 2),
        "min_pair_distance_unit": "m" if metric else "px",
        "min_pair": [labels[i], labels[j]],
        "tracks_compared": len(labels),
    }