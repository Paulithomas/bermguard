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


def plot_vehicle_map(positions: list[dict[str, Any]],
                     frame_size: tuple[int, int],
                     output_path: Path) -> None:
    """Distribucion espacial de maquinaria y matriz de distancias minimas.

    El panel izquierdo mapea los puntos de contacto con el suelo (SUP-05)
    coloreados por nivel de riesgo. El derecho resume la ocupacion por track.
    """
    width, height = frame_size
    fig, (ax_map, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 5), dpi=150, gridspec_kw={"width_ratios": [2, 1]})

    if positions:
        for risk, color in RISK_COLORS.items():
            pts = [p for p in positions if p["risk"] == risk]
            if pts:
                ax_map.scatter([p["x"] for p in pts], [p["y"] for p in pts],
                               s=14, c=color, alpha=0.65,
                               label=f"{risk} ({len(pts)})",
                               edgecolors="none")
        ax_map.legend(fontsize=8, loc="upper right", framealpha=0.9)

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
    else:
        for ax, msg in ((ax_map, "Sin detecciones"), (ax_bar, "Sin datos")):
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#64748B")

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