"""Resolucion del dispositivo de computo (SUP-23)."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def resolve_device(priority: list[str] | None = None) -> str:
    """Devuelve el primer dispositivo disponible segun la prioridad dada.

    El fallback a CPU es obligatorio: el evaluador podria ejecutar sin GPU,
    y un contenedor que entrega artefactos lentos vale mas que uno que aborta.
    """
    if priority is None:
        priority = ["cuda", "mps", "cpu"]

    try:
        import torch
    except ImportError:
        logger.warning("PyTorch no disponible, se usa CPU")
        return "cpu"

    for device in priority:
        if device == "cuda" and torch.cuda.is_available():
            return "cuda"
        if device == "mps" and torch.backends.mps.is_available():
            return "mps"
        if device == "cpu":
            return "cpu"

    return "cpu"