"""Contrato comun para los metodos de deteccion (SUP-27).

La abstraccion permite que --method 1|2|all seleccione implementaciones
intercambiables sin ramificar la logica del pipeline, y habilita el analisis
de concordancia entre metodos como proxy de confiabilidad (SUP-21).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.detection.types import FrameResult


class VehicleDetector(ABC):
    """Interfaz que todo metodo de deteccion debe implementar."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nombre legible del metodo, usado en reportes y metadatos."""

    @abstractmethod
    def load(self) -> None:
        """Carga pesos y prepara el modelo.

        Se invoca una sola vez antes de procesar. Los pesos deben estar
        disponibles localmente: no se permiten descargas en runtime (SUP-24).

        Raises:
            FileNotFoundError: si los pesos no estan en la imagen.
        """

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        """Detecta maquinaria en un frame.

        Args:
            frame: Imagen BGR tal como la entrega OpenCV.
            frame_index: Indice del frame dentro del video.

        Returns:
            Detecciones del frame junto con la latencia de inferencia.
        """

    def warmup(self, shape: tuple[int, int] = (640, 640)) -> None:
        """Ejecuta una inferencia en vacio para excluir el costo de la
        primera llamada de las mediciones de latencia del benchmark."""
        dummy = np.zeros((*shape, 3), dtype=np.uint8)
        self.detect(dummy, -1)