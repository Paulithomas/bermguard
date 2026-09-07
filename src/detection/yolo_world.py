"""Metodo 2: YOLO-World, deteccion open-vocabulary zero-shot.

Contraste con el metodo 1. YOLOv11-seg opera sobre un vocabulario cerrado
heredado de COCO, que carece de clase para maquinaria de oruga (SUP-29).
YOLO-World acepta prompts de texto arbitrarios, lo que permite definir las
clases del dominio sin etiquetar un solo frame.

Hipotesis contrastable (SUP-30). El metodo 1 cae de 0.88 a 0.39 detecciones
por frame en escenas nocturnas con faros directos, donde los vehiculos son
siluetas sin textura interna. Un modelo fundacional que razona sobre semantica
visual amplia deberia degradarse menos que uno ajustado a patrones de COCO.
El benchmark contrasta esta hipotesis; el reporte consigna el resultado sea
cual sea.

Costo esperado. La inferencia open-vocabulary es mas lenta y el modelo no
produce mascaras de instancia, solo cajas. Ambos aspectos son parte del
trade-off que el reporte debe cuantificar.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.detection.base import VehicleDetector
from src.detection.types import Detection, FrameResult, VehicleClass

logger = logging.getLogger(__name__)


class YoloWorldDetector(VehicleDetector):
    """Detector open-vocabulary basado en YOLO-World v2."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._model: Any = None
        self._device: str = "cpu"
        self._tracking: bool = True

        ov = config.get("open_vocabulary", {})
        self._confidence: float = ov.get("confidence", 0.05)
        self._prompt_map: dict[str, VehicleClass] = {}
        for class_name, prompts in ov.get("prompts", {}).items():
            try:
                vehicle_class = VehicleClass(class_name)
            except ValueError:
                logger.warning("Clase desconocida en config: %s", class_name)
                continue
            for prompt in prompts:
                self._prompt_map[prompt] = vehicle_class

        self._weights_path = Path(
            config.get("weights", {}).get("yolo_world",
                                          "weights/yolov8s-worldv2.pt"))

    @property
    def name(self) -> str:
        return "yolov8s-worldv2 (open-vocabulary)"

    def load(self) -> None:
        """Carga el modelo y fija el vocabulario desde los prompts.

        Raises:
            FileNotFoundError: si los pesos no estan disponibles (SUP-24).
        """
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"Pesos no encontrados en {self._weights_path}. "
                "Deben incluirse en la imagen durante el build.")
        if not self._prompt_map:
            raise ValueError("No hay prompts configurados para el metodo 2")

        from ultralytics import YOLOWorld

        from src.utils.device import resolve_device
        self._device = resolve_device(
            self._config.get("runtime", {}).get("device_priority"))

        self._model = YOLOWorld(str(self._weights_path))
        self._model.set_classes(list(self._prompt_map.keys()))
        logger.info("Modelo %s cargado en %s con %d prompts",
                    self.name, self._device, len(self._prompt_map))

    def reset_tracker(self) -> None:
        """Reinicia el estado del tracker en cada corte de toma (SUP-19)."""
        if self._model is None:
            return
        predictor = getattr(self._model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor else None
        if trackers:
            for tracker in trackers:
                tracker.reset()

    def set_tracking(self, enabled: bool) -> None:
        """Activa o desactiva el seguimiento entre frames."""
        self._tracking = enabled

    def detect(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        if self._model is None:
            raise RuntimeError("load() debe invocarse antes de detect()")

        start = time.perf_counter()
        if self._tracking:
            results = self._model.track(
                frame, conf=self._confidence, device=self._device,
                persist=True, tracker="bytetrack.yaml", verbose=False)
        else:
            results = self._model.predict(
                frame, conf=self._confidence, device=self._device,
                verbose=False)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        detections = self._parse(results[0]) if results else []
        return FrameResult(frame_index=frame_index,
                           detections=detections,
                           inference_ms=elapsed_ms)

    def _parse(self, result: Any) -> list[Detection]:
        """Traduce la salida al modelo de dominio via el mapa de prompts."""
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        names: dict[int, str] = result.names

        for i in range(len(boxes)):
            prompt = names.get(int(boxes.cls[i]), "")
            vehicle_class = self._prompt_map.get(prompt)
            if vehicle_class is None:
                continue

            track_id = None
            if getattr(boxes, "id", None) is not None:
                track_id = int(boxes.id[i])

            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
            detections.append(Detection(
                vehicle_class=vehicle_class,
                confidence=float(boxes.conf[i]),
                bbox=(x1, y1, x2, y2),
                mask=None,          # el modelo no produce mascaras
                track_id=track_id,
            ))

        return detections