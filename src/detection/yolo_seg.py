"""Metodo 1: YOLOv11-seg, arquitectura especializada.

Estrategia: modelo preentrenado en COCO como linea base, con mapeo de clases
COCO a la taxonomia del dominio (SUP-06). COCO no contiene una clase para
bulldozer, limitacion documentada en SUP-29 que motiva el fine-tune posterior.

Umbral de confianza bajo por diseno (SUP-18): el material generativo produce
geometrias irregulares que penalizan a detectores estrictos. El rechazo de
falsos positivos se delega al tracker por consistencia temporal.
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

COCO_TO_DOMAIN: dict[str, VehicleClass] = {
    "truck": VehicleClass.CAEX,
    "bus": VehicleClass.OTHER_HEAVY,
    "train": VehicleClass.OTHER_HEAVY,
    "car": VehicleClass.OTHER_HEAVY,
}


class YoloSegDetector(VehicleDetector):
    """Detector de instancias basado en YOLOv11-seg con seguimiento ByteTrack."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._model: Any = None
        self._device: str = "cpu"
        self._tracking: bool = True
        runtime = config.get("runtime", {})
        self._confidence: float = runtime.get("detection_confidence", 0.25)
        self._weights_path = Path(
            config.get("weights", {}).get("yolo_seg", "weights/yolo11n-seg.pt"))

    @property
    def name(self) -> str:
        return "yolo11n-seg (COCO) + ByteTrack"

    def load(self) -> None:
        """Carga el modelo desde pesos locales.

        Raises:
            FileNotFoundError: si los pesos no estan disponibles (SUP-24).
        """
        if not self._weights_path.exists():
            raise FileNotFoundError(
                f"Pesos no encontrados en {self._weights_path}. "
                "Deben incluirse en la imagen durante el build.")

        from ultralytics import YOLO

        from src.utils.device import resolve_device
        self._device = resolve_device(
            self._config.get("runtime", {}).get("device_priority"))

        self._model = YOLO(str(self._weights_path))
        logger.info("Modelo %s cargado en %s", self.name, self._device)

    def reset_tracker(self) -> None:
        """Reinicia el estado del tracker.

        Obligatorio en cada corte de toma (SUP-19): mantener los IDs a traves
        de un corte asocia vehiculos pertenecientes a escenas distintas.
        """
        if self._model is None:
            return
        predictor = getattr(self._model, "predictor", None)
        trackers = getattr(predictor, "trackers", None) if predictor else None
        if trackers:
            for tracker in trackers:
                tracker.reset()
            logger.debug("Tracker reinicializado")

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

        detections: list[Detection] = []
        if results:
            detections = self._parse(results[0])

        return FrameResult(frame_index=frame_index,
                           detections=detections,
                           inference_ms=elapsed_ms)

    def _parse(self, result: Any) -> list[Detection]:
        """Traduce la salida de Ultralytics al modelo de dominio."""
        detections: list[Detection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        masks = getattr(result, "masks", None)
        names: dict[int, str] = result.names

        for i in range(len(boxes)):
            coco_name = names[int(boxes.cls[i])]
            vehicle_class = COCO_TO_DOMAIN.get(coco_name)
            if vehicle_class is None:
                continue

            mask = None
            if masks is not None and i < len(masks.data):
                mask = masks.data[i].cpu().numpy().astype(np.uint8)

            track_id = None
            if getattr(boxes, "id", None) is not None:
                track_id = int(boxes.id[i])

            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
            detections.append(Detection(
                vehicle_class=vehicle_class,
                confidence=float(boxes.conf[i]),
                bbox=(x1, y1, x2, y2),
                mask=mask,
                track_id=track_id,
            ))

        return detections