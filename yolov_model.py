"""YOLO model loading and inference wrapper."""

from pathlib import Path
from typing import Any

from ultralytics import YOLO

import config


class YOLOv8Detector:
    """Loads YOLOv8n once and exposes a simple prediction method."""

    def __init__(self, model_path: str = config.MODEL_PATH) -> None:
        self.model_path = model_path
        self.model = YOLO(model_path)

    def predict(self, image: Any):
        """Run YOLO inference and return the first Results object."""
        results = self.model.predict(
            source=image,
            imgsz=config.YOLO_IMAGE_SIZE,
            conf=config.YOLO_INFERENCE_CONF,
            iou=config.YOLO_IOU,
            verbose=False,
        )
        return results[0]
