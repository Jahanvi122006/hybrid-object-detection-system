"""Convert Ultralytics Results into plain Python/NumPy detections."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Detection:
    """One object detection returned by YOLO."""

    box: tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


def extract_detections(result: Any) -> list[Detection]:
    """Extract boxes, confidence scores and class IDs from a YOLO Results object."""
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    names = result.names

    detections: list[Detection] = []
    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = np.round(box).astype(int).tolist()
        class_name = names.get(int(class_id), str(class_id)) if isinstance(names, dict) else str(class_id)
        detections.append(
            Detection(
                box=(x1, y1, x2, y2),
                confidence=float(confidence),
                class_id=int(class_id),
                class_name=class_name,
            )
        )
    return detections
