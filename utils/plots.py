"""OpenCV and Matplotlib visualization helpers."""

from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np

import config
from yolo_module.detection import Detection


def draw_detections(
    image: np.ndarray,
    detections: Sequence[Detection],
    keep_mask: Sequence[bool],
    title: str = "Hybrid Detection",
) -> np.ndarray:
    """Draw only detections whose keep_mask entry is True."""
    output = image.copy()
    for detection, keep in zip(detections, keep_mask):
        if not keep:
            continue
        x1, y1, x2, y2 = detection.box
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), config.BOX_THICKNESS)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, config.FONT_SCALE, 1)
        cv2.rectangle(output, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(
            output,
            label,
            (x1 + 2, max(th, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.FONT_SCALE,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        output,
        title,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def stage_counts(yolo_count: int, fuzzy_count: int, hybrid_count: int) -> dict[str, int]:
    return {"YOLO": yolo_count, "Fuzzy": fuzzy_count, "Hybrid": hybrid_count}


def plot_stage_comparison(counts: dict[str, int]):
    """Return a Matplotlib figure comparing stage-wise detection counts."""
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = list(counts.keys())
    values = list(counts.values())
    ax.bar(labels, values)
    ax.set_title("Stage-wise Detection Count")
    ax.set_ylabel("Number of Detections")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_ga_history(history: Sequence[float]):
    """Return a Matplotlib figure showing best fitness over GA generations."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(history) + 1), history, marker="o", markersize=3)
    ax.set_title("Genetic Algorithm Fitness")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig
