"""Detection data schemas for SkyWatch AI.

Defines the domain model for detection results, decoupled from
ultralytics and supervision. These dataclasses travel through the
entire pipeline (detect -> track -> evaluate -> serve).

Using frozen dataclasses ensures immutability: once a detection is
produced, it cannot be accidentally mutated downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
import supervision as sv


@dataclass(frozen=True, slots=True)
class Detection:
    """A single object detection result.

    Attributes:
        bbox: Bounding box as [x1, y1, x2, y2] in pixel coordinates (xyxy).
        confidence: Detection confidence score in [0, 1].
        class_id: Integer class ID from the model.
        class_name: Human-readable class name.
    """

    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """Detection results for a single image.

    Attributes:
        detections: List of individual detections.
        image_path: Source image path, if applicable.
        image_shape: Original image dimensions (H, W, C).
        inference_time_ms: Inference duration in milliseconds.
    """

    detections: list[Detection]
    image_path: Path | None
    image_shape: tuple[int, int, int]
    inference_time_ms: float


def detections_to_supervision(result: DetectionResult) -> sv.Detections:
    """Convert a DetectionResult to a supervision Detections object.

    This bridges our domain model to the supervision library for
    visualization, without coupling the rest of the codebase to it.

    Args:
        result: Our detection result to convert.

    Returns:
        A ``supervision.Detections`` instance ready for annotation.
    """
    if not result.detections:
        return sv.Detections.empty()

    xyxy = np.array([d.bbox for d in result.detections], dtype=np.float32)
    confidence = np.array([d.confidence for d in result.detections], dtype=np.float32)
    class_id = np.array([d.class_id for d in result.detections], dtype=int)

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
    )
