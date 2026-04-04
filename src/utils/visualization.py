"""Visualization utilities for SkyWatch AI.

Uses the supervision library for production-quality annotations
(anti-aliased boxes, label backgrounds, per-class color palettes).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import supervision as sv

if TYPE_CHECKING:
    import numpy as np

from src.models.schema import DetectionResult, detections_to_supervision
from src.tracking.schema import TrackingResult, tracked_detections_to_supervision
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def load_image(path: str | Path) -> np.ndarray:
    """Load an image from disk and convert BGR to RGB.

    Args:
        path: Path to the image file.

    Returns:
        Image as a numpy array in RGB format.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If OpenCV cannot decode the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")

    logger.info("Loaded image %s shape=%s", path.name, image.shape)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def annotate_detections(
    image: np.ndarray,
    result: DetectionResult,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Draw detection boxes and labels on an image.

    Does not mutate the input image.

    Args:
        image: Source image as a numpy array (RGB).
        result: Detection results to annotate.
        config: Visualization config section from detect.yaml.
            Uses sensible defaults if not provided.

    Returns:
        A new annotated image (RGB).
    """
    cfg = config or {}
    sv_detections = detections_to_supervision(result)

    annotated = image.copy()

    box_annotator = sv.BoxAnnotator(
        thickness=cfg.get("box_thickness", 2),
    )
    annotated = box_annotator.annotate(annotated, sv_detections)

    if cfg.get("show_labels", True):
        labels = _build_labels(result, show_confidence=cfg.get("show_confidence", True))
        label_annotator = sv.LabelAnnotator(
            text_scale=cfg.get("text_scale", 0.5),
            text_thickness=cfg.get("text_thickness", 1),
        )
        annotated = label_annotator.annotate(annotated, sv_detections, labels=labels)

    return annotated  # type: ignore[no-any-return]


def save_annotated_image(image: np.ndarray, output_path: str | Path) -> Path:
    """Save an annotated image to disk.

    Converts RGB back to BGR for OpenCV, and creates parent
    directories if needed.

    Args:
        image: Annotated image in RGB format.
        output_path: Destination file path.

    Returns:
        The resolved output path.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)

    logger.info("Saved annotated image to %s", path)
    return path


def annotate_tracks(
    image: np.ndarray,
    result: TrackingResult,
    config: dict[str, Any] | None = None,
) -> np.ndarray:
    """Draw tracked detection boxes with track IDs on an image.

    Similar to ``annotate_detections`` but labels include the track ID
    (e.g. "car #5 87%") so you can see which object is which across frames.

    Optionally draws trace trails — lines showing the recent trajectory
    of each tracked object. This uses supervision's TraceAnnotator which
    needs the tracker_id field to connect positions across frames.

    Does not mutate the input image.

    Args:
        image: Source image as a numpy array (RGB).
        result: Tracking results to annotate.
        config: Visualization config section from track.yaml.

    Returns:
        A new annotated image (RGB).
    """
    cfg = config or {}
    sv_detections = tracked_detections_to_supervision(result)

    annotated = image.copy()

    box_annotator = sv.BoxAnnotator(
        thickness=cfg.get("box_thickness", 2),
    )
    annotated = box_annotator.annotate(annotated, sv_detections)

    if cfg.get("show_labels", True):
        labels = _build_track_labels(
            result,
            show_confidence=cfg.get("show_confidence", True),
            show_track_ids=cfg.get("show_track_ids", True),
        )
        label_annotator = sv.LabelAnnotator(
            text_scale=cfg.get("text_scale", 0.5),
            text_thickness=cfg.get("text_thickness", 1),
        )
        annotated = label_annotator.annotate(annotated, sv_detections, labels=labels)

    if cfg.get("show_traces", False):
        trace_annotator = sv.TraceAnnotator(
            trace_length=cfg.get("trace_length", 30),
            thickness=cfg.get("box_thickness", 2),
        )
        annotated = trace_annotator.annotate(annotated, sv_detections)

    return annotated  # type: ignore[no-any-return]


def _build_labels(result: DetectionResult, *, show_confidence: bool) -> list[str]:
    """Build label strings for each detection."""
    labels = []
    for det in result.detections:
        if show_confidence:
            labels.append(f"{det.class_name} {det.confidence:.0%}")
        else:
            labels.append(det.class_name)
    return labels


def _build_track_labels(
    result: TrackingResult,
    *,
    show_confidence: bool,
    show_track_ids: bool,
) -> list[str]:
    """Build label strings for tracked detections.

    Format examples:
    - show_track_ids + show_confidence: "car #5 87%"
    - show_track_ids only: "car #5"
    - show_confidence only: "car 87%"
    - neither: "car"
    """
    labels = []
    for td in result.tracked_detections:
        parts = [td.class_name]
        if show_track_ids:
            parts.append(f"#{td.tracker_id}")
        if show_confidence:
            parts.append(f"{td.confidence:.0%}")
        labels.append(" ".join(parts))
    return labels
