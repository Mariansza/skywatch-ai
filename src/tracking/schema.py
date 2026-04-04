"""Tracking data schemas for SkyWatch AI.

Extends the detection schema with tracking-specific fields.
TrackedDetection adds a tracker_id to each detection, linking it
to a persistent track across video frames.

These are separate from the detection dataclasses because tracking
is an additional layer on top of detection — not a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import supervision as sv

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """A detection associated with a persistent track.

    Attributes:
        bbox: Bounding box as [x1, y1, x2, y2] in pixel coordinates (xyxy).
        confidence: Detection confidence score in [0, 1].
        class_id: Integer class ID from the model.
        class_name: Human-readable class name.
        tracker_id: Unique ID assigned by the tracker, consistent across frames.
    """

    bbox: np.ndarray
    confidence: float
    class_id: int
    class_name: str
    tracker_id: int


@dataclass(frozen=True, slots=True)
class TrackingResult:
    """Tracking results for a single video frame.

    Attributes:
        tracked_detections: List of detections with assigned track IDs.
        frame_index: Zero-based index of the frame in the video.
        image_shape: Frame dimensions (H, W, C).
        inference_time_ms: Combined detection + tracking time in milliseconds.
    """

    tracked_detections: list[TrackedDetection]
    frame_index: int
    image_shape: tuple[int, int, int]
    inference_time_ms: float


@dataclass(frozen=True, slots=True)
class VideoInfo:
    """Metadata about a video file.

    Attributes:
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        total_frames: Total number of frames in the video.
    """

    fps: float
    width: int
    height: int
    total_frames: int


@dataclass(frozen=True, slots=True)
class PipelineStats:
    """Summary statistics from processing a video.

    Attributes:
        frames_processed: Number of frames processed.
        total_time_ms: Total processing time in milliseconds.
        unique_tracks: Number of unique track IDs seen.
        output_path: Path to the output video file, if saved.
    """

    frames_processed: int
    total_time_ms: float
    unique_tracks: int
    output_path: Path | None


def tracked_detections_to_supervision(
    result: TrackingResult,
) -> sv.Detections:
    """Convert a TrackingResult to a supervision Detections object.

    The returned Detections includes tracker_id, which supervision
    annotators use to draw consistent colors per track and to
    render trace trails.

    Args:
        result: Tracking result to convert.

    Returns:
        A ``supervision.Detections`` with tracker_id populated.
    """
    if not result.tracked_detections:
        return sv.Detections.empty()

    xyxy = np.array([d.bbox for d in result.tracked_detections], dtype=np.float32)
    confidence = np.array(
        [d.confidence for d in result.tracked_detections], dtype=np.float32
    )
    class_id = np.array([d.class_id for d in result.tracked_detections], dtype=int)
    tracker_id = np.array([d.tracker_id for d in result.tracked_detections], dtype=int)

    return sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        tracker_id=tracker_id,
    )
