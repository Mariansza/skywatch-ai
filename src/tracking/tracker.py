"""ByteTrack multi-object tracker wrapper for SkyWatch AI.

Same philosophy as Detector: wrap the external library (supervision's
ByteTrack) behind a clean, typed interface. This keeps ByteTrack
internals isolated — if the library changes or we switch algorithms,
only this file needs updating.

ByteTrack is *stateful*: it remembers active tracks across frames.
Call ``update()`` once per frame, in order. Call ``reset()`` between
videos to clear the internal state.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import supervision as sv
import yaml

from src.models.schema import DetectionResult, detections_to_supervision
from src.tracking.schema import TrackedDetection, TrackingResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Tracker:
    """Wrapper around supervision's ByteTrack for multi-object tracking.

    Args:
        config: Dictionary with a ``tracking`` section matching
            the structure of ``configs/track.yaml``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        track_cfg = config.get("tracking", {})

        self._track_activation_threshold = track_cfg.get(
            "track_activation_threshold", 0.25
        )
        self._lost_track_buffer = track_cfg.get("lost_track_buffer", 30)
        self._minimum_matching_threshold = track_cfg.get(
            "minimum_matching_threshold", 0.8
        )
        self._frame_rate = track_cfg.get("frame_rate", 30)
        self._minimum_consecutive_frames = track_cfg.get(
            "minimum_consecutive_frames", 1
        )

        self._byte_track = self._create_byte_track()
        self._frame_index = 0

        logger.info(
            "Tracker initialized: activation=%.2f buffer=%d matching=%.2f fps=%d",
            self._track_activation_threshold,
            self._lost_track_buffer,
            self._minimum_matching_threshold,
            self._frame_rate,
        )

    def _create_byte_track(self) -> sv.ByteTrack:
        """Create a fresh ByteTrack instance with current parameters."""
        return sv.ByteTrack(
            track_activation_threshold=self._track_activation_threshold,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=self._minimum_matching_threshold,
            frame_rate=self._frame_rate,
            minimum_consecutive_frames=self._minimum_consecutive_frames,
        )

    def update(self, detection_result: DetectionResult) -> TrackingResult:
        """Process detections from one frame and assign track IDs.

        ByteTrack compares these detections against its internal state
        (the positions of previously tracked objects) using IoU (Intersection
        over Union — how much two boxes overlap). It then decides:
        - Which detections match existing tracks (same object, new position)
        - Which are new objects (no matching track → create a new one)
        - Which tracks are lost (no matching detection → keep in buffer)

        Args:
            detection_result: Detection results from the current frame.

        Returns:
            A ``TrackingResult`` with each detection enriched by a tracker_id.
        """
        start = time.perf_counter()

        sv_detections = detections_to_supervision(detection_result)
        tracked_sv = self._byte_track.update_with_detections(sv_detections)

        tracked_detections = self._build_tracked_detections(
            tracked_sv, detection_result
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = TrackingResult(
            tracked_detections=tracked_detections,
            frame_index=self._frame_index,
            image_shape=detection_result.image_shape,
            inference_time_ms=detection_result.inference_time_ms + elapsed_ms,
        )

        logger.debug(
            "Frame %d: %d detections → %d tracks (%.1f ms tracking)",
            self._frame_index,
            len(detection_result.detections),
            len(tracked_detections),
            elapsed_ms,
        )

        self._frame_index += 1
        return result

    def reset(self) -> None:
        """Reset the tracker state for a new video.

        Creates a fresh ByteTrack instance, clearing all active and
        lost tracks. Also resets the frame counter.
        """
        self._byte_track = self._create_byte_track()
        self._frame_index = 0
        logger.info("Tracker state reset")

    @classmethod
    def from_config(cls, config_path: str | Path) -> Tracker:
        """Create a Tracker from a YAML config file.

        Args:
            config_path: Path to a YAML configuration file.

        Returns:
            A configured ``Tracker`` instance.
        """
        path = Path(config_path)
        with path.open() as f:
            config = yaml.safe_load(f)
        return cls(config)

    @staticmethod
    def _build_tracked_detections(
        tracked_sv: sv.Detections,
        original: DetectionResult,
    ) -> list[TrackedDetection]:
        """Convert supervision tracked detections back to our domain model.

        ByteTrack may drop some detections (low confidence ones that
        don't match any track). So the output can have fewer items
        than the input.
        """
        if len(tracked_sv) == 0:
            return []

        names = {d.class_id: d.class_name for d in original.detections}
        detections: list[TrackedDetection] = []

        for i in range(len(tracked_sv)):
            class_id = int(tracked_sv.class_id[i])  # type: ignore[index]
            detections.append(
                TrackedDetection(
                    bbox=tracked_sv.xyxy[i],
                    confidence=float(tracked_sv.confidence[i]),  # type: ignore[index]
                    class_id=class_id,
                    class_name=names.get(class_id, f"class_{class_id}"),
                    tracker_id=int(tracked_sv.tracker_id[i]),  # type: ignore[index]
                )
            )

        return detections
