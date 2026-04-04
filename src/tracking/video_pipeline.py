"""Video processing pipeline for SkyWatch AI.

Orchestrates the full detect → track → annotate → save workflow
on a video file. This is the "conductor" that wires together
the Detector, Tracker, visualization, and video I/O modules.

The pipeline is also designed for reuse: ``process_frame()`` handles
a single frame, which the future FastAPI endpoint can call directly
for real-time streaming.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from src.models.detector import Detector
    from src.tracking.tracker import Tracker

from src.tracking.schema import PipelineStats, TrackingResult
from src.tracking.video_io import (
    create_video_writer,
    iterate_frames,
    open_video,
    write_frame,
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class VideoPipeline:
    """End-to-end video detection and tracking pipeline.

    Composes a Detector and a Tracker, processing each frame
    sequentially. ByteTrack needs sequential frames to maintain
    track continuity — you can't parallelize frame processing
    because frame N's tracking depends on frame N-1's state.

    Args:
        detector: A configured Detector instance.
        tracker: A configured Tracker instance.
        config: Full pipeline config (used for visualization settings).
    """

    def __init__(
        self,
        detector: Detector,
        tracker: Tracker,
        config: dict[str, Any],
    ) -> None:
        self._detector = detector
        self._tracker = tracker
        self._config = config

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> TrackingResult:
        """Run detection + tracking on a single frame.

        This is the core unit of work. The pipeline is:
        1. Detector.predict(frame) → DetectionResult
        2. Tracker.update(detections) → TrackingResult (with track IDs)

        Args:
            frame: Video frame as RGB numpy array.

        Returns:
            Tracking result with enriched detections for this frame.
        """
        detection_result = self._detector.predict(frame)
        return self._tracker.update(detection_result)

    def process_video(
        self,
        source: str | Path,
        output_path: str | Path | None = None,
        annotate_fn: Any | None = None,
    ) -> PipelineStats:
        """Process an entire video file.

        For each frame: detect → track → (optionally) annotate → save.

        Args:
            source: Path to the input video file.
            output_path: Where to save the annotated video. If None,
                no output video is written.
            annotate_fn: Optional function(frame, tracking_result, config)
                that returns an annotated frame. Passed from the CLI to
                keep this module free of visualization imports.

        Returns:
            Pipeline statistics (frames, time, unique tracks).
        """
        self._tracker.reset()

        cap, info = open_video(source)
        writer = None
        if output_path is not None:
            writer = create_video_writer(output_path, info)

        unique_track_ids: set[int] = set()
        frames_processed = 0
        start_time = time.perf_counter()

        logger.info("Processing video: %s (%d frames)", source, info.total_frames)

        for frame in iterate_frames(cap):
            tracking_result = self.process_frame(frame)

            for td in tracking_result.tracked_detections:
                unique_track_ids.add(td.tracker_id)

            if writer is not None and annotate_fn is not None:
                annotated = annotate_fn(
                    frame, tracking_result, self._config.get("visualization", {})
                )
                write_frame(writer, annotated)

            frames_processed += 1

            if frames_processed % 100 == 0:
                logger.info(
                    "Progress: %d/%d frames (%.0f%%)",
                    frames_processed,
                    info.total_frames,
                    frames_processed / max(info.total_frames, 1) * 100,
                )

        if writer is not None:
            writer.release()

        total_ms = (time.perf_counter() - start_time) * 1000
        resolved_output = Path(output_path) if output_path is not None else None

        logger.info(
            "Finished: %d frames, %d unique tracks, %.1f s total (%.1f FPS)",
            frames_processed,
            len(unique_track_ids),
            total_ms / 1000,
            frames_processed / max(total_ms / 1000, 0.001),
        )

        return PipelineStats(
            frames_processed=frames_processed,
            total_time_ms=total_ms,
            unique_tracks=len(unique_track_ids),
            output_path=resolved_output,
        )
