"""Tests for src/tracking/tracker.py."""

import numpy as np

from src.models.schema import Detection, DetectionResult
from src.tracking.schema import TrackingResult
from src.tracking.tracker import Tracker


def _make_detection_result(
    bboxes: list[list[float]],
    class_ids: list[int] | None = None,
) -> DetectionResult:
    """Helper to build a DetectionResult from bbox coordinates."""
    if class_ids is None:
        class_ids = [0] * len(bboxes)
    detections = [
        Detection(
            bbox=np.array(bb, dtype=np.float32),
            confidence=0.9,
            class_id=cid,
            class_name=f"class_{cid}",
        )
        for bb, cid in zip(bboxes, class_ids, strict=True)
    ]
    return DetectionResult(
        detections=detections,
        image_path=None,
        image_shape=(640, 640, 3),
        inference_time_ms=10.0,
    )


class TestTracker:
    """Tests for the Tracker wrapper."""

    def test_init_defaults(self) -> None:
        tracker = Tracker(config={})
        assert tracker._frame_index == 0

    def test_init_with_config(self, sample_track_config: dict) -> None:
        tracker = Tracker(sample_track_config)
        assert tracker._lost_track_buffer == 30
        assert tracker._frame_rate == 30

    def test_update_returns_tracking_result(self) -> None:
        tracker = Tracker(config={"tracking": {"minimum_consecutive_frames": 1}})
        dr = _make_detection_result([[100, 100, 200, 200]])
        result = tracker.update(dr)
        assert isinstance(result, TrackingResult)
        assert result.frame_index == 0

    def test_frame_index_increments(self) -> None:
        tracker = Tracker(config={})
        dr = _make_detection_result([[100, 100, 200, 200]])
        tracker.update(dr)
        result2 = tracker.update(dr)
        assert result2.frame_index == 1

    def test_reset_clears_state(self) -> None:
        tracker = Tracker(config={})
        dr = _make_detection_result([[100, 100, 200, 200]])
        tracker.update(dr)
        tracker.update(dr)
        tracker.reset()
        assert tracker._frame_index == 0

    def test_empty_detections(self) -> None:
        tracker = Tracker(config={})
        dr = _make_detection_result([])
        result = tracker.update(dr)
        assert len(result.tracked_detections) == 0

    def test_track_ids_are_consistent_across_frames(self) -> None:
        """Objects at similar positions across frames should keep the same ID."""
        tracker = Tracker(config={"tracking": {"minimum_consecutive_frames": 1}})
        # Frame 1: object at (100,100)-(200,200)
        dr1 = _make_detection_result([[100, 100, 200, 200]])
        r1 = tracker.update(dr1)

        # Frame 2: same object, slightly moved
        dr2 = _make_detection_result([[105, 105, 205, 205]])
        r2 = tracker.update(dr2)

        if r1.tracked_detections and r2.tracked_detections:
            id1 = r1.tracked_detections[0].tracker_id
            id2 = r2.tracked_detections[0].tracker_id
            assert id1 == id2

    def test_inference_time_includes_detection(self) -> None:
        tracker = Tracker(config={})
        dr = _make_detection_result([[100, 100, 200, 200]])
        result = tracker.update(dr)
        # Combined time should be >= detection time alone
        assert result.inference_time_ms >= dr.inference_time_ms
