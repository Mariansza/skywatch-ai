"""Tests for src/tracking/schema.py."""

import numpy as np
import pytest
import supervision as sv

from src.tracking.schema import (
    PipelineStats,
    TrackedDetection,
    TrackingResult,
    VideoInfo,
    tracked_detections_to_supervision,
)


class TestTrackedDetection:
    """Tests for the TrackedDetection dataclass."""

    def test_create(self, sample_tracked_detection: TrackedDetection) -> None:
        assert sample_tracked_detection.tracker_id == 1
        assert sample_tracked_detection.class_name == "person"
        assert sample_tracked_detection.confidence == 0.85

    def test_immutable(self, sample_tracked_detection: TrackedDetection) -> None:
        with pytest.raises(AttributeError):
            sample_tracked_detection.tracker_id = 99  # type: ignore[misc]

    def test_bbox_is_xyxy(self, sample_tracked_detection: TrackedDetection) -> None:
        assert sample_tracked_detection.bbox.shape == (4,)


class TestTrackingResult:
    """Tests for the TrackingResult dataclass."""

    def test_create(self, sample_tracking_result: TrackingResult) -> None:
        assert len(sample_tracking_result.tracked_detections) == 3
        assert sample_tracking_result.frame_index == 0

    def test_immutable(self, sample_tracking_result: TrackingResult) -> None:
        with pytest.raises(AttributeError):
            sample_tracking_result.frame_index = 5  # type: ignore[misc]

    def test_empty(self) -> None:
        result = TrackingResult(
            tracked_detections=[],
            frame_index=0,
            image_shape=(640, 640, 3),
            inference_time_ms=0.0,
        )
        assert len(result.tracked_detections) == 0


class TestVideoInfo:
    """Tests for the VideoInfo dataclass."""

    def test_create(self) -> None:
        info = VideoInfo(fps=30.0, width=1920, height=1080, total_frames=300)
        assert info.fps == 30.0
        assert info.width == 1920


class TestPipelineStats:
    """Tests for the PipelineStats dataclass."""

    def test_create(self) -> None:
        stats = PipelineStats(
            frames_processed=100,
            total_time_ms=5000.0,
            unique_tracks=12,
            output_path=None,
        )
        assert stats.unique_tracks == 12
        assert stats.output_path is None


class TestTrackedDetectionsToSupervision:
    """Tests for the tracked_detections_to_supervision converter."""

    def test_converts_correctly(self, sample_tracking_result: TrackingResult) -> None:
        sv_dets = tracked_detections_to_supervision(sample_tracking_result)
        assert isinstance(sv_dets, sv.Detections)
        assert len(sv_dets) == 3
        assert sv_dets.tracker_id is not None
        np.testing.assert_array_equal(sv_dets.tracker_id, [1, 2, 3])

    def test_empty_result(self) -> None:
        result = TrackingResult(
            tracked_detections=[],
            frame_index=0,
            image_shape=(640, 640, 3),
            inference_time_ms=0.0,
        )
        sv_dets = tracked_detections_to_supervision(result)
        assert len(sv_dets) == 0
