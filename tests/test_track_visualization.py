"""Tests for tracking visualization (annotate_tracks)."""

import numpy as np

from src.tracking.schema import TrackingResult
from src.utils.visualization import annotate_tracks


class TestAnnotateTracks:
    """Tests for annotate_tracks()."""

    def test_returns_new_image(
        self, sample_image: np.ndarray, sample_tracking_result: TrackingResult
    ) -> None:
        annotated = annotate_tracks(sample_image, sample_tracking_result)
        assert annotated.shape == sample_image.shape
        assert annotated is not sample_image

    def test_does_not_mutate_input(
        self, sample_image: np.ndarray, sample_tracking_result: TrackingResult
    ) -> None:
        original = sample_image.copy()
        annotate_tracks(sample_image, sample_tracking_result)
        np.testing.assert_array_equal(sample_image, original)

    def test_empty_tracking_result(self, sample_image: np.ndarray) -> None:
        empty = TrackingResult(
            tracked_detections=[],
            frame_index=0,
            image_shape=(640, 640, 3),
            inference_time_ms=0.0,
        )
        annotated = annotate_tracks(sample_image, empty)
        assert annotated.shape == sample_image.shape

    def test_with_track_ids_disabled(
        self, sample_image: np.ndarray, sample_tracking_result: TrackingResult
    ) -> None:
        config = {"show_track_ids": False, "show_labels": True, "show_confidence": True}
        annotated = annotate_tracks(sample_image, sample_tracking_result, config)
        assert annotated.shape == sample_image.shape

    def test_with_traces_enabled(
        self, sample_image: np.ndarray, sample_tracking_result: TrackingResult
    ) -> None:
        config = {"show_traces": True, "trace_length": 10}
        annotated = annotate_tracks(sample_image, sample_tracking_result, config)
        assert annotated.shape == sample_image.shape
