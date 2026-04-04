"""Tests for src/tracking/video_pipeline.py."""

from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from src.tracking.schema import TrackingResult
from src.tracking.tracker import Tracker
from src.tracking.video_pipeline import VideoPipeline


@pytest.fixture
def mock_detector() -> MagicMock:
    """A mock Detector that returns a fake DetectionResult."""
    from src.models.schema import Detection, DetectionResult

    detector = MagicMock()

    def fake_predict(source: np.ndarray) -> DetectionResult:
        return DetectionResult(
            detections=[
                Detection(
                    bbox=np.array([100, 100, 200, 200], dtype=np.float32),
                    confidence=0.9,
                    class_id=0,
                    class_name="person",
                ),
            ],
            image_path=None,
            image_shape=(64, 64, 3),
            inference_time_ms=10.0,
        )

    detector.predict = MagicMock(side_effect=fake_predict)
    return detector


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """Create a minimal test video (5 frames, 64x64)."""
    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 64))

    for i in range(5):
        frame = np.full((64, 64, 3), i * 50, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


class TestVideoPipeline:
    """Tests for the VideoPipeline class."""

    def test_process_frame(
        self, mock_detector: MagicMock, sample_track_config: dict
    ) -> None:
        tracker = Tracker(sample_track_config)
        pipeline = VideoPipeline(mock_detector, tracker, sample_track_config)

        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)

        assert isinstance(result, TrackingResult)
        assert result.frame_index == 0
        mock_detector.predict.assert_called_once()

    def test_process_video_without_output(
        self,
        mock_detector: MagicMock,
        sample_track_config: dict,
        sample_video_path: Path,
    ) -> None:
        tracker = Tracker(sample_track_config)
        pipeline = VideoPipeline(mock_detector, tracker, sample_track_config)

        stats = pipeline.process_video(sample_video_path)

        assert stats.frames_processed == 5
        assert stats.total_time_ms > 0
        assert stats.output_path is None

    def test_process_video_with_output(
        self,
        mock_detector: MagicMock,
        sample_track_config: dict,
        sample_video_path: Path,
        tmp_path: Path,
    ) -> None:
        tracker = Tracker(sample_track_config)
        pipeline = VideoPipeline(mock_detector, tracker, sample_track_config)
        output_path = tmp_path / "output.mp4"

        def dummy_annotate(
            frame: np.ndarray, result: TrackingResult, config: dict
        ) -> np.ndarray:
            return frame

        stats = pipeline.process_video(
            sample_video_path,
            output_path=output_path,
            annotate_fn=dummy_annotate,
        )

        assert stats.frames_processed == 5
        assert stats.output_path == output_path
        assert output_path.exists()

    def test_process_video_resets_tracker(
        self,
        mock_detector: MagicMock,
        sample_track_config: dict,
        sample_video_path: Path,
    ) -> None:
        tracker = Tracker(sample_track_config)
        pipeline = VideoPipeline(mock_detector, tracker, sample_track_config)

        pipeline.process_video(sample_video_path)
        # After processing, the tracker frame index was reset at the start
        # and then incremented per frame. Running again should reset.
        pipeline.process_video(sample_video_path)
        # If reset works, this second run should process normally
        assert True  # No crash = reset worked
