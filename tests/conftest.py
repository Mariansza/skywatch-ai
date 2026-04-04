"""Shared test fixtures for SkyWatch AI."""

from pathlib import Path

import numpy as np
import pytest

from src.models.schema import Detection, DetectionResult
from src.tracking.schema import TrackedDetection, TrackingResult


@pytest.fixture
def sample_image() -> np.ndarray:
    """Synthetic 640x640 RGB image for testing."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_config() -> dict:
    """Minimal detection config for testing."""
    return {
        "model": {
            "weights": "yolov8n.pt",
            "task": "detect",
            "device": "cpu",
        },
        "inference": {
            "conf_threshold": 0.1,
            "iou_threshold": 0.45,
            "max_detections": 300,
            "image_size": 640,
            "classes": None,
        },
        "visualization": {
            "box_thickness": 2,
            "text_scale": 0.5,
            "text_thickness": 1,
            "show_confidence": True,
            "show_labels": True,
        },
    }


@pytest.fixture
def sample_detection() -> Detection:
    """A single pre-built Detection instance."""
    return Detection(
        bbox=np.array([100.0, 150.0, 200.0, 300.0], dtype=np.float32),
        confidence=0.85,
        class_id=0,
        class_name="person",
    )


@pytest.fixture
def sample_detection_result(sample_detection: Detection) -> DetectionResult:
    """A DetectionResult with 3 detections."""
    det1 = sample_detection
    det2 = Detection(
        bbox=np.array([300.0, 100.0, 450.0, 250.0], dtype=np.float32),
        confidence=0.72,
        class_id=2,
        class_name="car",
    )
    det3 = Detection(
        bbox=np.array([50.0, 400.0, 120.0, 500.0], dtype=np.float32),
        confidence=0.61,
        class_id=7,
        class_name="truck",
    )
    return DetectionResult(
        detections=[det1, det2, det3],
        image_path=Path("test_image.jpg"),
        image_shape=(640, 640, 3),
        inference_time_ms=42.5,
    )


@pytest.fixture
def sample_tracked_detection() -> TrackedDetection:
    """A single pre-built TrackedDetection instance."""
    return TrackedDetection(
        bbox=np.array([100.0, 150.0, 200.0, 300.0], dtype=np.float32),
        confidence=0.85,
        class_id=0,
        class_name="person",
        tracker_id=1,
    )


@pytest.fixture
def sample_tracking_result(
    sample_tracked_detection: TrackedDetection,
) -> TrackingResult:
    """A TrackingResult with 3 tracked detections."""
    td1 = sample_tracked_detection
    td2 = TrackedDetection(
        bbox=np.array([300.0, 100.0, 450.0, 250.0], dtype=np.float32),
        confidence=0.72,
        class_id=2,
        class_name="car",
        tracker_id=2,
    )
    td3 = TrackedDetection(
        bbox=np.array([50.0, 400.0, 120.0, 500.0], dtype=np.float32),
        confidence=0.61,
        class_id=7,
        class_name="truck",
        tracker_id=3,
    )
    return TrackingResult(
        tracked_detections=[td1, td2, td3],
        frame_index=0,
        image_shape=(640, 640, 3),
        inference_time_ms=45.0,
    )


@pytest.fixture
def sample_track_config() -> dict:
    """Minimal tracking config for testing."""
    return {
        "model": {
            "weights": "yolov8n.pt",
            "task": "detect",
            "device": "cpu",
        },
        "inference": {
            "conf_threshold": 0.1,
            "iou_threshold": 0.45,
            "max_detections": 300,
            "image_size": 640,
            "classes": None,
        },
        "tracking": {
            "track_activation_threshold": 0.25,
            "lost_track_buffer": 30,
            "minimum_matching_threshold": 0.8,
            "frame_rate": 30,
            "minimum_consecutive_frames": 1,
        },
        "visualization": {
            "box_thickness": 2,
            "text_scale": 0.5,
            "text_thickness": 1,
            "show_confidence": True,
            "show_labels": True,
            "show_track_ids": True,
            "show_traces": False,
        },
    }
