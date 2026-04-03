"""Tests for src.models.detector.

These tests load a real YOLOv8 model (yolov8n.pt, ~6MB).
The model is downloaded once and cached by ultralytics.
Marked as slow so they can be skipped in quick CI runs.
"""

import numpy as np
import pytest

from src.models.detector import Detector
from src.models.schema import DetectionResult

slow = pytest.mark.slow


@slow
def test_detector_init(sample_config: dict) -> None:
    detector = Detector(sample_config)
    assert detector is not None


@slow
def test_detector_predict_returns_detection_result(
    sample_config: dict,
    sample_image: np.ndarray,
) -> None:
    detector = Detector(sample_config)
    result = detector.predict(sample_image)
    assert isinstance(result, DetectionResult)
    assert result.image_shape[2] == 3
    assert result.inference_time_ms > 0


@slow
def test_detector_predict_confidence_filter(
    sample_image: np.ndarray,
) -> None:
    low_conf_config = {
        "model": {"weights": "yolov8n.pt", "task": "detect", "device": "cpu"},
        "inference": {"conf_threshold": 0.01, "image_size": 640},
    }
    high_conf_config = {
        "model": {"weights": "yolov8n.pt", "task": "detect", "device": "cpu"},
        "inference": {"conf_threshold": 0.99, "image_size": 640},
    }
    det_low = Detector(low_conf_config)
    det_high = Detector(high_conf_config)

    result_low = det_low.predict(sample_image)
    result_high = det_high.predict(sample_image)

    assert len(result_high.detections) <= len(result_low.detections)


@slow
def test_detector_from_config() -> None:
    detector = Detector.from_config("configs/detect.yaml")
    assert detector is not None
