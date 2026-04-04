"""Tests for src/models/onnx_detector.py."""

from pathlib import Path

import numpy as np
import pytest

from src.models.schema import DetectionResult


@pytest.fixture
def onnx_model_path(tmp_path: Path) -> Path:
    """Export yolov8n to ONNX for testing.

    This fixture is slow (downloads model, runs export) but is needed
    to test the full ONNX pipeline end-to-end.
    """
    from src.models.export import export_to_onnx

    return export_to_onnx(weights_path="yolov8n.pt", output_dir=tmp_path)


@pytest.fixture
def onnx_config(onnx_model_path: Path) -> dict:
    """Config dict pointing to the exported ONNX model."""
    return {
        "model": {
            "weights": str(onnx_model_path),
            "device": "cpu",
        },
        "inference": {
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
            "max_detections": 300,
            "image_size": 640,
        },
    }


@pytest.mark.slow
class TestOnnxDetector:
    """Integration tests for OnnxDetector."""

    def test_predict_returns_detection_result(
        self, onnx_config: dict, sample_image: np.ndarray
    ) -> None:
        from src.models.onnx_detector import OnnxDetector

        detector = OnnxDetector(onnx_config)
        result = detector.predict(sample_image)
        assert isinstance(result, DetectionResult)
        assert result.inference_time_ms > 0

    def test_predict_batch(self, onnx_config: dict, sample_image: np.ndarray) -> None:
        from src.models.onnx_detector import OnnxDetector

        detector = OnnxDetector(onnx_config)
        results = detector.predict_batch([sample_image, sample_image])
        assert len(results) == 2

    def test_high_threshold_returns_fewer(
        self, onnx_config: dict, sample_image: np.ndarray
    ) -> None:
        from src.models.onnx_detector import OnnxDetector

        onnx_config["inference"]["conf_threshold"] = 0.99
        detector = OnnxDetector(onnx_config)
        result = detector.predict(sample_image)
        assert len(result.detections) == 0


@pytest.mark.slow
class TestCreateDetectorFactory:
    """Tests for the create_detector factory."""

    def test_pt_returns_detector(self, sample_config: dict) -> None:
        from src.models.detector import Detector, create_detector

        d = create_detector(sample_config)
        assert isinstance(d, Detector)

    def test_onnx_returns_onnx_detector(self, onnx_config: dict) -> None:
        from src.models.detector import create_detector
        from src.models.onnx_detector import OnnxDetector

        d = create_detector(onnx_config)
        assert isinstance(d, OnnxDetector)
