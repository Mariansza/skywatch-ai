"""Tests for src.models.schema."""

import numpy as np
import pytest
import supervision as sv

from src.models.schema import Detection, DetectionResult, detections_to_supervision


def test_detection_creation() -> None:
    det = Detection(
        bbox=np.array([10.0, 20.0, 30.0, 40.0]),
        confidence=0.9,
        class_id=1,
        class_name="car",
    )
    assert det.confidence == 0.9
    assert det.class_id == 1
    assert det.class_name == "car"
    assert det.bbox.shape == (4,)


def test_detection_immutability() -> None:
    det = Detection(
        bbox=np.array([10.0, 20.0, 30.0, 40.0]),
        confidence=0.9,
        class_id=1,
        class_name="car",
    )
    with pytest.raises(AttributeError):
        det.confidence = 0.5  # type: ignore[misc]


def test_detection_result_creation(
    sample_detection_result: DetectionResult,
) -> None:
    assert len(sample_detection_result.detections) == 3
    assert sample_detection_result.image_shape == (640, 640, 3)
    assert sample_detection_result.inference_time_ms == 42.5


def test_detections_to_supervision(
    sample_detection_result: DetectionResult,
) -> None:
    sv_dets = detections_to_supervision(sample_detection_result)
    assert isinstance(sv_dets, sv.Detections)
    assert sv_dets.xyxy.shape == (3, 4)
    assert sv_dets.confidence is not None
    assert len(sv_dets.confidence) == 3
    assert sv_dets.class_id is not None
    assert len(sv_dets.class_id) == 3


def test_detections_to_supervision_empty() -> None:
    empty_result = DetectionResult(
        detections=[],
        image_path=None,
        image_shape=(640, 640, 3),
        inference_time_ms=0.0,
    )
    sv_dets = detections_to_supervision(empty_result)
    assert isinstance(sv_dets, sv.Detections)
    assert len(sv_dets) == 0
