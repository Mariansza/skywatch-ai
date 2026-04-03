"""Tests for src.utils.visualization."""

from pathlib import Path

import numpy as np
import pytest

from src.models.schema import DetectionResult
from src.utils.visualization import (
    annotate_detections,
    load_image,
    save_annotated_image,
)


def test_annotate_detections_returns_image(
    sample_image: np.ndarray,
    sample_detection_result: DetectionResult,
) -> None:
    annotated = annotate_detections(sample_image, sample_detection_result)
    assert annotated.shape == sample_image.shape
    assert annotated.dtype == sample_image.dtype


def test_annotate_detections_does_not_mutate_input(
    sample_image: np.ndarray,
    sample_detection_result: DetectionResult,
) -> None:
    original = sample_image.copy()
    annotate_detections(sample_image, sample_detection_result)
    np.testing.assert_array_equal(sample_image, original)


def test_annotate_detections_empty_detections(
    sample_image: np.ndarray,
) -> None:
    empty_result = DetectionResult(
        detections=[],
        image_path=None,
        image_shape=(640, 640, 3),
        inference_time_ms=0.0,
    )
    annotated = annotate_detections(sample_image, empty_result)
    assert annotated.shape == sample_image.shape


def test_save_annotated_image_creates_file(
    sample_image: np.ndarray,
    tmp_path: Path,
) -> None:
    output = tmp_path / "subdir" / "output.jpg"
    result = save_annotated_image(sample_image, output)
    assert result.exists()
    assert result == output


def test_load_image_file_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_image("/nonexistent/path/image.jpg")


def test_load_image_invalid_file(tmp_path: Path) -> None:
    bad_file = tmp_path / "not_an_image.txt"
    bad_file.write_text("this is not an image")
    with pytest.raises(ValueError, match="Failed to decode"):
        load_image(bad_file)
