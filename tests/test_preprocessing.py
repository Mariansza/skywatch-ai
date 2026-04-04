"""Tests for src/models/preprocessing.py."""

import numpy as np

from src.models.preprocessing import LetterboxInfo, letterbox, preprocess_image


class TestLetterbox:
    """Tests for letterbox resizing."""

    def test_square_image_no_padding(self) -> None:
        """A 640x640 image should not be resized or padded."""
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        result, info = letterbox(image, 640)
        assert result.shape == (640, 640, 3)
        assert info.ratio == 1.0
        assert info.pad_w == 0.0
        assert info.pad_h == 0.0

    def test_landscape_image(self) -> None:
        """A wide image should be padded vertically."""
        image = np.zeros((360, 640, 3), dtype=np.uint8)
        result, info = letterbox(image, 640)
        assert result.shape == (640, 640, 3)
        assert info.ratio == 1.0  # 640/640 = 1.0
        assert info.pad_w == 0.0
        assert info.pad_h > 0  # padded top/bottom

    def test_portrait_image(self) -> None:
        """A tall image should be padded horizontally."""
        image = np.zeros((640, 360, 3), dtype=np.uint8)
        result, info = letterbox(image, 640)
        assert result.shape == (640, 640, 3)
        assert info.pad_w > 0  # padded left/right
        assert info.pad_h == 0.0

    def test_large_image_downscaled(self) -> None:
        """A 1920x1080 image should be scaled down to fit 640x640."""
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result, info = letterbox(image, 640)
        assert result.shape == (640, 640, 3)
        assert info.ratio < 1.0
        assert info.original_shape == (1080, 1920)

    def test_preserves_dtype(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result, _ = letterbox(image, 640)
        assert result.dtype == np.uint8

    def test_letterbox_info_is_frozen(self) -> None:
        info = LetterboxInfo(ratio=0.5, pad_w=10, pad_h=20, original_shape=(100, 200))
        import pytest

        with pytest.raises(AttributeError):
            info.ratio = 1.0  # type: ignore[misc]


class TestPreprocessImage:
    """Tests for the full preprocessing pipeline."""

    def test_output_shape(self) -> None:
        """Output should be (1, 3, 640, 640) — BCHW format."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        blob, _ = preprocess_image(image, 640)
        assert blob.shape == (1, 3, 640, 640)

    def test_output_dtype(self) -> None:
        """Output should be float32."""
        image = np.zeros((640, 640, 3), dtype=np.uint8)
        blob, _ = preprocess_image(image, 640)
        assert blob.dtype == np.float32

    def test_output_range(self) -> None:
        """Pixel values should be in [0.0, 1.0]."""
        image = np.full((640, 640, 3), 255, dtype=np.uint8)
        blob, _ = preprocess_image(image, 640)
        assert blob.min() >= 0.0
        assert blob.max() <= 1.0

    def test_contiguous_memory(self) -> None:
        """ONNX Runtime needs contiguous arrays."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        blob, _ = preprocess_image(image, 640)
        assert blob.flags["C_CONTIGUOUS"]

    def test_returns_letterbox_info(self) -> None:
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        _, info = preprocess_image(image, 640)
        assert isinstance(info, LetterboxInfo)
        assert info.original_shape == (1080, 1920)
