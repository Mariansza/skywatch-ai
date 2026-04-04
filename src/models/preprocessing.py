"""Image preprocessing for ONNX inference.

When using ultralytics, preprocessing (resize, normalize, pad) is
handled automatically. With raw ONNX Runtime, we must do it ourselves.
These functions replicate what ultralytics does internally, so the
ONNX backend produces identical results to the PyTorch backend.

The key concept is **letterbox** resizing: instead of stretching the
image to 640x640 (which distorts proportions), we resize to fit inside
640x640 while keeping the aspect ratio, then pad the remaining space
with gray. This preserves object shapes, which matters for detection
accuracy — a stretched car looks different than a real car.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    """Information needed to reverse the letterbox transformation.

    After detection, bounding box coordinates are in the 640x640
    letterboxed space. We need these values to convert them back
    to the original image coordinates.

    Attributes:
        ratio: Scale factor applied during resize.
        pad_w: Horizontal padding added (pixels, one side).
        pad_h: Vertical padding added (pixels, one side).
        original_shape: Original image (H, W) before letterbox.
    """

    ratio: float
    pad_w: float
    pad_h: float
    original_shape: tuple[int, int]


def letterbox(
    image: np.ndarray,
    target_size: int = 640,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize an image with letterbox padding (preserving aspect ratio).

    Example: a 1920x1080 image → scaled to 640x360, then padded
    vertically to 640x640 with gray bars top and bottom.

    Args:
        image: Input image as HWC numpy array (any color space).
        target_size: Target square dimension (default 640).
        color: Padding fill color (default gray 114).

    Returns:
        A tuple of (resized padded image, LetterboxInfo for unscaling).
    """
    h, w = image.shape[:2]

    ratio = min(target_size / h, target_size / w)
    new_w = int(round(w * ratio))
    new_h = int(round(h * ratio))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) / 2
    pad_h = (target_size - new_h) / 2

    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=color,
    )

    info = LetterboxInfo(
        ratio=ratio,
        pad_w=pad_w,
        pad_h=pad_h,
        original_shape=(h, w),
    )

    return padded, info


def preprocess_image(
    image: np.ndarray,
    target_size: int = 640,
) -> tuple[np.ndarray, LetterboxInfo]:
    """Full preprocessing pipeline for ONNX YOLO inference.

    Steps:
    1. Letterbox resize (pad to square, preserve aspect ratio)
    2. HWC → CHW (height/width/channels → channels/height/width)
       ONNX models expect channels-first layout.
    3. Normalize pixel values: [0, 255] → [0.0, 1.0]
    4. Add batch dimension: (3, 640, 640) → (1, 3, 640, 640)

    Args:
        image: Input image as HWC RGB uint8 numpy array.
        target_size: Model input size (default 640).

    Returns:
        A tuple of (preprocessed float32 tensor, LetterboxInfo).
    """
    padded, info = letterbox(image, target_size)

    # HWC → CHW
    blob = padded.transpose(2, 0, 1)

    # Normalize [0, 255] → [0.0, 1.0]
    blob = blob.astype(np.float32) / 255.0

    # Add batch dimension
    blob = np.expand_dims(blob, axis=0)

    # Ensure contiguous memory layout for ONNX Runtime
    blob = np.ascontiguousarray(blob)

    return blob, info
