"""YOLO output postprocessing for ONNX inference.

The raw ONNX output of YOLOv8 is a tensor of shape [1, 84, 8400].
This is NOT a list of detections — it's a grid of "proposals" that
needs decoding:

  84 = 4 (x_center, y_center, width, height) + 80 (class scores)
  8400 = number of anchor points across the feature maps

This module decodes that raw tensor into a list of Detection objects,
applying confidence filtering and NMS (Non-Maximum Suppression).

NMS explained: When multiple overlapping boxes detect the same object,
NMS keeps only the best one. It measures overlap using IoU (Intersection
over Union) — the area of overlap divided by the area of union. If two
boxes overlap more than the IoU threshold, the lower-confidence one
is removed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from src.models.preprocessing import LetterboxInfo

from src.models.schema import Detection


def decode_yolo_output(
    output: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
    letterbox_info: LetterboxInfo,
    class_names: dict[int, str],
    max_detections: int = 300,
) -> list[Detection]:
    """Decode raw YOLOv8 ONNX output into Detection objects.

    Args:
        output: Raw model output, shape [1, 84, 8400].
        conf_threshold: Minimum confidence to keep a detection.
        iou_threshold: IoU threshold for NMS.
        letterbox_info: Transform info to rescale boxes to original image.
        class_names: Mapping of class_id → class_name.
        max_detections: Maximum number of detections to return.

    Returns:
        List of Detection objects in original image coordinates.
    """
    # [1, 84, 8400] → [8400, 84] — one row per proposal
    predictions = output[0].T

    # Split into coordinates and class scores
    # First 4 values: x_center, y_center, width, height
    # Remaining 80: per-class confidence scores
    boxes_xywh = predictions[:, :4]
    class_scores = predictions[:, 4:]

    # For each proposal, find the best class and its score
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # Filter by confidence threshold
    mask = max_scores > conf_threshold
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    # Convert xywh → xyxy (center format → corner format)
    boxes_xyxy = _xywh_to_xyxy(boxes_xywh)

    # Rescale from letterboxed coordinates to original image coordinates
    boxes_xyxy = _rescale_boxes(boxes_xyxy, letterbox_info)

    # Apply NMS using OpenCV (no extra dependency needed)
    indices = _nms(boxes_xyxy, max_scores, iou_threshold)

    # Limit to max_detections
    indices = indices[:max_detections]

    # Build Detection objects
    detections: list[Detection] = []
    for i in indices:
        cid = int(class_ids[i])
        detections.append(
            Detection(
                bbox=boxes_xyxy[i].astype(np.float32),
                confidence=float(max_scores[i]),
                class_id=cid,
                class_name=class_names.get(cid, f"class_{cid}"),
            )
        )

    return detections


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from center format to corner format.

    [x_center, y_center, width, height] → [x1, y1, x2, y2]
    """
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return xyxy


def _rescale_boxes(
    boxes: np.ndarray,
    info: LetterboxInfo,
) -> np.ndarray:
    """Rescale boxes from letterboxed 640x640 space to original image space.

    Reverses the letterbox transformation: subtract padding, then
    divide by the scale ratio.
    """
    boxes[:, 0] = (boxes[:, 0] - info.pad_w) / info.ratio  # x1
    boxes[:, 1] = (boxes[:, 1] - info.pad_h) / info.ratio  # y1
    boxes[:, 2] = (boxes[:, 2] - info.pad_w) / info.ratio  # x2
    boxes[:, 3] = (boxes[:, 3] - info.pad_h) / info.ratio  # y2

    # Clip to image bounds
    orig_h, orig_w = info.original_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

    return boxes


def _nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> list[int]:
    """Apply Non-Maximum Suppression using OpenCV.

    cv2.dnn.NMSBoxes expects boxes in [x, y, w, h] format,
    so we convert from xyxy before calling it.
    """
    # Convert xyxy → xywh for OpenCV NMS
    xywh = np.empty_like(boxes_xyxy)
    xywh[:, 0] = boxes_xyxy[:, 0]
    xywh[:, 1] = boxes_xyxy[:, 1]
    xywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]  # width
    xywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]  # height

    indices = cv2.dnn.NMSBoxes(
        xywh.tolist(),
        scores.tolist(),
        score_threshold=0.0,  # Already filtered by conf_threshold
        nms_threshold=iou_threshold,
    )

    if len(indices) == 0:
        return []

    return indices.flatten().tolist()
