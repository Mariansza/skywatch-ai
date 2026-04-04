"""Tests for src/models/postprocessing.py."""

import numpy as np

from src.models.postprocessing import (
    _nms,
    _rescale_boxes,
    _xywh_to_xyxy,
    decode_yolo_output,
)
from src.models.preprocessing import LetterboxInfo


def _make_mock_output(
    n_proposals: int = 3,
    n_classes: int = 80,
) -> np.ndarray:
    """Create a mock YOLOv8 output tensor [1, 4+n_classes, n_proposals]."""
    output = np.zeros((1, 4 + n_classes, n_proposals), dtype=np.float32)

    for i in range(n_proposals):
        # x_center, y_center, width, height
        output[0, 0, i] = 320 + i * 50  # x_center
        output[0, 1, i] = 320 + i * 30  # y_center
        output[0, 2, i] = 100  # width
        output[0, 3, i] = 80  # height

        # Class scores — set one class high
        class_idx = i % n_classes
        output[0, 4 + class_idx, i] = 0.9 - i * 0.1

    return output


MOCK_CLASS_NAMES = {0: "person", 1: "bicycle", 2: "car"}
IDENTITY_LETTERBOX = LetterboxInfo(
    ratio=1.0, pad_w=0.0, pad_h=0.0, original_shape=(640, 640)
)


class TestXywhToXyxy:
    """Tests for center → corner box conversion."""

    def test_single_box(self) -> None:
        boxes = np.array([[320, 240, 100, 80]], dtype=np.float32)
        xyxy = _xywh_to_xyxy(boxes)
        np.testing.assert_allclose(xyxy, [[270, 200, 370, 280]])

    def test_multiple_boxes(self) -> None:
        boxes = np.array(
            [
                [100, 100, 50, 50],
                [200, 200, 100, 100],
            ],
            dtype=np.float32,
        )
        xyxy = _xywh_to_xyxy(boxes)
        assert xyxy.shape == (2, 4)
        np.testing.assert_allclose(xyxy[0], [75, 75, 125, 125])
        np.testing.assert_allclose(xyxy[1], [150, 150, 250, 250])


class TestRescaleBoxes:
    """Tests for reversing letterbox transformation."""

    def test_no_transform(self) -> None:
        """Identity letterbox should not change coordinates."""
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        result = _rescale_boxes(boxes.copy(), IDENTITY_LETTERBOX)
        np.testing.assert_allclose(result, [[100, 100, 200, 200]])

    def test_with_padding_and_scale(self) -> None:
        """Rescaling should reverse padding and scale."""
        info = LetterboxInfo(
            ratio=0.5, pad_w=10.0, pad_h=20.0, original_shape=(1080, 1920)
        )
        boxes = np.array([[110, 120, 210, 220]], dtype=np.float32)
        result = _rescale_boxes(boxes.copy(), info)
        # (110 - 10) / 0.5 = 200, (120 - 20) / 0.5 = 200
        np.testing.assert_allclose(result[0, :2], [200, 200])

    def test_clips_to_image_bounds(self) -> None:
        info = LetterboxInfo(ratio=1.0, pad_w=0.0, pad_h=0.0, original_shape=(640, 640))
        boxes = np.array([[-50, -50, 700, 700]], dtype=np.float32)
        result = _rescale_boxes(boxes.copy(), info)
        assert result[0, 0] == 0  # x1 clipped
        assert result[0, 1] == 0  # y1 clipped
        assert result[0, 2] == 640  # x2 clipped
        assert result[0, 3] == 640  # y2 clipped


class TestNms:
    """Tests for Non-Maximum Suppression."""

    def test_no_overlap(self) -> None:
        """Non-overlapping boxes should all be kept."""
        boxes = np.array(
            [
                [0, 0, 100, 100],
                [200, 200, 300, 300],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.8])
        indices = _nms(boxes, scores, iou_threshold=0.5)
        assert len(indices) == 2

    def test_full_overlap(self) -> None:
        """Identical boxes — only the highest score kept."""
        boxes = np.array(
            [
                [100, 100, 200, 200],
                [100, 100, 200, 200],
            ],
            dtype=np.float32,
        )
        scores = np.array([0.9, 0.5])
        indices = _nms(boxes, scores, iou_threshold=0.5)
        assert len(indices) == 1
        assert indices[0] == 0  # highest score kept

    def test_empty(self) -> None:
        boxes = np.empty((0, 4), dtype=np.float32)
        scores = np.empty(0, dtype=np.float32)
        indices = _nms(boxes, scores, iou_threshold=0.5)
        assert indices == []


class TestDecodeYoloOutput:
    """Tests for the full decode pipeline."""

    def test_returns_detections(self) -> None:
        output = _make_mock_output(n_proposals=3)
        detections = decode_yolo_output(
            output=output,
            conf_threshold=0.1,
            iou_threshold=0.5,
            letterbox_info=IDENTITY_LETTERBOX,
            class_names=MOCK_CLASS_NAMES,
        )
        assert len(detections) > 0
        assert all(d.confidence > 0.1 for d in detections)

    def test_high_threshold_filters_all(self) -> None:
        output = _make_mock_output(n_proposals=3)
        detections = decode_yolo_output(
            output=output,
            conf_threshold=0.99,
            iou_threshold=0.5,
            letterbox_info=IDENTITY_LETTERBOX,
            class_names=MOCK_CLASS_NAMES,
        )
        assert len(detections) == 0

    def test_detections_have_class_names(self) -> None:
        output = _make_mock_output(n_proposals=1)
        detections = decode_yolo_output(
            output=output,
            conf_threshold=0.1,
            iou_threshold=0.5,
            letterbox_info=IDENTITY_LETTERBOX,
            class_names=MOCK_CLASS_NAMES,
        )
        if detections:
            assert detections[0].class_name in MOCK_CLASS_NAMES.values()

    def test_bbox_is_xyxy(self) -> None:
        output = _make_mock_output(n_proposals=1)
        detections = decode_yolo_output(
            output=output,
            conf_threshold=0.1,
            iou_threshold=0.5,
            letterbox_info=IDENTITY_LETTERBOX,
            class_names=MOCK_CLASS_NAMES,
        )
        if detections:
            bbox = detections[0].bbox
            assert bbox.shape == (4,)
            assert bbox[0] < bbox[2]  # x1 < x2
            assert bbox[1] < bbox[3]  # y1 < y2

    def test_max_detections(self) -> None:
        output = _make_mock_output(n_proposals=500)
        detections = decode_yolo_output(
            output=output,
            conf_threshold=0.01,
            iou_threshold=0.99,
            letterbox_info=IDENTITY_LETTERBOX,
            class_names=MOCK_CLASS_NAMES,
            max_detections=5,
        )
        assert len(detections) <= 5
