"""YOLOv8 detection wrapper for SkyWatch AI.

Wraps the ultralytics YOLO API behind a clean, typed interface.
This abstraction exists for three reasons:
1. Testability — calling code can mock Detector without importing ultralytics.
2. Backend swap — ONNX Runtime inference will plug in behind the same interface.
3. API stability — absorbs ultralytics version changes internally.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from ultralytics import YOLO

if TYPE_CHECKING:
    import numpy as np

from src.models.schema import Detection, DetectionResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Detector:
    """Wrapper around YOLOv8 for object detection inference.

    Args:
        config: Dictionary with ``model`` and ``inference`` sections
            matching the structure of ``configs/detect.yaml``.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        model_cfg = config["model"]
        infer_cfg = config.get("inference", {})

        self._weights = model_cfg["weights"]
        self._device = model_cfg.get("device", "auto")
        if self._device == "auto":
            self._device = None  # Let ultralytics auto-select

        self._conf = infer_cfg.get("conf_threshold", 0.25)
        self._iou = infer_cfg.get("iou_threshold", 0.45)
        self._max_det = infer_cfg.get("max_detections", 300)
        self._imgsz = infer_cfg.get("image_size", 640)
        self._classes = infer_cfg.get("classes")

        logger.info(
            "Loading model weights=%s device=%s conf=%.2f iou=%.2f",
            self._weights,
            self._device or "auto",
            self._conf,
            self._iou,
        )
        self._model = YOLO(self._weights, task=model_cfg.get("task", "detect"))

    def predict(self, source: str | Path | np.ndarray) -> DetectionResult:
        """Run detection on a single image.

        Args:
            source: Image file path or numpy array (RGB or BGR).

        Returns:
            A ``DetectionResult`` with all detections found.
        """
        start = time.perf_counter()
        results = self._model.predict(
            source,
            conf=self._conf,
            iou=self._iou,
            max_det=self._max_det,
            imgsz=self._imgsz,
            device=self._device,
            classes=self._classes,
            verbose=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        result = results[0]
        boxes = result.boxes
        names = result.names

        detections: list[Detection] = []
        if boxes is not None:
            for i in range(len(boxes)):
                detections.append(
                    Detection(
                        bbox=boxes.xyxy[i].cpu().numpy(),
                        confidence=float(boxes.conf[i].cpu()),
                        class_id=int(boxes.cls[i].cpu()),
                        class_name=names[int(boxes.cls[i].cpu())],
                    )
                )

        image_path = Path(source) if isinstance(source, str) else None
        image_shape = result.orig_shape + (3,)

        logger.info(
            "Detected %d objects in %.1f ms",
            len(detections),
            elapsed_ms,
        )

        return DetectionResult(
            detections=detections,
            image_path=image_path,
            image_shape=image_shape,
            inference_time_ms=elapsed_ms,
        )

    def predict_batch(
        self, sources: list[str | Path | np.ndarray]
    ) -> list[DetectionResult]:
        """Run detection on multiple images.

        Args:
            sources: List of image file paths or numpy arrays.

        Returns:
            A list of ``DetectionResult``, one per input image.
        """
        return [self.predict(s) for s in sources]

    @classmethod
    def from_config(cls, config_path: str | Path) -> Detector:
        """Create a Detector from a YAML config file.

        Args:
            config_path: Path to a YAML configuration file.

        Returns:
            A configured ``Detector`` instance.
        """
        path = Path(config_path)
        with path.open() as f:
            config = yaml.safe_load(f)
        return cls(config)


def create_detector(config: dict[str, Any]) -> Detector:
    """Factory that selects the right backend based on weights file extension.

    - ``.pt`` → PyTorch backend (Detector, requires ultralytics)
    - ``.onnx`` → ONNX Runtime backend (OnnxDetector, lightweight)

    This lets you switch backends by changing one line in the config:
    ``weights: "yolov8n.onnx"`` instead of ``weights: "yolov8n.pt"``

    Args:
        config: Dictionary with ``model`` and ``inference`` sections.

    Returns:
        A Detector or OnnxDetector instance.
    """
    weights = config["model"]["weights"]
    if weights.endswith(".onnx"):
        from src.models.onnx_detector import OnnxDetector

        return OnnxDetector(config)  # type: ignore[return-value]
    return Detector(config)
