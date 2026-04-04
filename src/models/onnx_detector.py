"""ONNX Runtime detection backend for SkyWatch AI.

Drop-in replacement for the PyTorch Detector — same ``predict()``
interface, same ``DetectionResult`` output, but runs on ONNX Runtime
instead of PyTorch+ultralytics.

Why this matters for edge/defense deployment:
- PyTorch + ultralytics ≈ 2 GB of dependencies
- ONNX Runtime ≈ 50 MB
- Same model accuracy, often faster inference
- Runs on CPU, GPU, ARM (Jetson), even in browsers via WASM

The tradeoff is that we must handle preprocessing and postprocessing
ourselves (see preprocessing.py and postprocessing.py), whereas
ultralytics does it automatically.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import onnxruntime as ort
import yaml

if TYPE_CHECKING:
    import numpy as np

from src.models.export import load_model_metadata
from src.models.postprocessing import decode_yolo_output
from src.models.preprocessing import preprocess_image
from src.models.schema import DetectionResult
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OnnxDetector:
    """ONNX Runtime backend for YOLOv8 object detection.

    Provides the same interface as ``Detector`` so it can be used
    as a drop-in replacement anywhere in the pipeline.

    Args:
        config: Dictionary with ``model`` and ``inference`` sections
            matching the structure of ``configs/detect.yaml``.
            The ``weights`` field should point to a ``.onnx`` file.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        model_cfg = config["model"]
        infer_cfg = config.get("inference", {})

        self._weights = Path(model_cfg["weights"])
        self._conf = infer_cfg.get("conf_threshold", 0.25)
        self._iou = infer_cfg.get("iou_threshold", 0.45)
        self._max_det = infer_cfg.get("max_detections", 300)
        self._imgsz = infer_cfg.get("image_size", 640)

        # Select execution provider based on config
        device = model_cfg.get("device", "auto")
        providers = _select_providers(device)

        logger.info(
            "Loading ONNX model %s (providers=%s conf=%.2f iou=%.2f)",
            self._weights.name,
            [p if isinstance(p, str) else p[0] for p in providers],
            self._conf,
            self._iou,
        )

        self._session = ort.InferenceSession(str(self._weights), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

        # Load class names from sidecar metadata file
        self._class_names = load_model_metadata(self._weights)

    def predict(self, source: str | Path | np.ndarray) -> DetectionResult:
        """Run detection on a single image.

        Args:
            source: Image file path or numpy array (RGB).

        Returns:
            A ``DetectionResult`` identical in structure to the
            PyTorch Detector output.
        """
        start = time.perf_counter()

        # Load image if path
        if isinstance(source, (str, Path)):
            image = cv2.imread(str(source))
            if image is None:
                raise ValueError(f"Failed to read image: {source}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_path = Path(source)
        else:
            image = source
            image_path = None

        original_shape = image.shape

        # Preprocess: letterbox + normalize + CHW + batch
        blob, letterbox_info = preprocess_image(image, self._imgsz)

        # Run ONNX inference
        outputs = self._session.run(None, {self._input_name: blob})
        raw_output = outputs[0]  # shape: [1, 84, 8400]

        # Decode raw output into Detection objects
        detections = decode_yolo_output(
            output=raw_output,
            conf_threshold=self._conf,
            iou_threshold=self._iou,
            letterbox_info=letterbox_info,
            class_names=self._class_names,
            max_detections=self._max_det,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Detected %d objects in %.1f ms (ONNX)",
            len(detections),
            elapsed_ms,
        )

        return DetectionResult(
            detections=detections,
            image_path=image_path,
            image_shape=original_shape,
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
    def from_config(cls, config_path: str | Path) -> OnnxDetector:
        """Create an OnnxDetector from a YAML config file.

        Args:
            config_path: Path to a YAML configuration file.

        Returns:
            A configured ``OnnxDetector`` instance.
        """
        path = Path(config_path)
        with path.open() as f:
            config = yaml.safe_load(f)
        return cls(config)


def _select_providers(device: str | None) -> list[str]:
    """Select ONNX Runtime execution providers based on device preference.

    Falls back gracefully: if CUDA is requested but not available,
    falls back to CPU. The provider list is ordered by preference.
    """
    available = ort.get_available_providers()

    if device in ("cuda", "gpu"):
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.warning("CUDA provider not available, falling back to CPU")
        return ["CPUExecutionProvider"]

    if device == "cpu":
        return ["CPUExecutionProvider"]

    # "auto" or None — try CUDA first, fall back to CPU
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
