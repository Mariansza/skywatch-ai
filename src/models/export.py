"""ONNX export utilities for SkyWatch AI.

Converts a PyTorch YOLOv8 model (.pt) to ONNX format (.onnx).
The export is done via ultralytics' built-in export API, which
handles graph optimization (operator fusion, dead code elimination)
and produces a clean, portable ONNX file.

Additionally saves model metadata (class names) to a JSON sidecar
file. The ONNX file itself doesn't carry class name information,
so we extract it from the PyTorch model during export. The ONNX
detector loads this sidecar at inference time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExportConfig:
    """Configuration for ONNX export.

    Attributes:
        simplify: Simplify the ONNX graph (smaller, faster).
        half: Export in FP16 half-precision (smaller, GPU-only).
        int8: Export in INT8 precision (smallest, needs calibration).
        opset: ONNX opset version.
        image_size: Model input size.
    """

    simplify: bool = True
    half: bool = False
    int8: bool = False
    opset: int = 17
    image_size: int = 640


def export_to_onnx(
    weights_path: str | Path,
    output_dir: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    """Export a YOLOv8 PyTorch model to ONNX format.

    Args:
        weights_path: Path to the .pt model file.
        output_dir: Directory for the output .onnx file.
            If None, saves next to the input weights.
        config: Export config dict (from export.yaml).
            Uses defaults if not provided.

    Returns:
        Path to the exported .onnx file.
    """
    weights_path = Path(weights_path)
    export_cfg = _parse_export_config(config)

    logger.info(
        "Exporting %s to ONNX (simplify=%s half=%s int8=%s opset=%d)",
        weights_path.name,
        export_cfg.simplify,
        export_cfg.half,
        export_cfg.int8,
        export_cfg.opset,
    )

    model = YOLO(str(weights_path))

    exported_path = model.export(
        format="onnx",
        simplify=export_cfg.simplify,
        half=export_cfg.half,
        int8=export_cfg.int8,
        opset=export_cfg.opset,
        imgsz=export_cfg.image_size,
    )
    exported_path = Path(exported_path)

    # Move to output_dir if specified
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        dest = output_dir / exported_path.name
        exported_path.rename(dest)
        exported_path = dest

    # Save class names metadata
    meta_path = _save_model_metadata(model, exported_path)

    logger.info("Exported ONNX model to %s", exported_path)
    logger.info("Saved model metadata to %s", meta_path)

    return exported_path


def _save_model_metadata(model: YOLO, onnx_path: Path) -> Path:
    """Save class names and model info to a JSON sidecar file.

    The sidecar is named ``<model>_meta.json`` and lives next to
    the .onnx file. It contains the class name mapping that the
    ONNX detector needs at inference time.
    """
    meta_path = onnx_path.with_name(onnx_path.stem + "_meta.json")

    metadata = {
        "class_names": model.names,
        "num_classes": len(model.names),
        "source_weights": onnx_path.stem,
    }

    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    return meta_path


def load_model_metadata(onnx_path: str | Path) -> dict[int, str]:
    """Load class names from the JSON sidecar file.

    Args:
        onnx_path: Path to the .onnx model file.

    Returns:
        Dictionary mapping class_id (int) to class_name (str).

    Raises:
        FileNotFoundError: If the metadata file doesn't exist.
    """
    onnx_path = Path(onnx_path)
    meta_path = onnx_path.with_name(onnx_path.stem + "_meta.json")

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {meta_path}. "
            f"Re-export the model with scripts/export.py."
        )

    with meta_path.open() as f:
        metadata = json.load(f)

    # JSON keys are strings, convert to int
    return {int(k): v for k, v in metadata["class_names"].items()}


def _parse_export_config(config: dict[str, Any] | None) -> ExportConfig:
    """Parse export config from a dictionary."""
    if config is None:
        return ExportConfig()

    cfg = config.get("export", config)
    return ExportConfig(
        simplify=cfg.get("simplify", True),
        half=cfg.get("half", False),
        int8=cfg.get("int8", False),
        opset=cfg.get("opset", 17),
        image_size=cfg.get("image_size", 640),
    )
