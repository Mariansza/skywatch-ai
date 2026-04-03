#!/usr/bin/env python3
"""Run YOLOv8 object detection on images.

Usage:
    python scripts/detect.py --source path/to/image.jpg
    python scripts/detect.py --source path/to/images/ --conf 0.5
    python scripts/detect.py --source img.jpg --config configs/detect.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.models.detector import Detector
from src.models.schema import DetectionResult
from src.utils.logger import setup_logger
from src.utils.visualization import (
    annotate_detections,
    load_image,
    save_annotated_image,
)

logger = setup_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 detection on images.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to an image file or directory of images.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detect.yaml",
        help="Path to YAML config file (default: configs/detect.yaml).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Override confidence threshold.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cpu, cuda, auto).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save annotated images.",
    )
    return parser.parse_args(argv)


def load_config(config_path: str, args: argparse.Namespace) -> dict:
    """Load YAML config and apply CLI overrides."""
    with Path(config_path).open() as f:
        config = yaml.safe_load(f)

    if args.conf is not None:
        config["inference"]["conf_threshold"] = args.conf
    if args.device is not None:
        config["model"]["device"] = args.device
    if args.output_dir is not None:
        config["output"]["output_dir"] = args.output_dir

    return config


def collect_images(source: str) -> list[Path]:
    """Collect image paths from a file or directory."""
    source_path = Path(source)
    if source_path.is_file():
        return [source_path]
    if source_path.is_dir():
        images = sorted(
            p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            logger.warning("No images found in %s", source_path)
        return images
    logger.error("Source not found: %s", source_path)
    return []


def main(argv: list[str] | None = None) -> int:
    """Main entry point for detection CLI."""
    args = parse_args(argv)
    config = load_config(args.config, args)

    detector = Detector(config)
    images = collect_images(args.source)
    if not images:
        return 1

    output_dir = Path(config["output"]["output_dir"])
    save = config["output"].get("save_images", True) and not args.no_save
    viz_config = config.get("visualization", {})

    total_detections = 0
    total_time_ms = 0.0

    for image_path in images:
        image = load_image(image_path)
        result = detector.predict(image)
        result = DetectionResult(
            detections=result.detections,
            image_path=image_path,
            image_shape=result.image_shape,
            inference_time_ms=result.inference_time_ms,
        )

        total_detections += len(result.detections)
        total_time_ms += result.inference_time_ms

        if save:
            annotated = annotate_detections(image, result, viz_config)
            out_path = output_dir / image_path.name
            save_annotated_image(annotated, out_path)

    avg_time = total_time_ms / len(images) if images else 0
    logger.info(
        "Done — %d images, %d detections, avg %.1f ms/image",
        len(images),
        total_detections,
        avg_time,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
