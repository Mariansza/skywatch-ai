#!/usr/bin/env python3
"""Export a YOLOv8 model to ONNX format.

Usage:
    python scripts/export.py --weights yolov8n.pt
    python scripts/export.py --weights yolov8n.pt --half
    python scripts/export.py --weights yolov8n.pt --output-dir models/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.models.export import export_to_onnx
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 model to ONNX format.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to .pt model file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/export.yaml",
        help="Path to YAML config file (default: configs/export.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 half-precision.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export in INT8 precision.",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Skip ONNX graph simplification.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for export CLI."""
    args = parse_args(argv)

    config_path = Path(args.config)
    if config_path.exists():
        with config_path.open() as f:
            config = yaml.safe_load(f)
    else:
        config = {"export": {}}

    # Apply CLI overrides
    export_cfg = config.setdefault("export", {})
    export_cfg["weights"] = args.weights
    if args.half:
        export_cfg["half"] = True
    if args.int8:
        export_cfg["int8"] = True
    if args.no_simplify:
        export_cfg["simplify"] = False

    output_dir = args.output_dir or export_cfg.get("output_dir", "runs/export")

    onnx_path = export_to_onnx(
        weights_path=args.weights,
        output_dir=output_dir,
        config=config,
    )

    logger.info("Export complete: %s", onnx_path)
    logger.info("File size: %.1f MB", onnx_path.stat().st_size / 1024 / 1024)
    return 0


if __name__ == "__main__":
    sys.exit(main())
