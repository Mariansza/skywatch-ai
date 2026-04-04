#!/usr/bin/env python3
"""Benchmark PyTorch vs ONNX Runtime inference speed.

Compares the two backends on the same images to measure latency
differences. Useful for deciding which backend to deploy on edge.

Usage:
    python scripts/benchmark.py --source path/to/image.jpg
    python scripts/benchmark.py --source path/to/images/ --runs 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.models.detector import Detector, create_detector
from src.utils.logger import setup_logger
from src.utils.visualization import load_image

logger = setup_logger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark PyTorch vs ONNX inference.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image file or directory.",
    )
    parser.add_argument(
        "--weights-pt",
        type=str,
        default="yolov8n.pt",
        help="PyTorch model weights (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--weights-onnx",
        type=str,
        required=True,
        help="ONNX model weights.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of inference runs per image (default: 5).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup runs before timing (default: 2).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Save results to JSON file.",
    )
    return parser.parse_args(argv)


def collect_images(source: str) -> list[Path]:
    """Collect image paths from a file or directory."""
    source_path = Path(source)
    if source_path.is_file():
        return [source_path]
    if source_path.is_dir():
        return sorted(
            p for p in source_path.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    return []


def benchmark_backend(
    detector: Detector,
    images: list[np.ndarray],
    runs: int,
    warmup: int,
) -> list[float]:
    """Run inference multiple times and collect timing data.

    Returns list of inference times in ms (excluding warmup).
    """
    # Warmup runs — the first few inferences are slower due to
    # memory allocation, JIT compilation, etc.
    for _ in range(warmup):
        for img in images:
            detector.predict(img)

    # Timed runs
    times: list[float] = []
    for _ in range(runs):
        for img in images:
            result = detector.predict(img)
            times.append(result.inference_time_ms)

    return times


def compute_stats(times: list[float]) -> dict[str, float]:
    """Compute summary statistics from timing data."""
    arr = np.array(times)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p95_ms": float(np.percentile(arr, 95)),
        "n_runs": len(times),
    }


def main(argv: list[str] | None = None) -> int:
    """Main entry point for benchmark CLI."""
    args = parse_args(argv)

    image_paths = collect_images(args.source)
    if not image_paths:
        logger.error("No images found at %s", args.source)
        return 1

    images = [load_image(p) for p in image_paths]
    logger.info("Loaded %d images", len(images))

    base_config = {
        "model": {"device": "cpu"},
        "inference": {
            "conf_threshold": args.conf,
            "iou_threshold": 0.45,
            "max_detections": 300,
            "image_size": 640,
        },
    }

    # PyTorch benchmark
    logger.info("Benchmarking PyTorch (%s)...", args.weights_pt)
    pt_model = {**base_config["model"], "weights": args.weights_pt}
    pt_config = {**base_config, "model": pt_model}
    pt_detector = create_detector(pt_config)
    pt_times = benchmark_backend(pt_detector, images, args.runs, args.warmup)
    pt_stats = compute_stats(pt_times)

    # ONNX benchmark
    logger.info("Benchmarking ONNX (%s)...", args.weights_onnx)
    onnx_model = {**base_config["model"], "weights": args.weights_onnx}
    onnx_config = {**base_config, "model": onnx_model}
    onnx_detector = create_detector(onnx_config)
    onnx_times = benchmark_backend(onnx_detector, images, args.runs, args.warmup)
    onnx_stats = compute_stats(onnx_times)

    # Report
    speedup = pt_stats["mean_ms"] / max(onnx_stats["mean_ms"], 0.001)

    logger.info("")
    logger.info("=" * 55)
    logger.info("BENCHMARK RESULTS")
    logger.info("=" * 55)
    logger.info(
        "%-12s %8s %8s %8s %8s",
        "Backend",
        "Mean",
        "Std",
        "Min",
        "P95",
    )
    logger.info("-" * 55)
    logger.info(
        "%-12s %7.1fms %7.1fms %7.1fms %7.1fms",
        "PyTorch",
        pt_stats["mean_ms"],
        pt_stats["std_ms"],
        pt_stats["min_ms"],
        pt_stats["p95_ms"],
    )
    logger.info(
        "%-12s %7.1fms %7.1fms %7.1fms %7.1fms",
        "ONNX",
        onnx_stats["mean_ms"],
        onnx_stats["std_ms"],
        onnx_stats["min_ms"],
        onnx_stats["p95_ms"],
    )
    logger.info("-" * 55)
    logger.info("Speedup: %.2fx", speedup)
    logger.info("=" * 55)

    if args.save_json:
        results = {
            "pytorch": pt_stats,
            "onnx": onnx_stats,
            "speedup": speedup,
            "images": len(images),
            "runs_per_image": args.runs,
        }
        json_path = Path(args.save_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Results saved to %s", json_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
