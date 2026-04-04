#!/usr/bin/env python3
"""Run YOLOv8 detection + ByteTrack tracking on video.

Usage:
    python scripts/track.py --source path/to/video.mp4
    python scripts/track.py --source video.mp4 --conf 0.5
    python scripts/track.py --source video.mp4 --config configs/track.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from src.models.detector import Detector
from src.tracking.tracker import Tracker
from src.tracking.video_io import is_video_file
from src.tracking.video_pipeline import VideoPipeline
from src.utils.logger import setup_logger
from src.utils.visualization import annotate_tracks

logger = setup_logger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run detection + tracking on video.",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a video file (.mp4, .avi, .mov, .mkv).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/track.yaml",
        help="Path to YAML config file (default: configs/track.yaml).",
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
        help="Do not save annotated video.",
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


def main(argv: list[str] | None = None) -> int:
    """Main entry point for tracking CLI."""
    args = parse_args(argv)

    source = Path(args.source)
    if not source.exists():
        logger.error("Source not found: %s", source)
        return 1
    if not is_video_file(source):
        logger.error("Not a recognized video format: %s", source)
        return 1

    config = load_config(args.config, args)

    detector = Detector(config)
    tracker = Tracker(config)
    pipeline = VideoPipeline(detector, tracker, config)

    output_dir = Path(config["output"]["output_dir"])
    save = config["output"].get("save_video", True) and not args.no_save

    output_path = None
    annotate_fn = None
    if save:
        output_path = output_dir / source.name
        annotate_fn = annotate_tracks

    stats = pipeline.process_video(
        source=source,
        output_path=output_path,
        annotate_fn=annotate_fn,
    )

    logger.info(
        "Done — %d frames, %d unique tracks, %.1f s (%.1f FPS)",
        stats.frames_processed,
        stats.unique_tracks,
        stats.total_time_ms / 1000,
        stats.frames_processed / max(stats.total_time_ms / 1000, 0.001),
    )

    if stats.output_path:
        logger.info("Output saved to %s", stats.output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
