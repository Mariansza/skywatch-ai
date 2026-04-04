"""Video I/O utilities for SkyWatch AI.

Wraps OpenCV's VideoCapture and VideoWriter behind simple functions
that return our domain types (VideoInfo) instead of raw OpenCV objects.

OpenCV reads video frames in BGR format (its historical default).
We convert to RGB at the boundary, same as we do for images in
visualization.py — the rest of the codebase works in RGB only.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from collections.abc import Generator

    import numpy as np

from src.tracking.schema import VideoInfo
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}


def open_video(path: str | Path) -> tuple[cv2.VideoCapture, VideoInfo]:
    """Open a video file and return its capture object and metadata.

    Args:
        path: Path to the video file.

    Returns:
        A tuple of (VideoCapture, VideoInfo).

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If OpenCV cannot open the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")

    info = VideoInfo(
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    )

    logger.info(
        "Opened video %s: %dx%d @ %.1f fps (%d frames)",
        path.name,
        info.width,
        info.height,
        info.fps,
        info.total_frames,
    )

    return cap, info


def iterate_frames(
    cap: cv2.VideoCapture,
) -> Generator[np.ndarray, None, None]:
    """Yield video frames as RGB numpy arrays.

    Converts each frame from BGR (OpenCV default) to RGB so the rest
    of the codebase doesn't have to worry about color spaces.

    Args:
        cap: An opened VideoCapture object.

    Yields:
        Frames as numpy arrays in RGB format.
    """
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cap.release()


def create_video_writer(
    path: str | Path,
    info: VideoInfo,
) -> cv2.VideoWriter:
    """Create a video writer for saving annotated output.

    Uses mp4v codec for broad compatibility. Creates parent
    directories if they don't exist.

    Args:
        path: Output file path.
        info: Video metadata (fps, dimensions) to match the source.

    Returns:
        An opened VideoWriter ready for writing frames.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        info.fps,
        (info.width, info.height),
    )

    logger.info(
        "Created video writer: %s (%dx%d @ %.1f fps)",
        path,
        info.width,
        info.height,
        info.fps,
    )
    return writer


def write_frame(writer: cv2.VideoWriter, frame: np.ndarray) -> None:
    """Write an RGB frame to the video writer.

    Converts RGB back to BGR for OpenCV, matching the pattern
    used in save_annotated_image().

    Args:
        writer: An opened VideoWriter.
        frame: Frame in RGB format.
    """
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(bgr)


def is_video_file(path: str | Path) -> bool:
    """Check if a path looks like a video file based on extension.

    Args:
        path: File path to check.

    Returns:
        True if the extension is a known video format.
    """
    return Path(path).suffix.lower() in _VIDEO_EXTENSIONS
