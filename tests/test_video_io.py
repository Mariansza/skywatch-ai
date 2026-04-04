"""Tests for src/tracking/video_io.py."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.tracking.schema import VideoInfo
from src.tracking.video_io import (
    create_video_writer,
    is_video_file,
    iterate_frames,
    open_video,
    write_frame,
)


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """Create a minimal test video (10 frames, 64x64, 30 fps)."""
    video_path = tmp_path / "test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (64, 64))

    for i in range(10):
        frame = np.full((64, 64, 3), i * 25, dtype=np.uint8)
        writer.write(frame)

    writer.release()
    return video_path


class TestOpenVideo:
    """Tests for open_video()."""

    def test_returns_capture_and_info(self, sample_video_path: Path) -> None:
        cap, info = open_video(sample_video_path)
        assert isinstance(info, VideoInfo)
        assert info.width == 64
        assert info.height == 64
        assert info.fps == 30.0
        assert info.total_frames == 10
        cap.release()

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            open_video("/nonexistent/video.mp4")

    def test_invalid_file(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "not_a_video.mp4"
        bad_file.write_text("not a video")
        with pytest.raises(ValueError, match="Failed to open"):
            open_video(bad_file)


class TestIterateFrames:
    """Tests for iterate_frames()."""

    def test_yields_all_frames(self, sample_video_path: Path) -> None:
        cap, info = open_video(sample_video_path)
        frames = list(iterate_frames(cap))
        assert len(frames) == 10

    def test_frames_are_rgb(self, sample_video_path: Path) -> None:
        cap, _ = open_video(sample_video_path)
        frames = list(iterate_frames(cap))
        assert frames[0].shape == (64, 64, 3)
        assert frames[0].dtype == np.uint8


class TestCreateVideoWriter:
    """Tests for create_video_writer()."""

    def test_creates_writer(self, tmp_path: Path) -> None:
        info = VideoInfo(fps=30.0, width=64, height=64, total_frames=10)
        output_path = tmp_path / "output.mp4"
        writer = create_video_writer(output_path, info)
        assert writer.isOpened()
        writer.release()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        info = VideoInfo(fps=30.0, width=64, height=64, total_frames=10)
        output_path = tmp_path / "nested" / "dir" / "output.mp4"
        writer = create_video_writer(output_path, info)
        assert writer.isOpened()
        writer.release()


class TestWriteFrame:
    """Tests for write_frame()."""

    def test_writes_frame(self, tmp_path: Path) -> None:
        info = VideoInfo(fps=30.0, width=64, height=64, total_frames=1)
        output_path = tmp_path / "output.mp4"
        writer = create_video_writer(output_path, info)

        frame_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        write_frame(writer, frame_rgb)
        writer.release()

        # Verify the output file is not empty
        assert output_path.stat().st_size > 0


class TestIsVideoFile:
    """Tests for is_video_file()."""

    def test_video_extensions(self) -> None:
        assert is_video_file("test.mp4") is True
        assert is_video_file("test.avi") is True
        assert is_video_file("test.mov") is True
        assert is_video_file(Path("test.mkv")) is True

    def test_non_video_extensions(self) -> None:
        assert is_video_file("test.jpg") is False
        assert is_video_file("test.png") is False
        assert is_video_file("test.txt") is False
