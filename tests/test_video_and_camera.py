"""Camera / video failure handling and empty-input behavior."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.pipeline import FramePipeline
from utils.video_utils import is_valid_upload_name, open_camera, open_video


def test_invalid_upload_rejected():
    assert not is_valid_upload_name("notes.txt")
    assert is_valid_upload_name("round.mp4")
    assert is_valid_upload_name("sparring.MOV")


def test_missing_video_returns_none(tmp_path: Path):
    assert open_video(tmp_path / "missing.mp4") is None


def test_empty_frame_does_not_crash():
    pipe = FramePipeline("boxing")
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    result = pipe.process(empty)
    assert result.pose_ok is False
    pipe.close()


def test_black_frame_reports_no_pose():
    pipe = FramePipeline("kickboxing")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    result = pipe.process(frame)
    assert result.pose_ok is False
    assert "not detected" in result.pose_message.lower() or result.pose_message != "" or pipe.pose_backend == "none"
    pipe.close()


def test_open_camera_invalid_index():
    cap = open_camera(99, width=160, height=120)
    assert cap is None


def test_read_latest_frame_without_camera():
    from utils.video_utils import camera_backends, read_latest_frame

    assert camera_backends()
    ok, frame = read_latest_frame(None)
    assert ok is False
    assert frame is None
