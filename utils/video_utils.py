"""Camera and video helpers."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger("video")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def camera_backends() -> list[int]:
    """Prefer the native capture API. The default backend is a common source of flicker."""
    ordered: list[int] = []
    if sys.platform == "darwin":
        ordered.append(int(getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)))
    elif sys.platform.startswith("win"):
        ordered.append(int(getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY)))
        ordered.append(int(getattr(cv2, "CAP_MSMF", cv2.CAP_ANY)))
    else:
        ordered.append(int(getattr(cv2, "CAP_V4L2", cv2.CAP_ANY)))
    ordered.append(int(cv2.CAP_ANY))
    unique: list[int] = []
    for backend in ordered:
        if backend not in unique:
            unique.append(backend)
    return unique


def list_camera_indices(max_index: int = 5) -> list[int]:
    """Probe a small range of camera indexes. Missing cameras are skipped."""
    found: list[int] = []
    for index in range(max_index):
        cap = open_camera(index, width=320, height=240, warmup_frames=2)
        if cap is None:
            continue
        found.append(index)
        cap.release()
    return found


def open_camera(
    index: int,
    width: int = 640,
    height: int = 480,
    fps: int | None = None,
    warmup_frames: int = 8,
) -> cv2.VideoCapture | None:
    """Open a webcam with a stable native backend.

    Do not force FPS unless the caller asks — renegotiating format mid-stream
    is a frequent cause of macOS camera flicker and dropped frames.
    """
    if index < 0 or index > 20:
        logger.warning("Camera index %s is out of range", index)
        return None

    last_error = ""
    for backend in camera_backends():
        cap = None
        try:
            cap = cv2.VideoCapture(index, backend)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            continue
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            continue
        _configure_capture(cap, width=width, height=height, fps=fps)
        if _warmup(cap, warmup_frames):
            logger.info("Opened camera %s with backend %s", index, backend)
            return cap
        cap.release()
        last_error = f"backend {backend} opened but produced no frames"

    logger.warning("Camera index %s is unavailable (%s)", index, last_error or "no backend worked")
    return None


def _configure_capture(
    cap: cv2.VideoCapture,
    width: int,
    height: int,
    fps: int | None,
) -> None:
    try:
        if sys.platform != "darwin":
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:  # noqa: BLE001
        pass
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    # Only hint height; many cameras ignore it or become unstable if both are forced.
    if height > 0 and width >= 800:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    if sys.platform != "darwin":
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:  # noqa: BLE001
            pass


def _warmup(cap: cv2.VideoCapture, frames: int) -> bool:
    ok = False
    for _ in range(max(1, frames)):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return True
    return ok


def camera_is_open(cap: cv2.VideoCapture | None) -> bool:
    if cap is None:
        return False
    try:
        return bool(cap.isOpened())
    except Exception:  # noqa: BLE001
        return False


def read_latest_frame(cap: cv2.VideoCapture | None) -> tuple[bool, np.ndarray | None]:
    """Read one camera frame. Avoid extra grab/retrieve cycles that strobe the feed."""
    if not camera_is_open(cap):
        return False, None
    assert cap is not None
    try:
        ok, frame = cap.read()
        if ok and frame is not None and getattr(frame, "size", 0) > 0:
            return True, frame
        return False, None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Camera read failed: %s", exc)
        return False, None


def open_video(path: str | Path) -> cv2.VideoCapture | None:
    video_path = Path(path)
    if not video_path.exists():
        logger.warning("Video does not exist: %s", video_path)
        return None
    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        logger.warning("Unsupported video extension: %s", video_path.suffix)
        return None
    cap = cv2.VideoCapture(str(video_path))
    if cap is None or not cap.isOpened():
        logger.warning("Could not open video: %s", video_path)
        if cap is not None:
            cap.release()
        return None
    return cap


def video_properties(cap: cv2.VideoCapture) -> dict[str, float | int]:
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = (frame_count / fps) if fps > 1e-3 else 0.0
    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width or max_width <= 0:
        return frame
    scale = max_width / float(width)
    new_size = (max_width, max(1, int(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def iter_frames(
    cap: cv2.VideoCapture,
    max_width: int | None = None,
    frame_skip: int = 0,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame_index, BGR frame). frame_skip=1 means process every other frame."""
    index = 0
    skip = max(0, int(frame_skip))
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if skip and index % (skip + 1) != 0:
            index += 1
            continue
        if max_width:
            frame = resize_frame(frame, max_width)
        yield index, frame
        index += 1


def is_valid_upload_name(filename: str) -> bool:
    return Path(filename).suffix.lower() in VIDEO_EXTENSIONS
