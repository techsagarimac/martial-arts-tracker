"""Pose detection wrappers for MediaPipe Solutions and Tasks APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.request import urlretrieve

import numpy as np

from config.settings import get_settings
from models.technique import Landmark
from utils.logger import get_logger

logger = get_logger("pose")

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

TRACKED_LANDMARKS = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# MediaPipe BlazePose 33-landmark indices.
MP_INDEX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

SKELETON_EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "nose"),
    ("right_shoulder", "nose"),
)


@dataclass
class PoseResult:
    detected: bool
    landmarks: dict[str, Landmark] = field(default_factory=dict)
    world_landmarks: dict[str, Landmark] = field(default_factory=dict)
    average_visibility: float = 0.0
    message: str = ""
    person_count: int = 0


class PoseDetector:
    """Single-person pose estimator.

    Prefers the classic MediaPipe Solutions API. Falls back to the Tasks
    PoseLandmarker API when Solutions is not installed.
    """

    def __init__(
        self,
        min_detection_confidence: float | None = None,
        min_tracking_confidence: float | None = None,
        min_visibility: float | None = None,
    ) -> None:
        settings = get_settings()
        self.min_detection = (
            min_detection_confidence
            if min_detection_confidence is not None
            else settings.min_detection_confidence
        )
        self.min_tracking = (
            min_tracking_confidence
            if min_tracking_confidence is not None
            else settings.min_tracking_confidence
        )
        self.min_visibility = (
            min_visibility if min_visibility is not None else settings.min_visibility
        )
        self._backend = "none"
        self._pose: Any = None
        self._landmarker: Any = None
        self._mp: Any = None
        self._timestamp_ms = 0
        self._init_backend(settings.pose_model_path)

    @property
    def backend(self) -> str:
        return self._backend

    def _init_backend(self, model_path: Any) -> None:
        try:
            import mediapipe as mp

            self._mp = mp
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    enable_segmentation=False,
                    min_detection_confidence=self.min_detection,
                    min_tracking_confidence=self.min_tracking,
                )
                self._backend = "solutions"
                logger.info("Pose backend: MediaPipe Solutions")
                return
        except Exception as exc:  # noqa: BLE001 — backend probe
            logger.warning("MediaPipe Solutions unavailable: %s", exc)

        try:
            self._init_tasks_backend(model_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to initialize any pose backend: %s", exc)
            self._backend = "none"

    def _init_tasks_backend(self, model_path: Any) -> None:
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        path = Path(model_path)
        ensure_pose_model(path)
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_detection,
            min_pose_presence_confidence=self.min_detection,
            min_tracking_confidence=self.min_tracking,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._backend = "tasks"
        logger.info("Pose backend: MediaPipe Tasks")

    def is_ready(self) -> bool:
        return self._backend in {"solutions", "tasks"}

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> PoseResult:
        if frame_bgr is None or frame_bgr.size == 0:
            return PoseResult(detected=False, message="Empty frame")
        if not self.is_ready():
            return PoseResult(
                detected=False,
                message="Pose engine is unavailable. Check MediaPipe installation.",
            )
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        height, width = frame_bgr.shape[:2]
        try:
            if self._backend == "solutions":
                return self._detect_solutions(rgb, width, height)
            return self._detect_tasks(rgb, width, height, timestamp_ms)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pose detection failed: %s", exc)
            return PoseResult(detected=False, message="Pose detection failed")

    def _detect_solutions(self, rgb: np.ndarray, width: int, height: int) -> PoseResult:
        results = self._pose.process(rgb)
        if not results or not results.pose_landmarks:
            return PoseResult(detected=False, person_count=0, message="Pose not detected clearly")
        world = results.pose_world_landmarks
        return self._from_landmark_list(
            results.pose_landmarks.landmark,
            world.landmark if world else None,
            width,
            height,
            person_count=1,
        )

    def _detect_tasks(
        self,
        rgb: np.ndarray,
        width: int,
        height: int,
        timestamp_ms: int | None,
    ) -> PoseResult:
        import mediapipe as mp

        if timestamp_ms is None or timestamp_ms <= self._timestamp_ms:
            self._timestamp_ms += 33
            timestamp_ms = self._timestamp_ms
        else:
            self._timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(image, int(timestamp_ms))
        poses = result.pose_landmarks or []
        if not poses:
            return PoseResult(detected=False, person_count=0, message="Pose not detected clearly")
        person_count = len(poses)
        world_list = None
        if result.pose_world_landmarks:
            world_list = result.pose_world_landmarks[0]
        parsed = self._from_landmark_list(poses[0], world_list, width, height, person_count)
        if person_count > 1:
            parsed.message = (
                "Multiple people visible — tracking the most prominent person only."
            )
        return parsed

    def _from_landmark_list(
        self,
        landmarks: Any,
        world_landmarks: Any,
        width: int,
        height: int,
        person_count: int,
    ) -> PoseResult:
        mapped: dict[str, Landmark] = {}
        visibilities: list[float] = []
        for name, index in MP_INDEX.items():
            if index >= len(landmarks):
                continue
            lm = landmarks[index]
            vis = float(getattr(lm, "visibility", getattr(lm, "presence", 1.0)) or 0.0)
            mapped[name] = Landmark(
                x=float(lm.x),
                y=float(lm.y),
                z=float(getattr(lm, "z", 0.0) or 0.0),
                visibility=vis,
                px=float(lm.x) * width,
                py=float(lm.y) * height,
            )
            if name in TRACKED_LANDMARKS:
                visibilities.append(vis)

        world_mapped: dict[str, Landmark] = {}
        if world_landmarks is not None:
            for name, index in MP_INDEX.items():
                if index >= len(world_landmarks):
                    continue
                lm = world_landmarks[index]
                world_mapped[name] = Landmark(
                    x=float(lm.x),
                    y=float(lm.y),
                    z=float(getattr(lm, "z", 0.0) or 0.0),
                    visibility=float(getattr(lm, "visibility", 1.0) or 1.0),
                )

        avg_vis = float(np.mean(visibilities)) if visibilities else 0.0
        required = [mapped[n].visibility for n in TRACKED_LANDMARKS if n in mapped]
        visible_count = sum(1 for v in required if v >= self.min_visibility)
        if avg_vis < self.min_visibility or visible_count < 8:
            return PoseResult(
                detected=False,
                landmarks=mapped,
                world_landmarks=world_mapped,
                average_visibility=avg_vis,
                person_count=person_count,
                message="Pose not detected clearly",
            )
        return PoseResult(
            detected=True,
            landmarks=mapped,
            world_landmarks=world_mapped,
            average_visibility=avg_vis,
            person_count=person_count,
            message="",
        )

    def close(self) -> None:
        try:
            if self._pose is not None:
                self._pose.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._landmarker is not None:
                self._landmarker.close()
        except Exception:  # noqa: BLE001
            pass
        self._pose = None
        self._landmarker = None
        self._backend = "none"

    def __enter__(self) -> PoseDetector:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


def ensure_pose_model(path: Path) -> Path:
    """Download the lite Pose Landmarker if it is not already on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    logger.info("Downloading pose landmarker model to %s", path)
    urlretrieve(POSE_MODEL_URL, path)
    return path

