"""Convert raw pose landmarks into arrays the rest of the pipeline can use."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.pose_detector import TRACKED_LANDMARKS, PoseResult
from models.technique import Landmark


@dataclass
class ProcessedPose:
    detected: bool
    message: str
    average_visibility: float
    points: dict[str, np.ndarray]
    pixels: dict[str, np.ndarray]
    visibility: dict[str, float]
    world_points: dict[str, np.ndarray]
    image_width: int
    image_height: int

    def get(self, name: str) -> np.ndarray | None:
        return self.points.get(name)

    def visible(self, name: str, threshold: float) -> bool:
        return self.visibility.get(name, 0.0) >= threshold

    def require(self, names: tuple[str, ...], threshold: float) -> bool:
        return all(self.visible(name, threshold) for name in names)


def _lm_to_xy(lm: Landmark) -> np.ndarray:
    return np.array([lm.x, lm.y], dtype=float)


def _lm_to_xyz(lm: Landmark) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=float)


def _lm_to_px(lm: Landmark) -> np.ndarray:
    return np.array([lm.px, lm.py], dtype=float)


class LandmarkProcessor:
    def __init__(self, min_visibility: float = 0.5) -> None:
        self.min_visibility = min_visibility

    def process(self, pose: PoseResult, image_width: int, image_height: int) -> ProcessedPose:
        points: dict[str, np.ndarray] = {}
        pixels: dict[str, np.ndarray] = {}
        visibility: dict[str, float] = {}
        world_points: dict[str, np.ndarray] = {}

        for name, lm in pose.landmarks.items():
            visibility[name] = lm.visibility
            if lm.visibility >= self.min_visibility * 0.4:
                points[name] = _lm_to_xyz(lm)
                pixels[name] = _lm_to_px(lm)

        for name, lm in pose.world_landmarks.items():
            world_points[name] = _lm_to_xyz(lm)

        body_ok = pose.detected and all(
            visibility.get(name, 0.0) >= self.min_visibility for name in ("left_hip", "right_hip")
        )
        tracked_visible = sum(
            1 for name in TRACKED_LANDMARKS if visibility.get(name, 0.0) >= self.min_visibility
        )
        if not body_ok or tracked_visible < 8:
            return ProcessedPose(
                detected=False,
                message=pose.message or "Pose not detected clearly",
                average_visibility=pose.average_visibility,
                points=points,
                pixels=pixels,
                visibility=visibility,
                world_points=world_points,
                image_width=image_width,
                image_height=image_height,
            )
        return ProcessedPose(
            detected=True,
            message=pose.message,
            average_visibility=pose.average_visibility,
            points=points,
            pixels=pixels,
            visibility=visibility,
            world_points=world_points,
            image_width=image_width,
            image_height=image_height,
        )
