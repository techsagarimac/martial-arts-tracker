"""Synthetic pose helpers shared by unit tests (no camera required)."""

from __future__ import annotations

import numpy as np

from core.angle_calculator import AngleCalculator
from core.landmark_processor import ProcessedPose
from core.motion_tracker import MotionTracker
from models.technique import Landmark


def point(x: float, y: float, z: float = 0.0) -> np.ndarray:
    return np.array([x, y, z], dtype=float)


def make_pose(points: dict[str, np.ndarray], detected: bool = True) -> ProcessedPose:
    vis = {k: 0.95 for k in points}
    pixels = {k: np.array([v[0] * 640, v[1] * 480]) for k, v in points.items()}
    return ProcessedPose(
        detected=detected,
        message="" if detected else "Pose not detected clearly",
        average_visibility=0.95 if detected else 0.1,
        points=points,
        pixels=pixels,
        visibility=vis,
        world_points=dict(points),
        image_width=640,
        image_height=480,
    )


def standing_guard(offset_x: float = 0.0, lead: str = "left") -> dict[str, np.ndarray]:
    """Frontal standing pose. Elbows ~90° so extension during a punch is measurable."""
    z_lead = -0.08 if lead == "left" else 0.05
    z_rear = 0.05 if lead == "left" else -0.08
    return {
        "nose": point(0.50 + offset_x, 0.10, 0.0),
        "left_shoulder": point(0.38 + offset_x, 0.26, z_lead),
        "right_shoulder": point(0.62 + offset_x, 0.26, z_rear),
        "left_elbow": point(0.38 + offset_x, 0.46, z_lead),
        "right_elbow": point(0.62 + offset_x, 0.46, z_rear),
        "left_wrist": point(0.52 + offset_x, 0.46, z_lead - 0.03),
        "right_wrist": point(0.48 + offset_x, 0.46, z_rear - 0.03),
        "left_hip": point(0.43 + offset_x, 0.54, z_lead),
        "right_hip": point(0.57 + offset_x, 0.54, z_rear),
        "left_knee": point(0.42 + offset_x, 0.74, z_lead),
        "right_knee": point(0.58 + offset_x, 0.74, z_rear),
        "left_ankle": point(0.36 + offset_x, 0.93, z_lead),
        "right_ankle": point(0.64 + offset_x, 0.93, z_rear),
    }


def _lerp_points(a: dict[str, np.ndarray], b: dict[str, np.ndarray], t: float) -> dict[str, np.ndarray]:
    return {k: a[k] * (1 - t) + b[k] * t for k in a}


def punch_sequence(side: str = "left", kind: str = "jab", n: int = 10) -> list[ProcessedPose]:
    start = standing_guard(lead="left")
    end = standing_guard(lead="left")
    if kind == "hook":
        # Lateral swing, elbow stays near 90°.
        if side == "left":
            end["left_elbow"] = point(0.30, 0.46, -0.08)
            end["left_wrist"] = point(0.78, 0.44, -0.10)
        else:
            end["right_elbow"] = point(0.70, 0.46, 0.05)
            end["right_wrist"] = point(0.22, 0.44, 0.02)
    else:
        if side == "left":
            end["left_elbow"] = point(0.42, 0.30, -0.18)
            end["left_wrist"] = point(0.48, 0.32, -0.35)
        else:
            end["right_elbow"] = point(0.58, 0.30, -0.18)
            end["right_wrist"] = point(0.52, 0.32, -0.35)
        if kind == "cross" and side == "right":
            end["left_shoulder"] = point(0.38, 0.26, 0.10)
            end["right_shoulder"] = point(0.62, 0.26, -0.12)
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        frames.append(make_pose(_lerp_points(start, end, t)))
    return frames


def kick_sequence(side: str = "right", kind: str = "front_kick", n: int = 12) -> list[ProcessedPose]:
    start = standing_guard()
    chamber = standing_guard()
    finish = standing_guard()
    if kind == "front_kick":
        chamber[f"{side}_knee"] = point(0.58 if side == "right" else 0.42, 0.58, -0.08)
        chamber[f"{side}_ankle"] = point(0.60 if side == "right" else 0.40, 0.48, -0.10)
        finish[f"{side}_knee"] = point(0.60 if side == "right" else 0.40, 0.50, -0.12)
        finish[f"{side}_ankle"] = point(0.62 if side == "right" else 0.38, 0.38, -0.28)
    elif kind == "roundhouse_kick":
        chamber[f"{side}_knee"] = point(0.70, 0.60, 0.0) if side == "right" else point(0.30, 0.60, 0.0)
        chamber[f"{side}_ankle"] = point(0.72, 0.46, 0.0) if side == "right" else point(0.28, 0.46, 0.0)
        chamber["left_hip"] = point(0.40, 0.54, -0.08)
        chamber["right_hip"] = point(0.62, 0.52, 0.05)
        finish[f"{side}_knee"] = point(0.80, 0.48, 0.0) if side == "right" else point(0.20, 0.48, 0.0)
        finish[f"{side}_ankle"] = point(0.96, 0.40, 0.04) if side == "right" else point(0.04, 0.40, 0.04)
        finish["left_hip"] = point(0.38, 0.54, -0.08)
        finish["right_hip"] = point(0.64, 0.50, 0.08)
    else:
        finish[f"{side}_knee"] = point(0.74, 0.58, 0.0) if side == "right" else point(0.26, 0.58, 0.0)
        finish[f"{side}_ankle"] = point(0.88, 0.56, 0.0) if side == "right" else point(0.12, 0.56, 0.0)
        finish[f"{side}_hip"] = point(0.62, 0.48, 0.05) if side == "right" else point(0.38, 0.48, -0.08)
        chamber = finish
    frames = []
    for i in range(n):
        t = i / max(n - 1, 1)
        if t < 0.45:
            pts = _lerp_points(start, chamber, t / 0.45)
        else:
            pts = _lerp_points(chamber, finish, (t - 0.45) / 0.55)
        frames.append(make_pose(pts))
    return frames


def feed_tracker(frames: list[ProcessedPose], dt: float = 1 / 30) -> MotionTracker:
    tracker = MotionTracker(window_seconds=0.6)
    calc = AngleCalculator()
    t = 0.0
    for pose in frames:
        angles = calc.calculate(pose)
        tracker.update(pose, angles, t)
        t += dt
    return tracker


def landmarks_from_points(points: dict[str, np.ndarray]) -> dict[str, Landmark]:
    out = {}
    for name, p in points.items():
        out[name] = Landmark(
            x=float(p[0]),
            y=float(p[1]),
            z=float(p[2]) if p.size > 2 else 0.0,
            visibility=0.9,
            px=float(p[0]) * 640,
            py=float(p[1]) * 480,
        )
    return out
