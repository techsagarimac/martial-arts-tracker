"""Frame-to-frame motion: displacements, rotation, and speed estimates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from config.settings import CalibrationSettings
from core.landmark_processor import ProcessedPose
from models.technique import JointAngles, MotionMetrics
from utils.geometry import (
    distance,
    line_angle_deg,
    midpoint,
    nanmean,
    normalize_angle_delta,
    velocity,
)


@dataclass
class PoseSample:
    timestamp: float
    points: dict[str, np.ndarray]
    pixels: dict[str, np.ndarray]
    world_points: dict[str, np.ndarray]
    visibility: dict[str, float]
    angles: JointAngles
    width: int
    height: int


@dataclass
class SpeedReading:
    """Three units of speed. Only estimated_mps is physical, and only if calibrated."""

    pixel_per_sec: float
    normalized_per_sec: float
    estimated_mps: float | None
    label: str


class MotionTracker:
    def __init__(self, window_seconds: float = 0.45, calibration: CalibrationSettings | None = None) -> None:
        self.window_seconds = window_seconds
        self.calibration = calibration or CalibrationSettings()
        self.history: deque[PoseSample] = deque()
        self.peak_normalized_speed = 0.0
        self._last_fps = 0.0

    def reset(self) -> None:
        self.history.clear()
        self.peak_normalized_speed = 0.0
        self._last_fps = 0.0

    def update(self, pose: ProcessedPose, angles: JointAngles, timestamp: float) -> MotionMetrics:
        if not pose.detected:
            return MotionMetrics(fps=self._last_fps)
        sample = PoseSample(
            timestamp=timestamp,
            points=dict(pose.points),
            pixels=dict(pose.pixels),
            world_points=dict(pose.world_points),
            visibility=dict(pose.visibility),
            angles=angles,
            width=pose.image_width,
            height=pose.image_height,
        )
        self.history.append(sample)
        self._trim(timestamp)
        if len(self.history) < 2:
            return MotionMetrics(fps=self._last_fps)

        oldest = self.history[0]
        newest = self.history[-1]
        dt = newest.timestamp - oldest.timestamp
        inst_dt = newest.timestamp - self.history[-2].timestamp
        if inst_dt > 1e-6:
            self._last_fps = 1.0 / inst_dt

        hand = self._pair_disp(oldest, newest, "left_wrist", "right_wrist")
        foot = self._pair_disp(oldest, newest, "left_ankle", "right_ankle")
        knee = self._pair_disp(oldest, newest, "left_knee", "right_knee")
        hip = self._pair_disp(oldest, newest, "left_hip", "right_hip")
        center_old = self._body_center(oldest)
        center_new = self._body_center(newest)
        center_move = distance(center_old, center_new) if center_old is not None else 0.0

        shoulder_old = line_angle_deg(oldest.points.get("left_shoulder"), oldest.points.get("right_shoulder"))
        shoulder_new = line_angle_deg(newest.points.get("left_shoulder"), newest.points.get("right_shoulder"))
        shoulder_rot = abs(normalize_angle_delta(shoulder_new - shoulder_old))

        peak_norm, peak_px = self._peak_limb_speed()
        self.peak_normalized_speed = max(self.peak_normalized_speed, peak_norm)
        estimated = self.estimate_physical_speed(peak_norm, newest)

        return MotionMetrics(
            hand_displacement=hand,
            foot_displacement=foot,
            knee_displacement=knee,
            hip_displacement=hip,
            shoulder_rotation_deg=shoulder_rot if np.isfinite(shoulder_rot) else 0.0,
            body_center_movement=center_move if np.isfinite(center_move) else 0.0,
            pixel_speed=peak_px,
            normalized_speed=peak_norm,
            estimated_mps=estimated,
            peak_normalized_speed=self.peak_normalized_speed,
            fps=self._last_fps,
        )

    def speed_reading(self, metrics: MotionMetrics) -> SpeedReading:
        if metrics.estimated_mps is not None:
            label = "Estimated physical speed (calibration-dependent)"
        else:
            label = "Normalized visual speed (not m/s)"
        return SpeedReading(
            pixel_per_sec=metrics.pixel_speed,
            normalized_per_sec=metrics.normalized_speed,
            estimated_mps=metrics.estimated_mps,
            label=label,
        )

    def estimate_physical_speed(self, normalized_speed: float, sample: PoseSample | None = None) -> float | None:
        """Convert normalized speed to m/s only when a scale is available.

        Uses athlete height vs observed head-to-ankle length, or an explicit
        pixels-per-meter value. Ordinary webcams cannot measure force.
        """
        if normalized_speed is None or not np.isfinite(normalized_speed):
            return None
        sample = sample or (self.history[-1] if self.history else None)
        if sample is None:
            return None
        ppm = self.calibration.pixels_per_meter
        if ppm and ppm > 1:
            # normalized * image_height pixels / sec / pixels_per_meter
            return float(normalized_speed * sample.height / ppm)
        height_cm = self.calibration.athlete_height_cm
        if height_cm and height_cm > 50:
            body_len = self._observed_body_length(sample)
            if body_len and body_len > 0.05:
                meters_per_norm = (height_cm / 100.0) / body_len
                return float(normalized_speed * meters_per_norm)
        return None

    def samples_in_window(self) -> list[PoseSample]:
        return list(self.history)

    def current(self) -> PoseSample | None:
        return self.history[-1] if self.history else None

    def limb_path(self, name: str) -> list[np.ndarray]:
        return [s.points[name] for s in self.history if name in s.points]

    def limb_speed(self, name: str) -> float:
        if len(self.history) < 2:
            return 0.0
        a, b = self.history[-2], self.history[-1]
        if name not in a.points or name not in b.points:
            return 0.0
        dt = b.timestamp - a.timestamp
        value = velocity(a.points[name], b.points[name], dt)
        return 0.0 if value != value else float(value)

    def _trim(self, timestamp: float) -> None:
        cutoff = timestamp - max(self.window_seconds, 0.2)
        while len(self.history) > 2 and self.history[0].timestamp < cutoff:
            self.history.popleft()

    def _pair_disp(self, old: PoseSample, new: PoseSample, left: str, right: str) -> float:
        values = []
        for name in (left, right):
            if name in old.points and name in new.points:
                values.append(distance(old.points[name], new.points[name]))
        return nanmean(values) if values else 0.0

    def _body_center(self, sample: PoseSample) -> np.ndarray | None:
        return midpoint(sample.points.get("left_hip"), sample.points.get("right_hip"))

    def _observed_body_length(self, sample: PoseSample) -> float | None:
        nose = sample.points.get("nose")
        left_ank = sample.points.get("left_ankle")
        right_ank = sample.points.get("right_ankle")
        feet = midpoint(left_ank, right_ank)
        if nose is None or feet is None:
            return None
        return distance(nose, feet)

    def _peak_limb_speed(self) -> tuple[float, float]:
        names = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        peak_n, peak_p = 0.0, 0.0
        samples = list(self.history)
        for i in range(1, len(samples)):
            dt = samples[i].timestamp - samples[i - 1].timestamp
            if dt <= 1e-6:
                continue
            for name in names:
                if name not in samples[i].points or name not in samples[i - 1].points:
                    continue
                n_speed = velocity(samples[i - 1].points[name], samples[i].points[name], dt)
                if n_speed == n_speed:
                    peak_n = max(peak_n, n_speed)
                if name in samples[i].pixels and name in samples[i - 1].pixels:
                    p_speed = velocity(samples[i - 1].pixels[name], samples[i].pixels[name], dt)
                    if p_speed == p_speed:
                        peak_p = max(peak_p, p_speed)
        return peak_n, peak_p
