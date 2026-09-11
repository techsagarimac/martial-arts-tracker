"""Motion, velocity units, calibration, and low-FPS behavior."""

from __future__ import annotations

from config.settings import CalibrationSettings
from core.motion_tracker import MotionTracker
from models.technique import JointAngles
from tests.helpers import feed_tracker, make_pose, punch_sequence, standing_guard
from utils.geometry import velocity
import pytest


def test_hand_displacement_increases_during_punch():
    frames = punch_sequence("left", "jab")
    tracker = feed_tracker(frames, dt=1 / 30)
    sample = tracker.current()
    assert sample is not None
    metrics = tracker.update(frames[-1], sample.angles, sample.timestamp + 1 / 30)
    assert metrics.hand_displacement > 0.02
    assert metrics.normalized_speed > 0.0
    assert metrics.pixel_speed > 0.0
    assert metrics.estimated_mps is None  # uncalibrated


def test_uncalibrated_speed_is_not_mps():
    frames = punch_sequence("right", "cross")
    tracker = feed_tracker(frames)
    sample = tracker.current()
    assert sample is not None
    mps = tracker.estimate_physical_speed(1.0, sample)
    assert mps is None


def test_calibrated_height_produces_estimate():
    frames = punch_sequence("left", "jab")
    tracker = feed_tracker(frames)
    tracker.calibration = CalibrationSettings(athlete_height_cm=175)
    sample = tracker.current()
    mps = tracker.estimate_physical_speed(0.5, sample)
    assert mps is not None
    assert mps > 0


def test_low_fps_still_computes_velocity():
    speed = velocity((0.0, 0.0), (0.2, 0.0), 0.5)  # 2 fps
    assert speed == pytest.approx(0.4)


def test_empty_history_is_safe():
    tracker = MotionTracker()
    metrics = tracker.update(make_pose(standing_guard(), detected=False), JointAngles(), 0.0)
    assert metrics.normalized_speed == 0.0
