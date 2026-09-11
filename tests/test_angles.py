"""Angle and distance geometry tests, including missing-landmark cases."""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.angle_calculator import AngleCalculator
from tests.helpers import make_pose, point, standing_guard
from utils.geometry import angle_at_joint, distance, midpoint, velocity


def test_right_angle():
    a = (0.0, 1.0)
    b = (0.0, 0.0)
    c = (1.0, 0.0)
    assert angle_at_joint(a, b, c) == pytest.approx(90.0, abs=1e-6)


def test_straight_angle():
    a = (0.0, 0.0)
    b = (1.0, 0.0)
    c = (2.0, 0.0)
    assert angle_at_joint(a, b, c) == pytest.approx(180.0, abs=1e-6)


def test_angle_missing_point_is_nan():
    assert math.isnan(angle_at_joint(None, (0, 0), (1, 0)))
    assert math.isnan(angle_at_joint((0, 0), (0, 0), (1, 0)))  # degenerate zero-length


def test_distance_and_midpoint():
    assert distance((0, 0), (3, 4)) == pytest.approx(5.0)
    mid = midpoint((0, 0, 0), (2, 2, 2))
    assert mid is not None
    assert np.allclose(mid, [1, 1, 1])


def test_velocity_rejects_zero_dt():
    assert math.isnan(velocity((0, 0), (1, 0), 0.0))
    assert velocity((0, 0), (2, 0), 0.5) == pytest.approx(4.0)


def test_joint_angles_from_standing_pose():
    pose = make_pose(standing_guard())
    angles = AngleCalculator().calculate(pose)
    for value in angles.as_dict().values():
        assert 0 < value < 180
    # Guard elbows should be bent, not locked.
    assert angles.left_elbow < 175
    assert angles.right_elbow < 175


def test_missing_landmarks_yield_nan_angles():
    pose = make_pose({"nose": point(0.5, 0.1)})
    angles = AngleCalculator().calculate(pose)
    assert math.isnan(angles.left_elbow)
    assert math.isnan(angles.left_knee)
