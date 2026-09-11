"""Reusable geometry helpers for pose landmarks.

All functions are pure and tolerate missing or degenerate inputs by returning
NaN / empty arrays rather than raising.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

EPS = 1e-8


def as_vector(point: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    if point is None:
        return None
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return None
    return arr


def distance(a: Sequence[float] | np.ndarray | None, b: Sequence[float] | np.ndarray | None) -> float:
    va, vb = as_vector(a), as_vector(b)
    if va is None or vb is None:
        return float("nan")
    n = min(va.size, vb.size)
    return float(np.linalg.norm(va[:n] - vb[:n]))


def midpoint(a: Sequence[float] | np.ndarray | None, b: Sequence[float] | np.ndarray | None) -> np.ndarray | None:
    va, vb = as_vector(a), as_vector(b)
    if va is None or vb is None:
        return None
    n = min(va.size, vb.size)
    return (va[:n] + vb[:n]) / 2.0


def displacement(
    start: Sequence[float] | np.ndarray | None,
    end: Sequence[float] | np.ndarray | None,
) -> np.ndarray | None:
    va, vb = as_vector(start), as_vector(end)
    if va is None or vb is None:
        return None
    n = min(va.size, vb.size)
    return vb[:n] - va[:n]


def velocity(
    start: Sequence[float] | np.ndarray | None,
    end: Sequence[float] | np.ndarray | None,
    dt_seconds: float,
) -> float:
    """Scalar speed = distance / dt. Returns NaN if dt is non-positive."""
    if dt_seconds is None or dt_seconds <= EPS:
        return float("nan")
    dist = distance(start, end)
    if math.isnan(dist):
        return float("nan")
    return dist / dt_seconds


def velocity_vector(
    start: Sequence[float] | np.ndarray | None,
    end: Sequence[float] | np.ndarray | None,
    dt_seconds: float,
) -> np.ndarray | None:
    if dt_seconds is None or dt_seconds <= EPS:
        return None
    delta = displacement(start, end)
    if delta is None:
        return None
    return delta / dt_seconds


def angle_at_joint(
    a: Sequence[float] | np.ndarray | None,
    b: Sequence[float] | np.ndarray | None,
    c: Sequence[float] | np.ndarray | None,
) -> float:
    """Interior angle ABC in degrees, where B is the joint vertex."""
    va, vb, vc = as_vector(a), as_vector(b), as_vector(c)
    if va is None or vb is None or vc is None:
        return float("nan")
    n = min(va.size, vb.size, vc.size)
    ba = va[:n] - vb[:n]
    bc = vc[:n] - vb[:n]
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na < EPS or nc < EPS:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def line_angle_deg(a: Sequence[float] | np.ndarray | None, b: Sequence[float] | np.ndarray | None) -> float:
    """Angle of vector A→B versus the +X axis, in degrees (−180, 180]."""
    delta = displacement(a, b)
    if delta is None or delta.size < 2:
        return float("nan")
    return float(np.degrees(np.arctan2(delta[1], delta[0])))


def normalize_angle_delta(delta_deg: float) -> float:
    """Wrap an angle difference into (−180, 180]."""
    if math.isnan(delta_deg):
        return float("nan")
    wrapped = (delta_deg + 180.0) % 360.0 - 180.0
    return wrapped


def clamp(value: float, lo: float, hi: float) -> float:
    if math.isnan(value):
        return float("nan")
    return max(lo, min(hi, value))


def nanmean(values: Sequence[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))
