"""Feature extraction scaffold for a future sequence classifier.

Do not train without a labeled dataset. This module only prepares features.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.dataset_collector import ML_CLASSES


FEATURE_KEYS = (
    "normalized_joint_coordinates",
    "joint_angles",
    "velocities",
    "acceleration",
    "body_orientation",
)

SUGGESTED_MODELS = (
    "Random Forest",
    "SVM",
    "LSTM",
    "Temporal CNN",
    "Transformer sequence classifier",
)


def extract_sequence_features(frames: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Convert a 30–60 frame window into numeric feature matrices."""
    if not frames:
        return {
            "normalized_joint_coordinates": np.zeros((0, 0)),
            "joint_angles": np.zeros((0, 0)),
            "velocities": np.zeros((0,)),
            "acceleration": np.zeros((0,)),
            "body_orientation": np.zeros((0,)),
        }
    coords = []
    angles = []
    speeds = []
    rotations = []
    for frame in frames:
        points = frame.get("points") or {}
        flat: list[float] = []
        for name in sorted(points.keys()):
            flat.extend(points[name])
        coords.append(flat)
        angle_dict = frame.get("angles") or {}
        angles.append([float(angle_dict.get(k, float("nan"))) for k in sorted(angle_dict.keys())])
        speeds.append(float(frame.get("normalized_speed") or 0.0))
        rotations.append(float(frame.get("shoulder_rotation_deg") or 0.0))
    coord_arr = _pad_rows(coords)
    angle_arr = _pad_rows(angles)
    speed_arr = np.asarray(speeds, dtype=float)
    accel = np.diff(speed_arr, prepend=speed_arr[:1])
    return {
        "normalized_joint_coordinates": coord_arr,
        "joint_angles": angle_arr,
        "velocities": speed_arr,
        "acceleration": accel,
        "body_orientation": np.asarray(rotations, dtype=float),
    }


def _pad_rows(rows: list[list[float]]) -> np.ndarray:
    width = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), width), np.nan, dtype=float)
    for i, row in enumerate(rows):
        out[i, : len(row)] = row
    return out


def describe_pipeline() -> str:
    return (
        "ML extension is dataset-first. Collect labeled windows, extract features "
        f"{', '.join(FEATURE_KEYS)}, then train one of: {', '.join(SUGGESTED_MODELS)}. "
        f"Classes: {', '.join(ML_CLASSES)}. No weights are shipped in this MVP."
    )
