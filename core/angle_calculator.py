"""Joint-angle calculations. Kept out of the UI on purpose."""

from __future__ import annotations

from core.landmark_processor import ProcessedPose
from models.technique import JointAngles
from utils.geometry import angle_at_joint


JOINT_TRIPLETS: dict[str, tuple[str, str, str]] = {
    "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "left_shoulder": ("left_elbow", "left_shoulder", "left_hip"),
    "right_shoulder": ("right_elbow", "right_shoulder", "right_hip"),
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
}


class AngleCalculator:
    def calculate(self, pose: ProcessedPose) -> JointAngles:
        values: dict[str, float] = {}
        for joint, (a, b, c) in JOINT_TRIPLETS.items():
            values[joint] = angle_at_joint(pose.get(a), pose.get(b), pose.get(c))
        return JointAngles(**values)

    def calculate_named(
        self,
        pose: ProcessedPose,
        a: str,
        b: str,
        c: str,
    ) -> float:
        return angle_at_joint(pose.get(a), pose.get(b), pose.get(c))
