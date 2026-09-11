"""Value objects produced by the vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FeedbackLevel = Literal["green", "yellow", "red", "info"]


@dataclass
class Landmark:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 0.0
    px: float = 0.0
    py: float = 0.0

    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)

    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def pixel(self) -> tuple[float, float]:
        return (self.px, self.py)


@dataclass
class JointAngles:
    left_elbow: float = float("nan")
    right_elbow: float = float("nan")
    left_shoulder: float = float("nan")
    right_shoulder: float = float("nan")
    left_hip: float = float("nan")
    right_hip: float = float("nan")
    left_knee: float = float("nan")
    right_knee: float = float("nan")

    def as_dict(self) -> dict[str, float]:
        return {
            "left_elbow": self.left_elbow,
            "right_elbow": self.right_elbow,
            "left_shoulder": self.left_shoulder,
            "right_shoulder": self.right_shoulder,
            "left_hip": self.left_hip,
            "right_hip": self.right_hip,
            "left_knee": self.left_knee,
            "right_knee": self.right_knee,
        }


@dataclass
class MotionMetrics:
    hand_displacement: float = 0.0
    foot_displacement: float = 0.0
    knee_displacement: float = 0.0
    hip_displacement: float = 0.0
    shoulder_rotation_deg: float = 0.0
    body_center_movement: float = 0.0
    pixel_speed: float = 0.0
    normalized_speed: float = 0.0
    estimated_mps: float | None = None
    peak_normalized_speed: float = 0.0
    fps: float = 0.0


@dataclass
class TechniqueDetection:
    name: str | None
    display_name: str
    category: str | None
    confidence: float
    side: str | None = None
    uncertain: bool = False
    notes: str = ""


@dataclass
class StanceResult:
    label: Literal["GOOD", "NEEDS IMPROVEMENT", "UNKNOWN"]
    score: float
    feedback: list[str] = field(default_factory=list)
    feet_separation: float = float("nan")
    guard_height: float = float("nan")


@dataclass
class FormScores:
    accuracy: float
    alignment: float
    extension: float
    balance: float
    consistency: float
    overall: float
    heuristic_note: str = "Training heuristics, not official competition scores."

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "alignment": self.alignment,
            "extension": self.extension,
            "balance": self.balance,
            "consistency": self.consistency,
            "overall": self.overall,
        }


@dataclass
class FeedbackItem:
    message: str
    level: FeedbackLevel
    key: str
    timestamp: float
