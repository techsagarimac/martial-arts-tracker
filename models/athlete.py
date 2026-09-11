"""Athlete profile. No biometric identity and no sensitive documents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


EXPERIENCE_LEVELS = ("Beginner", "Intermediate", "Advanced", "Instructor")
TRAINING_GOALS = (
    "General fitness",
    "Technique accuracy",
    "Speed development",
    "Sparring preparation",
    "Flexibility and form",
)


@dataclass
class Athlete:
    id: int | None
    name: str
    martial_art: str
    experience_level: str = "Beginner"
    training_goal: str = "Technique accuracy"
    age: int | None = None
    height_cm: float | None = None
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now().isoformat(timespec="seconds")
        self.name = (self.name or "").strip() or "Athlete"
