"""Training session and per-technique event records."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TechniqueEvent:
    id: int | None = None
    session_id: int | None = None
    detected_at: str = ""
    technique_name: str = ""
    category: str = ""
    score: float = 0.0
    accuracy: float = 0.0
    alignment: float = 0.0
    extension: float = 0.0
    balance: float = 0.0
    consistency: float = 0.0
    confidence: float = 0.0
    speed_normalized: float = 0.0
    speed_pixel: float = 0.0
    speed_estimated_mps: float | None = None
    feedback: str = ""
    side: str = ""


@dataclass
class SessionRecord:
    id: int | None = None
    athlete_id: int | None = None
    athlete_name: str = ""
    martial_art: str = "Kickboxing"
    started_at: str = ""
    ended_at: str | None = None
    duration_seconds: float = 0.0
    punch_count: int = 0
    kick_count: int = 0
    technique_distribution: dict[str, int] = field(default_factory=dict)
    average_score: float = 0.0
    best_score: float = 0.0
    average_speed: float = 0.0
    max_speed: float = 0.0
    average_balance: float = 0.0
    average_form: float = 0.0
    form_warnings: list[str] = field(default_factory=list)
    notes: str = ""
    source: str = "live"
    video_name: str | None = None
    events: list[TechniqueEvent] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.started_at:
            self.started_at = datetime.now().isoformat(timespec="seconds")
