"""Future-ready sparring interfaces. Two-person tracking is not implemented."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.landmark_processor import ProcessedPose


@dataclass
class FighterState:
    label: str  # "A" or "B"
    pose: ProcessedPose | None = None
    attack_count: int = 0
    last_technique: str | None = None


@dataclass
class SparringMetrics:
    distance_estimate: float | None = None
    attack_frequency_a: float = 0.0
    attack_frequency_b: float = 0.0
    defensive_movement_a: float = 0.0
    defensive_movement_b: float = 0.0
    reaction_time_a: float | None = None
    reaction_time_b: float | None = None
    notes: list[str] = field(default_factory=list)


class SparringAnalyzer(Protocol):
    """Contract for a future two-person tracker.

    Do not invent fighter identities from a single-pose model.
    """

    def reset(self) -> None:
        ...

    def update(self, fighter_a: FighterState, fighter_b: FighterState, timestamp: float) -> SparringMetrics:
        ...


class UnimplementedSparringAnalyzer:
    """Placeholder that refuses to guess two-person labels from one skeleton."""

    def reset(self) -> None:
        return None

    def update(self, fighter_a: FighterState, fighter_b: FighterState, timestamp: float) -> SparringMetrics:
        return SparringMetrics(
            notes=[
                "Sparring analysis requires a dedicated multi-person pose model.",
                "Single-person MediaPipe Pose cannot reliably separate Fighter A and Fighter B.",
            ]
        )
