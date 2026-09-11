"""Combine form subscores into a transparent training score."""

from __future__ import annotations

from models.technique import FormScores, TechniqueDetection
from utils.geometry import clamp


class ScoringEngine:
    """Weighted heuristic scorer. Not a federation or gym official score."""

    WEIGHTS = {
        "accuracy": 0.30,
        "alignment": 0.20,
        "extension": 0.20,
        "balance": 0.15,
        "consistency": 0.15,
    }

    def score(self, form: FormScores, detection: TechniqueDetection | None = None) -> FormScores:
        overall = (
            self.WEIGHTS["accuracy"] * form.accuracy
            + self.WEIGHTS["alignment"] * form.alignment
            + self.WEIGHTS["extension"] * form.extension
            + self.WEIGHTS["balance"] * form.balance
            + self.WEIGHTS["consistency"] * form.consistency
        )
        if detection is not None and (detection.uncertain or not detection.name):
            overall *= 0.85
        form.overall = round(float(clamp(overall, 0, 100)))
        return form

    def explanation(self) -> str:
        parts = [f"{name.title()} {int(w * 100)}%" for name, w in self.WEIGHTS.items()]
        return "Overall = " + " + ".join(parts) + ". Training heuristic only."
