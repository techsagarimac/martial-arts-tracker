"""Accumulate a live or video session and persist it."""

from __future__ import annotations

from collections import Counter
from datetime import datetime

from core.pipeline import FrameAnalysis
from database.repositories import SessionRepository
from models.athlete import Athlete
from models.session import SessionRecord, TechniqueEvent
from utils.logger import get_logger

logger = get_logger("session")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


class SessionManager:
    def __init__(self, athlete: Athlete, martial_art: str, source: str = "live", video_name: str | None = None) -> None:
        self.athlete = athlete
        self.record = SessionRecord(
            athlete_id=athlete.id,
            athlete_name=athlete.name,
            martial_art=martial_art,
            started_at=_now(),
            source=source,
            video_name=video_name,
        )
        self._t0 = datetime.now()
        self._scores: list[float] = []
        self._speeds: list[float] = []
        self._balances: list[float] = []
        self._forms: list[float] = []
        self._warnings: list[str] = []
        self._counts: Counter[str] = Counter()
        self.active = True

    def ingest(self, analysis: FrameAnalysis) -> TechniqueEvent | None:
        if not self.active:
            return None
        self.record.duration_seconds = (datetime.now() - self._t0).total_seconds()
        if analysis.motion.normalized_speed:
            self._speeds.append(analysis.motion.normalized_speed)
        if analysis.form.balance:
            self._balances.append(analysis.form.balance)
        if analysis.form.overall:
            self._forms.append(analysis.form.overall)
        for item in analysis.feedback:
            if item.level in {"yellow", "red"} and item.message not in self._warnings:
                self._warnings.append(item.message)

        detection = analysis.technique
        if detection is None or detection.uncertain or not detection.name:
            self._refresh_aggregates()
            return None

        event = TechniqueEvent(
            detected_at=_now(),
            technique_name=detection.display_name,
            category=detection.category or "",
            score=analysis.form.overall,
            accuracy=analysis.form.accuracy,
            alignment=analysis.form.alignment,
            extension=analysis.form.extension,
            balance=analysis.form.balance,
            consistency=analysis.form.consistency,
            confidence=detection.confidence,
            speed_normalized=analysis.motion.normalized_speed,
            speed_pixel=analysis.motion.pixel_speed,
            speed_estimated_mps=analysis.motion.estimated_mps,
            feedback="; ".join(item.message for item in analysis.feedback[:3]),
            side=detection.side or "",
        )
        self.record.events.append(event)
        self._counts[detection.display_name] += 1
        self._scores.append(analysis.form.overall)
        if detection.category == "punch":
            self.record.punch_count += 1
        elif detection.category == "kick":
            self.record.kick_count += 1
        self._refresh_aggregates()
        return event

    def _refresh_aggregates(self) -> None:
        self.record.technique_distribution = dict(self._counts)
        self.record.average_score = sum(self._scores) / len(self._scores) if self._scores else 0.0
        self.record.best_score = max(self._scores) if self._scores else 0.0
        self.record.average_speed = sum(self._speeds) / len(self._speeds) if self._speeds else 0.0
        self.record.max_speed = max(self._speeds) if self._speeds else 0.0
        self.record.average_balance = sum(self._balances) / len(self._balances) if self._balances else 0.0
        self.record.average_form = sum(self._forms) / len(self._forms) if self._forms else 0.0
        self.record.form_warnings = self._warnings[:12]
        self.record.duration_seconds = (datetime.now() - self._t0).total_seconds()

    def stop(self) -> SessionRecord:
        self.active = False
        self._refresh_aggregates()
        self.record.ended_at = _now()
        return self.record

    def save(self, notes: str = "") -> SessionRecord:
        if self.active:
            self.stop()
        if notes:
            self.record.notes = notes
        repo = SessionRepository()
        saved = repo.create(self.record)
        logger.info("Saved session %s with %s events", saved.id, len(saved.events))
        return saved
