"""Persistence for athletes, sessions, and technique events."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime

from database.db import connect, init_db
from models.athlete import Athlete
from models.session import SessionRecord, TechniqueEvent
from utils.logger import get_logger

logger = get_logger("repositories")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class AthleteRepository:
    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn = conn or init_db()

    def create(self, athlete: Athlete) -> Athlete:
        cur = self.conn.execute(
            """
            INSERT INTO athletes (name, age, height_cm, martial_art, experience_level, training_goal, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                athlete.name,
                athlete.age,
                athlete.height_cm,
                athlete.martial_art,
                athlete.experience_level,
                athlete.training_goal,
                athlete.created_at or _now(),
            ),
        )
        self.conn.commit()
        athlete.id = int(cur.lastrowid)
        return athlete

    def update(self, athlete: Athlete) -> Athlete:
        if athlete.id is None:
            return self.create(athlete)
        self.conn.execute(
            """
            UPDATE athletes
            SET name=?, age=?, height_cm=?, martial_art=?, experience_level=?, training_goal=?
            WHERE id=?
            """,
            (
                athlete.name,
                athlete.age,
                athlete.height_cm,
                athlete.martial_art,
                athlete.experience_level,
                athlete.training_goal,
                athlete.id,
            ),
        )
        self.conn.commit()
        return athlete

    def get(self, athlete_id: int) -> Athlete | None:
        row = self.conn.execute("SELECT * FROM athletes WHERE id=?", (athlete_id,)).fetchone()
        return self._from_row(row) if row else None

    def list_all(self) -> list[Athlete]:
        rows = self.conn.execute("SELECT * FROM athletes ORDER BY name").fetchall()
        return [self._from_row(r) for r in rows]

    def delete(self, athlete_id: int) -> None:
        self.conn.execute("DELETE FROM technique_events WHERE session_id IN (SELECT id FROM sessions WHERE athlete_id=?)", (athlete_id,))
        self.conn.execute("DELETE FROM sessions WHERE athlete_id=?", (athlete_id,))
        self.conn.execute("DELETE FROM athletes WHERE id=?", (athlete_id,))
        self.conn.commit()

    def get_or_create_default(self) -> Athlete:
        rows = self.list_all()
        if rows:
            return rows[0]
        return self.create(
            Athlete(
                id=None,
                name="Athlete",
                martial_art="Kickboxing",
                experience_level="Beginner",
                training_goal="Technique accuracy",
            )
        )

    @staticmethod
    def _from_row(row: sqlite3.Row) -> Athlete:
        return Athlete(
            id=row["id"],
            name=row["name"],
            age=row["age"],
            height_cm=row["height_cm"],
            martial_art=row["martial_art"] or "Kickboxing",
            experience_level=row["experience_level"] or "Beginner",
            training_goal=row["training_goal"] or "Technique accuracy",
            created_at=row["created_at"],
        )


class SessionRepository:
    def __init__(self, conn: sqlite3.Connection | None = None) -> None:
        self.conn = conn or init_db()

    def create(self, session: SessionRecord) -> SessionRecord:
        cur = self.conn.execute(
            """
            INSERT INTO sessions (
                athlete_id, martial_art, started_at, ended_at, duration_seconds,
                punch_count, kick_count, technique_distribution, average_score, best_score,
                average_speed, max_speed, average_balance, average_form, form_warnings,
                notes, source, video_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.athlete_id,
                session.martial_art,
                session.started_at,
                session.ended_at,
                session.duration_seconds,
                session.punch_count,
                session.kick_count,
                json.dumps(session.technique_distribution),
                session.average_score,
                session.best_score,
                session.average_speed,
                session.max_speed,
                session.average_balance,
                session.average_form,
                json.dumps(session.form_warnings),
                session.notes,
                session.source,
                session.video_name,
            ),
        )
        self.conn.commit()
        session.id = int(cur.lastrowid)
        for event in session.events:
            event.session_id = session.id
            self.add_event(event)
        return session

    def add_event(self, event: TechniqueEvent) -> TechniqueEvent:
        cur = self.conn.execute(
            """
            INSERT INTO technique_events (
                session_id, detected_at, technique_name, category, score, accuracy,
                alignment, extension, balance, consistency, confidence, speed_normalized,
                speed_pixel, speed_estimated_mps, feedback, side
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.session_id,
                event.detected_at,
                event.technique_name,
                event.category,
                event.score,
                event.accuracy,
                event.alignment,
                event.extension,
                event.balance,
                event.consistency,
                event.confidence,
                event.speed_normalized,
                event.speed_pixel,
                event.speed_estimated_mps,
                event.feedback,
                event.side,
            ),
        )
        self.conn.commit()
        event.id = int(cur.lastrowid)
        return event

    def list_sessions(self, athlete_id: int | None = None) -> list[SessionRecord]:
        if athlete_id is None:
            rows = self.conn.execute(
                """
                SELECT s.*, a.name AS athlete_name
                FROM sessions s
                LEFT JOIN athletes a ON a.id = s.athlete_id
                ORDER BY s.started_at DESC
                """
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT s.*, a.name AS athlete_name
                FROM sessions s
                LEFT JOIN athletes a ON a.id = s.athlete_id
                WHERE s.athlete_id=?
                ORDER BY s.started_at DESC
                """,
                (athlete_id,),
            ).fetchall()
        return [self._session_from_row(r) for r in rows]

    def get(self, session_id: int) -> SessionRecord | None:
        row = self.conn.execute(
            """
            SELECT s.*, a.name AS athlete_name
            FROM sessions s
            LEFT JOIN athletes a ON a.id = s.athlete_id
            WHERE s.id=?
            """,
            (session_id,),
        ).fetchone()
        if not row:
            return None
        session = self._session_from_row(row)
        session.events = self.list_events(session_id)
        return session

    def list_events(
        self,
        session_id: int | None = None,
        technique_name: str | None = None,
        athlete_id: int | None = None,
    ) -> list[TechniqueEvent]:
        sql = """
            SELECT e.* FROM technique_events e
            JOIN sessions s ON s.id = e.session_id
            WHERE 1=1
        """
        params: list[object] = []
        if session_id is not None:
            sql += " AND e.session_id=?"
            params.append(session_id)
        if athlete_id is not None:
            sql += " AND s.athlete_id=?"
            params.append(athlete_id)
        if technique_name:
            sql += " AND e.technique_name=?"
            params.append(technique_name)
        sql += " ORDER BY e.detected_at DESC"
        rows = self.conn.execute(sql, params).fetchall()
        return [self._event_from_row(r) for r in rows]

    def delete_session(self, session_id: int) -> None:
        self.conn.execute("DELETE FROM technique_events WHERE session_id=?", (session_id,))
        self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        self.conn.commit()

    def delete_all(self) -> None:
        self.conn.execute("DELETE FROM technique_events")
        self.conn.execute("DELETE FROM sessions")
        self.conn.commit()

    @staticmethod
    def _session_from_row(row: sqlite3.Row) -> SessionRecord:
        dist_raw = row["technique_distribution"] or "{}"
        warn_raw = row["form_warnings"] or "[]"
        try:
            dist = json.loads(dist_raw)
        except json.JSONDecodeError:
            dist = {}
        try:
            warnings = json.loads(warn_raw)
        except json.JSONDecodeError:
            warnings = []
        return SessionRecord(
            id=row["id"],
            athlete_id=row["athlete_id"],
            athlete_name=row["athlete_name"] if "athlete_name" in row.keys() else "",
            martial_art=row["martial_art"] or "",
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            duration_seconds=row["duration_seconds"] or 0,
            punch_count=row["punch_count"] or 0,
            kick_count=row["kick_count"] or 0,
            technique_distribution=dist,
            average_score=row["average_score"] or 0,
            best_score=row["best_score"] or 0,
            average_speed=row["average_speed"] or 0,
            max_speed=row["max_speed"] or 0,
            average_balance=row["average_balance"] or 0,
            average_form=row["average_form"] or 0,
            form_warnings=warnings,
            notes=row["notes"] or "",
            source=row["source"] or "live",
            video_name=row["video_name"],
        )

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> TechniqueEvent:
        return TechniqueEvent(
            id=row["id"],
            session_id=row["session_id"],
            detected_at=row["detected_at"] or "",
            technique_name=row["technique_name"] or "",
            category=row["category"] or "",
            score=row["score"] or 0,
            accuracy=row["accuracy"] or 0,
            alignment=row["alignment"] or 0,
            extension=row["extension"] or 0,
            balance=row["balance"] or 0,
            consistency=row["consistency"] or 0,
            confidence=row["confidence"] or 0,
            speed_normalized=row["speed_normalized"] or 0,
            speed_pixel=row["speed_pixel"] or 0,
            speed_estimated_mps=row["speed_estimated_mps"],
            feedback=row["feedback"] or "",
            side=row["side"] or "",
        )
