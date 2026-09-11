"""SQLite session persistence and athlete CRUD."""

from __future__ import annotations

from pathlib import Path

from database.db import init_db
from database.repositories import AthleteRepository, SessionRepository
from models.athlete import Athlete
from models.session import SessionRecord, TechniqueEvent


def test_athlete_and_session_roundtrip(tmp_path: Path):
    conn = init_db(tmp_path / "test.db")
    athletes = AthleteRepository(conn)
    sessions = SessionRepository(conn)
    athlete = athletes.create(Athlete(id=None, name="Alex", martial_art="Karate"))
    assert athlete.id is not None

    record = SessionRecord(
        athlete_id=athlete.id,
        athlete_name="Alex",
        martial_art="Karate",
        punch_count=2,
        kick_count=1,
        average_score=87,
        best_score=93,
        technique_distribution={"Jab": 2, "Front kick": 1},
        events=[
            TechniqueEvent(
                technique_name="Jab",
                category="punch",
                score=82,
                confidence=0.91,
                speed_normalized=1.2,
            )
        ],
    )
    saved = sessions.create(record)
    loaded = sessions.get(saved.id)
    assert loaded is not None
    assert loaded.punch_count == 2
    assert loaded.athlete_name == "Alex"
    assert len(loaded.events) == 1
    assert loaded.events[0].technique_name == "Jab"

    sessions.delete_session(saved.id)
    assert sessions.get(saved.id) is None


def test_delete_athlete_cascades(tmp_path: Path):
    conn = init_db(tmp_path / "test2.db")
    athletes = AthleteRepository(conn)
    sessions = SessionRepository(conn)
    athlete = athletes.create(Athlete(id=None, name="Sam", martial_art="Boxing"))
    sessions.create(SessionRecord(athlete_id=athlete.id, martial_art="Boxing"))
    athletes.delete(athlete.id)
    assert athletes.get(athlete.id) is None
    assert sessions.list_sessions(athlete.id) == []
