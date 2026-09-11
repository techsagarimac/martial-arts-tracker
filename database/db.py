"""SQLite connection helpers. All data stays on the local machine."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger("db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS athletes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER,
    height_cm REAL,
    martial_art TEXT,
    experience_level TEXT,
    training_goal TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    athlete_id INTEGER,
    martial_art TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_seconds REAL DEFAULT 0,
    punch_count INTEGER DEFAULT 0,
    kick_count INTEGER DEFAULT 0,
    technique_distribution TEXT,
    average_score REAL DEFAULT 0,
    best_score REAL DEFAULT 0,
    average_speed REAL DEFAULT 0,
    max_speed REAL DEFAULT 0,
    average_balance REAL DEFAULT 0,
    average_form REAL DEFAULT 0,
    form_warnings TEXT,
    notes TEXT,
    source TEXT DEFAULT 'live',
    video_name TEXT,
    FOREIGN KEY (athlete_id) REFERENCES athletes(id)
);

CREATE TABLE IF NOT EXISTS technique_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    detected_at TEXT,
    technique_name TEXT,
    category TEXT,
    score REAL,
    accuracy REAL,
    alignment REAL,
    extension REAL,
    balance REAL,
    consistency REAL,
    confidence REAL,
    speed_normalized REAL,
    speed_pixel REAL,
    speed_estimated_mps REAL,
    feedback TEXT,
    side TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    settings = get_settings()
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    except sqlite3.Error as exc:
        logger.error("Database initialization failed: %s", exc)
        raise
    return conn
