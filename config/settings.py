"""Runtime settings loaded from environment variables and sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    path = Path(raw) if raw else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


@dataclass
class CalibrationSettings:
    """Optional scale hints. Uncalibrated speed is never shown as exact m/s."""

    athlete_height_cm: float | None = None
    reference_distance_m: float | None = None
    camera_distance_m: float | None = None
    pixels_per_meter: float | None = None


@dataclass
class Settings:
    project_root: Path = PROJECT_ROOT
    db_path: Path = field(default_factory=lambda: _env_path("MAT_DB_PATH", Path("data/tracker.db")))
    log_path: Path = field(default_factory=lambda: _env_path("MAT_LOG_PATH", Path("data/logs/app.log")))
    log_level: str = field(default_factory=lambda: os.getenv("MAT_LOG_LEVEL", "INFO"))
    min_detection_confidence: float = field(
        default_factory=lambda: _env_float("MAT_MIN_DETECTION_CONFIDENCE", 0.5)
    )
    min_tracking_confidence: float = field(
        default_factory=lambda: _env_float("MAT_MIN_TRACKING_CONFIDENCE", 0.5)
    )
    min_visibility: float = field(default_factory=lambda: _env_float("MAT_MIN_VISIBILITY", 0.5))
    technique_confidence_threshold: float = 0.55
    camera_index: int = field(default_factory=lambda: _env_int("MAT_CAMERA_INDEX", 0))
    frame_width: int = field(default_factory=lambda: _env_int("MAT_FRAME_WIDTH", 640))
    target_fps: int = field(default_factory=lambda: _env_int("MAT_TARGET_FPS", 30))
    pose_model_path: Path = field(
        default_factory=lambda: _env_path(
            "MAT_POSE_MODEL_PATH", Path("data/models/pose_landmarker_lite.task")
        )
    )
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    uploads_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")
    sessions_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "sessions")
    exports_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "exports")
    ml_dataset_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "ml_dataset")
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    feedback_cooldown_seconds: float = 2.8
    motion_window_seconds: float = 0.45
    pose_history_limit: int = 300

    def ensure_directories(self) -> None:
        for path in (
            self.db_path.parent,
            self.log_path.parent,
            self.uploads_dir,
            self.sessions_dir,
            self.exports_dir,
            self.ml_dataset_dir,
            self.pose_model_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
