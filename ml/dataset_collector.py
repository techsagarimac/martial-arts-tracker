"""Labeled pose-sequence collector for a future ML classifier.

No model is trained here. Sequences are stored locally as JSON lines.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from config.settings import get_settings
from core.landmark_processor import ProcessedPose
from models.technique import JointAngles, MotionMetrics

ML_CLASSES = (
    "jab",
    "cross",
    "hook",
    "front_kick",
    "roundhouse_kick",
    "side_kick",
    "stance",
)


class DatasetCollector:
    def __init__(self, directory: Path | None = None, window_frames: int = 45) -> None:
        settings = get_settings()
        self.directory = directory or settings.ml_dataset_dir
        self.directory.mkdir(parents=True, exist_ok=True)
        self.window_frames = window_frames
        self._buffer: list[dict] = []
        self.label: str | None = None
        self.recording = False

    def start(self, label: str) -> None:
        if label not in ML_CLASSES:
            raise ValueError(f"Unknown label {label}. Expected one of {ML_CLASSES}")
        self.label = label
        self._buffer = []
        self.recording = True

    def add(
        self,
        pose: ProcessedPose,
        angles: JointAngles,
        motion: MotionMetrics,
        timestamp: float,
    ) -> bool:
        if not self.recording or not pose.detected:
            return False
        self._buffer.append(
            {
                "t": timestamp,
                "points": {k: v.tolist() for k, v in pose.points.items()},
                "angles": angles.as_dict(),
                "normalized_speed": motion.normalized_speed,
                "shoulder_rotation_deg": motion.shoulder_rotation_deg,
            }
        )
        return len(self._buffer) >= self.window_frames

    def save(self) -> Path | None:
        if not self.label or not self._buffer:
            self.recording = False
            return None
        path = self.directory / f"{self.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        record = {
            "label": self.label,
            "frames": self._buffer,
            "frame_count": len(self._buffer),
        }
        path.write_text(json.dumps(record), encoding="utf-8")
        self.recording = False
        self._buffer = []
        return path

    def cancel(self) -> None:
        self.recording = False
        self._buffer = []
        self.label = None
