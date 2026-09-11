"""End-to-end video file handling with a generated clip."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.pipeline import FramePipeline
from core.session_manager import SessionManager
from database.db import init_db
from database.repositories import AthleteRepository, SessionRepository
from models.athlete import Athlete
from utils.video_utils import open_video, video_properties


def _write_clip(path: Path, n: int = 6) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    assert writer.isOpened()
    for i in range(n):
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        cv2.putText(frame, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_generated_video_analysis(tmp_path: Path):
    clip = tmp_path / "clip.mp4"
    _write_clip(clip)
    cap = open_video(clip)
    assert cap is not None
    props = video_properties(cap)
    assert props["frame_count"] >= 1
    conn = init_db(tmp_path / "v.db")
    athlete = AthleteRepository(conn).create(Athlete(id=None, name="Pat", martial_art="Boxing"))
    pipe = FramePipeline("boxing")
    mgr = SessionManager(athlete, "Boxing", source="video", video_name="clip.mp4")
    frames = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        mgr.ingest(pipe.process(frame))
        frames += 1
    cap.release()
    pipe.close()
    saved = SessionRepository(conn).create(mgr.stop())
    assert frames >= 1
    assert saved.id is not None
    assert saved.source == "video"
