"""Per-frame orchestration: pose → angles → motion → technique → form → overlay."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from config.settings import CalibrationSettings, Settings, get_settings
from core.angle_calculator import AngleCalculator
from core.feedback import FeedbackManager
from core.form_analyzer import FormAnalyzer, StanceAnalyzer
from core.landmark_processor import LandmarkProcessor, ProcessedPose
from core.motion_tracker import MotionTracker
from core.overlay import draw_hud, draw_skeleton
from core.pose_detector import PoseDetector
from core.scoring_engine import ScoringEngine
from core.technique_detector import TechniqueDetector
from martial_arts.base import StylePlugin, get_style
from models.technique import (
    FeedbackItem,
    FormScores,
    JointAngles,
    MotionMetrics,
    StanceResult,
    TechniqueDetection,
)
from utils.video_utils import resize_frame


@dataclass
class FrameAnalysis:
    overlay_bgr: np.ndarray
    pose_ok: bool
    pose_message: str
    processed: ProcessedPose | None
    angles: JointAngles
    motion: MotionMetrics
    stance: StanceResult
    technique: TechniqueDetection | None
    form: FormScores
    feedback: list[FeedbackItem] = field(default_factory=list)
    fps: float = 0.0
    world_landmarks: dict = field(default_factory=dict)


class FramePipeline:
    def __init__(
        self,
        style: StylePlugin | str = "kickboxing",
        settings: Settings | None = None,
        min_detection_confidence: float | None = None,
        calibration: CalibrationSettings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        plugin = get_style(style) if isinstance(style, str) else style
        self.style = plugin
        self.detector = PoseDetector(
            min_detection_confidence=min_detection_confidence,
            min_visibility=self.settings.min_visibility,
        )
        self.landmarks = LandmarkProcessor(self.settings.min_visibility)
        self.angles = AngleCalculator()
        self.motion = MotionTracker(self.settings.motion_window_seconds, calibration)
        self.techniques = TechniqueDetector(plugin, self.settings.technique_confidence_threshold)
        self.stance = StanceAnalyzer()
        self.form = FormAnalyzer()
        self.scoring = ScoringEngine()
        self.feedback = FeedbackManager(self.settings.feedback_cooldown_seconds)
        self._t0 = perf_counter()
        self._frames = 0
        self._skip_counter = 0
        self.frame_skip = 0
        self.max_width = self.settings.frame_width
        self.last_technique: TechniqueDetection | None = None
        self.pose_history: list[dict] = []
        self._last_points: dict = {}
        self._last_visibility: dict = {}
        self._pose_misses = 0

    def set_style(self, style: StylePlugin | str) -> None:
        plugin = get_style(style) if isinstance(style, str) else style
        self.style = plugin
        self.techniques.set_style(plugin)

    def set_calibration(self, calibration: CalibrationSettings) -> None:
        self.motion.calibration = calibration

    @property
    def pose_backend(self) -> str:
        return self.detector.backend

    def process(self, frame_bgr: np.ndarray, recording: bool = False) -> FrameAnalysis:
        self._frames += 1
        if frame_bgr is None or frame_bgr.size == 0 or min(frame_bgr.shape[:2]) < 8:
            blank = np.zeros((240, 320, 3), dtype=np.uint8)
            return FrameAnalysis(
                overlay_bgr=blank,
                pose_ok=False,
                pose_message="Empty frame",
                processed=None,
                angles=JointAngles(),
                motion=MotionMetrics(),
                stance=StanceResult(label="UNKNOWN", score=0.0, feedback=["Empty frame"]),
                technique=None,
                form=FormScores(0, 0, 0, 0, 0, 0),
                fps=self._fps(),
            )
        if self.max_width:
            frame_bgr = resize_frame(frame_bgr, self.max_width)

        self._skip_counter += 1

        elapsed = perf_counter() - self._t0
        fps = self._frames / max(elapsed, 1e-3)
        # Shrink processing size on slow machines instead of skipping frames
        # (skipped frames previously flashed a raw image and looked like flicker).
        if fps < 8 and self._frames > 20 and self.max_width > 400:
            self.max_width = 400

        timestamp = elapsed
        pose = self.detector.detect(frame_bgr, timestamp_ms=int(elapsed * 1000))
        h, w = frame_bgr.shape[:2]
        processed = self.landmarks.process(pose, w, h)
        angles = self.angles.calculate(processed) if processed.detected else JointAngles()
        motion = self.motion.update(processed, angles, timestamp)
        stance = self.stance.analyze(processed, angles)
        technique = self.techniques.detect(self.motion) if processed.detected else None
        if technique and technique.name:
            self.last_technique = technique
        form = self.form.analyze(processed, angles, stance, technique or self.last_technique, self.motion)
        form = self.scoring.score(form, technique)
        cues = self.feedback.update(
            timestamp,
            processed.detected,
            processed.message,
            stance,
            technique,
            form,
        )

        overlay = frame_bgr.copy()
        if processed.detected and processed.points:
            self._last_points = processed.points
            self._last_visibility = processed.visibility
            self._pose_misses = 0
            overlay = draw_skeleton(
                overlay,
                processed.points,
                processed.visibility,
                self.settings.min_visibility,
            )
            pose_ok = True
            pose_message = processed.message
        elif self._pose_misses < 8 and self._last_points:
            self._pose_misses += 1
            overlay = draw_skeleton(
                overlay,
                self._last_points,
                self._last_visibility,
                self.settings.min_visibility,
            )
            pose_ok = True
            pose_message = processed.message
        else:
            self._pose_misses += 1
            pose_ok = False
            pose_message = processed.message

        overlay = draw_hud(
            overlay,
            self._fps(),
            recording,
            technique or self.last_technique,
            angles,
            pose_ok,
            pose_message,
        )

        if processed.detected and pose.world_landmarks:
            if len(self.pose_history) >= self.settings.pose_history_limit:
                self.pose_history.pop(0)
            self.pose_history.append(
                {
                    "t": timestamp,
                    "world": {k: v.xyz() for k, v in pose.world_landmarks.items()},
                    "image": {k: (float(v[0]), float(v[1]), float(v[2]) if v.size > 2 else 0.0) for k, v in processed.points.items()},
                }
            )

        return FrameAnalysis(
            overlay_bgr=overlay,
            pose_ok=pose_ok,
            pose_message=pose_message,
            processed=processed,
            angles=angles,
            motion=motion,
            stance=stance,
            technique=technique,
            form=form,
            feedback=cues or self.feedback.active,
            fps=self._fps(),
            world_landmarks={k: v.xyz() for k, v in pose.world_landmarks.items()} if pose.world_landmarks else {},
        )

    def _fps(self) -> float:
        elapsed = perf_counter() - self._t0
        if elapsed <= 0:
            return 0.0
        return self._frames / elapsed

    def close(self) -> None:
        self.detector.close()
