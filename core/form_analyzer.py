"""Stance and form heuristics. Training feedback only — not medical advice."""

from __future__ import annotations

from collections import deque
from typing import Literal

import numpy as np

from core.landmark_processor import ProcessedPose
from core.motion_tracker import MotionTracker
from models.technique import FormScores, JointAngles, StanceResult, TechniqueDetection
from utils.geometry import clamp, distance, midpoint


class StanceAnalyzer:
    def analyze(self, pose: ProcessedPose, angles: JointAngles) -> StanceResult:
        if not pose.detected:
            return StanceResult(label="UNKNOWN", score=0.0, feedback=["Pose not detected clearly"])

        notes: list[str] = []
        score = 80.0

        la, ra = pose.get("left_ankle"), pose.get("right_ankle")
        ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
        lh, rh = pose.get("left_hip"), pose.get("right_hip")
        lk, rk = pose.get("left_knee"), pose.get("right_knee")
        lw, rw = pose.get("left_wrist"), pose.get("right_wrist")

        shoulder_w = distance(ls, rs)
        feet_sep = distance(la, ra)
        stance_ratio = feet_sep / shoulder_w if shoulder_w and shoulder_w == shoulder_w and shoulder_w > 1e-4 else float("nan")

        if stance_ratio == stance_ratio:
            if stance_ratio < 0.85:
                notes.append("Feet are too close together.")
                score -= 14
            elif stance_ratio > 2.4:
                notes.append("Stance is very wide — recover toward a balanced base.")
                score -= 8

        if lh is not None and rh is not None:
            hip_tilt = abs(lh[1] - rh[1])
            if hip_tilt > 0.07:
                notes.append("Hips appear uneven.")
                score -= 6
        if ls is not None and rs is not None:
            if abs(ls[1] - rs[1]) > 0.07:
                notes.append("Shoulders appear uneven.")
                score -= 6

        # Knee tracking: knee x should sit roughly between hip and ankle.
        for label, hip, knee, ankle in (
            ("left", lh, lk, la),
            ("right", rh, rk, ra),
        ):
            if hip is None or knee is None or ankle is None:
                continue
            inward = (knee[0] - hip[0]) * (ankle[0] - hip[0]) < 0 and abs(knee[0] - hip[0]) > 0.04
            if inward:
                notes.append(f"Front knee appears excessively inward ({label}).")
                score -= 8

        guard = float("nan")
        if lw is not None and rw is not None and ls is not None and rs is not None:
            shoulder_y = (ls[1] + rs[1]) / 2.0
            wrist_y = (lw[1] + rw[1]) / 2.0
            guard = shoulder_y - wrist_y  # positive = hands higher than shoulders
            if wrist_y > shoulder_y + 0.08:
                notes.append("Guard position is low.")
                score -= 12
            elif wrist_y > shoulder_y + 0.02:
                notes.append("Keep your guard a little higher.")
                score -= 5

        hips = midpoint(lh, rh)
        feet = midpoint(la, ra)
        if hips is not None and feet is not None:
            if abs(hips[0] - feet[0]) > 0.08:
                notes.append("Weight looks shifted off your base.")
                score -= 6

        score = float(clamp(score, 0, 100))
        status: Literal["GOOD", "NEEDS IMPROVEMENT", "UNKNOWN"]
        if not notes:
            notes.append("Stance looks balanced.")
            status = "GOOD"
        else:
            status = "GOOD" if score >= 75 else "NEEDS IMPROVEMENT"

        return StanceResult(
            label=status,
            score=score,
            feedback=notes,
            feet_separation=stance_ratio if stance_ratio == stance_ratio else float("nan"),
            guard_height=guard,
        )


class FormAnalyzer:
    def __init__(self) -> None:
        self._recent_overall: deque[float] = deque(maxlen=12)

    def reset(self) -> None:
        self._recent_overall.clear()

    def analyze(
        self,
        pose: ProcessedPose,
        angles: JointAngles,
        stance: StanceResult,
        detection: TechniqueDetection | None,
        tracker: MotionTracker,
    ) -> FormScores:
        if not pose.detected:
            return FormScores(0, 0, 0, 0, 0, 0)

        alignment = self._alignment(pose, stance)
        balance = self._balance(pose, stance)
        extension = self._extension(angles, detection)
        accuracy = self._accuracy(detection, tracker)
        consistency = self._consistency()

        overall = (
            0.30 * accuracy
            + 0.20 * alignment
            + 0.20 * extension
            + 0.15 * balance
            + 0.15 * consistency
        )
        scores = FormScores(
            accuracy=round(accuracy),
            alignment=round(alignment),
            extension=round(extension),
            balance=round(balance),
            consistency=round(consistency),
            overall=round(float(clamp(overall, 0, 100))),
        )
        if detection and detection.name and not detection.uncertain:
            self._recent_overall.append(scores.overall)
        return scores

    def _accuracy(self, detection: TechniqueDetection | None, tracker: MotionTracker) -> float:
        if detection is None:
            return 62.0
        if detection.uncertain or not detection.name:
            return float(clamp(40 + detection.confidence * 30, 35, 70))
        speed = tracker.limb_speed(f"{detection.side}_wrist") if detection.side else 0.0
        if detection.category == "kick" and detection.side:
            speed = tracker.limb_speed(f"{detection.side}_ankle")
        speed_bonus = 8 if speed > 0.8 else 0
        return float(clamp(detection.confidence * 100 + speed_bonus, 50, 98))

    def _alignment(self, pose: ProcessedPose, stance: StanceResult) -> float:
        base = stance.score
        ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
        if ls is not None and rs is not None:
            tilt = abs(ls[1] - rs[1])
            base -= min(20, tilt * 180)
        return float(clamp(base, 0, 100))

    def _balance(self, pose: ProcessedPose, stance: StanceResult) -> float:
        hips = midpoint(pose.get("left_hip"), pose.get("right_hip"))
        feet = midpoint(pose.get("left_ankle"), pose.get("right_ankle"))
        score = 88.0
        if hips is None or feet is None:
            return stance.score
        offset = abs(hips[0] - feet[0])
        score -= min(35, offset * 250)
        return float(clamp(score, 0, 100))

    def _extension(self, angles: JointAngles, detection: TechniqueDetection | None) -> float:
        if detection is None or detection.uncertain:
            elbows = [a for a in (angles.left_elbow, angles.right_elbow) if a == a]
            return float(clamp(np.nanmean(elbows) * 0.5, 40, 90)) if elbows else 60.0
        if detection.category == "punch":
            angle = angles.left_elbow if detection.side == "left" else angles.right_elbow
            if detection.name == "hook":
                # Hooks should stay bent.
                if angle != angle:
                    return 70.0
                return float(clamp(100 - abs(angle - 95) * 0.8, 40, 95))
            if angle != angle:
                return 70.0
            return float(clamp((angle - 90) * 1.4, 40, 97))
        knee = angles.left_knee if detection.side == "left" else angles.right_knee
        if knee != knee:
            return 70.0
        return float(clamp((knee - 80) * 1.1, 40, 97))

    def _consistency(self) -> float:
        if len(self._recent_overall) < 2:
            return 80.0
        arr = np.asarray(self._recent_overall, dtype=float)
        spread = float(np.std(arr))
        return float(clamp(95 - spread * 1.8, 50, 96))
