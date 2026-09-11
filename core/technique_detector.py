"""Rule-based martial-arts technique detector.

This is a geometric heuristic, not a trained classifier. Low-confidence
windows are reported as uncertain instead of forcing a label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.motion_tracker import MotionTracker, PoseSample
from martial_arts.base import StylePlugin
from models.technique import TechniqueDetection
from utils.geometry import distance, line_angle_deg, normalize_angle_delta


PUNCH_SPEED_MIN = 0.40
KICK_SPEED_MIN = 0.40
MIN_EMIT_CONFIDENCE = 0.42


@dataclass
class _Candidate:
    name: str
    display: str
    category: str
    confidence: float
    side: str
    notes: str = ""


class TechniqueDetector:
    def __init__(
        self,
        style: StylePlugin,
        min_confidence: float = 0.55,
        cooldown_seconds: float = 0.55,
    ) -> None:
        self.style = style
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self._last_emit_time = -1e9
        self._last_name: str | None = None
        self.enabled = {spec.id for spec in style.enabled_techniques()}

    def set_style(self, style: StylePlugin) -> None:
        self.style = style
        self.enabled = {spec.id for spec in style.enabled_techniques()}

    def detect(self, tracker: MotionTracker) -> TechniqueDetection | None:
        samples = tracker.samples_in_window()
        if len(samples) < 4:
            return None
        newest = samples[-1]
        if newest.timestamp - self._last_emit_time < self.cooldown_seconds:
            return None
        if not self._body_visible(newest):
            return TechniqueDetection(
                name=None,
                display_name="Technique uncertain",
                category=None,
                confidence=0.0,
                uncertain=True,
                notes="Body is not sufficiently visible for classification.",
            )

        if not self._is_peak(tracker):
            return None

        candidates = [c for c in self._score_all(samples) if c.confidence >= MIN_EMIT_CONFIDENCE]
        candidates = [c for c in candidates if c.name in self.enabled]
        if not candidates:
            return None

        best = max(candidates, key=lambda c: c.confidence)
        if best.confidence < self.min_confidence:
            self._last_emit_time = newest.timestamp
            return TechniqueDetection(
                name=None,
                display_name="Technique uncertain",
                category=best.category,
                confidence=best.confidence,
                side=best.side,
                uncertain=True,
                notes="Movement detected, but the pattern did not match a technique clearly.",
            )

        self._last_emit_time = newest.timestamp
        self._last_name = best.name
        return TechniqueDetection(
            name=best.name,
            display_name=best.display,
            category=best.category,
            confidence=float(np.clip(best.confidence, 0.0, 0.99)),
            side=best.side,
            uncertain=False,
            notes=best.notes,
        )

    def _body_visible(self, sample: PoseSample) -> bool:
        needed = (
            "left_shoulder",
            "right_shoulder",
            "left_hip",
            "right_hip",
        )
        return all(name in sample.points and sample.visibility.get(name, 0) >= 0.4 for name in needed)

    def _is_peak(self, tracker: MotionTracker) -> bool:
        """Emit near a local speed maximum so one strike is not counted many times."""
        names = ("left_wrist", "right_wrist", "left_ankle", "right_ankle")
        speeds = [tracker.limb_speed(n) for n in names]
        current = max(speeds) if speeds else 0.0
        if current < min(PUNCH_SPEED_MIN, KICK_SPEED_MIN) * 0.7:
            return False
        samples = tracker.samples_in_window()
        if len(samples) < 5:
            return True
        # Compare mid-window peak vs latest speed.
        mid = samples[len(samples) // 2]
        late = samples[-1]
        dt = late.timestamp - mid.timestamp
        if dt <= 1e-3:
            return True
        return current >= PUNCH_SPEED_MIN * 0.65

    def _score_all(self, samples: list[PoseSample]) -> list[_Candidate]:
        out: list[_Candidate] = []
        for side in ("left", "right"):
            punch = self._score_punch(samples, side)
            if punch:
                out.append(punch)
            kick = self._score_kick(samples, side)
            if kick:
                out.append(kick)
        return out

    def _score_punch(self, samples: list[PoseSample], side: str) -> _Candidate | None:
        wrist = f"{side}_wrist"
        elbow = f"{side}_elbow"
        path = [s.points[wrist] for s in samples if wrist in s.points]
        if len(path) < 4:
            return None
        start, end = samples[0], samples[-1]
        dt = max(end.timestamp - start.timestamp, 1e-3)
        disp = distance(path[0], path[-1])
        speed = disp / dt
        if speed < PUNCH_SPEED_MIN * 0.8:
            return None

        elbow_angles = [getattr(s.angles, elbow) for s in samples]
        elbow_angles = [a for a in elbow_angles if a == a]
        if len(elbow_angles) < 3:
            return None
        elbow_min, elbow_max = min(elbow_angles), max(elbow_angles)
        elbow_end = elbow_angles[-1]
        delta = np.asarray(path[-1][:2]) - np.asarray(path[0][:2])
        dx, dy = float(delta[0]), float(delta[1])
        dz = float(path[-1][2] - path[0][2]) if path[0].size > 2 and path[-1].size > 2 else 0.0
        shoulder_rot = abs(self._shoulder_rotation(samples))
        lead = self._lead_side(end)

        hook_score = 0.0
        if elbow_max < 140 and abs(dx) > abs(dy) and abs(dx) > 0.04:
            hook_score = 0.58
            hook_score += 0.12 if 50 <= elbow_min <= 125 else 0.0
            hook_score += 0.10 if speed > 0.55 else 0.0
            hook_score += 0.08 if shoulder_rot > 8 else 0.0
            hook_score += min(0.12, abs(dx) * 1.5)

        straight_score = 0.0
        if elbow_max >= 120 and (elbow_max - elbow_min) >= 12:
            straight_score = 0.56
            straight_score += 0.12 if elbow_end >= 135 else 0.04
            straight_score += 0.10 if speed > 0.55 else 0.0
            if dz < -0.02 or dy < -0.02:
                straight_score += 0.10
            if abs(dx) < abs(dy) + 0.04:
                straight_score += 0.06
            straight_score += min(0.08, shoulder_rot / 80.0)

        if hook_score < MIN_EMIT_CONFIDENCE and straight_score < MIN_EMIT_CONFIDENCE:
            return None

        if hook_score >= straight_score and hook_score >= MIN_EMIT_CONFIDENCE:
            return _Candidate("hook", "Hook", "punch", min(hook_score, 0.95), side, "Bent-arm lateral punch pattern")

        # Distinguish jab / cross / generic straight.
        if lead == side:
            name, display = "jab", "Jab"
            conf = straight_score + 0.04
            note = "Lead-hand linear punch"
        elif lead and lead != side:
            name, display = "cross", "Cross"
            conf = straight_score + (0.08 if shoulder_rot > 10 else 0.02)
            note = "Rear-hand linear punch"
        else:
            name, display = "straight_punch", "Straight punch"
            conf = straight_score
            note = "Linear punch (lead/rear unclear)"

        if shoulder_rot < 6 and lead and lead != side:
            # Rear hand without rotation — keep as straight rather than a confident cross.
            name, display = "straight_punch", "Straight punch"
            note = "Linear punch with limited shoulder turn"

        other_wrist = "right_wrist" if side == "left" else "left_wrist"
        other_disp = 0.0
        if other_wrist in start.points and other_wrist in end.points:
            other_disp = distance(start.points[other_wrist], end.points[other_wrist])
        if other_disp > disp * 0.9:
            conf -= 0.08  # both hands moving — less like a single punch

        return _Candidate(name, display, "punch", float(np.clip(conf, 0, 0.96)), side, note)

    def _score_kick(self, samples: list[PoseSample], side: str) -> _Candidate | None:
        ankle = f"{side}_ankle"
        knee = f"{side}_knee"
        hip = f"{side}_hip"
        path = [s.points[ankle] for s in samples if ankle in s.points]
        if len(path) < 4:
            return None
        start, end = samples[0], samples[-1]
        dt = max(end.timestamp - start.timestamp, 1e-3)
        disp = distance(path[0], path[-1])
        speed = disp / dt
        if speed < KICK_SPEED_MIN * 0.75:
            return None

        knee_angles = [getattr(s.angles, knee) for s in samples]
        knee_angles = [a for a in knee_angles if a == a]
        if len(knee_angles) < 3:
            return None
        knee_min, knee_max = min(knee_angles), max(knee_angles)
        chambered = knee_min < 140
        extended = knee_max > 125
        if not (chambered or speed > 0.7):
            return None

        delta = np.asarray(path[-1][:2]) - np.asarray(path[0][:2])
        dx, dy = float(delta[0]), float(delta[1])
        hip_rot = abs(self._hip_rotation(samples))
        mid_y = [p[1] for p in path]
        lifted = (min(mid_y) < path[0][1] - 0.03) if mid_y else False

        # Front kick: lift + extend, mostly vertical / toward camera.
        front = 0.0
        if chambered and lifted and abs(dy) >= abs(dx) * 0.7:
            front = 0.55
            front += 0.12 if extended else 0.04
            front += 0.10 if speed > 0.7 else 0.0
            front += 0.08 if hip_rot < 18 else 0.0
            front += 0.06 if dy < -0.04 else 0.0

        # Roundhouse: hip turn + lateral arc.
        roundhouse = 0.0
        if chambered and (abs(dx) > 0.04 or hip_rot > 12):
            roundhouse = 0.52
            roundhouse += 0.14 if hip_rot > 14 else 0.04
            roundhouse += 0.10 if abs(dx) > 0.05 else 0.0
            roundhouse += 0.08 if lifted else 0.0
            roundhouse += 0.08 if speed > 0.75 else 0.0

        # Side kick: abduction / sideways extend, less whipping rotation.
        sidekick = 0.0
        if extended and abs(dx) > 0.06 and abs(dx) > abs(dy):
            sidekick = 0.52
            sidekick += 0.12 if knee_max > 150 else 0.04
            sidekick += 0.10 if hip_rot < 22 else 0.0
            sidekick += 0.08 if speed > 0.7 else 0.0
            # Torso lean approximation: kicking hip higher (smaller y) than support hip.
            other_hip = "right_hip" if side == "left" else "left_hip"
            if hip in end.points and other_hip in end.points:
                if end.points[hip][1] < end.points[other_hip][1] - 0.02:
                    sidekick += 0.08

        scores = {
            "front_kick": front,
            "roundhouse_kick": roundhouse,
            "side_kick": sidekick,
        }
        name = max(scores, key=lambda k: scores[k])
        conf = scores[name]
        if conf < MIN_EMIT_CONFIDENCE:
            return None
        labels = {
            "front_kick": ("Front kick", "Chamber-then-extend kick, mostly sagittal"),
            "roundhouse_kick": ("Roundhouse-style kick", "Hip rotation with a lateral kicking arc"),
            "side_kick": ("Side kick", "Sideways extension / abduction pattern"),
        }
        display, note = labels[name]
        return _Candidate(name, display, "kick", float(np.clip(conf, 0, 0.95)), side, note)

    def _shoulder_rotation(self, samples: list[PoseSample]) -> float:
        first = line_angle_deg(samples[0].points.get("left_shoulder"), samples[0].points.get("right_shoulder"))
        last = line_angle_deg(samples[-1].points.get("left_shoulder"), samples[-1].points.get("right_shoulder"))
        return normalize_angle_delta(last - first)

    def _hip_rotation(self, samples: list[PoseSample]) -> float:
        first = line_angle_deg(samples[0].points.get("left_hip"), samples[0].points.get("right_hip"))
        last = line_angle_deg(samples[-1].points.get("left_hip"), samples[-1].points.get("right_hip"))
        return normalize_angle_delta(last - first)

    def _lead_side(self, sample: PoseSample) -> str | None:
        """Approximate orthodox/southpaw from which shoulder/foot is closer to camera (smaller z)."""
        ls, rs = sample.points.get("left_shoulder"), sample.points.get("right_shoulder")
        la, ra = sample.points.get("left_ankle"), sample.points.get("right_ankle")
        votes: list[str] = []
        if ls is not None and rs is not None and ls.size > 2 and rs.size > 2:
            votes.append("left" if ls[2] < rs[2] else "right")
        if la is not None and ra is not None and la.size > 2 and ra.size > 2:
            votes.append("left" if la[2] < ra[2] else "right")
        if not votes:
            return None
        return "left" if votes.count("left") >= votes.count("right") else "right"
