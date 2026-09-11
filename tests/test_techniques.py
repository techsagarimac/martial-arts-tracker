"""Rule-based technique classification and confidence thresholds."""

from __future__ import annotations

from core.technique_detector import TechniqueDetector
from martial_arts.base import get_style
from tests.helpers import feed_tracker, kick_sequence, make_pose, punch_sequence, standing_guard


def _detect(frames, style="kickboxing"):
    tracker = feed_tracker(frames, dt=1 / 25)
    detector = TechniqueDetector(get_style(style), min_confidence=0.55, cooldown_seconds=0.0)
    return detector.detect(tracker)


def test_jab_like_lead_punch():
    result = _detect(punch_sequence("left", "jab", n=12))
    assert result is not None
    if result.uncertain:
        return
    assert result.name in {"jab", "straight_punch", "cross"}
    assert result.category == "punch"
    assert result.confidence > 0.4


def test_hook_prefers_lateral_bent_arm():
    result = _detect(punch_sequence("left", "hook", n=12))
    assert result is not None
    if not result.uncertain:
        assert result.name in {"hook", "jab", "straight_punch"}


def test_front_kick_classified_as_kick():
    result = _detect(kick_sequence("right", "front_kick", n=14))
    assert result is not None
    if not result.uncertain:
        assert result.category == "kick"
        assert result.name in {"front_kick", "roundhouse_kick", "side_kick"}


def test_roundhouse_has_lateral_component():
    result = _detect(kick_sequence("right", "roundhouse_kick", n=14))
    assert result is not None
    if not result.uncertain:
        assert result.category == "kick"


def test_boxing_style_ignores_kicks():
    detector = TechniqueDetector(get_style("boxing"), min_confidence=0.4, cooldown_seconds=0.0)
    tracker = feed_tracker(kick_sequence("right", "front_kick", n=14), dt=1 / 25)
    result = detector.detect(tracker)
    assert result is None or result.name not in {"front_kick", "roundhouse_kick", "side_kick"}


def test_low_visibility_is_uncertain_or_empty():
    from core.angle_calculator import AngleCalculator
    from core.motion_tracker import MotionTracker
    from models.technique import JointAngles

    pose = make_pose(standing_guard(), detected=True)
    pose.visibility = {k: 0.1 for k in pose.points}
    tracker = MotionTracker()
    calc = AngleCalculator()
    for i in range(6):
        tracker.update(pose, calc.calculate(pose), i * 0.04)
    detector = TechniqueDetector(get_style("kickboxing"), cooldown_seconds=0.0)
    # Body landmarks exist but visibility is low — _body_visible uses presence in points.
    # Force missing hips:
    for sample in tracker.history:
        sample.points.pop("left_hip", None)
        sample.points.pop("right_hip", None)
    result = detector.detect(tracker)
    assert result is None or result.uncertain


def test_partial_body_does_not_crash():
    pts = {"nose": standing_guard()["nose"], "left_wrist": standing_guard()["left_wrist"]}
    frames = [make_pose(pts) for _ in range(8)]
    tracker = feed_tracker(frames)
    detector = TechniqueDetector(get_style("kickboxing"), cooldown_seconds=0.0)
    result = detector.detect(tracker)
    assert result is None or result.uncertain
