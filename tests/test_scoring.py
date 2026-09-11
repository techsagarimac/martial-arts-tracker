"""Scoring engine, form analyzer, and confidence clipping."""

from __future__ import annotations

from core.form_analyzer import FormAnalyzer, StanceAnalyzer
from core.scoring_engine import ScoringEngine
from models.technique import FormScores, TechniqueDetection
from tests.helpers import feed_tracker, make_pose, standing_guard


def test_overall_is_weighted_and_bounded():
    form = FormScores(accuracy=80, alignment=90, extension=70, balance=60, consistency=100, overall=0)
    scored = ScoringEngine().score(form)
    expected = 0.30 * 80 + 0.20 * 90 + 0.20 * 70 + 0.15 * 60 + 0.15 * 100
    assert scored.overall == round(expected)
    assert 0 <= scored.overall <= 100


def test_uncertain_technique_reduces_score():
    form = FormScores(80, 80, 80, 80, 80, 80)
    detection = TechniqueDetection(None, "Technique uncertain", "punch", 0.4, uncertain=True)
    scored = ScoringEngine().score(form, detection)
    assert scored.overall < 80


def test_stance_feet_too_close():
    pts = standing_guard()
    pts["left_ankle"][0] = 0.49
    pts["right_ankle"][0] = 0.51
    pose = make_pose(pts)
    from core.angle_calculator import AngleCalculator

    stance = StanceAnalyzer().analyze(pose, AngleCalculator().calculate(pose))
    assert stance.label in {"NEEDS IMPROVEMENT", "GOOD"}
    assert any("close" in f.lower() or "wide" in f.lower() or "balanced" in f.lower() for f in stance.feedback)


def test_undetected_pose_has_zero_form():
    pose = make_pose({}, detected=False)
    from core.angle_calculator import AngleCalculator
    from models.technique import JointAngles

    tracker = feed_tracker([pose])
    form = FormAnalyzer().analyze(pose, JointAngles(), StanceAnalyzer().analyze(pose, JointAngles()), None, tracker)
    assert form.overall == 0
