"""Report text and style plugin isolation."""

from __future__ import annotations

from core.report import build_report, improvement_vs_previous
from martial_arts.base import get_style
from models.session import SessionRecord


def test_report_contains_required_sections():
    session = SessionRecord(
        athlete_name="Alex",
        martial_art="Kickboxing",
        duration_seconds=762,
        punch_count=84,
        kick_count=39,
        average_score=87,
        best_score=93,
        form_warnings=["Guard position is low.", "Rotate your hip more"],
    )
    previous = SessionRecord(average_score=78)
    text = build_report(session, previous)
    assert "Alex" in text
    assert "Punches: 84" in text
    assert "Kicks: 39" in text
    assert "87/100" in text
    assert "not an official competition score" in text.lower() or "training aid" in text.lower()


def test_improvement_percent():
    current = SessionRecord(average_score=88)
    previous = SessionRecord(average_score=80)
    assert "+10%" in improvement_vs_previous(current, previous)


def test_styles_do_not_share_identical_enabled_sets():
    boxing = {t.id for t in get_style("boxing").enabled_techniques()}
    tkd = {t.id for t in get_style("taekwondo").enabled_techniques()}
    kick = {t.id for t in get_style("kickboxing").enabled_techniques()}
    assert "jab" in boxing
    assert "front_kick" not in boxing
    assert "front_kick" in tkd
    assert "jab" in kick and "roundhouse_kick" in kick
