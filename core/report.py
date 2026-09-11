"""Plain-text / Markdown session reports for local export."""

from __future__ import annotations

from core.session_manager import format_duration
from models.session import SessionRecord


def improvement_vs_previous(current: SessionRecord, previous: SessionRecord | None) -> str:
    if previous is None or previous.average_score <= 0 or current.average_score <= 0:
        return "Not enough history to compare."
    delta = ((current.average_score - previous.average_score) / previous.average_score) * 100.0
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.0f}% compared with previous session"


def build_report(session: SessionRecord, previous: SessionRecord | None = None) -> str:
    best_name = "—"
    best_score = session.best_score
    if session.events:
        top = max(session.events, key=lambda e: e.score)
        best_name = f"{top.technique_name} — {top.score:.0f}/100"
        best_score = top.score
    warnings = session.form_warnings[:6] or ["No recurring form warnings recorded."]
    dist_lines = (
        "\n".join(f"  • {name}: {count}" for name, count in session.technique_distribution.items())
        or "  • None recorded"
    )
    progress = improvement_vs_previous(session, previous)
    speed_note = (
        "Normalized visual speed (not m/s unless calibrated)."
    )
    return f"""MARTIAL ARTS TRAINING REPORT
================================

Athlete:
{session.athlete_name or "Athlete"}

Martial art:
{session.martial_art}

Date:
{session.started_at}

Duration:
{format_duration(session.duration_seconds)}

Source:
{session.source}{f" ({session.video_name})" if session.video_name else ""}

Techniques:
Punches: {session.punch_count}
Kicks: {session.kick_count}

Technique distribution:
{dist_lines}

Average Form Score:
{session.average_score:.0f}/100

Best Technique:
{best_name}

Best score:
{best_score:.0f}/100

Average estimated movement speed:
{session.average_speed:.2f}  ({speed_note})

Maximum estimated movement speed:
{session.max_speed:.2f}

Areas to Improve:
{chr(10).join("• " + w for w in warnings)}

Progress:
{progress}

Notes:
{session.notes or "—"}

--------------------------------
This report is a training aid generated from camera-based pose estimation.
It is not an official competition score, medical assessment, or impact-force measurement.
"""
