"""Home dashboard: athlete performance snapshot and recent sessions."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.session_manager import format_duration
from database.repositories import AthleteRepository, SessionRepository
from ui.components import hero, metric_card


def _execution_times(events) -> list[float]:
    """Onset-to-save gaps are not true lab reaction times; we expose execution confidence proxy."""
    return [e.confidence for e in events if e.confidence]


def render_dashboard() -> None:
    hero(
        "AI Martial Arts Motion Tracker",
        "Local camera training assistant — pose, technique heuristics, and session history.",
    )
    athletes = AthleteRepository().list_all()
    sessions = SessionRepository().list_sessions()
    events = SessionRepository().list_events()

    athlete_name = athletes[0].name if athletes else "—"
    total_tech = sum(s.punch_count + s.kick_count for s in sessions)
    avg_score = sum(s.average_score for s in sessions) / len(sessions) if sessions else 0
    best = max((s.best_score for s in sessions), default=0)
    exec_proxy = _execution_times(events)
    avg_exec = f"{sum(exec_proxy)/len(exec_proxy)*100:.0f}%" if exec_proxy else "—"

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        metric_card("Athlete", athlete_name)
    with c2:
        metric_card("Total Sessions", str(len(sessions)))
    with c3:
        metric_card("Total Techniques", str(total_tech))
    with c4:
        metric_card("Average Score", f"{avg_score:.0f}" if sessions else "—", "Training heuristic")
    with c5:
        metric_card("Best Score", f"{best:.0f}" if sessions else "—")

    st.caption(
        "Average reaction time is not measured unless a stimulus cue is shown. "
        f"Mean detection confidence across stored techniques: {avg_exec}."
    )

    if not sessions:
        st.info("No sessions yet. Open Live Tracker or Video Analysis to record one.")
        return

    recent = sessions[:8]
    df = pd.DataFrame(
        {
            "Session": [f"S{len(sessions)-i}" for i in range(len(recent))][::-1],
            "Score": [s.average_score for s in recent][::-1],
            "Punches": [s.punch_count for s in recent][::-1],
            "Kicks": [s.kick_count for s in recent][::-1],
            "Duration": [format_duration(s.duration_seconds) for s in recent][::-1],
        }
    )
    left, right = st.columns((1.4, 1))
    with left:
        fig = px.line(df, x="Session", y="Score", markers=True, title="Session progress")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0B0F14", plot_bgcolor="#151B23")
        fig.update_traces(line_color="#F5C518")
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Recent sessions")
        st.dataframe(
            pd.DataFrame(
                {
                    "Date": [s.started_at.replace("T", " ") for s in recent],
                    "Style": [s.martial_art for s in recent],
                    "Score": [round(s.average_score) for s in recent],
                    "P": [s.punch_count for s in recent],
                    "K": [s.kick_count for s in recent],
                }
            ),
            hide_index=True,
            width="stretch",
        )

    if len(sessions) >= 2 and sessions[1].average_score > 0:
        delta = (sessions[0].average_score - sessions[1].average_score) / sessions[1].average_score * 100
        st.success(f"Latest session vs previous: {delta:+.0f}% average score.")
