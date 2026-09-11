"""Progress charts across saved sessions."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from core.session_manager import format_duration
from database.repositories import SessionRepository
from ui.components import hero, metric_card


def render_performance() -> None:
    hero("Performance", "Session-to-session trends. Scores are training heuristics.")
    sessions = list(reversed(SessionRepository().list_sessions()))
    if not sessions:
        st.info("Train a few sessions to unlock charts.")
        return

    df = pd.DataFrame(
        {
            "Session": [f"{i+1}" for i in range(len(sessions))],
            "Score": [s.average_score for s in sessions],
            "Punches": [s.punch_count for s in sessions],
            "Kicks": [s.kick_count for s in sessions],
            "Speed": [s.average_speed for s in sessions],
            "Balance": [s.average_balance for s in sessions],
            "Form": [s.average_form for s in sessions],
            "Duration min": [s.duration_seconds / 60.0 for s in sessions],
        }
    )
    first, last = df["Score"].iloc[0], df["Score"].iloc[-1]
    improvement = ((last - first) / first * 100) if first else 0
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Sessions", str(len(sessions)))
    with c2:
        metric_card("Latest score", f"{last:.0f}")
    with c3:
        metric_card("Improvement", f"{improvement:+.0f}%", f"Session 1 → {len(sessions)}")

    charts = [
        ("Technique score over time", "Score"),
        ("Punch count", "Punches"),
        ("Kick count", "Kicks"),
        ("Movement speed estimate (normalized / s)", "Speed"),
        ("Balance score", "Balance"),
        ("Form score", "Form"),
        ("Training duration (minutes)", "Duration min"),
    ]
    for title, col in charts:
        fig = px.bar(df, x="Session", y=col, title=title) if col != "Score" else px.line(
            df, x="Session", y=col, markers=True, title=title
        )
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0B0F14", plot_bgcolor="#151B23", height=320)
        fig.update_traces(marker_color="#F5C518")
        st.plotly_chart(fig, width="stretch")

    st.dataframe(
        pd.DataFrame(
            {
                "When": [s.started_at.replace("T", " ") for s in sessions],
                "Duration": [format_duration(s.duration_seconds) for s in sessions],
                "Score": [round(s.average_score) for s in sessions],
            }
        ),
        hide_index=True,
        width="stretch",
    )
