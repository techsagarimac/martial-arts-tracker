"""Saved training sessions with delete and report export."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from core.report import build_report
from core.session_manager import format_duration
from database.repositories import SessionRepository
from ui.components import hero


def render_session_history() -> None:
    hero("Session History", "Local SQLite records. Delete anytime — nothing is synced.")
    repo = SessionRepository()
    sessions = repo.list_sessions()
    if not sessions:
        st.info("No saved sessions.")
        return

    rows = [
        {
            "ID": s.id,
            "Date": s.started_at.replace("T", " "),
            "Athlete": s.athlete_name,
            "Style": s.martial_art,
            "Duration": format_duration(s.duration_seconds),
            "Punches": s.punch_count,
            "Kicks": s.kick_count,
            "Avg score": round(s.average_score),
            "Best": round(s.best_score),
            "Source": s.source,
        }
        for s in sessions
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    ids = [s.id for s in sessions if s.id is not None]
    selected = st.selectbox("Open session", ids)
    session = repo.get(int(selected))
    if session is None:
        return
    st.write(session.technique_distribution or {})
    if session.form_warnings:
        st.markdown("**Form warnings**")
        for w in session.form_warnings:
            st.caption(w)
    if session.notes:
        st.markdown(f"**Notes:** {session.notes}")

    previous = None
    idx = ids.index(session.id)
    if idx + 1 < len(ids):
        previous = repo.get(int(ids[idx + 1]))
    report = build_report(session, previous)
    st.download_button("Download report", report, file_name=f"session_{session.id}.txt")
    if session.events:
        st.dataframe(
            pd.DataFrame(
                {
                    "Technique": [e.technique_name for e in session.events],
                    "Score": [round(e.score) for e in session.events],
                    "Confidence": [f"{int(e.confidence*100)}%" for e in session.events],
                }
            ),
            hide_index=True,
            width="stretch",
        )
    if st.button("Delete this session"):
        repo.delete_session(int(session.id))
        st.success("Session deleted.")
        st.rerun()
