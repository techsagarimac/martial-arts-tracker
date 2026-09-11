"""Filterable technique history table."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database.repositories import AthleteRepository, SessionRepository
from ui.components import hero


def render_technique_analysis() -> None:
    hero("Technique Analysis", "Every stored detection with score, speed estimate, and confidence.")
    athletes = AthleteRepository().list_all()
    athlete_id = None
    if athletes:
        choice = st.selectbox("Athlete", ["All"] + [a.name for a in athletes])
        if choice != "All":
            athlete_id = next(a.id for a in athletes if a.name == choice)

    events = SessionRepository().list_events(athlete_id=athlete_id)
    if not events:
        st.info("No techniques stored yet.")
        return

    names = sorted({e.technique_name for e in events})
    selected = st.multiselect("Filter techniques", names, default=names)
    rows = []
    for event in events:
        if event.technique_name not in selected:
            continue
        speed = "—"
        if event.speed_estimated_mps:
            speed = f"~{event.speed_estimated_mps:.2f} m/s est."
        elif event.speed_normalized:
            speed = f"{event.speed_normalized:.2f} n/s"
        rows.append(
            {
                "Date": (event.detected_at or "")[:16].replace("T", " "),
                "Technique": event.technique_name,
                "Score": round(event.score),
                "Speed": speed,
                "Confidence": f"{int(event.confidence * 100)}%",
                "Side": event.side,
                "Category": event.category,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Speed is a visual estimate unless calibration is set. Confidence is a rule-match score.")
