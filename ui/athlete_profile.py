"""Athlete profiles without biometric identity."""

from __future__ import annotations

import streamlit as st

from database.repositories import AthleteRepository, SessionRepository
from martial_arts.base import STYLE_LABELS
from models.athlete import EXPERIENCE_LEVELS, TRAINING_GOALS, Athlete
from ui.components import hero, metric_card


def render_athlete_profile() -> None:
    hero("Athlete Profile", "Optional height is used only for speed scale. No face identity is stored.")
    repo = AthleteRepository()
    athletes = repo.list_all()
    if not athletes:
        athletes = [repo.get_or_create_default()]

    names = [a.name for a in athletes] + ["＋ New athlete"]
    choice = st.selectbox("Select", names)
    current = None if choice == "＋ New athlete" else next(a for a in athletes if a.name == choice)

    with st.form("athlete_form"):
        name = st.text_input("Name", current.name if current else "")
        c1, c2, c3 = st.columns(3)
        age = c1.number_input("Age (optional)", min_value=0, max_value=100, value=int(current.age or 0) if current else 0)
        height = c2.number_input(
            "Height cm (optional)",
            min_value=0.0,
            max_value=250.0,
            value=float(current.height_cm or 0) if current else 0.0,
        )
        art = c3.selectbox(
            "Martial art",
            STYLE_LABELS,
            index=STYLE_LABELS.index(current.martial_art) if current and current.martial_art in STYLE_LABELS else 1,
        )
        exp = st.selectbox(
            "Experience",
            EXPERIENCE_LEVELS,
            index=EXPERIENCE_LEVELS.index(current.experience_level) if current and current.experience_level in EXPERIENCE_LEVELS else 0,
        )
        goal = st.selectbox(
            "Training goal",
            TRAINING_GOALS,
            index=TRAINING_GOALS.index(current.training_goal) if current and current.training_goal in TRAINING_GOALS else 1,
        )
        submitted = st.form_submit_button("Save profile")
    if submitted:
        athlete = Athlete(
            id=None if current is None else current.id,
            name=name or "Athlete",
            age=int(age) or None,
            height_cm=float(height) or None,
            martial_art=art,
            experience_level=exp,
            training_goal=goal,
        )
        saved = repo.update(athlete) if athlete.id else repo.create(athlete)
        st.success(f"Saved {saved.name}.")
        st.rerun()

    if current and current.id:
        sessions = SessionRepository().list_sessions(current.id)
        events = SessionRepository().list_events(athlete_id=current.id)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            metric_card("Sessions", str(len(sessions)))
        with c2:
            metric_card("Techniques", str(sum(s.punch_count + s.kick_count for s in sessions)))
        with c3:
            avg = sum(s.average_score for s in sessions) / len(sessions) if sessions else 0
            metric_card("Average score", f"{avg:.0f}" if sessions else "—")
        with c4:
            best = max((s.best_score for s in sessions), default=0)
            metric_card("Best score", f"{best:.0f}" if sessions else "—")
        with c5:
            conf = [e.confidence for e in events]
            metric_card(
                "Avg. detection conf.",
                f"{sum(conf)/len(conf)*100:.0f}%" if conf else "—",
                "Not lab reaction time",
            )
        if st.button("Delete this athlete and their sessions"):
            repo.delete(current.id)
            st.success("Deleted locally.")
            st.rerun()
