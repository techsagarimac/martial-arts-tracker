"""AI Martial Arts Motion Tracker — Streamlit entry point."""

from __future__ import annotations

import streamlit as st

from config.settings import get_settings
from database.db import init_db
from database.repositories import AthleteRepository
from ui.athlete_profile import render_athlete_profile
from ui.components import inject_theme
from ui.dashboard import render_dashboard
from ui.live_tracker import render_live_tracker
from ui.performance import render_performance
from ui.session_history import render_session_history
from ui.settings_page import render_settings
from ui.technique_analysis import render_technique_analysis
from ui.video_analysis import render_video_analysis
from utils.logger import get_logger

PAGES = {
    "🏠 Dashboard": render_dashboard,
    "📷 Live Tracker": render_live_tracker,
    "🎥 Video Analysis": render_video_analysis,
    "🥋 Technique Analysis": render_technique_analysis,
    "📊 Performance": render_performance,
    "📅 Session History": render_session_history,
    "👤 Athlete Profile": render_athlete_profile,
    "⚙️ Settings": render_settings,
}


def main() -> None:
    st.set_page_config(
        page_title="AI Martial Arts Motion Tracker",
        page_icon="🥋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    settings = get_settings()
    logger = get_logger("app")
    try:
        init_db(settings.db_path)
        AthleteRepository().get_or_create_default()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Database unavailable: %s", exc)
        st.error("Local database could not be opened. Check that the data/ folder is writable.")
        return

    with st.sidebar:
        st.markdown("## 🥋 Motion Tracker")
        st.caption("On-device training assistant")
        page = st.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
        st.divider()
        st.caption("Camera data stays on this computer.")
        st.caption("Not a medical device. Not official scoring.")

    PAGES[page]()


if __name__ == "__main__":
    main()
