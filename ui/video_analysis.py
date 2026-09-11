"""Upload a recorded training video and run the same local pipeline."""

from __future__ import annotations

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st

from config.settings import get_settings
from core.calibration import build_calibration
from core.pipeline import FramePipeline
from core.report import build_report
from core.session_manager import SessionManager, format_duration
from database.repositories import AthleteRepository, SessionRepository
from martial_arts.base import STYLE_LABELS, slug_from_display
from ui.components import hero, metric_card
from ui.pose3d import build_pose_figure
from utils.logger import get_logger
from utils.video_utils import is_valid_upload_name, open_video, video_properties

logger = get_logger("ui.video")


def render_video_analysis() -> None:
    settings = get_settings()
    hero("Video Analysis", "MP4, AVI, and MOV files are processed on this machine only.")
    athletes = AthleteRepository().list_all() or [AthleteRepository().get_or_create_default()]

    c1, c2, c3 = st.columns(3)
    athlete = c1.selectbox("Athlete", athletes, format_func=lambda a: a.name, key="vid_athlete")
    style_name = c2.selectbox("Martial art", STYLE_LABELS, index=1, key="vid_style")
    max_width = c3.select_slider("Process width", options=[320, 480, 640], value=480)

    uploaded = st.file_uploader("Training video", type=["mp4", "avi", "mov"])
    if uploaded is None:
        st.info("Upload a clip of a single athlete, ideally full body, from a stable camera.")
        return
    if not is_valid_upload_name(uploaded.name):
        st.error("Unsupported file type.")
        return

    dest = settings.uploads_dir / uploaded.name
    dest.write_bytes(uploaded.getbuffer())
    probe = open_video(dest)
    if probe is None:
        st.error("This file could not be opened as a video. Try another MP4 export.")
        return

    props = video_properties(probe)
    probe.release()
    st.caption(
        f"{uploaded.name} · {props['width']}×{props['height']} · "
        f"{props['fps']:.1f} fps · {format_duration(float(props['duration_seconds']))}"
    )
    if int(props["frame_count"]) == 0:
        st.warning("This video reports zero frames.")
        return

    skip = 1 if props["fps"] and props["fps"] > 40 else 0
    if st.button("Analyze video", type="primary"):
        cap = open_video(dest)
        if cap is None:
            st.error("Could not reopen the video for analysis.")
            return
        cal = build_calibration(
            athlete.height_cm,
            st.session_state.get("ref_distance"),
            st.session_state.get("cam_distance"),
            st.session_state.get("pixels_per_meter"),
        )
        pipeline = FramePipeline(
            style=slug_from_display(style_name),
            calibration=cal.settings,
        )
        pipeline.max_width = int(max_width)
        pipeline.frame_skip = skip
        mgr = SessionManager(athlete, style_name, source="video", video_name=uploaded.name)
        preview_frames: list = []
        timeline = []
        progress = st.progress(0, text="Analyzing…")
        total = max(int(props["frame_count"]), 1)
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if skip and idx % (skip + 1) != 0:
                    idx += 1
                    continue
                analysis = pipeline.process(frame, recording=False)
                event = mgr.ingest(analysis)
                t = idx / props["fps"] if props["fps"] else idx / 30.0
                if event:
                    timeline.append({"time": t, "technique": event.technique_name, "score": event.score})
                if len(preview_frames) < 12 and analysis.pose_ok:
                    preview_frames.append(cv2.cvtColor(analysis.overlay_bgr, cv2.COLOR_BGR2RGB))
                idx += 1
                if idx % 5 == 0:
                    progress.progress(min(idx / total, 1.0), text=f"Frame {idx}/{total}")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Video analysis failed: %s", exc)
            st.error("Analysis stopped because a frame could not be processed. Partial results were kept.")
        finally:
            cap.release()
            pipeline.close()
            progress.progress(1.0, text="Done")

        saved = mgr.save(notes=f"Video analysis of {uploaded.name}")
        st.session_state.video_session_id = saved.id
        st.session_state.video_previews = preview_frames
        st.session_state.video_timeline = timeline
        st.session_state.video_pose_history = pipeline.pose_history
        st.success(f"Saved local session {saved.id}.")

    session_id = st.session_state.get("video_session_id")
    if not session_id:
        return
    session = SessionRepository().get(session_id)
    if session is None:
        return

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Punches", str(session.punch_count))
    with k2:
        metric_card("Kicks", str(session.kick_count))
    with k3:
        metric_card("Average score", f"{session.average_score:.0f}")
    with k4:
        metric_card("Max speed (n/s)", f"{session.max_speed:.2f}", "Normalized estimate")
    with k5:
        metric_card("Duration", format_duration(session.duration_seconds))

    previews = st.session_state.get("video_previews") or []
    if previews:
        st.markdown("#### Pose overlay samples")
        cols = st.columns(min(4, len(previews)))
        for i, img in enumerate(previews[:4]):
            cols[i % 4].image(img, width="stretch")

    timeline = st.session_state.get("video_timeline") or []
    if timeline:
        df = pd.DataFrame(timeline)
        fig = px.scatter(df, x="time", y="score", color="technique", title="Detected techniques")
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0B0F14", plot_bgcolor="#151B23")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(df, hide_index=True, width="stretch")

    history = st.session_state.get("video_pose_history") or []
    if history:
        st.markdown("#### Estimated 3D pose")
        idx = st.slider("Timeline", 0, len(history) - 1, 0)
        slow = st.checkbox("Slow-motion step (use slider)")
        st.plotly_chart(build_pose_figure(history[idx]), width="stretch")
        st.caption("Rotate and zoom the 3D view. " + ("Scrub the timeline for replay." if slow else "Drag to rotate."))

    previous = None
    all_sessions = SessionRepository().list_sessions(athlete.id)
    if len(all_sessions) >= 2:
        previous = all_sessions[1]
    report = build_report(session, previous)
    st.download_button("Export session report", report, file_name=f"session_{session.id}_report.txt")
    st.text_area("Report preview", report, height=280)
