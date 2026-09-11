"""Live webcam tracker with pose overlay and session recording."""

from __future__ import annotations

import cv2
import numpy as np
import streamlit as st

from config.settings import get_settings
from core.calibration import build_calibration
from core.pipeline import FramePipeline
from core.session_manager import SessionManager, format_duration
from database.repositories import AthleteRepository
from martial_arts.base import STYLE_LABELS, slug_from_display
from ui.components import feedback_chips, hero, metric_card, recording_banner
from utils.logger import get_logger
from utils.video_utils import camera_is_open, open_camera, read_latest_frame

logger = get_logger("ui.live")

RECONNECT_AFTER_FAILS = 8
STOP_AFTER_FAILS = 20


def _ensure_pipeline(style_slug: str, confidence: float, calibration) -> FramePipeline:
    pipe: FramePipeline | None = st.session_state.get("pipeline")
    if pipe is None:
        pipe = FramePipeline(
            style=style_slug,
            min_detection_confidence=confidence,
            calibration=calibration,
        )
        st.session_state.pipeline = pipe
        return pipe
    pipe.set_calibration(calibration)
    return pipe


def _release_camera() -> None:
    cap = st.session_state.pop("camera", None)
    st.session_state.cam_fails = 0
    if cap is not None:
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            logger.warning("Camera release failed", exc_info=True)


def _open_live_camera(index: int, width: int):
    cap = open_camera(index, width=width, height=0, fps=None, warmup_frames=10)
    if cap is None:
        return None
    st.session_state.camera = cap
    st.session_state.cam_fails = 0
    st.session_state.live_cam_index = index
    st.session_state.live_width = width
    return cap


def _apply_live_metrics(analysis, pipeline: FramePipeline) -> None:
    live_metrics = {
        "technique": "—",
        "confidence": "—",
        "speed": f"{analysis.motion.normalized_speed:.2f} n/s",
        "angle": "—",
        "balance": f"{analysis.form.balance:.0f}",
        "form": f"{analysis.form.overall:.0f}/100",
        "fps": f"{analysis.fps:.1f}",
    }
    tech = analysis.technique or pipeline.last_technique
    if tech:
        live_metrics["technique"] = "Technique uncertain" if tech.uncertain else tech.display_name
        live_metrics["confidence"] = f"{int(tech.confidence * 100)}%"
    if analysis.motion.estimated_mps is not None:
        live_metrics["speed"] = f"~{analysis.motion.estimated_mps:.2f} m/s est."
    left_elbow = analysis.angles.left_elbow
    right_elbow = analysis.angles.right_elbow
    if left_elbow == left_elbow:
        live_metrics["angle"] = f"L elbow {left_elbow:.0f}°"
    elif right_elbow == right_elbow:
        live_metrics["angle"] = f"R elbow {right_elbow:.0f}°"
    st.session_state.last_overlay = analysis.overlay_bgr
    st.session_state.live_feedback = analysis.feedback
    st.session_state.live_metrics = live_metrics
    st.session_state.live_angles = analysis.angles.as_dict()
    st.session_state.live_stance = analysis.stance


def _placeholder_frame() -> np.ndarray:
    frame = np.full((360, 640, 3), 18, dtype=np.uint8)
    cv2.putText(
        frame,
        "Start session to open camera",
        (90, 185),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (200, 200, 200),
        1,
    )
    return frame


def _render_metrics() -> None:
    live_metrics = st.session_state.get("live_metrics") or {
        "technique": "—",
        "confidence": "—",
        "speed": "—",
        "angle": "—",
        "balance": "—",
        "form": "—",
        "fps": "—",
    }
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        metric_card("Technique", live_metrics["technique"], live_metrics["confidence"])
    with m2:
        metric_card("Speed", live_metrics["speed"], "Visual estimate")
    with m3:
        metric_card("Joint angle", live_metrics["angle"])
    with m4:
        metric_card("Balance", live_metrics["balance"])
    with m5:
        metric_card("Form score", live_metrics["form"], "Heuristic")
    st.markdown("#### Coaching cues")
    feedback_chips(st.session_state.get("live_feedback") or [])
    mgr: SessionManager | None = st.session_state.get("session_mgr")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Punches", str(mgr.record.punch_count if mgr else 0))
    with c2:
        metric_card("Kicks", str(mgr.record.kick_count if mgr else 0))
    with c3:
        metric_card("Session time", format_duration(mgr.record.duration_seconds) if mgr else "00:00")
    with c4:
        metric_card("Average score", f"{mgr.record.average_score:.0f}" if mgr and mgr.record.average_score else "—")
    stance = st.session_state.get("live_stance")
    if stance is not None:
        st.caption(f"Stance: {stance.label}. " + " ".join(stance.feedback[:2]))


@st.fragment(key="live_camera_panel")
def _live_panel() -> None:
    """Update one image slot in a loop so Streamlit does not remount the video each frame."""
    img_slot = st.empty()
    metrics_slot = st.empty()
    running = bool(st.session_state.get("live_running"))

    if not running:
        img_slot.image(_placeholder_frame(), channels="BGR")
        with metrics_slot.container():
            _render_metrics()
        return

    pipeline: FramePipeline | None = st.session_state.get("pipeline")
    cap = st.session_state.get("camera")
    stats: SessionManager | None = st.session_state.get("session_mgr")
    if pipeline is None or not camera_is_open(cap):
        st.session_state.live_running = False
        img_slot.image(_placeholder_frame(), channels="BGR")
        st.warning("Camera or pose pipeline was reset. Press Start session again.")
        return

    frames = 0
    while st.session_state.get("live_running"):
        ok, frame = read_latest_frame(cap)
        if not ok or frame is None:
            fails = int(st.session_state.get("cam_fails", 0)) + 1
            st.session_state.cam_fails = fails
            if fails == RECONNECT_AFTER_FAILS:
                _release_camera()
                cap = _open_live_camera(
                    int(st.session_state.get("live_cam_index", 0)),
                    int(st.session_state.get("live_width", 640)),
                )
            if fails >= STOP_AFTER_FAILS:
                st.session_state.live_running = False
                _release_camera()
                st.error(
                    "Camera feed stopped. Close other apps using the webcam, "
                    "then start the session again."
                )
                break
            continue

        st.session_state.cam_fails = 0
        frame = cv2.flip(frame, 1)
        analysis = pipeline.process(frame, recording=True)
        if stats is not None:
            stats.ingest(analysis)
        _apply_live_metrics(analysis, pipeline)
        img_slot.image(analysis.overlay_bgr, channels="BGR")
        frames += 1
        if frames % 15 == 0:
            with metrics_slot.container():
                _render_metrics()

    with metrics_slot.container():
        _render_metrics()


def render_live_tracker() -> None:
    settings = get_settings()
    hero("Live Tracker", "Webcam pose overlay, joint angles, and rule-based strike detection.")

    athletes = AthleteRepository().list_all()
    if not athletes:
        athletes = [AthleteRepository().get_or_create_default()]

    with st.sidebar:
        st.markdown("### Session")
        athlete = st.selectbox("Athlete", athletes, format_func=lambda a: a.name)
        default_style = 1 if "Kickboxing" in STYLE_LABELS else 0
        if "Kickboxing" in STYLE_LABELS:
            default_style = STYLE_LABELS.index("Kickboxing")
        style_name = st.selectbox("Martial art", STYLE_LABELS, index=default_style)
        camera_index = st.number_input("Camera", min_value=0, max_value=8, value=int(settings.camera_index))
        confidence = st.slider("Detection confidence", 0.2, 0.9, float(settings.min_detection_confidence), 0.05)
        resolution = st.select_slider("Resolution width", options=[320, 480, 640, 800], value=640)
        notes = st.text_input("Session notes", "")
        st.caption("Camera, style, and resolution lock when the session starts.")

    cal = build_calibration(
        athlete.height_cm,
        st.session_state.get("ref_distance"),
        st.session_state.get("cam_distance"),
        st.session_state.get("pixels_per_meter"),
    )
    style_slug = slug_from_display(style_name)
    pipeline: FramePipeline | None = st.session_state.get("pipeline")

    recording_banner(bool(st.session_state.get("live_running")))

    b1, b2, b3 = st.columns([1, 1, 2])
    start = b1.button("Start session", type="primary", width="stretch")
    stop = b2.button("Stop & save", width="stretch")
    backend = pipeline.pose_backend if pipeline is not None else "idle"
    b3.caption(f"Pose engine: `{backend}` · processed locally")

    if start:
        _release_camera()
        cap = _open_live_camera(int(camera_index), int(resolution))
        if cap is None:
            st.error(
                "Camera unavailable. On macOS, allow Camera access for Terminal/Cursor, "
                "quit Zoom/Meet/Photo Booth, then try another camera index. "
                "You can still analyze uploaded videos."
            )
            st.session_state.live_running = False
        else:
            pipeline = _ensure_pipeline(style_slug, float(confidence), cal.settings)
            pipeline.max_width = int(resolution)
            pipeline.set_style(style_slug)
            pipeline.feedback.reset()
            pipeline.form.reset()
            pipeline.motion.reset()
            pipeline.last_technique = None
            st.session_state.session_mgr = SessionManager(athlete, style_name, source="live")
            st.session_state.live_feedback = []
            st.session_state.last_overlay = None
            st.session_state.live_running = True

    if stop and (st.session_state.get("live_running") or st.session_state.get("session_mgr")):
        st.session_state.live_running = False
        mgr: SessionManager | None = st.session_state.get("session_mgr")
        _release_camera()
        if mgr is not None:
            saved = mgr.save(notes)
            st.success(f"Session saved locally (id {saved.id}). Nothing was uploaded.")
            st.session_state.session_mgr = None
        st.rerun()

    _live_panel()
