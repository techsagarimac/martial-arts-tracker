"""App settings, calibration, privacy, and ML dataset collection."""

from __future__ import annotations

import streamlit as st

from config.settings import get_settings
from core.calibration import CALIBRATION_DISCLAIMER, build_calibration
from database.repositories import SessionRepository
from ml.dataset_collector import ML_CLASSES
from ml.pipeline import describe_pipeline
from ui.components import hero
from ui.pose3d import DISCLAIMER as POSE3D_DISCLAIMER


def render_settings() -> None:
    hero("Settings", "Calibration, privacy, and future ML collection. All defaults stay on-device.")
    settings = get_settings()

    st.subheader("Camera calibration (optional)")
    st.caption(CALIBRATION_DISCLAIMER)
    c1, c2, c3 = st.columns(3)
    st.session_state.ref_distance = c1.number_input(
        "Known reference distance (m)",
        min_value=0.0,
        value=float(st.session_state.get("ref_distance") or 0.0),
    )
    st.session_state.cam_distance = c2.number_input(
        "Camera distance to athlete (m)",
        min_value=0.0,
        value=float(st.session_state.get("cam_distance") or 0.0),
    )
    st.session_state.pixels_per_meter = c3.number_input(
        "Pixels per meter (if measured)",
        min_value=0.0,
        value=float(st.session_state.get("pixels_per_meter") or 0.0),
    )
    result = build_calibration(
        None,
        st.session_state.ref_distance,
        st.session_state.cam_distance,
        st.session_state.pixels_per_meter,
    )
    st.info(result.message)

    st.subheader("Privacy")
    st.markdown(
        """
- Camera frames are processed locally.
- Videos are not uploaded to external servers by this app.
- Faces are not recognized and no biometric identity profile is created.
- You can delete session data below.
- Feedback is for training, not medical or injury diagnosis.
        """
    )
    st.caption(POSE3D_DISCLAIMER)
    if st.button("Delete all sessions and technique events"):
        SessionRepository().delete_all()
        st.success("Training history cleared.")

    st.subheader("Machine-learning extension")
    st.write(describe_pipeline())
    st.caption("Labels you can record later: " + ", ".join(ML_CLASSES))
    st.code(str(settings.ml_dataset_dir))

    st.subheader("About sparring mode")
    st.write(
        "Two-person tracking is architected (Fighter A / Fighter B interfaces) "
        "but not enabled. A single-pose model cannot reliably separate two athletes."
    )
