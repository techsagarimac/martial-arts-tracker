"""Draw pose overlays and HUD elements onto BGR frames."""

from __future__ import annotations

import cv2
import numpy as np

from core.pose_detector import SKELETON_EDGES, TRACKED_LANDMARKS
from models.technique import JointAngles, TechniqueDetection

CYAN = (245, 197, 24)
GOLD = (24, 197, 245)
WHITE = (236, 238, 244)
RED = (70, 70, 220)
GREEN = (80, 200, 120)
DIM = (40, 44, 52)


def draw_skeleton(
    frame_bgr: np.ndarray,
    landmarks: dict,
    visibility: dict[str, float],
    min_visibility: float = 0.5,
) -> np.ndarray:
    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]
    points: dict[str, tuple[int, int]] = {}
    for name in TRACKED_LANDMARKS:
        lm = landmarks.get(name)
        if lm is None or visibility.get(name, 0.0) < min_visibility * 0.6:
            continue
        x = int(np.clip(lm[0] * w if lm[0] <= 1.5 else lm[0], 0, w - 1))
        y = int(np.clip(lm[1] * h if lm[1] <= 1.5 else lm[1], 0, h - 1))
        # Processed points are already normalized x,y.
        if 0 <= lm[0] <= 1.5 and 0 <= lm[1] <= 1.5:
            x, y = int(lm[0] * w), int(lm[1] * h)
        points[name] = (x, y)

    for a, b in SKELETON_EDGES:
        if a in points and b in points:
            cv2.line(overlay, points[a], points[b], GOLD, 2, cv2.LINE_AA)
    for name, pt in points.items():
        color = CYAN if "wrist" in name or "ankle" in name else WHITE
        cv2.circle(overlay, pt, 5, color, -1, cv2.LINE_AA)

    return overlay


def draw_hud(
    frame_bgr: np.ndarray,
    fps: float,
    recording: bool,
    technique: TechniqueDetection | None,
    angles: JointAngles | None,
    pose_ok: bool,
    pose_message: str,
) -> np.ndarray:
    frame = frame_bgr.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 36), (11, 15, 20), -1)
    status = "REC" if recording else "LIVE"
    rec_color = RED if recording else GREEN
    cv2.circle(frame, (18, 18), 7, rec_color, -1)
    cv2.putText(frame, status, (32, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (w - 110, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        WHITE,
        1,
        cv2.LINE_AA,
    )
    if not pose_ok:
        cv2.putText(
            frame,
            pose_message or "Pose not detected clearly",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            RED,
            2,
            cv2.LINE_AA,
        )
    elif technique is not None:
        label = technique.display_name
        if technique.confidence:
            label += f"  {int(technique.confidence * 100)}%"
        cv2.putText(frame, label, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, GOLD, 2, cv2.LINE_AA)
    if angles is not None:
        y = 58
        for name, value in (
            ("L elbow", angles.left_elbow),
            ("R elbow", angles.right_elbow),
            ("L knee", angles.left_knee),
            ("R knee", angles.right_knee),
        ):
            if value == value:
                cv2.putText(
                    frame,
                    f"{name}: {value:.0f}°",
                    (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    WHITE,
                    1,
                    cv2.LINE_AA,
                )
                y += 18
    return frame
