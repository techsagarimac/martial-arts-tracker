"""Estimated 3D skeleton from monocular pose landmarks (Plotly)."""

from __future__ import annotations

import plotly.graph_objects as go

from core.pose_detector import SKELETON_EDGES, TRACKED_LANDMARKS

DISCLAIMER = (
    "Estimated 3D pose from a single camera — not an exact body scan or anatomical model."
)


def _coords(frame: dict, name: str) -> tuple[float, float, float] | None:
    world = (frame.get("world") or {}).get(name)
    if world is not None and len(world) >= 3:
        x, y, z = float(world[0]), float(world[1]), float(world[2])
        return x, -y, z  # lift Y so the head is up
    image = (frame.get("image") or {}).get(name)
    if image is not None and len(image) >= 2:
        x, y = float(image[0]), float(image[1])
        z = float(image[2]) if len(image) > 2 else 0.0
        return x, 1.0 - y, z
    return None


def build_pose_figure(frame: dict | None, title: str = "Estimated 3D pose") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=f"{title}<br><sup>{DISCLAIMER}</sup>",
        paper_bgcolor="#0B0F14",
        plot_bgcolor="#0B0F14",
        margin=dict(l=0, r=0, t=70, b=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y (up)",
            zaxis_title="Z (depth)",
            aspectmode="data",
            bgcolor="#0B0F14",
        ),
        showlegend=False,
        height=520,
    )
    if not frame:
        fig.add_annotation(text="No pose frames stored yet.", showarrow=False)
        return fig

    xs, ys, zs, labels = [], [], [], []
    points: dict[str, tuple[float, float, float]] = {}
    for name in TRACKED_LANDMARKS:
        xyz = _coords(frame, name)
        if xyz is None:
            continue
        points[name] = xyz
        xs.append(xyz[0])
        ys.append(xyz[1])
        zs.append(xyz[2])
        labels.append(name.replace("_", " "))

    for a, b in SKELETON_EDGES:
        if a in points and b in points:
            xa, ya, za = points[a]
            xb, yb, zb = points[b]
            fig.add_trace(
                go.Scatter3d(
                    x=[xa, xb],
                    y=[ya, yb],
                    z=[za, zb],
                    mode="lines",
                    line=dict(color="#F5C518", width=6),
                    hoverinfo="skip",
                )
            )
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker=dict(size=5, color="#4CC9F0"),
        )
    )
    return fig
