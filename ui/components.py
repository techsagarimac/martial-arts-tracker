"""Shared Streamlit chrome: theme CSS, metric cards, feedback chips."""

from __future__ import annotations

import streamlit as st

from models.technique import FeedbackItem


THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');
html, body, [class*="css"]  { font-family: 'Outfit', sans-serif; }
.block-container { padding-top: 1.2rem; max-width: 1400px; }
.mat-hero {
    background: linear-gradient(135deg, #151B23 0%, #1E2A3A 55%, #3A2E12 100%);
    border: 1px solid #2A3544;
    border-radius: 18px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.1rem;
}
.mat-hero h1 { margin: 0 0 .35rem 0; font-size: 1.85rem; letter-spacing: .02em; }
.mat-hero p { margin: 0; color: #A9B4C2; }
.mat-card {
    background: #151B23;
    border: 1px solid #2A3544;
    border-radius: 14px;
    padding: 1rem 1.1rem;
}
.mat-metric {
    background: #151B23;
    border: 1px solid #2A3544;
    border-radius: 14px;
    padding: .9rem 1rem;
    min-height: 96px;
}
.mat-metric .label { color: #8B97A8; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
.mat-metric .value { font-size: 1.7rem; font-weight: 700; color: #F5C518; line-height: 1.15; margin-top: .2rem; }
.mat-metric .hint { color: #7D8794; font-size: .75rem; }
.chip {
    display: inline-block;
    padding: .28rem .7rem;
    border-radius: 999px;
    font-size: .82rem;
    margin: .15rem .2rem .15rem 0;
    font-weight: 600;
}
.chip-green { background: #16351f; color: #7DDA92; border: 1px solid #2f6a3d; }
.chip-yellow { background: #3a3110; color: #F5C518; border: 1px solid #8a7018; }
.chip-red { background: #3a1518; color: #ff8a8a; border: 1px solid #8a3038; }
.chip-info { background: #152433; color: #8ecae6; border: 1px solid #2c5364; }
.rec-banner {
    background: #3a1518;
    color: #ffd0d0;
    border: 1px solid #8a3038;
    border-radius: 10px;
    padding: .55rem .8rem;
    margin-bottom: .6rem;
    font-weight: 600;
}
div[data-testid="stSidebar"] { background: #0E131A; }
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="mat-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, hint: str = "") -> None:
    hint_html = f'<div class="hint">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="mat-metric"><div class="label">{label}</div>'
        f'<div class="value">{value}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def feedback_chips(items: list[FeedbackItem]) -> None:
    if not items:
        st.caption("No coaching cues right now.")
        return
    html = []
    for item in items[-6:]:
        html.append(f'<span class="chip chip-{item.level}">{item.message}</span>')
    st.markdown(" ".join(html), unsafe_allow_html=True)


def recording_banner(active: bool) -> None:
    if active:
        st.markdown(
            '<div class="rec-banner">● CAMERA ACTIVE — recording is processed locally and is not uploaded.</div>',
            unsafe_allow_html=True,
        )
