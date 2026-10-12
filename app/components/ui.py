"""Presentation helpers: CSS, metric tiles, section headers, empty states."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.config import PALETTE

CSS_PATH = Path(__file__).resolve().parents[1] / "styles" / "app.css"


def inject_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", level: str | None = None) -> None:
    tag = f'<span class="fil-level">{level}</span>' if level else ""
    st.markdown(
        f'<div class="fil-header">{tag}<h1>{title}</h1>'
        f'<p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def section(title: str, note: str = "") -> None:
    st.markdown(
        f'<div class="fil-section"><h3>{title}</h3>'
        + (f'<span>{note}</span>' if note else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def metric_row(items: list[tuple[str, str, str]], columns: int | None = None) -> None:
    """``items`` is ``(label, value, caption)``."""
    cols = st.columns(columns or len(items))
    for col, (label, value, caption) in zip(cols, items):
        col.markdown(
            f'<div class="fil-metric"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="caption">{caption}</div></div>',
            unsafe_allow_html=True,
        )


_MD = (
    (r"\*\*(.+?)\*\*", r"<strong>\1</strong>"),
    (r"`(.+?)`", r"<code>\1</code>"),
    (r"(?<![*\w])\*(?!\s)(.+?)(?<!\s)\*(?![*\w])", r"<em>\1</em>"),
)


def _inline_md(text: str) -> str:
    """Render the tiny subset of markdown used in notes inside raw HTML."""
    import re

    for pattern, repl in _MD:
        text = re.sub(pattern, repl, text, flags=re.S)
    return text


def note(text: str, kind: str = "info") -> None:
    st.markdown(
        f'<div class="fil-note fil-note-{kind}">{_inline_md(text)}</div>',
        unsafe_allow_html=True,
    )


def caveat(text: str) -> None:
    note(text, "caveat")


def chip(text: str, colour: str | None = None) -> str:
    c = colour or PALETTE["accent"]
    return (
        f'<span class="fil-chip" style="border-color:{c};color:{c}">{text}</span>'
    )


def empty(message: str) -> None:
    st.markdown(f'<div class="fil-empty">{message}</div>', unsafe_allow_html=True)


def plot(fig, key: str | None = None, **kwargs) -> None:
    st.plotly_chart(
        fig, width="stretch", key=key,
        config={"displayModeBar": False, "scrollZoom": False}, **kwargs,
    )
