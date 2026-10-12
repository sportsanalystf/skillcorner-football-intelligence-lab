"""Plotly theme and shared chart furniture.

One template, registered once, so every figure in the app inherits the same
surface, grid weight, type scale and hover behaviour. Charts then only ever
specify what makes them different.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from src.config import PALETTE, SERIES

FONT = "Inter, SF Pro Text, Segoe UI, Helvetica Neue, Arial, sans-serif"

TEMPLATE_NAME = "fil_dark"


def _template() -> go.layout.Template:
    t = go.layout.Template()
    t.layout = go.Layout(
        font=dict(family=FONT, size=13, color=PALETTE["text"]),
        paper_bgcolor=PALETTE["surface"],
        plot_bgcolor=PALETTE["surface"],
        colorway=list(SERIES),
        margin=dict(l=48, r=24, t=48, b=44),
        title=dict(font=dict(size=16, color=PALETTE["text"]), x=0, xanchor="left", y=0.97),
        xaxis=dict(
            gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["line"],
            linecolor=PALETTE["line"], tickcolor=PALETTE["line"],
            tickfont=dict(size=12, color=PALETTE["text_secondary"]),
            title=dict(font=dict(size=12, color=PALETTE["muted"])),
            showspikes=False,
        ),
        yaxis=dict(
            gridcolor=PALETTE["grid"], zerolinecolor=PALETTE["line"],
            linecolor=PALETTE["line"], tickcolor=PALETTE["line"],
            tickfont=dict(size=12, color=PALETTE["text_secondary"]),
            title=dict(font=dict(size=12, color=PALETTE["muted"])),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", bordercolor=PALETTE["line"], borderwidth=0,
            font=dict(size=12, color=PALETTE["text_secondary"]),
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
        ),
        hoverlabel=dict(
            bgcolor=PALETTE["panel_alt"], bordercolor=PALETTE["line"],
            font=dict(family=FONT, size=12, color=PALETTE["text"]),
        ),
        hovermode="closest",
        dragmode=False,
    )
    return t


def register() -> None:
    pio.templates[TEMPLATE_NAME] = _template()
    pio.templates.default = TEMPLATE_NAME


register()


def apply(fig: go.Figure, *, title: str | None = None, height: int | None = None,
          showlegend: bool | None = None) -> go.Figure:
    fig.update_layout(template=TEMPLATE_NAME)
    if title is not None:
        fig.update_layout(title=title)
    if height is not None:
        fig.update_layout(height=height)
    if showlegend is not None:
        fig.update_layout(showlegend=showlegend)
    return fig


def team_colours(home_name: str, away_name: str) -> dict[str, str]:
    return {home_name: SERIES[0], away_name: SERIES[1]}
