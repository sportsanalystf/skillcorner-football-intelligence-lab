"""Non-pitch charts.

Deliberately a small vocabulary: ranked bars for magnitude, percentile rows for
a player profile, lines for change over time, scatter for two-variable
relationships, and stacked bars for composition. Every chart carries a hover
layer, a legend when it has two or more series, and direct labels where the
number is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import PALETTE, SERIES
from src.visualization.theme import apply

BAR_RADIUS = 4


def ranked_bar(
    df: pd.DataFrame, *, label: str, value: str, title: str | None = None,
    top: int = 15, colour: str = SERIES[0], value_fmt: str = ",.0f",
    height: int | None = None, ascending: bool = False,
    highlight: str | None = None, unit: str = "",
) -> go.Figure:
    d = df[[label, value]].dropna().sort_values(value, ascending=ascending).head(top)
    d = d.iloc[::-1]
    colours = [
        SERIES[3] if (highlight is not None and str(v) == str(highlight)) else colour
        for v in d[label]
    ]
    fig = go.Figure(
        go.Bar(
            x=d[value], y=d[label], orientation="h",
            marker=dict(color=colours, cornerradius=BAR_RADIUS),
            text=[format(v, value_fmt) + unit for v in d[value]],
            textposition="outside",
            textfont=dict(size=11, color=PALETTE["text_secondary"]),
            hovertemplate="<b>%{y}</b><br>%{x:" + value_fmt + "}" + unit + "<extra></extra>",
            showlegend=False,
        )
    )
    span = float(d[value].max() - min(0, d[value].min())) or 1.0
    fig.update_xaxes(showgrid=True, title=None, range=[min(0, d[value].min()), d[value].max() + span * 0.18])
    fig.update_yaxes(showgrid=False, title=None, automargin=True)
    return apply(fig, title=title, height=height or max(240, 26 * len(d) + 90), showlegend=False)


def percentile_rows(
    items: list[tuple[str, float, str]], *, title: str | None = None,
    height: int | None = None, colour: str = SERIES[0],
) -> go.Figure:
    """One row per metric: percentile as a bar, raw value as a direct label.

    ``items`` is ``(metric label, percentile 0-1, formatted raw value)``.
    Preferred over a radar: a radar's area is meaningless and its axis order
    changes the shape, while these rows are directly comparable.
    """
    items = list(items)[::-1]
    labels = [i[0] for i in items]
    pct = [100 * i[1] for i in items]
    raw = [i[2] for i in items]
    colours = [
        PALETTE["bad"] if p < 25 else PALETTE["warn"] if p < 50 else
        colour if p < 80 else PALETTE["good"] for p in pct
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[100] * len(pct), y=labels, orientation="h",
            marker=dict(color=PALETTE["panel_alt"], cornerradius=BAR_RADIUS),
            hoverinfo="skip", showlegend=False, name="",
        )
    )
    fig.add_trace(
        go.Bar(
            x=pct, y=labels, orientation="h",
            marker=dict(color=colours, cornerradius=BAR_RADIUS),
            text=raw, textposition="outside",
            textfont=dict(size=11, color=PALETTE["text"]),
            customdata=np.stack([raw, pct], axis=-1),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>%{customdata[1]:.0f}th percentile<extra></extra>",
            showlegend=False, name="",
        )
    )
    fig.update_layout(barmode="overlay", bargap=0.35)
    fig.update_xaxes(range=[0, 128], showgrid=False, visible=False)
    fig.update_yaxes(showgrid=False, title=None, automargin=True,
                     tickfont=dict(size=12, color=PALETTE["text_secondary"]))
    return apply(fig, title=title, height=height or max(220, 30 * len(items) + 80), showlegend=False)


def timeline(
    df: pd.DataFrame, *, x: str, series: dict[str, str], title: str | None = None,
    height: int = 300, x_title: str = "match clock (min)", y_title: str = "",
    fill: bool = False, hover_fmt: str = ".3f",
) -> go.Figure:
    """Lines over the match clock. ``series`` maps column -> display name."""
    fig = go.Figure()
    for i, (col, name) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=df[x], y=df[col], mode="lines", name=name,
                line=dict(color=SERIES[i % len(SERIES)], width=2),
                fill="tozeroy" if fill else None,
                fillcolor=None if not fill else _alpha(SERIES[i % len(SERIES)], 0.18),
                hovertemplate="%{y:" + hover_fmt + "}<extra>" + name + "</extra>",
            )
        )
    fig.update_layout(hovermode="x unified")
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    return apply(fig, title=title, height=height, showlegend=len(series) > 1)


def half_time_marker(fig: go.Figure, x: float = 45.0) -> go.Figure:
    fig.add_vline(x=x, line=dict(color=PALETTE["line"], width=1, dash="dot"))
    fig.add_annotation(x=x, yref="paper", y=1.02, text="HT", showarrow=False,
                       font=dict(size=10, color=PALETTE["muted"]))
    return fig


def scatter(
    df: pd.DataFrame, *, x: str, y: str, label: str, title: str | None = None,
    colour_col: str | None = None, size_col: str | None = None,
    height: int = 460, x_title: str | None = None, y_title: str | None = None,
    quadrants: bool = True, highlight: list[str] | None = None,
    annotate_top: int = 0,
) -> go.Figure:
    fig = go.Figure()
    groups = df.groupby(colour_col, observed=True) if colour_col else [(None, df)]
    for i, (key, g) in enumerate(groups):
        size = 9
        if size_col and g[size_col].notna().any():
            s = g[size_col].to_numpy(dtype=float)
            size = 8 + 16 * (s - np.nanmin(s)) / max(np.nanmax(s) - np.nanmin(s), 1e-9)
        fig.add_trace(
            go.Scatter(
                x=g[x], y=g[y], mode="markers", name=str(key) if key is not None else "",
                marker=dict(color=SERIES[i % len(SERIES)], size=size, opacity=0.85,
                            line=dict(color=PALETTE["surface"], width=1.5)),
                customdata=g[[label]].to_numpy(),
                hovertemplate="<b>%{customdata[0]}</b><br>" +
                              f"{x_title or x}: %{{x:.2f}}<br>{y_title or y}: %{{y:.2f}}<extra></extra>",
                showlegend=colour_col is not None,
            )
        )
    if quadrants and len(df) > 2:
        fig.add_hline(y=float(df[y].median()), line=dict(color=PALETTE["line"], width=1, dash="dot"))
        fig.add_vline(x=float(df[x].median()), line=dict(color=PALETTE["line"], width=1, dash="dot"))
    names = set(highlight or [])
    if annotate_top and len(df):
        top = df.assign(_r=df[x].rank(pct=True) + df[y].rank(pct=True)).nlargest(annotate_top, "_r")
        names |= set(top[label].astype(str))
    for _, r in df[df[label].astype(str).isin(names)].iterrows():
        fig.add_annotation(x=r[x], y=r[y], text=str(r[label]), showarrow=False,
                           yshift=13, font=dict(size=10, color=PALETTE["text"]))
    fig.update_xaxes(title=x_title or x)
    fig.update_yaxes(title=y_title or y)
    return apply(fig, title=title, height=height)


def stacked_bars(
    df: pd.DataFrame, *, index: str, columns: list[str], names: list[str] | None = None,
    title: str | None = None, height: int | None = None, horizontal: bool = True,
    unit: str = "",
) -> go.Figure:
    names = names or columns
    fig = go.Figure()
    for i, (c, n) in enumerate(zip(columns, names)):
        kw = dict(
            name=n, marker=dict(color=SERIES[i % len(SERIES)],
                                line=dict(color=PALETTE["surface"], width=2)),
            hovertemplate="<b>%{y}</b><br>" + n + ": %{x:,.0f}" + unit + "<extra></extra>",
        )
        if horizontal:
            fig.add_trace(go.Bar(y=df[index], x=df[c], orientation="h", **kw))
        else:
            kw["hovertemplate"] = "<b>%{x}</b><br>" + n + ": %{y:,.0f}" + unit + "<extra></extra>"
            fig.add_trace(go.Bar(x=df[index], y=df[c], **kw))
    fig.update_layout(barmode="stack", bargap=0.28)
    if horizontal:
        fig.update_yaxes(title=None, automargin=True, showgrid=False)
        fig.update_xaxes(title=None)
    return apply(fig, title=title, height=height or max(240, 26 * len(df) + 100),
                 showlegend=True)


def distribution(
    df: pd.DataFrame, *, value: str, group: str, title: str | None = None,
    height: int = 380, y_title: str | None = None,
) -> go.Figure:
    fig = go.Figure()
    for i, (key, g) in enumerate(df.groupby(group, observed=True)):
        fig.add_trace(
            go.Box(
                y=g[value], name=str(key), boxpoints="outliers",
                marker=dict(color=SERIES[i % len(SERIES)], size=5),
                line=dict(color=SERIES[i % len(SERIES)], width=1.6),
                fillcolor=_alpha(SERIES[i % len(SERIES)], 0.18),
                hovertemplate="%{y:.3f}<extra>" + str(key) + "</extra>",
                showlegend=False,
            )
        )
    fig.update_yaxes(title=y_title or value)
    return apply(fig, title=title, height=height, showlegend=False)


def bar_with_error(
    df: pd.DataFrame, *, x: str, y: str, err: str | None = None,
    title: str | None = None, height: int = 340, y_title: str = "",
    value_fmt: str = ".3f", colour: str = SERIES[0],
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=df[x].astype(str), y=df[y],
            marker=dict(color=colour, cornerradius=BAR_RADIUS),
            error_y=dict(type="data", array=df[err], color=PALETTE["muted"], thickness=1.2, width=4)
            if err and err in df else None,
            text=[format(v, value_fmt) for v in df[y]],
            textposition="outside", textfont=dict(size=11, color=PALETTE["text_secondary"]),
            hovertemplate="<b>%{x}</b><br>%{y:" + value_fmt + "}<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_yaxes(title=y_title)
    fig.update_xaxes(title=None, automargin=True)
    return apply(fig, title=title, height=height, showlegend=False)


def _alpha(hex_colour: str, a: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"
