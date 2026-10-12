"""Pitch-based Plotly figures.

Every function here returns a `go.Figure` on a correctly proportioned pitch,
in the coordinate space the caller asks for. Two conventions are enforced:

* the pitch is always drawn with equal aspect, so a 10 m run looks the same
  vertically and horizontally;
* nothing is drawn outside the pitch rectangle except deliberate margin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import PALETTE, SEQ_CONTROL, SEQ_MAGNITUDE, SERIES
from src.utils.pitch import STANDARD_PITCH, Pitch, penalty_spots, pitch_line_segments
from src.visualization.theme import apply

MARGIN = 4.0


def blank_pitch(pitch: Pitch = STANDARD_PITCH, *, height: int = 520,
                title: str | None = None, show_direction: bool = False,
                direction_label: str = "attacking direction") -> go.Figure:
    fig = go.Figure()
    for xs, ys in pitch_line_segments(pitch):
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=PALETTE["pitch_line"], width=1.4),
                hoverinfo="skip", showlegend=False,
            )
        )
    sx, sy = zip(*penalty_spots(pitch))
    fig.add_trace(
        go.Scatter(x=list(sx), y=list(sy), mode="markers",
                   marker=dict(color=PALETTE["pitch_line"], size=4),
                   hoverinfo="skip", showlegend=False)
    )
    if show_direction:
        fig.add_annotation(
            x=0, y=-pitch.half_width - MARGIN + 1.2, text=f"→ {direction_label}",
            showarrow=False, font=dict(size=11, color=PALETTE["muted"]),
        )
    fig.update_xaxes(
        range=[-pitch.half_length - MARGIN, pitch.half_length + MARGIN],
        showgrid=False, zeroline=False, visible=False, constrain="domain",
    )
    fig.update_yaxes(
        range=[-pitch.half_width - MARGIN, pitch.half_width + MARGIN],
        showgrid=False, zeroline=False, visible=False,
        scaleanchor="x", scaleratio=1, constrain="domain",
    )
    fig.update_layout(
        plot_bgcolor=PALETTE["pitch"], paper_bgcolor=PALETTE["surface"],
        margin=dict(l=8, r=8, t=44 if title else 12, b=12),
    )
    return apply(fig, title=title, height=height, showlegend=False)


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #

def tracking_snapshot(
    players: pd.DataFrame,
    ball: pd.Series | None = None,
    *,
    home_team_id: int,
    home_name: str = "Home",
    away_name: str = "Away",
    x_col: str = "x_std",
    y_col: str = "y_std",
    show_velocity: bool = True,
    highlight: set[int] | None = None,
    pitch: Pitch = STANDARD_PITCH,
    title: str | None = None,
    height: int = 560,
    control: "object | None" = None,
) -> go.Figure:
    """All 22 players and the ball at one frame, optionally over a control surface."""
    fig = blank_pitch(pitch, height=height, title=title)

    if control is not None:
        fig.add_trace(
            go.Heatmap(
                x=control.xs, y=control.ys, z=control.p_home,
                colorscale=SEQ_CONTROL, zmin=0, zmax=1, opacity=0.45,
                showscale=False, hoverinfo="skip", zsmooth="best",
            )
        )
        fig.data = tuple(list(fig.data[-1:]) + list(fig.data[:-1]))

    highlight = highlight or set()
    for is_home, colour, name in (
        (True, SERIES[0], home_name),
        (False, SERIES[1], away_name),
    ):
        sub = players[(players["team_id"] == home_team_id) == is_home]
        if sub.empty:
            continue
        sizes = [21 if pid in highlight else 16 for pid in sub["player_id"]]
        widths = [2.6 if pid in highlight else 1.4 for pid in sub["player_id"]]
        fig.add_trace(
            go.Scatter(
                x=sub[x_col], y=sub[y_col], mode="markers+text",
                text=sub.get("shirt_number", pd.Series("", index=sub.index)).astype("string").fillna(""),
                textposition="middle center",
                textfont=dict(size=10, color=PALETTE["bg"], family="Inter, sans-serif"),
                marker=dict(
                    color=colour, size=sizes,
                    line=dict(color=PALETTE["surface"], width=widths),
                ),
                name=name,
                customdata=np.stack([
                    sub.get("display_name", sub["player_id"].astype(str)).astype(str),
                    sub.get("position", pd.Series("", index=sub.index)).astype(str),
                    sub.get("speed", pd.Series(np.nan, index=sub.index)).astype(float).round(2),
                ], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]} m/s<extra></extra>",
                showlegend=True,
            )
        )
        if show_velocity and {"vx", "vy"} <= set(sub.columns):
            xs, ys = [], []
            for _, r in sub.iterrows():
                if not np.isfinite(r["vx"]) or not np.isfinite(r["vy"]):
                    continue
                xs += [r[x_col], r[x_col] + r["vx"] * 0.7, None]
                ys += [r[y_col], r[y_col] + r["vy"] * 0.7, None]
            if xs:
                fig.add_trace(
                    go.Scatter(x=xs, y=ys, mode="lines",
                               line=dict(color=colour, width=1.6),
                               opacity=0.75, hoverinfo="skip", showlegend=False)
                )

    if ball is not None and np.isfinite(ball.get("ball_x_std", np.nan)):
        fig.add_trace(
            go.Scatter(
                x=[ball["ball_x_std"]], y=[ball["ball_y_std"]], mode="markers",
                marker=dict(color="#FFFFFF", size=11, symbol="circle",
                            line=dict(color=PALETTE["bg"], width=2)),
                name="Ball", hovertemplate="ball<extra></extra>",
            )
        )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.01, xanchor="center", x=0.5),
        margin=dict(l=8, r=8, t=48, b=34),
    )
    return fig


def control_map(control, *, title: str | None = None, height: int = 480,
                pitch: Pitch = STANDARD_PITCH, home_name="Home", away_name="Away") -> go.Figure:
    fig = blank_pitch(pitch, height=height, title=title)
    fig.add_trace(
        go.Heatmap(
            x=control.xs, y=control.ys, z=control.p_home,
            colorscale=SEQ_CONTROL, zmin=0, zmax=1, zsmooth="best",
            colorbar=dict(
                title=dict(text="control", side="right"),
                tickvals=[0, 0.5, 1], ticktext=[away_name, "even", home_name],
                thickness=10, len=0.6, outlinewidth=0,
                tickfont=dict(size=11, color=PALETTE["text_secondary"]),
            ),
            hovertemplate="%{z:.0%} " + home_name + "<extra></extra>",
        )
    )
    fig.data = tuple(list(fig.data[-1:]) + list(fig.data[:-1]))
    return fig


# --------------------------------------------------------------------------- #
# Aggregate views
# --------------------------------------------------------------------------- #

def heatmap(x, y, *, pitch: Pitch = STANDARD_PITCH, nbins=(30, 20),
            title: str | None = None, height: int = 460,
            colorscale=None) -> go.Figure:
    fig = blank_pitch(pitch, height=height, title=title, show_direction=True)
    H, xe, ye = np.histogram2d(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), bins=nbins,
        range=[[-pitch.half_length, pitch.half_length], [-pitch.half_width, pitch.half_width]],
    )
    H = H / H.sum() if H.sum() else H
    fig.add_trace(
        go.Heatmap(
            x=(xe[:-1] + xe[1:]) / 2, y=(ye[:-1] + ye[1:]) / 2, z=H.T,
            colorscale=colorscale or SEQ_MAGNITUDE, zsmooth="best",
            showscale=False, hovertemplate="%{z:.2%} of time here<extra></extra>",
        )
    )
    fig.data = tuple(list(fig.data[-1:]) + list(fig.data[:-1]))
    return fig


def arrows(
    df: pd.DataFrame,
    *,
    x0="x_start_att", y0="y_start_att", x1="x_end_att", y1="y_end_att",
    colour_col: str | None = None, colour_map: dict | None = None,
    width_col: str | None = None, base_width: float = 1.6,
    hover: list[str] | None = None,
    pitch: Pitch = STANDARD_PITCH, title: str | None = None, height: int = 480,
    head: bool = True, opacity: float = 0.85, single_colour: str | None = None,
) -> go.Figure:
    """Draw a set of directed actions (passes, carries, runs) as arrows."""
    fig = blank_pitch(pitch, height=height, title=title, show_direction=True)
    if df.empty:
        return fig
    groups = df.groupby(colour_col, observed=True) if colour_col else [(None, df)]
    for i, (key, g) in enumerate(groups):
        colour = (
            single_colour or (colour_map or {}).get(key) or SERIES[i % len(SERIES)]
        )
        xs, ys = [], []
        for _, r in g.iterrows():
            xs += [r[x0], r[x1], None]
            ys += [r[y0], r[y1], None]
        w = base_width
        if width_col and g[width_col].notna().any():
            w = float(np.clip(g[width_col].mean(), 0.5, 6))
        fig.add_trace(
            go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=colour, width=w),
                       opacity=opacity, name=str(key) if key is not None else "",
                       hoverinfo="skip", showlegend=colour_col is not None)
        )
        if head:
            cd = None
            tmpl = "<extra></extra>"
            if hover:
                cd = g[hover].astype(str).to_numpy()
                tmpl = "<br>".join(f"{h}: %{{customdata[{k}]}}" for k, h in enumerate(hover)) + "<extra></extra>"
            fig.add_trace(
                go.Scatter(
                    x=g[x1], y=g[y1], mode="markers",
                    marker=dict(color=colour, size=6,
                                line=dict(color=PALETTE["surface"], width=1)),
                    customdata=cd, hovertemplate=tmpl,
                    showlegend=False, name="",
                )
            )
    return fig


def passing_network(
    nodes: pd.DataFrame, edges: pd.DataFrame, *,
    colour: str = SERIES[0], title: str | None = None, height: int = 520,
    pitch: Pitch = STANDARD_PITCH, label_col: str = "player_name",
) -> go.Figure:
    """Average-position passing network.

    ``nodes`` needs ``x``, ``y``, ``label_col``, ``involvement``;
    ``edges`` needs ``x0``, ``y0``, ``x1``, ``y1``, ``weight`` and labels.
    """
    fig = blank_pitch(pitch, height=height, title=title, show_direction=True)
    if edges.empty and nodes.empty:
        return fig
    wmax = max(float(edges["weight"].max()) if len(edges) else 1.0, 1.0)
    for _, e in edges.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[e["x0"], e["x1"]], y=[e["y0"], e["y1"]], mode="lines",
                line=dict(color=colour, width=0.8 + 5.0 * e["weight"] / wmax),
                opacity=0.16 + 0.5 * e["weight"] / wmax,
                hovertemplate=f"{e.get('label','')}: {int(e['weight'])} passes<extra></extra>",
                showlegend=False, name="",
            )
        )
    if not nodes.empty:
        inv = nodes["involvement"].to_numpy(dtype=float)
        size = 14 + 26 * (inv - inv.min()) / max(float(np.ptp(inv)), 1e-9)
        fig.add_trace(
            go.Scatter(
                x=nodes["x"], y=nodes["y"], mode="markers+text",
                marker=dict(color=colour, size=size, opacity=0.95,
                            line=dict(color=PALETTE["surface"], width=2)),
                text=nodes[label_col], textposition="top center",
                textfont=dict(size=10, color=PALETTE["text"]),
                customdata=np.stack([nodes[label_col].astype(str), inv], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.0f} involvements<extra></extra>",
                showlegend=False, name="",
            )
        )
    return fig


def team_shape_snapshot(
    players: pd.DataFrame, *, colour: str = SERIES[0], pitch: Pitch = STANDARD_PITCH,
    title: str | None = None, height: int = 460, x_col="xa", y_col="ya",
    show_hull: bool = True,
) -> go.Figure:
    from scipy.spatial import ConvexHull, QhullError

    fig = blank_pitch(pitch, height=height, title=title, show_direction=True)
    if players.empty:
        return fig
    pts = players[[x_col, y_col]].to_numpy(dtype=float)
    if show_hull and len(pts) >= 3:
        try:
            h = ConvexHull(pts)
            ring = np.append(h.vertices, h.vertices[0])
            fig.add_trace(
                go.Scatter(x=pts[ring, 0], y=pts[ring, 1], mode="lines",
                           fill="toself", fillcolor=colour, opacity=0.14,
                           line=dict(color=colour, width=1.4),
                           hoverinfo="skip", showlegend=False)
            )
        except (QhullError, ValueError):
            pass
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    fig.add_trace(
        go.Scatter(x=[cx], y=[cy], mode="markers",
                   marker=dict(color=colour, size=13, symbol="x-thin",
                               line=dict(color=colour, width=2.5)),
                   name="centroid", hovertemplate="centroid<extra></extra>",
                   showlegend=False)
    )
    fig.add_trace(
        go.Scatter(
            x=pts[:, 0], y=pts[:, 1], mode="markers+text",
            marker=dict(color=colour, size=13, line=dict(color=PALETTE["surface"], width=1.5)),
            text=players.get("position", pd.Series("", index=players.index)),
            textposition="top center", textfont=dict(size=9, color=PALETTE["text_secondary"]),
            hovertext=players.get("display_name", None),
            hovertemplate="%{hovertext}<extra></extra>",
            showlegend=False, name="",
        )
    )
    return fig


def trajectory(
    path: pd.DataFrame, *, x="x_std", y="y_std", colour_by: str | None = "speed",
    pitch: Pitch = STANDARD_PITCH, title: str | None = None, height: int = 460,
    line_colour: str = SERIES[0],
) -> go.Figure:
    fig = blank_pitch(pitch, height=height, title=title)
    if path.empty:
        return fig
    if colour_by and colour_by in path:
        fig.add_trace(
            go.Scatter(
                x=path[x], y=path[y], mode="markers",
                marker=dict(
                    color=path[colour_by], colorscale=SEQ_MAGNITUDE, size=5,
                    colorbar=dict(title=dict(text=colour_by, side="right"),
                                  thickness=9, len=0.55, outlinewidth=0),
                ),
                hovertemplate=f"{colour_by}: %{{marker.color:.2f}}<extra></extra>",
                showlegend=False, name="",
            )
        )
    fig.add_trace(
        go.Scatter(x=path[x], y=path[y], mode="lines",
                   line=dict(color=line_colour, width=1.2), opacity=0.6,
                   hoverinfo="skip", showlegend=False)
    )
    return fig
