"""3D skeleton and orientation figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config import PALETTE, SERIES
from src.data.pose_loading import SKELETON
from src.visualization.theme import apply


def skeleton_3d(
    joints: pd.DataFrame, *, title: str | None = None, height: int = 620,
    colour: str = SERIES[0], centre: bool = True, show_error: bool = True,
) -> go.Figure:
    """One player's 29 joints at one frame, as a 3D skeleton.

    ``joints`` is the long joint table filtered to a single player-frame, with
    ``joint``, ``jx``, ``jy``, ``jz`` and ``p90_mae_cm``.
    """
    fig = go.Figure()
    if joints.empty:
        return apply(fig, title=title, height=height)
    j = joints.set_index("joint")
    ox, oy = (float(j["jx"].mean()), float(j["jy"].mean())) if centre else (0.0, 0.0)

    xs, ys, zs = [], [], []
    for a, b in SKELETON:
        if a in j.index and b in j.index:
            xs += [j.loc[a, "jx"] - ox, j.loc[b, "jx"] - ox, None]
            ys += [j.loc[a, "jy"] - oy, j.loc[b, "jy"] - oy, None]
            zs += [j.loc[a, "jz"], j.loc[b, "jz"], None]
    fig.add_trace(
        go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                     line=dict(color=colour, width=5), hoverinfo="skip",
                     name="skeleton", showlegend=False)
    )
    size = 5
    if show_error and "p90_mae_cm" in joints:
        e = joints["p90_mae_cm"].to_numpy(dtype=float)
        size = 3 + 7 * (e - np.nanmin(e)) / max(np.nanmax(e) - np.nanmin(e), 1e-9)
    fig.add_trace(
        go.Scatter3d(
            x=joints["jx"] - ox, y=joints["jy"] - oy, z=joints["jz"], mode="markers",
            marker=dict(size=size, color=PALETTE["text"], opacity=0.9,
                        line=dict(color=colour, width=1)),
            customdata=np.stack([joints["joint"].astype(str),
                                 joints["p90_mae_cm"].astype(float)], axis=-1),
            hovertemplate="<b>%{customdata[0]}</b><br>±%{customdata[1]:.1f} cm (p90)<extra></extra>",
            name="joints", showlegend=False,
        )
    )
    r = 0.9
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="x (m)", range=[-r, r], backgroundcolor=PALETTE["pitch"],
                       gridcolor=PALETTE["line"], color=PALETTE["muted"], showspikes=False),
            yaxis=dict(title="y (m)", range=[-r, r], backgroundcolor=PALETTE["pitch"],
                       gridcolor=PALETTE["line"], color=PALETTE["muted"], showspikes=False),
            zaxis=dict(title="z (m)", range=[0, 2.1], backgroundcolor=PALETTE["pitch"],
                       gridcolor=PALETTE["line"], color=PALETTE["muted"], showspikes=False),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=1.15),
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.9)),
        ),
        margin=dict(l=0, r=0, t=44 if title else 8, b=0),
        paper_bgcolor=PALETTE["surface"],
    )
    return apply(fig, title=title, height=height, showlegend=False)


def orientation_rose(df: pd.DataFrame, *, angle_col: str = "facing_goal_deg",
                     title: str | None = None, height: int = 400,
                     colour: str = SERIES[0]) -> go.Figure:
    """Distribution of facing angle relative to the attacked goal."""
    bins = np.arange(0, 181, 15)
    counts, _ = np.histogram(df[angle_col].dropna(), bins=bins)
    fig = go.Figure(
        go.Barpolar(
            r=counts, theta=(bins[:-1] + bins[1:]) / 2, width=14,
            marker=dict(color=colour, line=dict(color=PALETTE["surface"], width=1.5)),
            hovertemplate="%{theta:.0f}° from goal<br>%{r} frames<extra></extra>",
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor=PALETTE["pitch"], sector=[0, 180],
            radialaxis=dict(gridcolor=PALETTE["line"], color=PALETTE["muted"],
                            tickfont=dict(size=10)),
            angularaxis=dict(gridcolor=PALETTE["line"], color=PALETTE["muted"],
                             direction="counterclockwise", rotation=0,
                             tickmode="array", tickvals=[0, 45, 90, 135, 180],
                             ticktext=["facing goal", "45°", "square", "135°", "facing own goal"]),
        ),
        paper_bgcolor=PALETTE["surface"], margin=dict(l=40, r=40, t=50, b=20),
    )
    return apply(fig, title=title, height=height, showlegend=False)


def orientation_on_pitch(df: pd.DataFrame, *, title: str | None = None, height: int = 520,
                         pitch=None) -> go.Figure:
    """Players' positions with a short arrow showing which way each is facing."""
    from src.utils.pitch import STANDARD_PITCH
    from src.visualization.pitch_plots import blank_pitch

    pitch = pitch or STANDARD_PITCH
    fig = blank_pitch(pitch, height=height, title=title)
    if df.empty:
        return fig
    for side, colour in (("home", SERIES[0]), ("away", SERIES[1])):
        s = df[df["side"] == side]
        if s.empty:
            continue
        xs, ys = [], []
        for _, r in s.iterrows():
            xs += [r["x"], r["x"] + r["facing_x"] * 3.0, None]
            ys += [r["y"], r["y"] + r["facing_y"] * 3.0, None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                 line=dict(color=colour, width=2.2),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(
            go.Scatter(
                x=s["x"], y=s["y"], mode="markers",
                marker=dict(color=colour, size=11,
                            line=dict(color=PALETTE["surface"], width=1.5)),
                customdata=np.stack([s["display_name"].astype(str),
                                     s["facing_goal_deg"].astype(float),
                                     s["body_orientation_class"].astype(str)], axis=-1),
                hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.0f}° from goal "
                              "(%{customdata[2]})<extra></extra>",
                name=side, showlegend=True,
            )
        )
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.01, xanchor="center", x=0.5),
        margin=dict(l=8, r=8, t=48, b=34),
    )
    return fig
