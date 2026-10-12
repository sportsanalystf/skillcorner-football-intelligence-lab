"""A pitch-control model for broadcast tracking.

**This is not a SkillCorner metric.** SkillCorner publishes pressure and
space-constraint bands; nothing in their open data is a pitch-control surface.
What follows is this project's own model, and it is deliberately the simplest
thing that is defensible on broadcast data.

Model
-----
For every player and every grid cell, estimate the time to reach that cell:

    t(p, c) = reaction + |c - (x_p + v_p * reaction)| / v_max

i.e. the player continues at their current velocity through a reaction period
and then runs in a straight line at ``v_max``. This is the Spearman-style
time-to-intercept simplification: no acceleration profile, no ball flight time.

Control of a cell is then a softmax over the two teams' best arrival times:

    P_home(c) = sigmoid( (t_away_best(c) - t_home_best(c)) / sigma )

with ``sigma = 0.45 s``. A team that gets there half a second earlier holds
roughly 75% of the cell; a dead heat splits it 50/50.

Why not something richer
------------------------
Broadcast tracking extrapolates about 44% of player positions and gives no
acceleration you would trust to 0.1 s. A model with per-player speed profiles
fitted to that data would be more precise about numbers that are not there.
The simple version has two parameters, both of which are stated above, and its
failure mode -- underestimating control for a player who is decelerating into
a cell -- is easy to describe.

Known limitations are listed in ``docs/LIMITATIONS.md``; the derivation and
parameter choices in ``docs/METHODOLOGY.md``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    PC_GRID_NX,
    PC_GRID_NY,
    PC_MAX_SPEED_MS,
    PC_REACTION_TIME_S,
    PC_TTI_SIGMA,
)
from src.utils.pitch import STANDARD_PITCH, Pitch, grid

__all__ = ["control_surface", "control_summary", "PitchControlResult"]


class PitchControlResult:
    """A control surface plus the geometry needed to plot or summarise it."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray, p_home: np.ndarray, pitch: Pitch):
        self.xs = xs
        self.ys = ys
        self.p_home = p_home          # shape (ny, nx), 1 = fully home-controlled
        self.pitch = pitch

    @property
    def home_share(self) -> float:
        return float(self.p_home.mean())

    def share_in_region(self, x_min: float, x_max: float) -> float:
        m = (self.xs >= x_min) & (self.xs <= x_max)
        return float(self.p_home[:, m].mean()) if m.any() else float("nan")

    def to_frame(self) -> pd.DataFrame:
        X, Y = np.meshgrid(self.xs, self.ys)
        return pd.DataFrame({"x": X.ravel(), "y": Y.ravel(), "p_home": self.p_home.ravel()})


def _time_to_point(px, py, vx, vy, X, Y, reaction: float, vmax: float) -> np.ndarray:
    """Arrival times for each player to each cell. Shape (n_players, ny, nx)."""
    ax = px + np.nan_to_num(vx) * reaction
    ay = py + np.nan_to_num(vy) * reaction
    d = np.sqrt((X[None, :, :] - ax[:, None, None]) ** 2 + (Y[None, :, :] - ay[:, None, None]) ** 2)
    return reaction + d / vmax


def control_surface(
    frame_players: pd.DataFrame,
    *,
    home_team_id: int,
    pitch: Pitch = STANDARD_PITCH,
    nx: int = PC_GRID_NX,
    ny: int = PC_GRID_NY,
    reaction: float = PC_REACTION_TIME_S,
    vmax: float = PC_MAX_SPEED_MS,
    sigma: float = PC_TTI_SIGMA,
    x_col: str = "x_std",
    y_col: str = "y_std",
) -> PitchControlResult:
    """Control surface for one frame.

    Args:
        frame_players: One row per player on the pitch at this frame, with
            position, velocity (``vx``/``vy``) and ``team_id``.
        home_team_id: Which team the returned probabilities refer to.

    Returns:
        A :class:`PitchControlResult` whose ``p_home`` is the probability that
        the home team reaches each cell first.
    """
    xs, ys, (X, Y) = grid(nx, ny, pitch)
    df = frame_players.dropna(subset=[x_col, y_col])
    home = df[df["team_id"] == home_team_id]
    away = df[df["team_id"] != home_team_id]
    if home.empty or away.empty:
        return PitchControlResult(xs, ys, np.full((ny, nx), 0.5), pitch)

    th = _time_to_point(
        home[x_col].to_numpy(), home[y_col].to_numpy(),
        home.get("vx", pd.Series(0.0, index=home.index)).to_numpy(),
        home.get("vy", pd.Series(0.0, index=home.index)).to_numpy(),
        X, Y, reaction, vmax,
    ).min(axis=0)
    ta = _time_to_point(
        away[x_col].to_numpy(), away[y_col].to_numpy(),
        away.get("vx", pd.Series(0.0, index=away.index)).to_numpy(),
        away.get("vy", pd.Series(0.0, index=away.index)).to_numpy(),
        X, Y, reaction, vmax,
    ).min(axis=0)

    p_home = 1.0 / (1.0 + np.exp(-(ta - th) / sigma))
    return PitchControlResult(xs, ys, p_home, pitch)


def control_summary(
    res: PitchControlResult,
    *,
    home_attacks_positive_x: bool,
) -> dict[str, float]:
    """Headline numbers from a control surface, in each team's own terms."""
    hl = res.pitch.half_length
    third = hl / 3 * 2  # boundary of the final third at +-35 m
    if home_attacks_positive_x:
        home_att = res.share_in_region(third, hl)
        home_def = res.share_in_region(-hl, -third)
    else:
        home_att = res.share_in_region(-hl, -third)
        home_def = res.share_in_region(third, hl)
    return {
        "home_control": res.home_share,
        "away_control": 1 - res.home_share,
        "home_control_att_third": home_att,
        "home_control_own_third": home_def,
        "away_control_att_third": 1 - home_def,
        "away_control_own_third": 1 - home_att,
    }


def control_at_frames(
    trk: pd.DataFrame,
    frames: list[int],
    *,
    home_team_id: int,
    home_attacks_positive_x_by_period: dict[int, bool],
    **kwargs,
) -> pd.DataFrame:
    """Summaries for a list of frames. Used for timelines, not for plotting."""
    out = []
    sub = trk[trk["frame"].isin(set(frames))]
    for fr, g in sub.groupby("frame", observed=True):
        per = int(g["period"].iloc[0])
        res = control_surface(g, home_team_id=home_team_id, **kwargs)
        rec = {"frame": int(fr), "period": per}
        rec.update(
            control_summary(
                res, home_attacks_positive_x=home_attacks_positive_x_by_period.get(per, True)
            )
        )
        out.append(rec)
    return pd.DataFrame(out)
