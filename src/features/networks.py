"""Tracking-informed passing networks.

The usual passing network places a node at a player's average position taken
from their own passing events, which biases every node towards where that
player happened to touch the ball. With tracking available there is no reason
to accept that: nodes are placed at the player's **mean tracked position over
the window**, so a full-back who never receives in the final third still shows
up where he actually stood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_network(
    passes: pd.DataFrame,
    tracking: pd.DataFrame,
    *,
    team_id: int,
    frame_range: tuple[int, int] | None = None,
    completed_only: bool = True,
    min_edge: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(nodes, edges)`` for one team.

    Args:
        passes: Rows from ``passes.parquet`` for the match.
        tracking: Tracking for the match with ``x_std``, ``y_std``,
            ``attack_sign``, ``team_id``, ``player_id``, ``position``.
        team_id: Team to build the network for.
        frame_range: Restrict to a frame window (e.g. until the first
            substitution, which is what makes a network readable).
        completed_only: Only count completed passes as edges.
        min_edge: Drop pairs below this many passes.
    """
    p = passes[passes["team_id"] == team_id].copy()
    t = tracking[(tracking["team_id"] == team_id)].copy()
    if frame_range:
        lo, hi = frame_range
        p = p[p["frame_start"].between(lo, hi)]
        t = t[t["frame"].between(lo, hi)]
    if completed_only:
        p = p[p["pass_outcome"] == "successful"]
    if t.empty:
        return pd.DataFrame(), pd.DataFrame()

    t = t.assign(xa=t["x_std"] * t["attack_sign"], ya=t["y_std"] * t["attack_sign"])
    nodes = (
        t.groupby("player_id", observed=True)
        .agg(x=("xa", "mean"), y=("ya", "mean"), frames=("xa", "size"),
             position=("position", "first"))
        .reset_index()
    )
    names = (
        p.groupby("player_id", observed=True)["player_name"].first()
        .rename("player_name").reset_index()
    )
    nodes = nodes.merge(names, on="player_id", how="left")
    nodes["player_name"] = nodes["player_name"].fillna(nodes["player_id"].astype(str))

    involvement = (
        pd.concat([
            p.groupby("player_id", observed=True).size().rename("as_passer"),
            p.dropna(subset=["receiver_id"]).groupby("receiver_id", observed=True)
            .size().rename("as_receiver"),
        ], axis=1).fillna(0)
    )
    involvement["involvement"] = involvement.sum(axis=1)
    nodes = nodes.merge(involvement.reset_index().rename(columns={"index": "player_id"}),
                        on="player_id", how="left")
    for c in ("as_passer", "as_receiver", "involvement"):
        nodes[c] = nodes[c].fillna(0)
    nodes = nodes[nodes["involvement"] > 0].copy()

    e = p.dropna(subset=["receiver_id"]).copy()
    e["receiver_id"] = e["receiver_id"].astype("int64")
    edges = (
        e.groupby(["player_id", "receiver_id"], observed=True)
        .agg(weight=("event_uid", "size"),
             epv=("pass_epv_total", "sum"),
             progression=("progression_m", "mean"),
             line_breaks=("fil_line_breaking", "sum"))
        .reset_index()
    )
    edges = edges[edges["weight"] >= min_edge]
    pos = nodes.set_index("player_id")[["x", "y", "player_name"]]
    edges = edges[edges["player_id"].isin(pos.index) & edges["receiver_id"].isin(pos.index)]
    if edges.empty:
        return nodes, edges
    edges["x0"] = pos.loc[edges["player_id"], "x"].to_numpy()
    edges["y0"] = pos.loc[edges["player_id"], "y"].to_numpy()
    edges["x1"] = pos.loc[edges["receiver_id"], "x"].to_numpy()
    edges["y1"] = pos.loc[edges["receiver_id"], "y"].to_numpy()
    edges["label"] = (
        pos.loc[edges["player_id"], "player_name"].to_numpy() + " → "
        + pos.loc[edges["receiver_id"], "player_name"].to_numpy()
    )
    return nodes.sort_values("involvement", ascending=False), edges.sort_values("weight", ascending=False)


def first_substitution_frame(squads: pd.DataFrame, match_id: int, team_id: int) -> int | None:
    """Frame of the team's first substitution, the usual cut-off for a network."""
    s = squads[(squads["match_id"] == match_id) & (squads["team_id"] == team_id)]
    subs = s[(~s["is_starter"]) & s["start_frame"].notna()]
    return int(subs["start_frame"].min()) if len(subs) else None
