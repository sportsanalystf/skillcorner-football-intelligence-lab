"""Per-frame team shape from tracking.

For every frame in which a team has players on the pitch, this computes the
geometry a coach would read off a still: how high the line is, how compact the
block is, how wide the team is, and where the units sit relative to each other.

All metrics are in the **attacking frame of the team they describe**, on the
standard 105 x 68 pitch, so "line height = 40" always means "40 m from their own
goal line" regardless of which way they happen to be kicking.

Goalkeepers are excluded from every shape metric. A keeper 40 m behind the
block turns team length into a measure of nothing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

from src.config import DERIVED_DIR
from src.preprocessing.build_tracking import load_tracking

log = logging.getLogger(__name__)

TEAM_FRAMES_DIR = DERIVED_DIR / "team_frames"

#: Number of players used to define each horizontal unit.
LINE_N = 4


def team_frames(match_id: int, stride: int = 1) -> pd.DataFrame:
    """Compute team shape for one match.

    Args:
        match_id: Match to process.
        stride: Take every ``stride``-th frame. 1 keeps the full 10 Hz.
    """
    trk = load_tracking(
        match_id,
        columns=["frame", "period", "match_clock_s", "player_id", "team_id", "side",
                 "position", "x_std", "y_std", "attack_sign", "speed", "is_detected"],
    )
    if stride > 1:
        trk = trk[trk["frame"] % stride == 0]

    trk = trk[trk["position"].astype("string") != "GK"].copy()
    # Attacking-frame coordinates for this player's own team.
    trk["xa"] = trk["x_std"] * trk["attack_sign"]
    trk["ya"] = trk["y_std"] * trk["attack_sign"]

    rows: list[dict] = []
    cols = ["frame", "period", "match_clock_s", "team_id", "xa", "ya", "speed", "is_detected"]
    arr = trk[cols].to_numpy()
    order = np.lexsort((arr[:, 3], arr[:, 0]))
    arr = arr[order]

    # Group boundaries on (frame, team_id).
    keys = arr[:, [0, 3]]
    change = np.any(keys[1:] != keys[:-1], axis=1)
    starts = np.concatenate(([0], np.flatnonzero(change) + 1))
    ends = np.concatenate((np.flatnonzero(change) + 1, [len(arr)]))

    for s, e in zip(starts, ends):
        blk = arr[s:e]
        n = e - s
        if n < 6:
            continue
        xa = blk[:, 4].astype("float64")
        ya = blk[:, 5].astype("float64")
        sp = blk[:, 6].astype("float64")
        xs = np.sort(xa)
        cx, cy = xa.mean(), ya.mean()
        d = np.hypot(xa - cx, ya - cy)

        pts = np.column_stack([xa, ya])
        try:
            hull = ConvexHull(pts)
            hull_area = float(hull.volume)  # 2-D "volume" is area
        except (QhullError, ValueError):
            hull_area = np.nan

        rows.append(
            {
                "frame": int(blk[0, 0]),
                "period": int(blk[0, 1]),
                "match_clock_s": float(blk[0, 2]),
                "team_id": int(blk[0, 3]),
                "n_players": int(n),
                "detected_frac": float(blk[:, 7].astype("float64").mean()),
                "centroid_x": float(cx),
                "centroid_y": float(cy),
                "width_m": float(ya.max() - ya.min()),
                "length_m": float(xa.max() - xa.min()),
                "hull_area_m2": hull_area,
                "compactness_m": float(d.mean()),
                "spread_std_m": float(d.std()),
                # Unit lines: the LINE_N deepest / highest outfielders.
                "def_line_x": float(xs[:LINE_N].mean()),
                "mid_line_x": float(np.median(xa)),
                "att_line_x": float(xs[-LINE_N:].mean()),
                "deepest_x": float(xs[0]),
                "highest_x": float(xs[-1]),
                "def_to_mid_m": float(np.median(xa) - xs[:LINE_N].mean()),
                "mid_to_att_m": float(xs[-LINE_N:].mean() - np.median(xa)),
                "mean_speed_ms": float(np.nanmean(sp)),
                # Occupation of the thirds, in this team's attacking frame.
                "n_own_def_third": int((xa < -17.5).sum()),
                "n_own_mid_third": int(((xa >= -17.5) & (xa <= 17.5)).sum()),
                "n_own_att_third": int((xa > 17.5).sum()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["vertical_compactness_m"] = out["att_line_x"] - out["def_line_x"]
    out["shape_area_ratio"] = out["hull_area_m2"] / (out["width_m"] * out["length_m"]).replace(0, np.nan)
    out["match_id"] = match_id
    return out


def add_opponent_context(tf: pd.DataFrame) -> pd.DataFrame:
    """Attach the opposing team's line positions to each row.

    Both teams' coordinates are in their *own* attacking frame, so an
    opponent's defensive line at x=30 sits at x=-30 in our frame: the sign
    flip is what makes ``space_between_lines_m`` mean anything.
    """
    if tf.empty:
        return tf
    cols = ["frame", "team_id", "def_line_x", "att_line_x", "centroid_x", "compactness_m",
            "width_m", "hull_area_m2", "n_own_def_third", "n_own_att_third"]
    other = tf[cols].copy()
    other = other.rename(
        columns={
            "team_id": "opp_team_id",
            "def_line_x": "opp_def_line_x",
            "att_line_x": "opp_att_line_x",
            "centroid_x": "opp_centroid_x",
            "compactness_m": "opp_compactness_m",
            "width_m": "opp_width_m",
            "hull_area_m2": "opp_hull_area_m2",
            # The opponent's attacking third is our defensive third.
            "n_own_att_third": "n_opp_in_our_def_third",
            "n_own_def_third": "n_opp_in_our_att_third",
        }
    )
    merged = tf.merge(other, on="frame", how="inner")
    merged = merged[merged["team_id"] != merged["opp_team_id"]].copy()
    # Flip the opponent's numbers into our attacking frame.
    for c in ("opp_def_line_x", "opp_att_line_x", "opp_centroid_x"):
        merged[c] = -merged[c]
    merged["space_between_lines_m"] = merged["opp_def_line_x"] - merged["att_line_x"]
    # Signed x-distance between the two centroids, read in our attacking frame.
    # Negative: the blocks are apart, each sitting in its own half.
    # Positive: the blocks overlap, the game compressed towards one goal.
    merged["block_overlap_m"] = merged["centroid_x"] - merged["opp_centroid_x"]
    merged["superiority_att_third"] = merged["n_own_att_third"] - merged["n_opp_in_our_att_third"]
    merged["superiority_def_third"] = merged["n_own_def_third"] - merged["n_opp_in_our_def_third"]
    return merged


def build(match_ids: list[int] | None = None, stride: int = 1) -> dict[int, int]:
    from src.data.loaders import available_match_ids

    match_ids = match_ids or available_match_ids()
    TEAM_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    counts = {}
    for mid in match_ids:
        tf = add_opponent_context(team_frames(mid, stride=stride))
        tf.to_parquet(TEAM_FRAMES_DIR / f"{mid}.parquet", index=False, compression="zstd")
        counts[mid] = len(tf)
        log.info("team frames %s: %d rows", mid, len(tf))
    return counts


def load_team_frames(match_id: int) -> pd.DataFrame:
    return pd.read_parquet(TEAM_FRAMES_DIR / f"{match_id}.parquet")
