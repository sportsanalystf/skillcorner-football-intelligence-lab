"""Turn the raw tracking JSONL into analytical Parquet.

One pass over each match file produces two tables:

``data/processed/tracking/match_id=<id>/tracking.parquet``
    One row per player-frame, with smoothed position and kinematics attached.
``data/processed/ball/match_id=<id>/ball.parquet``
    One row per frame: ball position, detection flag, and who was in possession.

Both carry ``match_clock_s`` so that tracking, events and phases join on a
single continuous clock rather than on frame numbers that restart their
meaning at half time.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, TRACKING_FPS
from src.data.loaders import PERIOD_OFFSET_S, iter_tracking, load_match_meta
from src.tracking.kinematics import add_kinematics

log = logging.getLogger(__name__)

TRACKING_DIR = PROCESSED_DIR / "tracking"
BALL_DIR = PROCESSED_DIR / "ball"


def read_tracking_long(match_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse one match file into (player_frames, ball_frames) long tables."""
    frames: list[int] = []
    periods: list[float] = []
    bx: list[float] = []
    by: list[float] = []
    bz: list[float] = []
    bdet: list[object] = []
    pgroup: list[object] = []
    pplayer: list[float] = []
    proj_area: list[float] = []

    pf_frame: list[int] = []
    pf_pid: list[int] = []
    pf_x: list[float] = []
    pf_y: list[float] = []
    pf_det: list[bool] = []

    for rec in iter_tracking(match_id):
        fr = int(rec["frame"])
        per = rec.get("period")
        frames.append(fr)
        periods.append(np.nan if per is None else float(per))

        b = rec.get("ball_data") or {}
        bx.append(b.get("x"))
        by.append(b.get("y"))
        bz.append(b.get("z"))
        bdet.append(b.get("is_detected"))

        poss = rec.get("possession") or {}
        pgroup.append(poss.get("group"))
        pid = poss.get("player_id")
        pplayer.append(np.nan if pid is None else float(pid))

        icp = rec.get("image_corners_projection") or {}
        proj_area.append(_polygon_area(icp))

        for p in rec.get("player_data") or []:
            pf_frame.append(fr)
            pf_pid.append(int(p["player_id"]))
            pf_x.append(p.get("x"))
            pf_y.append(p.get("y"))
            pf_det.append(bool(p.get("is_detected")))

    ball = pd.DataFrame(
        {
            "frame": np.asarray(frames, dtype="int32"),
            "period": pd.array(periods, dtype="Float64").astype("Int8"),
            "ball_x": np.asarray(bx, dtype="float32"),
            "ball_y": np.asarray(by, dtype="float32"),
            "ball_z": np.asarray(bz, dtype="float32"),
            "ball_detected": pd.array(bdet, dtype="boolean"),
            "possession_group": pd.array(pgroup, dtype="string"),
            "possession_player_id": pd.array(pplayer, dtype="Float64").astype("Int64"),
            "visible_area_m2": np.asarray(proj_area, dtype="float32"),
        }
    )

    players = pd.DataFrame(
        {
            "frame": np.asarray(pf_frame, dtype="int32"),
            "player_id": np.asarray(pf_pid, dtype="int64"),
            "x": np.asarray(pf_x, dtype="float64"),
            "y": np.asarray(pf_y, dtype="float64"),
            "is_detected": np.asarray(pf_det, dtype="bool"),
        }
    )
    players = players.merge(ball[["frame", "period"]], on="frame", how="left")
    return players, ball


def _polygon_area(icp: dict) -> float:
    """Shoelace area of the broadcast's projected visible quadrilateral."""
    keys = ("top_left", "bottom_left", "bottom_right", "top_right")
    pts = []
    for k in keys:
        x, y = icp.get(f"x_{k}"), icp.get(f"y_{k}")
        if x is None or y is None:
            return np.nan
        pts.append((x, y))
    a = 0.0
    for i in range(4):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % 4]
        a += x1 * y2 - x2 * y1
    return abs(a) / 2.0


def build_match(match_id: int, overwrite: bool = False) -> dict[str, int]:
    meta = load_match_meta(match_id)
    tdir = TRACKING_DIR / f"match_id={match_id}"
    bdir = BALL_DIR / f"match_id={match_id}"
    tpath, bpath = tdir / "tracking.parquet", bdir / "ball.parquet"
    if tpath.exists() and bpath.exists() and not overwrite:
        return {"skipped": 1}

    players, ball = read_tracking_long(match_id)
    squad = meta.players_frame()[["player_id", "team_id", "side", "position", "position_group"]]

    # Clock.
    ball["match_clock_s"] = meta.clock_seconds(ball["frame"], ball["period"].astype("float"))
    players = players.dropna(subset=["period"]).copy()
    players["period"] = players["period"].astype("int8")
    players["match_clock_s"] = meta.clock_seconds(players["frame"], players["period"])

    players = players.merge(squad, on="player_id", how="left")
    if players["team_id"].isna().any():
        n = int(players["team_id"].isna().sum())
        log.warning("match %s: %d player-frames with unknown team id", match_id, n)

    tracked = add_kinematics(players)

    # Standard-pitch coordinates, and the sign that puts each player's team
    # attacking towards +x.
    sx = 105.0 / meta.pitch.length
    sy = 68.0 / meta.pitch.width
    tracked["x_std"] = (tracked["xs"] * sx).astype("float32")
    tracked["y_std"] = (tracked["ys"] * sy).astype("float32")
    home_sign = np.array(
        [1 if meta.home_attacks_positive_x.get(int(p), True) else -1 for p in tracked["period"]]
    )
    is_home = (tracked["side"] == "home").to_numpy()
    tracked["attack_sign"] = np.where(is_home, home_sign, -home_sign).astype("int8")
    tracked["home_attack_sign"] = home_sign.astype("int8")

    tracked["match_id"] = np.int64(match_id)
    tracked["team_id"] = tracked["team_id"].astype("int64")
    keep = [
        "match_id", "frame", "period", "match_clock_s", "player_id", "team_id", "side",
        "position", "position_group", "x", "y", "is_detected",
        "xs", "ys", "x_std", "y_std", "vx", "vy", "speed", "accel",
        "heading", "step_m", "speed_zone", "attack_sign", "home_attack_sign",
    ]
    tracked = tracked[keep]
    for c in ("xs", "ys", "vx", "vy", "speed", "accel", "heading", "step_m", "x", "y"):
        tracked[c] = tracked[c].astype("float32")
    tracked["match_clock_s"] = tracked["match_clock_s"].astype("float32")
    for c in ("side", "position", "position_group", "speed_zone"):
        tracked[c] = tracked[c].astype("category")

    # Ball: standard coordinates + a simple smoothed speed.
    ball["ball_x_std"] = (ball["ball_x"] * sx).astype("float32")
    ball["ball_y_std"] = (ball["ball_y"] * sy).astype("float32")
    ball = _add_ball_kinematics(ball)
    ball["match_id"] = np.int64(match_id)

    tdir.mkdir(parents=True, exist_ok=True)
    bdir.mkdir(parents=True, exist_ok=True)
    tracked.to_parquet(tpath, index=False, compression="zstd")
    ball.to_parquet(bpath, index=False, compression="zstd")
    return {"player_frames": len(tracked), "ball_frames": len(ball)}


def _add_ball_kinematics(ball: pd.DataFrame) -> pd.DataFrame:
    """Ball speed, smoothed inside contiguous detected runs only.

    The ball is missing from 25-30% of frames and jumps hard when it reappears,
    so speed is only defined where consecutive frames both carry a position.
    """
    from src.tracking.kinematics import smooth_and_differentiate

    ball = ball.sort_values("frame", ignore_index=True)
    out_speed = np.full(len(ball), np.nan)
    out_vx = np.full(len(ball), np.nan)
    out_vy = np.full(len(ball), np.nan)
    for per, grp in ball.groupby("period", dropna=True):
        idx = grp.index.to_numpy()
        present = grp["ball_x"].notna().to_numpy()
        k = smooth_and_differentiate(
            grp["ball_x"].to_numpy(dtype="float64"),
            grp["ball_y"].to_numpy(dtype="float64"),
            present,
            window=7,
            max_gap=3,
            max_speed=45.0,     # a struck ball clears 30 m/s; 45 is the artefact line
            max_accel=400.0,    # a kick is ~1000 m/s^2 over 10 ms, i.e. unbounded at 10 Hz
        )
        out_speed[idx] = np.where(present, k["speed"], np.nan)
        out_vx[idx] = np.where(present, k["vx"], np.nan)
        out_vy[idx] = np.where(present, k["vy"], np.nan)
    ball["ball_speed"] = out_speed.astype("float32")
    ball["ball_vx"] = out_vx.astype("float32")
    ball["ball_vy"] = out_vy.astype("float32")
    return ball


def load_tracking(match_id: int, columns: list[str] | None = None) -> pd.DataFrame:
    return pd.read_parquet(TRACKING_DIR / f"match_id={match_id}" / "tracking.parquet", columns=columns)


def load_ball(match_id: int) -> pd.DataFrame:
    return pd.read_parquet(BALL_DIR / f"match_id={match_id}" / "ball.parquet")
