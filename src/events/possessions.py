"""Possession, passing, pressing and transition tables.

SkillCorner's dynamic events already carry a great deal of derived context --
EPV, pressure bands, line breaks, passing options. Recomputing any of that
would be worse than using it. What this module adds is the part that only
exists once events and tracking sit on the same clock:

* the spatial situation around the ball carrier at the moment of the event
  (nearest opponent, local numbers, space ahead), measured from tracking;
* possession *chains*, by stitching player possessions into the phase of play
  they belong to;
* transitions, defined by where team possession actually changes hands.

Derived columns are named so they can never be mistaken for SkillCorner
metrics: anything computed here that has no upstream equivalent carries a
``fil_`` prefix.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DERIVED_DIR, PROCESSED_DIR
from src.data.loaders import available_match_ids, load_match_meta
from src.preprocessing.build_tracking import load_ball, load_tracking

log = logging.getLogger(__name__)

POSSESSIONS_PATH = DERIVED_DIR / "possessions.parquet"
TEAM_POSSESSIONS_PATH = DERIVED_DIR / "team_possessions.parquet"
PASSES_PATH = DERIVED_DIR / "passes.parquet"
RUNS_PATH = DERIVED_DIR / "off_ball_runs.parquet"
PRESSURES_PATH = DERIVED_DIR / "pressures.parquet"
TRANSITIONS_PATH = DERIVED_DIR / "transitions.parquet"

PRESSURE_ORDER = ["no_pressure", "low_pressure", "medium_pressure",
                  "high_pressure", "very_high_pressure"]
DIFFICULTY_ORDER = ["very_easy", "easy", "medium", "hard", "very_hard"]


def _events(match_id: int) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "events.parquet",
                           filters=[("match_id", "==", match_id)])


# --------------------------------------------------------------------------- #
# Spatial context from tracking
# --------------------------------------------------------------------------- #

def spatial_context(match_id: int, keys: pd.DataFrame) -> pd.DataFrame:
    """Tracking-derived situation around a player at a given frame.

    Args:
        match_id: Match to read tracking for.
        keys: Frame with columns ``event_uid``, ``frame``, ``player_id``,
            ``team_id``, ``attack_sign``.

    Returns:
        One row per input key with the ``fil_*`` context columns. Keys whose
        player is not tracked at that frame come back with NaNs, which is the
        honest answer -- roughly 3% of events fall in an untracked frame.
    """
    if keys.empty:
        return pd.DataFrame()
    trk = load_tracking(
        match_id,
        columns=["frame", "player_id", "team_id", "x_std", "y_std", "speed", "position"],
    )
    trk = trk[trk["frame"].isin(set(keys["frame"]))]
    if trk.empty:
        return pd.DataFrame({"event_uid": keys["event_uid"]})

    by_frame: dict[int, np.ndarray] = {}
    g = trk[["frame", "player_id", "team_id", "x_std", "y_std", "speed"]].to_numpy()
    gk_mask = (trk["position"].astype("string") == "GK").to_numpy().astype(float)
    g = np.column_stack([g, gk_mask])
    order = np.argsort(g[:, 0], kind="stable")
    g = g[order]
    bounds = np.flatnonzero(np.diff(g[:, 0])) + 1
    for blk in np.split(g, bounds):
        by_frame[int(blk[0, 0])] = blk

    out = []
    for uid, fr, pid, tid, sign in keys[
        ["event_uid", "frame", "player_id", "team_id", "attack_sign"]
    ].itertuples(index=False):
        blk = by_frame.get(int(fr))
        rec = {"event_uid": uid}
        if blk is None:
            out.append(rec)
            continue
        me = blk[blk[:, 1] == pid]
        if len(me) == 0:
            out.append(rec)
            continue
        mx, my = me[0, 3], me[0, 4]
        opp = blk[(blk[:, 2] != tid) & (blk[:, 6] == 0)]
        mate = blk[(blk[:, 2] == tid) & (blk[:, 1] != pid) & (blk[:, 6] == 0)]
        if len(opp):
            d = np.hypot(opp[:, 3] - mx, opp[:, 4] - my)
            rec["fil_nearest_opponent_m"] = float(d.min())
            rec["fil_opponents_within_3m"] = int((d <= 3).sum())
            rec["fil_opponents_within_5m"] = int((d <= 5).sum())
            rec["fil_opponents_within_10m"] = int((d <= 10).sum())
            # Space ahead: opponents between the carrier and the goal being
            # attacked, inside a 20 m-wide corridor.
            ahead = (opp[:, 3] - mx) * sign > 0
            corridor = np.abs(opp[:, 4] - my) <= 10
            rec["fil_opponents_ahead_in_corridor"] = int((ahead & corridor).sum())
            rec["fil_opponents_ahead"] = int(ahead.sum())
        if len(mate):
            dm = np.hypot(mate[:, 3] - mx, mate[:, 4] - my)
            rec["fil_nearest_teammate_m"] = float(dm.min())
            rec["fil_teammates_within_10m"] = int((dm <= 10).sum())
            ahead_m = (mate[:, 3] - mx) * sign > 0
            rec["fil_teammates_ahead"] = int(ahead_m.sum())
        if len(opp) and len(mate):
            rec["fil_local_superiority_10m"] = int(
                rec.get("fil_teammates_within_10m", 0) - rec.get("fil_opponents_within_10m", 0)
            )
        rec["fil_carrier_speed_ms"] = float(me[0, 5]) if np.isfinite(me[0, 5]) else np.nan
        out.append(rec)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# Player possessions
# --------------------------------------------------------------------------- #

POSSESSION_KEEP = [
    "event_uid", "match_id", "event_id", "index", "frame_start", "frame_end",
    "match_clock_s", "match_clock_end_s", "period", "minute_start", "duration",
    "player_id", "player_name", "player_position", "team_id", "team_shortname",
    "phase_index", "player_possession_phase_index",
    "first_player_possession_in_team_possession",
    "last_player_possession_in_team_possession",
    "n_player_possessions_in_phase", "team_possession_loss_in_phase",
    "team_in_possession_phase_type", "team_out_of_possession_phase_type",
    "start_type", "end_type", "game_state", "team_score", "opponent_team_score",
    "x_start", "y_start", "x_end", "y_end",
    "x_start_att", "y_start_att", "x_end_att", "y_end_att",
    "x_start_std", "y_start_std", "x_end_std", "y_end_std",
    "x_start_trk", "y_start_trk",
    "third_start", "third_end", "channel_start", "channel_end",
    "penalty_area_start", "penalty_area_end",
    "progression_m", "goal_distance_gain_m", "dist_to_goal_start", "dist_to_goal_end",
    "attack_sign", "carry", "one_touch", "quick_pass", "is_header",
    "pass_outcome", "pass_direction", "pass_range", "pass_distance", "pass_angle",
    "player_targeted_id", "player_targeted_name", "player_targeted_position",
    "player_targeted_x_reception_att", "player_targeted_y_reception_att",
    "lead_to_shot", "lead_to_goal", "received_in_space", "possession_danger",
    # NOTE: `xthreat` and `xpass_completion` are populated on passing_option and
    # off_ball_run rows, never on player_possession. The equivalents for a pass
    # actually played are the player_targeted_* columns below (81.5% populated).
    "player_targeted_xpass_completion", "player_targeted_xthreat",
    "player_targeted_dangerous", "player_targeted_difficult_pass_target",
    "passing_option_at_pass_moment", "predicted_passing_option",
    "xshot_player_possession_max", "xloss_player_possession_max",
    "first_line_break", "second_last_line_break", "last_line_break",
    "furthest_line_break", "n_opponents_bypassed",
    "last_defensive_line_x_start_att", "last_defensive_line_x_end_att",
    "delta_to_last_defensive_line_start", "delta_to_last_defensive_line_gain",
    "inside_defensive_shape_start", "n_passing_options", "n_off_ball_runs",
    "n_passing_options_ahead", "n_opponents_ahead_start", "n_opponents_ahead_end",
    "n_teammates_ahead_start", "fully_extrapolated",
    "possession_epv_for_start", "possession_epv_for_end",
    "possession_epv_against_start", "possession_epv_against_end",
    "possession_epv_delta_for", "possession_epv_delta_against",
    "possession_epv_total", "pass_epv_total", "pass_epv_delta_for",
    "pass_epv_delta_against", "pass_reception_epv_for",
    "overall_pressure_start", "overall_pressure_start_id",
    "overall_pressure_end", "overall_pressure_end_id",
    "reception_difficulty_start", "reception_difficulty_start_id",
    "space_constraint_start", "space_constraint_start_id",
    "time_to_impact_start", "time_to_impact_start_id",
    "passing_option_ease_start", "passing_option_ease_start_id",
]


def build_possessions(match_id: int) -> pd.DataFrame:
    ev = _events(match_id)
    pp = ev[ev["event_type"] == "player_possession"].copy()
    keep = [c for c in POSSESSION_KEEP if c in pp.columns]
    pp = pp[keep]
    ctx = spatial_context(
        match_id,
        pp[["event_uid", "frame_start", "player_id", "team_id", "attack_sign"]]
        .rename(columns={"frame_start": "frame"})
        .dropna(subset=["player_id"]),
    )
    if not ctx.empty:
        pp = pp.merge(ctx, on="event_uid", how="left")
    pp["fil_is_progressive"] = (pp["progression_m"] >= 5.0) & (pp["end_type"] == "pass")
    pp["fil_pressure_rank"] = pp["overall_pressure_start_id"]
    pp["fil_under_pressure"] = pp["overall_pressure_start_id"] >= 4
    return pp


# --------------------------------------------------------------------------- #
# Team possessions (chains)
# --------------------------------------------------------------------------- #

def build_team_possessions(poss: pd.DataFrame, match_id: int) -> pd.DataFrame:
    """Aggregate player possessions into one row per phase of play."""
    p = poss[poss["match_id"] == match_id].sort_values(["phase_index", "frame_start"])
    if p.empty:
        return pd.DataFrame()
    meta = load_match_meta(match_id)
    phases = pd.read_parquet(
        PROCESSED_DIR / "phases.parquet", filters=[("match_id", "==", match_id)]
    )

    def agg(g: pd.DataFrame) -> pd.Series:
        first, last = g.iloc[0], g.iloc[-1]
        passes = g[g["end_type"] == "pass"]
        return pd.Series(
            {
                "match_id": match_id,
                "team_id": first["team_id"],
                "team": first["team_shortname"],
                "period": first["period"],
                "frame_start": int(first["frame_start"]),
                "frame_end": int(last["frame_end"]),
                "match_clock_s": float(first["match_clock_s"]),
                "duration_s": float(last["match_clock_end_s"] - first["match_clock_s"]),
                "n_player_possessions": len(g),
                "n_passes": len(passes),
                "n_passes_completed": int((passes["pass_outcome"] == "successful").sum()),
                "n_players_involved": int(g["player_id"].nunique()),
                "x_start_att": float(first["x_start_att"]),
                "y_start_att": float(first["y_start_att"]),
                "x_end_att": float(last["x_end_att"]),
                "y_end_att": float(last["y_end_att"]),
                "territory_gain_m": float(last["x_end_att"] - first["x_start_att"]),
                "goal_distance_gain_m": float(first["dist_to_goal_start"] - last["dist_to_goal_end"]),
                "max_x_att": float(g["x_end_att"].max()),
                "width_used_m": float(
                    max(g["y_start_att"].max(), g["y_end_att"].max())
                    - min(g["y_start_att"].min(), g["y_end_att"].min())
                ),
                "start_type": first["start_type"],
                "end_type": last["end_type"],
                "lead_to_shot": bool(g["lead_to_shot"].fillna(False).any()),
                "lead_to_goal": bool(g["lead_to_goal"].fillna(False).any()),
                "n_line_breaks": int(
                    g[["first_line_break", "second_last_line_break", "last_line_break"]]
                    .fillna(False).to_numpy().sum()
                ),
                "epv_for_start": float(first["possession_epv_for_start"]),
                "epv_for_end": float(last["possession_epv_for_end"]),
                "epv_total": float(g["possession_epv_total"].sum(min_count=1)),
                "epv_delta_for": float(g["possession_epv_delta_for"].sum(min_count=1)),
                "epv_delta_against": float(g["possession_epv_delta_against"].sum(min_count=1)),
                "mean_pressure_id": float(g["overall_pressure_start_id"].mean()),
                "max_pressure_id": float(g["overall_pressure_start_id"].max()),
                "pressure_at_start_id": float(first["overall_pressure_start_id"]),
                "pressure_at_end_id": float(last["overall_pressure_end_id"]),
                "mean_nearest_opponent_m": float(g.get("fil_nearest_opponent_m", pd.Series(dtype=float)).mean()),
                "players": ", ".join(pd.unique(g["player_name"].dropna().astype(str))[:8]),
            }
        )

    out = (
        p.groupby("phase_index", dropna=True)
        .apply(agg, include_groups=False)
        .reset_index()  # groupby key comes back as a column
    )
    out["speed_of_progression_mps"] = out["territory_gain_m"] / out["duration_s"].replace(0, np.nan)
    out["passes_per_10s"] = out["n_passes"] / out["duration_s"].replace(0, np.nan) * 10
    ph = phases[["index", "team_in_possession_phase_type", "team_out_of_possession_phase_type",
                 "team_in_possession_width_start", "team_in_possession_length_start",
                 "team_out_of_possession_width_start", "team_out_of_possession_length_start"]]
    out = out.merge(ph, left_on="phase_index", right_on="index", how="left").drop(columns=["index"])
    out["is_home"] = out["team_id"] == meta.home_team_id
    return out


# --------------------------------------------------------------------------- #
# Passes, runs, pressures
# --------------------------------------------------------------------------- #

def build_passes(poss: pd.DataFrame) -> pd.DataFrame:
    """One row per attempted pass, taken from the possession it ended."""
    p = poss[poss["end_type"] == "pass"].copy()
    p["completed"] = p["pass_outcome"] == "successful"
    p["receiver_id"] = p["player_targeted_id"]
    p["fil_progressive_pass"] = (
        p["completed"]
        & (p["progression_m"] >= 5.0)
        & (p["pass_direction"] == "forward")
    )
    p["fil_line_breaking"] = p[["first_line_break", "second_last_line_break", "last_line_break"]].fillna(False).any(axis=1)
    p["fil_epv_added"] = p["pass_epv_total"]
    return p


def build_off_ball_runs(match_id: int) -> pd.DataFrame:
    ev = _events(match_id)
    r = ev[ev["event_type"] == "off_ball_run"].copy()
    cols = [
        "event_uid", "match_id", "frame_start", "frame_end", "match_clock_s", "period",
        "player_id", "player_name", "player_position", "team_id", "team_shortname",
        "event_subtype", "phase_index", "duration",
        "x_start_att", "y_start_att", "x_end_att", "y_end_att",
        "distance_covered", "speed_avg", "speed_avg_band", "trajectory_direction",
        "targeted", "received", "received_in_space", "lead_to_shot", "lead_to_goal",
        "dangerous", "xthreat", "break_defensive_line", "intended_run_behind",
        "push_defensive_line", "separation_start", "separation_end", "separation_gain",
        "delta_to_last_defensive_line_start", "delta_to_last_defensive_line_gain",
        "player_in_possession_id", "player_in_possession_name",
        "n_simultaneous_runs", "in_to_out", "out_to_in", "progression_m",
        "attack_sign", "third_start", "third_end", "channel_start", "channel_end",
    ]
    return r[[c for c in cols if c in r.columns]]


def build_pressures(match_id: int) -> pd.DataFrame:
    ev = _events(match_id)
    e = ev[ev["event_type"] == "on_ball_engagement"].copy()
    cols = [
        "event_uid", "match_id", "frame_start", "frame_end", "match_clock_s", "period",
        "player_id", "player_name", "player_position", "team_id", "team_shortname",
        "event_subtype", "phase_index", "duration",
        "x_start_att", "y_start_att", "x_end_att", "y_end_att",
        "player_in_possession_id", "player_in_possession_name",
        "interplayer_distance_start", "interplayer_distance_end", "interplayer_distance_min",
        "angle_of_engagement", "goal_side_start", "goal_side_end",
        "beaten_by_possession", "beaten_by_movement", "stop_possession_danger",
        "reduce_possession_danger", "force_backward", "possession_danger",
        "pressing_chain", "pressing_chain_length", "pressing_chain_end_type",
        "index_in_pressing_chain", "distance_covered", "speed_avg",
        "third_start", "third_end", "attack_sign",
    ]
    out = e[[c for c in cols if c in e.columns]].copy()
    out["fil_is_counterpress"] = out["event_subtype"] == "counter_press"
    return out


# --------------------------------------------------------------------------- #
# Transitions
# --------------------------------------------------------------------------- #

def build_transitions(match_id: int, window_s: float = 8.0) -> pd.DataFrame:
    """One row per change of team possession, with what each side did next.

    A transition is located at the first frame of the new team's possession.
    The window that follows is measured on tracking: how far the winning team
    moved the ball up the pitch, and how quickly the losing team re-formed.
    """
    from src.features.team_shape import load_team_frames

    ball = load_ball(match_id)
    meta = load_match_meta(match_id)
    b = ball.dropna(subset=["possession_group"]).copy()
    if b.empty:
        return pd.DataFrame()
    grp = b["possession_group"].astype(str)
    change = grp != grp.shift()
    starts = b[change]

    try:
        tf = load_team_frames(match_id)
    except FileNotFoundError:
        tf = pd.DataFrame()

    recs = []
    prev_group = None
    for row in starts.itertuples(index=False):
        g = str(row.possession_group)
        if prev_group is None or g == prev_group:
            prev_group = g
            continue
        winner_side = "home" if g.startswith("home") else "away"
        winner_id = meta.home_team_id if winner_side == "home" else meta.away_team_id
        loser_id = meta.away_team_id if winner_side == "home" else meta.home_team_id
        f0 = int(row.frame)
        f1 = f0 + int(window_s * 10)
        seg = ball[(ball["frame"] >= f0) & (ball["frame"] <= f1)].dropna(subset=["ball_x_std"])
        if seg.empty:
            prev_group = g
            continue
        sign = meta.attack_sign(winner_id, int(row.period)) if pd.notna(row.period) else 1
        bx = seg["ball_x_std"].to_numpy() * sign
        rec = {
            "match_id": match_id,
            "frame": f0,
            "period": int(row.period) if pd.notna(row.period) else None,
            "match_clock_s": float(row.match_clock_s),
            "winning_team_id": winner_id,
            "losing_team_id": loser_id,
            "regain_x_att": float(bx[0]),
            "regain_y_att": float(seg["ball_y_std"].to_numpy()[0] * sign),
            "max_progress_m": float(bx.max() - bx[0]),
            "progress_at_window_m": float(bx[-1] - bx[0]),
            "time_to_max_progress_s": float(
                (seg["frame"].to_numpy()[int(np.argmax(bx))] - f0) / 10.0
            ),
            "held_possession_s": float(_hold_duration(ball, f0, g)),
        }
        if not tf.empty:
            w = tf[(tf["team_id"] == winner_id) & (tf["frame"].between(f0, f1))]
            l = tf[(tf["team_id"] == loser_id) & (tf["frame"].between(f0, f1))]
            if len(w) > 2:
                rec["winner_compactness_start_m"] = float(w["compactness_m"].iloc[0])
                rec["winner_length_gain_m"] = float(w["length_m"].iloc[-1] - w["length_m"].iloc[0])
                rec["winner_line_advance_m"] = float(w["def_line_x"].iloc[-1] - w["def_line_x"].iloc[0])
            if len(l) > 2:
                rec["loser_compactness_start_m"] = float(l["compactness_m"].iloc[0])
                rec["loser_compactness_end_m"] = float(l["compactness_m"].iloc[-1])
                rec["loser_recovery_m"] = float(l["compactness_m"].iloc[0] - l["compactness_m"].iloc[-1])
                rec["loser_line_drop_m"] = float(l["def_line_x"].iloc[0] - l["def_line_x"].iloc[-1])
        recs.append(rec)
        prev_group = g
    return pd.DataFrame(recs)


def _hold_duration(ball: pd.DataFrame, f0: int, group: str) -> float:
    sub = ball[ball["frame"] >= f0]
    g = sub["possession_group"].astype("object").fillna("").astype(str).to_numpy()
    same = g == group
    if not same.any():
        return 0.0
    end = int(np.argmax(~same)) if (~same).any() else len(same)
    return end / 10.0


# --------------------------------------------------------------------------- #

def build(match_ids: list[int] | None = None) -> dict[str, int]:
    match_ids = match_ids or available_match_ids()
    poss_all, tp_all, run_all, pr_all, tr_all = [], [], [], [], []
    for mid in match_ids:
        p = build_possessions(mid)
        poss_all.append(p)
        tp_all.append(build_team_possessions(p, mid))
        run_all.append(build_off_ball_runs(mid))
        pr_all.append(build_pressures(mid))
        tr_all.append(build_transitions(mid))
        log.info("possessions %s: %d", mid, len(p))

    poss = pd.concat(poss_all, ignore_index=True)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    poss.to_parquet(POSSESSIONS_PATH, index=False, compression="zstd")
    pd.concat(tp_all, ignore_index=True).to_parquet(TEAM_POSSESSIONS_PATH, index=False, compression="zstd")
    build_passes(poss).to_parquet(PASSES_PATH, index=False, compression="zstd")
    pd.concat(run_all, ignore_index=True).to_parquet(RUNS_PATH, index=False, compression="zstd")
    pd.concat(pr_all, ignore_index=True).to_parquet(PRESSURES_PATH, index=False, compression="zstd")
    pd.concat(tr_all, ignore_index=True).to_parquet(TRANSITIONS_PATH, index=False, compression="zstd")
    return {
        "possessions": len(poss),
        "team_possessions": sum(len(t) for t in tp_all),
        "runs": sum(len(r) for r in run_all),
        "pressures": sum(len(p) for p in pr_all),
        "transitions": sum(len(t) for t in tr_all),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(build())
