"""Normalise dynamic events, phases of play, squads and season aggregates.

The dynamic-events CSV is 322 columns wide and mixes four different event
types into one table, each populating a different subset of those columns.
Rather than throw columns away, this module keeps the whole table but:

* adds a single ``match_clock_s`` so events join to tracking without frame
  arithmetic at the call site,
* rescales x/y onto the 105 x 68 standard pitch,
* adds attacking-direction-normalised coordinates,
* fixes the dtypes pandas guesses wrong (booleans stored as ``True``/``False``
  strings, ids stored as floats),
* and splits out three narrow, strongly-typed views that most analysis wants:
  ``possessions``, ``passing_options`` and ``off_ball_runs``.

Nothing is invented. Every column that appears downstream either exists in the
raw file or is derived here with the derivation documented in
``docs/METHODOLOGY.md``.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR
from src.data.loaders import (
    available_match_ids,
    load_aggregate,
    load_events_raw,
    load_match_meta,
    load_phases_raw,
    match_index,
)

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

EVENT_TYPES = ("player_possession", "passing_option", "off_ball_run", "on_ball_engagement")

#: Columns that only make sense on ``player_possession`` rows.
EPV_COLS = [
    "possession_epv_for_start", "possession_epv_for_end",
    "possession_epv_against_start", "possession_epv_against_end",
    "possession_epv_delta_for", "possession_epv_delta_against",
    "possession_epv_total",
    "pass_reception_epv_for", "pass_reception_epv_against",
    "pass_epv_delta_for", "pass_epv_delta_against", "pass_epv_total",
]

PRESSURE_COLS = [
    "overall_pressure_start", "overall_pressure_end",
    "reception_difficulty_start", "space_constraint_start",
    "time_to_impact_start", "time_to_impact_end",
    "passing_option_ease_start", "passing_option_ease_end",
]

PRESSURE_ID_COLS = [c + "_id" for c in PRESSURE_COLS]

XY_PAIRS = [
    ("x_start", "y_start"),
    ("x_end", "y_end"),
    ("player_in_possession_x_start", "player_in_possession_y_start"),
    ("player_in_possession_x_end", "player_in_possession_y_end"),
    ("player_targeted_x_pass", "player_targeted_y_pass"),
    ("player_targeted_x_reception", "player_targeted_y_reception"),
]

X_ONLY = ["last_defensive_line_x_start", "last_defensive_line_x_end"]

BOOL_LIKE = [
    "penalty_area_start", "penalty_area_end", "high_pass", "lead_to_shot",
    "lead_to_goal", "targeted", "received", "received_in_space",
    "possession_danger", "beaten_by_possession", "beaten_by_movement",
    "stop_possession_danger", "reduce_possession_danger", "force_backward",
    "in_to_out", "out_to_in", "one_touch", "quick_pass", "carry",
    "forward_momentum", "is_header", "hand_pass", "initiate_give_and_go",
    "give_and_go", "intended_run_behind", "push_defensive_line",
    "break_defensive_line", "first_line_break", "second_last_line_break",
    "last_line_break", "organised_defense", "pass_ahead", "dangerous",
    "difficult_pass_target", "inside_defensive_shape_start",
    "inside_defensive_shape_end", "goal_side_start", "goal_side_end",
    "fully_extrapolated", "pressing_chain", "team_possession_loss_in_phase",
    "first_player_possession_in_team_possession",
    "last_player_possession_in_team_possession",
    "lead_to_different_phase", "issued_from_different_phase",
    "close_at_player_possession_start",
    "player_targeted_dangerous", "player_targeted_difficult_pass_target",
    "predicted_passing_option", "is_player_possession_start_matched",
    "is_player_possession_end_matched", "is_previous_pass_matched",
    "is_pass_reception_matched",
]


def _coerce_bools(df: pd.DataFrame) -> pd.DataFrame:
    for c in BOOL_LIKE:
        if c in df.columns:
            s = df[c]
            if s.dtype == object or str(s.dtype) == "string":
                s = s.astype("string").str.lower().map(
                    {"true": True, "false": False, "1": True, "0": False}
                )
            df[c] = s.astype("boolean")
    return df


def build_events(match_id: int) -> pd.DataFrame:
    meta = load_match_meta(match_id)
    df = load_events_raw(match_id)
    df["match_id"] = match_id
    df["match_clock_s"] = meta.clock_seconds(df["frame_start"], df["period"])
    df["match_clock_end_s"] = meta.clock_seconds(df["frame_end"], df["period"])

    sx = 105.0 / meta.pitch.length
    sy = 68.0 / meta.pitch.width

    # attack_sign: +1 when the event's team attacks towards +x this period.
    sign = np.array(
        [meta.attack_sign(int(t), int(p)) for t, p in zip(df["team_id"], df["period"])],
        dtype="int8",
    )
    df["attack_sign"] = sign

    # ---------------------------------------------------------------- #
    # Orientation. Verified empirically (tests/test_event_orientation.py):
    # the x/y on every dynamic-event and phase row are ALREADY expressed in
    # the attacking frame of that row's own team -- a team attacking
    # right-to-left in the broadcast has its event coordinates negated
    # relative to the tracking file. Matching a player_possession row against
    # the tracked position of its own player at frame_start reproduces the
    # tracking coordinate to ~5e-7 m after multiplying by ``attack_sign``,
    # and not at all without it.
    #
    #   <col>          raw, attacking frame, this match's metres
    #   <col>_att      attacking frame, standard 105x68 pitch
    #   <col>_std      broadcast frame, standard 105x68 pitch (joins to x_std)
    #   <col>_trk      broadcast frame, this match's metres (joins to x/y)
    # ---------------------------------------------------------------- #
    for xc, yc in XY_PAIRS:
        if xc in df.columns:
            df[xc + "_att"] = (df[xc] * sx).astype("float32")
            df[yc + "_att"] = (df[yc] * sy).astype("float32")
            df[xc + "_std"] = (df[xc + "_att"] * sign).astype("float32")
            df[yc + "_std"] = (df[yc + "_att"] * sign).astype("float32")
            df[xc + "_trk"] = (df[xc] * sign).astype("float32")
            df[yc + "_trk"] = (df[yc] * sign).astype("float32")
    for xc in X_ONLY:
        if xc in df.columns:
            df[xc + "_att"] = (df[xc] * sx).astype("float32")
            df[xc + "_std"] = (df[xc + "_att"] * sign).astype("float32")

    df = _coerce_bools(df)

    # Progression, in attacking space: how much closer to the opposition goal.
    if "x_start_att" in df.columns:
        df["progression_m"] = (df["x_end_att"] - df["x_start_att"]).astype("float32")
        df["dist_to_goal_start"] = np.hypot(52.5 - df["x_start_att"], df["y_start_att"]).astype("float32")
        df["dist_to_goal_end"] = np.hypot(52.5 - df["x_end_att"], df["y_end_att"]).astype("float32")
        df["goal_distance_gain_m"] = (df["dist_to_goal_start"] - df["dist_to_goal_end"]).astype("float32")

    # ``event_id`` is a string of the form "{event_type_id}_{n}" and is unique
    # only within a match, so a globally addressable key is added here.
    df["event_uid"] = df["match_id"].astype(str) + ":" + df["event_id"].astype(str)
    df["assoc_possession_uid"] = np.where(
        df["associated_player_possession_event_id"].notna(),
        df["match_id"].astype(str) + ":" + df["associated_player_possession_event_id"].astype(str),
        None,
    )
    df["targeted_option_uid"] = np.where(
        df["targeted_passing_option_event_id"].notna(),
        df["match_id"].astype(str) + ":" + df["targeted_passing_option_event_id"].astype(str),
        None,
    )

    for c in ("player_id", "team_id", "player_targeted_id", "player_in_possession_id",
              "phase_index", "player_possession_phase_index"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float64").astype("Int64")

    return df.copy()  # de-fragment after the wide column additions


def build_phases(match_id: int) -> pd.DataFrame:
    meta = load_match_meta(match_id)
    df = load_phases_raw(match_id)
    df["match_id"] = match_id
    df["match_clock_s"] = meta.clock_seconds(df["frame_start"], df["period"])
    df["match_clock_end_s"] = meta.clock_seconds(df["frame_end"], df["period"])
    sx = 105.0 / meta.pitch.length
    sy = 68.0 / meta.pitch.width
    sign = np.array(
        [meta.attack_sign(int(t), int(p)) for t, p in zip(df["team_in_possession_id"], df["period"])],
        dtype="int8",
    )
    df["attack_sign"] = sign
    for xc, yc in (("x_start", "y_start"), ("x_end", "y_end")):
        # Same convention as the events file: raw is already attacking-frame.
        df[xc + "_att"] = (df[xc] * sx).astype("float32")
        df[yc + "_att"] = (df[yc] * sy).astype("float32")
        df[xc + "_std"] = (df[xc + "_att"] * sign).astype("float32")
        df[yc + "_std"] = (df[yc + "_att"] * sign).astype("float32")
        df[xc + "_trk"] = (df[xc] * sign).astype("float32")
        df[yc + "_trk"] = (df[yc] * sign).astype("float32")
    df["territory_gain_m"] = (df["x_end_att"] - df["x_start_att"]).astype("float32")
    df = _coerce_bools(df)
    df["phase_uid"] = df["match_id"].astype(str) + ":" + df["index"].astype(str)
    df["team_out_of_possession_id"] = np.where(
        df["team_in_possession_id"] == meta.home_team_id, meta.away_team_id, meta.home_team_id
    )
    return df


def build_squads() -> pd.DataFrame:
    rows = [load_match_meta(m).players_frame() for m in available_match_ids()]
    df = pd.concat(rows, ignore_index=True)
    return df


def build_match_table() -> pd.DataFrame:
    recs = []
    for mid in available_match_ids():
        m = load_match_meta(mid)
        h, a = m.score
        recs.append(
            {
                "match_id": mid,
                "date": m.date,
                "competition": (m.raw.get("competition_edition") or {}).get("name"),
                "round": (m.raw.get("competition_round") or {}).get("name"),
                "stadium": (m.raw.get("stadium") or {}).get("name"),
                "home_team_id": m.home_team_id,
                "home_team": m.home_team_name,
                "away_team_id": m.away_team_id,
                "away_team": m.away_team_name,
                "home_score": h,
                "away_score": a,
                "label": m.label,
                "pitch_length": m.pitch.length,
                "pitch_width": m.pitch.width,
                "period1_start_frame": m.period_start_frames.get(1),
                "period1_end_frame": m.period_end_frames.get(1),
                "period2_start_frame": m.period_start_frames.get(2),
                "period2_end_frame": m.period_end_frames.get(2),
                "home_side_p1": (m.raw.get("home_team_side") or [None])[0],
                "has_bodypose": mid in (1925299, 1996435),
            }
        )
    df = pd.DataFrame(recs).sort_values("date", ignore_index=True)
    return df


def run(match_ids: list[int] | None = None) -> dict[str, int]:
    match_ids = match_ids or available_match_ids()
    ev = pd.concat([build_events(m) for m in match_ids], ignore_index=True)
    ph = pd.concat([build_phases(m) for m in match_ids], ignore_index=True)
    squads = build_squads()
    matches = build_match_table()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(PROCESSED_DIR / "events.parquet", index=False, compression="zstd")
    ph.to_parquet(PROCESSED_DIR / "phases.parquet", index=False, compression="zstd")
    squads.to_parquet(PROCESSED_DIR / "squads.parquet", index=False, compression="zstd")
    matches.to_parquet(PROCESSED_DIR / "matches.parquet", index=False, compression="zstd")
    match_index().to_parquet(PROCESSED_DIR / "match_index.parquet", index=False)

    for kind in ("physical", "passing", "obr"):
        load_aggregate(kind).to_parquet(
            PROCESSED_DIR / f"season_{kind}.parquet", index=False, compression="zstd"
        )

    return {
        "events": len(ev), "phases": len(ph),
        "squad_rows": len(squads), "matches": len(matches),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())
