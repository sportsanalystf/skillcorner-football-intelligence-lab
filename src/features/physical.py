"""Match-level physical output computed from tracking.

This is deliberately kept separate from the season aggregate files. The two
are *not* the same quantity and should never be summed together:

* the season aggregates are SkillCorner's own, filtered to performances of
  60+ minutes and computed on their full pipeline;
* what is here is computed from the open broadcast tracking in this repo, over
  whatever frames a player happened to be tracked in, with the smoothing
  described in :mod:`src.tracking.kinematics`.

Broadcast tracking only sees the part of the pitch the camera is showing, so a
player's tracked time is always less than their played time. Every row records
``tracked_minutes`` and ``coverage`` next to the totals so that a distance can
be read as "distance while visible", which is what it is.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.config import DERIVED_DIR, DT
from src.data.loaders import available_match_ids, load_match_meta
from src.preprocessing.build_tracking import load_ball, load_tracking
from src.tracking.kinematics import summarise_player_physical

log = logging.getLogger(__name__)

OUT_PATH = DERIVED_DIR / "player_match_physical.parquet"
SPRINTS_PATH = DERIVED_DIR / "sprints.parquet"


def player_match_physical(match_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta = load_match_meta(match_id)
    trk = load_tracking(
        match_id,
        columns=["frame", "period", "match_clock_s", "player_id", "team_id", "side",
                 "position", "position_group", "speed", "accel", "step_m",
                 "is_detected", "x_std", "y_std", "attack_sign"],
    )
    ball = load_ball(match_id)
    poss = ball.set_index("frame")["possession_group"]

    squad = meta.players_frame().set_index("player_id")
    rows = []
    sprint_rows = []

    for pid, grp in trk.groupby("player_id", observed=True):
        s = summarise_player_physical(grp)
        info = squad.loc[pid] if pid in squad.index else None
        side = str(grp["side"].iloc[0])
        team_poss = poss.reindex(grp["frame"]).astype("object").fillna("").to_numpy().astype(str)
        in_poss = team_poss == f"{side} team"
        out_poss = team_poss == ("away team" if side == "home" else "home team")

        rec = {
            "match_id": match_id,
            "player_id": int(pid),
            "team_id": int(grp["team_id"].iloc[0]),
            "side": side,
            "position": str(grp["position"].iloc[0]),
            "position_group": str(grp["position_group"].iloc[0]),
            "player_name": (info["display_name"] if info is not None else str(pid)),
            "minutes_played": (float(info["minutes_played"]) if info is not None
                               and pd.notna(info["minutes_played"]) else np.nan),
            **s.to_dict(),
        }
        rec["coverage"] = rec["tracked_minutes"] / rec["minutes_played"] if rec["minutes_played"] else np.nan
        step = grp["step_m"].to_numpy()
        rec["distance_tip_m"] = float(np.nansum(np.where(in_poss, step, 0.0)))
        rec["distance_otip_m"] = float(np.nansum(np.where(out_poss, step, 0.0)))
        rec["minutes_tip_tracked"] = float(in_poss.sum() * DT / 60)
        rec["minutes_otip_tracked"] = float(out_poss.sum() * DT / 60)

        # Average attacking-frame position while tracked: a data-driven
        # "average position", which is what a heat-map centroid shows.
        xa = grp["x_std"].to_numpy() * grp["attack_sign"].to_numpy()
        ya = grp["y_std"].to_numpy() * grp["attack_sign"].to_numpy()
        rec["avg_x_att"] = float(np.nanmean(xa))
        rec["avg_y_att"] = float(np.nanmean(ya))
        rec["x_att_std"] = float(np.nanstd(xa))
        rec["y_att_std"] = float(np.nanstd(ya))
        rows.append(rec)

        from src.tracking.kinematics import detect_efforts
        from src.config import SPRINT_THRESHOLD_MS

        for eff in detect_efforts(grp["speed"].to_numpy(), grp["frame"].to_numpy(),
                                  SPRINT_THRESHOLD_MS):
            sprint_rows.append({"match_id": match_id, "player_id": int(pid),
                                "team_id": int(grp["team_id"].iloc[0]), **eff})

    return pd.DataFrame(rows), pd.DataFrame(sprint_rows)


def build(match_ids: list[int] | None = None) -> dict[str, int]:
    match_ids = match_ids or available_match_ids()
    frames, sprints = [], []
    for mid in match_ids:
        f, s = player_match_physical(mid)
        frames.append(f)
        if len(s):
            sprints.append(s)
        log.info("physical %s: %d players", mid, len(f))
    out = pd.concat(frames, ignore_index=True)
    spr = pd.concat(sprints, ignore_index=True) if sprints else pd.DataFrame()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PATH, index=False, compression="zstd")
    spr.to_parquet(SPRINTS_PATH, index=False, compression="zstd")
    return {"player_matches": len(out), "sprints": len(spr)}


def load() -> pd.DataFrame:
    return pd.read_parquet(OUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(build())
