"""Data-quality validation for the SkillCorner Open Data tables.

SkillCorner's own README is explicit that "around 97% of the player identity we
provide are accurate" and that "some speed or acceleration smoothing and
control should be applied to the raw data". This module measures both, plus
everything else that can quietly poison a tracking analysis, and writes the
numbers to ``reports/validation.parquet`` and ``reports/validation.md``.

Every check returns a number, never a pass/fail alone, because the useful
question is not "is the data clean" but "how much of it can I trust for this
particular metric".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import DT, MAX_PLAUSIBLE_SPEED_MS, PROCESSED_DIR, REPORTS_DIR
from src.data.loaders import load_match_meta
from src.preprocessing.build_tracking import load_ball, load_tracking

log = logging.getLogger(__name__)

#: A jump larger than this between consecutive frames is, at 10 Hz, a 25 m/s
#: translation. It is an identity swap or a homography glitch, not a player.
TELEPORT_STEP_M = 2.5
#: How far outside the touchline a coordinate may sit before it is impossible.
PITCH_MARGIN_M = 5.0


@dataclass
class Check:
    name: str
    value: float
    unit: str = ""
    note: str = ""
    severity: str = "info"  # info | warn | error


@dataclass
class MatchValidation:
    match_id: int
    checks: list[Check] = field(default_factory=list)

    def add(self, *args, **kwargs) -> None:
        self.checks.append(Check(*args, **kwargs))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"match_id": self.match_id, **c.__dict__} for c in self.checks]
        )


# --------------------------------------------------------------------------- #

def validate_match(match_id: int) -> MatchValidation:
    v = MatchValidation(match_id)
    meta = load_match_meta(match_id)
    trk = load_tracking(match_id)
    ball = load_ball(match_id)
    events = pd.read_parquet(
        PROCESSED_DIR / "events.parquet",
        filters=[("match_id", "==", match_id)],
    )
    phases = pd.read_parquet(
        PROCESSED_DIR / "phases.parquet",
        filters=[("match_id", "==", match_id)],
    )

    # --- frame integrity ------------------------------------------------- #
    frames = ball["frame"].to_numpy()
    v.add("ball_frames", len(ball), "frames")
    v.add("duplicate_frames", int(len(frames) - len(np.unique(frames))), "frames",
          severity="error" if len(frames) != len(np.unique(frames)) else "info")
    gaps = np.diff(np.sort(frames))
    v.add("frame_sequence_breaks", int((gaps != 1).sum()), "breaks",
          "frames should increment by exactly 1",
          severity="error" if (gaps != 1).any() else "info")

    # --- clock consistency ------------------------------------------------ #
    # Reconstruct SkillCorner's own timestamp from the frame number and compare.
    recon = meta.clock_seconds(ball["frame"], ball["period"].astype("float"))
    delta = np.abs(recon - ball["match_clock_s"].to_numpy())
    v.add("clock_max_abs_error", float(np.nanmax(delta)) if np.isfinite(delta).any() else 0.0,
          "s", "derived match clock vs the file's own period offsets")

    # --- coverage --------------------------------------------------------- #
    n_untracked = int(ball["ball_x"].isna().sum())
    v.add("frames_without_tracking", n_untracked, "frames",
          f"{n_untracked / len(ball):.1%} of the file carries no players or ball")
    per_frame = trk.groupby("frame", observed=True).size()
    v.add("frames_with_22_players", int((per_frame == 22).sum()), "frames")
    v.add("frames_with_other_counts", int((per_frame != 22).sum()), "frames",
          severity="warn" if (per_frame != 22).any() else "info")
    v.add("player_frames", len(trk), "rows")
    v.add("detected_fraction", float(trk["is_detected"].mean()), "frac",
          "the rest are SkillCorner extrapolations")

    # --- identity continuity ---------------------------------------------- #
    d = trk.sort_values(["player_id", "frame"])
    dx = d.groupby("player_id", observed=True)["x"].diff()
    dy = d.groupby("player_id", observed=True)["y"].diff()
    dfr = d.groupby("player_id", observed=True)["frame"].diff()
    step = np.hypot(dx, dy).where(dfr == 1)
    n_tp = int((step > TELEPORT_STEP_M).sum())
    v.add("teleport_steps", n_tp, "steps",
          f">{TELEPORT_STEP_M} m in one 0.1 s frame ({n_tp / max(len(trk),1):.4%} of steps)",
          severity="warn" if n_tp / max(len(trk), 1) > 0.001 else "info")
    v.add("max_single_frame_step", float(np.nanmax(step)) if step.notna().any() else 0.0, "m")

    # --- coordinates ------------------------------------------------------ #
    hl, hw = meta.pitch.length / 2 + PITCH_MARGIN_M, meta.pitch.width / 2 + PITCH_MARGIN_M
    oob = ((trk["x"].abs() > hl) | (trk["y"].abs() > hw)).sum()
    v.add("players_out_of_bounds", int(oob), "rows",
          f"beyond {PITCH_MARGIN_M} m outside the touchline",
          severity="warn" if oob else "info")

    # --- kinematics -------------------------------------------------------- #
    v.add("speed_masked_implausible", int(trk["speed"].isna().sum()), "rows",
          f"speed > {MAX_PLAUSIBLE_SPEED_MS} m/s or |a| > 8 m/s^2 after smoothing")
    v.add("max_speed_ms", float(trk["speed"].max()), "m/s")
    v.add("p999_speed_ms", float(trk["speed"].quantile(0.999)), "m/s")
    v.add("max_abs_accel_ms2", float(trk["accel"].abs().max()), "m/s2")

    # --- ball -------------------------------------------------------------- #
    tracked_frames = ball["ball_x"].notna()
    v.add("ball_missing_within_tracked", int((~ball.loc[tracked_frames, "ball_x"].notna()).sum()),
          "frames")
    v.add("ball_detected_fraction", float(ball.loc[tracked_frames, "ball_detected"].mean()), "frac")
    v.add("ball_z_min", float(ball["ball_z"].min()), "m",
          "negative z is a reconstruction artefact")
    v.add("ball_z_max", float(ball["ball_z"].max()), "m")
    v.add("ball_max_speed_ms", float(ball["ball_speed"].max()), "m/s")

    # --- possession -------------------------------------------------------- #
    v.add("frames_with_possession_group", int(ball["possession_group"].notna().sum()), "frames")
    v.add("frames_with_possession_player", int(ball["possession_player_id"].notna().sum()), "frames")

    # --- events ------------------------------------------------------------ #
    v.add("events", len(events), "rows")
    for et, n in events["event_type"].value_counts().items():
        v.add(f"events_{et}", int(n), "rows")
    squad_ids = set(meta.players_frame()["player_id"])
    unknown = set(events["player_id"].dropna().astype(int)) - squad_ids
    v.add("events_with_unknown_player", len(unknown), "players",
          severity="error" if unknown else "info")
    pp = events[events["event_type"] == "player_possession"]
    v.add("possession_epv_populated", float(pp["possession_epv_total"].notna().mean()),
          "frac", "share of player_possession rows carrying EPV")
    v.add("pressure_populated", float(pp["overall_pressure_start"].notna().mean()), "frac")
    v.add("fully_extrapolated_events", int(events["fully_extrapolated"].fillna(False).sum()),
          "rows", "events whose whole window is extrapolated tracking")

    # --- event <-> tracking synchronisation --------------------------------- #
    sync = event_tracking_offset(events, trk)
    v.add("event_sync_matched", int(sync["n"]), "events",
          "player_possession events whose player is tracked at frame_start")
    v.add("event_sync_median_error_m", float(sync["median"]), "m",
          "distance between the event's own x/y and the tracked position")
    v.add("event_sync_p90_error_m", float(sync["p90"]), "m",
          severity="warn" if sync["p90"] > 3.0 else "info")

    # --- phases ------------------------------------------------------------- #
    v.add("phases", len(phases), "rows")
    ph = phases.sort_values("frame_start")
    overlap = int((ph["frame_start"].shift(-1) < ph["frame_end"]).sum())
    v.add("phase_overlaps", overlap, "pairs", severity="warn" if overlap else "info")
    covered = float((ph["frame_end"] - ph["frame_start"]).sum())
    v.add("phase_coverage_of_tracked_frames",
          covered / max(int(tracked_frames.sum()), 1), "frac",
          "phases are only defined while the ball is in play")

    return v


def event_tracking_offset(events: pd.DataFrame, trk: pd.DataFrame) -> dict[str, float]:
    """How far an event's stated x/y sits from the tracked player at that frame.

    This is the single most important cross-source check in the project: every
    "show me the tracking context of this event" feature depends on it.
    """
    pp = events[(events["event_type"] == "player_possession") & events["player_id"].notna()]
    if pp.empty:
        return {"n": 0, "median": np.nan, "p90": np.nan}
    key = pp[["frame_start", "player_id", "x_start_trk", "y_start_trk"]].rename(
        columns={"frame_start": "frame", "x_start_trk": "x_start", "y_start_trk": "y_start"}
    )
    key["player_id"] = key["player_id"].astype("int64")
    m = key.merge(trk[["frame", "player_id", "x", "y"]], on=["frame", "player_id"], how="inner")
    if m.empty:
        return {"n": 0, "median": np.nan, "p90": np.nan}
    dist = np.hypot(m["x_start"] - m["x"], m["y_start"] - m["y"])
    return {
        "n": len(m),
        "median": float(np.nanmedian(dist)),
        "p90": float(np.nanpercentile(dist, 90)),
    }


def run(match_ids: list[int] | None = None) -> pd.DataFrame:
    from src.data.loaders import available_match_ids

    match_ids = match_ids or available_match_ids()
    frames = []
    for mid in match_ids:
        try:
            frames.append(validate_match(mid).to_frame())
            log.info("validated %s", mid)
        except FileNotFoundError as exc:
            log.warning("skipping %s: %s", mid, exc)
    out = pd.concat(frames, ignore_index=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(REPORTS_DIR / "validation.parquet", index=False)
    _write_markdown(out)
    return out


def _write_markdown(df: pd.DataFrame) -> None:
    piv = df.pivot_table(index="name", columns="match_id", values="value", aggfunc="first")
    summary = (
        df.groupby("name")
        .agg(unit=("unit", "first"), min=("value", "min"),
             median=("value", "median"), max=("value", "max"),
             note=("note", "first"))
        .reset_index()
    )
    lines = [
        "# Validation report",
        "",
        f"Generated over {df.match_id.nunique()} matches.",
        "",
        "## Across matches",
        "",
        "| check | unit | min | median | max | note |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| `{r['name']}` | {r['unit']} | {r['min']:,.4g} | {r['median']:,.4g} "
            f"| {r['max']:,.4g} | {r['note']} |"
        )
    lines += ["", "## Per match", "", piv.round(4).to_markdown()]
    (REPORTS_DIR / "validation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = run()
    print(out.groupby("name")["value"].median().to_string())
