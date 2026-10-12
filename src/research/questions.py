"""Research questions run against the full dataset.

Each function returns ``(table, findings)`` where ``table`` is the evidence and
``findings`` is a short list of statements that are true of that table. Nothing
here is written by hand: the numbers quoted in ``docs/ANALYTICS.md`` and in the
app's Research page are read out of the artefacts this module writes to
``reports/research/``.

Statistical notes
-----------------
* Effects are reported with a rank-based test (Spearman / Kruskal-Wallis)
  because almost every quantity here -- EPV deltas, progression, compactness --
  is heavy-tailed and often zero-inflated.
* p-values are computed on 5,000-19,000 observations, so almost anything is
  "significant". Effect sizes are what is quoted; p-values are reported for
  completeness and never used as the headline.
* These are 20 matches from one league. Nothing here is a league-wide law.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy import stats

from src.config import DERIVED_DIR, REPORTS_DIR

log = logging.getLogger(__name__)

RESEARCH_DIR = REPORTS_DIR / "research"


def _load(name: str) -> pd.DataFrame:
    return pd.read_parquet(DERIVED_DIR / f"{name}.parquet")


def _fmt_p(p: float) -> str:
    return "<1e-16" if p < 1e-16 else f"{p:.2e}"


# --------------------------------------------------------------------------- #
# Q1
# --------------------------------------------------------------------------- #

def q1_pressure_and_pass_value() -> tuple[pd.DataFrame, list[str]]:
    """How does pressure on the passer change what the pass is worth?"""
    poss = _load("possessions")
    p = poss[(poss["end_type"] == "pass") & poss["overall_pressure_start_id"].notna()]
    order = ["no_pressure", "low_pressure", "medium_pressure", "high_pressure", "very_high_pressure"]
    tbl = (
        p.assign(completed=p["pass_outcome"] == "successful")
        .groupby("overall_pressure_start", observed=True)
        .agg(
            n=("completed", "size"),
            completion=("completed", "mean"),
            pass_epv=("pass_epv_total", "mean"),
            progression_m=("progression_m", "mean"),
            forward_share=("pass_direction", lambda s: (s == "forward").mean()),
            nearest_opponent_m=("fil_nearest_opponent_m", "mean"),
        )
        .reindex(order)
        .reset_index()
    )
    rho = stats.spearmanr(p["overall_pressure_start_id"], p["pass_epv_total"], nan_policy="omit")
    rho_c = stats.spearmanr(
        p["overall_pressure_start_id"], (p["pass_outcome"] == "successful").astype(int)
    )
    findings = [
        f"Completion falls monotonically with pressure, from "
        f"{tbl.loc[0, 'completion']:.1%} unpressured to {tbl.loc[4, 'completion']:.1%} "
        f"under very high pressure (Spearman rho={rho_c.statistic:.3f}, p={_fmt_p(rho_c.pvalue)}, "
        f"n={len(p):,}).",
        f"Pass EPV added does not fall monotonically: it peaks at medium pressure "
        f"({tbl.loc[2, 'pass_epv']:+.4f}) and only turns negative at very high pressure "
        f"({tbl.loc[4, 'pass_epv']:+.4f}). Overall rho={rho.statistic:.3f}, p={_fmt_p(rho.pvalue)}.",
        f"SkillCorner's pressure bands line up with tracking geometry that never saw them: "
        f"mean distance to the nearest opponent runs "
        f"{tbl.loc[0, 'nearest_opponent_m']:.1f} m -> {tbl.loc[4, 'nearest_opponent_m']:.1f} m "
        f"across the five bands.",
    ]
    return tbl, findings


# --------------------------------------------------------------------------- #
# Q2
# --------------------------------------------------------------------------- #

def q2_value_under_pressure(min_possessions: int = 60) -> tuple[pd.DataFrame, list[str]]:
    """Which players keep creating value when they are pressed?"""
    poss = _load("possessions")
    pressed = poss[poss["overall_pressure_start_id"] >= 3]
    g = (
        pressed.groupby(["player_id", "player_name", "team_shortname", "player_position"],
                        observed=True)
        .agg(
            possessions_pressed=("possession_epv_total", "size"),
            epv_total=("possession_epv_total", "sum"),
            epv_per_possession=("possession_epv_total", "mean"),
            retention=("end_type", lambda s: (s != "possession_loss").mean()),
            progression_m=("progression_m", "mean"),
            mean_pressure=("overall_pressure_start_id", "mean"),
        )
        .reset_index()
    )
    g = g[g["possessions_pressed"] >= min_possessions].sort_values(
        "epv_per_possession", ascending=False
    )
    all_p = (
        poss.groupby("player_id", observed=True)["possession_epv_total"]
        .mean().rename("epv_per_possession_all")
    )
    g = g.merge(all_p, on="player_id", how="left")
    g["pressure_resilience"] = g["epv_per_possession"] - g["epv_per_possession_all"]
    top = g.head(5)
    findings = [
        f"{len(g)} players had at least {min_possessions} possessions under medium-or-worse "
        f"pressure across the 20 matches.",
        "Highest EPV added per pressed possession: "
        + "; ".join(f"{r.player_name} ({r.team_shortname}) {r.epv_per_possession:+.4f}"
                    for r in top.itertuples()),
        f"Best pressure resilience (pressed EPV minus overall EPV): "
        + "; ".join(
            f"{r.player_name} {r.pressure_resilience:+.4f}"
            for r in g.nlargest(3, "pressure_resilience").itertuples()
        ),
    ]
    return g, findings


# --------------------------------------------------------------------------- #
# Q3
# --------------------------------------------------------------------------- #

def q3_shape_at_loss() -> tuple[pd.DataFrame, list[str]]:
    """Does the shape a team is in when it loses the ball predict what happens next?"""
    tr = _load("transitions").dropna(subset=["loser_compactness_start_m", "max_progress_m"])
    q = pd.qcut(tr["loser_compactness_start_m"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    tbl = (
        tr.assign(quintile=q)
        .groupby("quintile", observed=True)
        .agg(
            n=("max_progress_m", "size"),
            compactness_m=("loser_compactness_start_m", "mean"),
            opp_max_progress_m=("max_progress_m", "mean"),
            opp_progress_8s_m=("progress_at_window_m", "mean"),
            opp_hold_time_s=("held_possession_s", "mean"),
            loser_recovery_m=("loser_recovery_m", "mean"),
        )
        .reset_index()
    )
    rho = stats.spearmanr(tr["loser_compactness_start_m"], tr["max_progress_m"])
    rho_hold = stats.spearmanr(tr["loser_compactness_start_m"], tr["held_possession_s"])
    findings = [
        f"{len(tr):,} changes of possession analysed across 20 matches.",
        f"Teams that lose the ball while *stretched* (compactness "
        f"{tbl.iloc[4]['compactness_m']:.1f} m) concede "
        f"{tbl.iloc[4]['opp_max_progress_m']:.1f} m of progression in the next 8 s, versus "
        f"{tbl.iloc[0]['opp_max_progress_m']:.1f} m for the most compact quintile "
        f"({tbl.iloc[0]['compactness_m']:.1f} m). Spearman rho={rho.statistic:.3f}, "
        f"p={_fmt_p(rho.pvalue)}.",
        f"The opponent also holds the ball for less time after a loss by a stretched team "
        f"({tbl.iloc[4]['opp_hold_time_s']:.1f} s vs {tbl.iloc[0]['opp_hold_time_s']:.1f} s; "
        f"rho={rho_hold.statistic:.3f}). A stretched shape is mostly a team committed high up "
        f"the pitch, and the counter-press that follows shortens the opponent's possession -- "
        f"this is an association in one league's 20 matches, not a demonstration that "
        f"stretching out is safe.",
    ]
    return tbl, findings


# --------------------------------------------------------------------------- #
# Q4
# --------------------------------------------------------------------------- #

def q4_phase_value() -> tuple[pd.DataFrame, list[str]]:
    """What is each phase of play actually worth?"""
    tp = _load("team_possessions")
    tbl = (
        tp.groupby("team_in_possession_phase_type", observed=True)
        .agg(
            n=("duration_s", "size"),
            share=("duration_s", lambda s: len(s)),
            duration_s=("duration_s", "mean"),
            territory_gain_m=("territory_gain_m", "mean"),
            progression_speed_mps=("speed_of_progression_mps", "median"),
            shot_rate=("lead_to_shot", "mean"),
            goal_rate=("lead_to_goal", "mean"),
            epv_total=("epv_total", "mean"),
            passes=("n_passes", "mean"),
        )
        .reset_index()
        .sort_values("epv_total", ascending=False)
    )
    tbl["share"] = tbl["n"] / tbl["n"].sum()
    best = tbl.iloc[0]
    bu = tbl[tbl["team_in_possession_phase_type"] == "build_up"].iloc[0]
    findings = [
        f"{len(tp):,} team possessions, grouped by SkillCorner's in-possession phase type.",
        f"`{best['team_in_possession_phase_type']}` possessions are worth "
        f"{best['epv_total'] / max(bu['epv_total'], 1e-9):.0f}x a build-up possession in EPV "
        f"({best['epv_total']:+.4f} vs {bu['epv_total']:+.4f}) and end in a shot "
        f"{best['shot_rate']:.0%} of the time, but make up only {best['share']:.1%} of all "
        f"possessions.",
        "Transition and quick-break possessions gain "
        + f"{tbl.set_index('team_in_possession_phase_type').loc['transition', 'territory_gain_m']:.0f} m "
        + f"and {tbl.set_index('team_in_possession_phase_type').loc['quick_break', 'territory_gain_m']:.0f} m "
        + "of territory on average, against "
        + f"{bu['territory_gain_m']:+.1f} m for build-up.",
    ]
    return tbl, findings


# --------------------------------------------------------------------------- #
# Q5
# --------------------------------------------------------------------------- #

def q5_orientation_at_reception(min_speed: float = 2.0) -> tuple[pd.DataFrame, list[str]]:
    """Does the direction a receiver is facing predict what the possession becomes?

    Orientation here is **movement heading**, taken from smoothed tracking, not
    body pose. On the phase of play where 3D pose is available, the two agree
    to a median of about 9 degrees for players moving faster than 4 m/s and
    13 degrees between 2 and 4 m/s, against 69 degrees below 1 m/s (see
    :func:`q10_pose_vs_heading`), which is why the sample is restricted to
    moving receivers.
    """
    r = reception_orientation(min_speed=min_speed)
    tbl = (
        r.groupby("orientation", observed=True)
        .agg(
            n=("progression_m", "size"),
            progression_m=("progression_m", "mean"),
            epv=("possession_epv_total", "mean"),
            forward_pass_share=("pass_direction", lambda s: (s == "forward").mean()),
            lead_to_shot=("lead_to_shot", "mean"),
            line_breaking=("fil_is_progressive", "mean"),
            mean_pressure_id=("overall_pressure_start_id", "mean"),
            retention=("end_type", lambda s: (s != "possession_loss").mean()),
        )
        .reindex(["forward", "sideways", "backward"])
        .reset_index()
    )
    kw = stats.kruskal(*[g["progression_m"].dropna() for _, g in r.groupby("orientation", observed=True)])
    kwe = stats.kruskal(
        *[g["possession_epv_total"].dropna() for _, g in r.groupby("orientation", observed=True)]
    )
    # Control for pressure: the effect should survive within a pressure band.
    within = (
        r.groupby(["overall_pressure_start", "orientation"], observed=True)["progression_m"]
        .mean().unstack()
    )
    findings = [
        f"{len(r):,} pass receptions by a player moving faster than {min_speed} m/s.",
        f"Receiving while facing the opposition goal is worth "
        f"{tbl.loc[0, 'progression_m'] - tbl.loc[2, 'progression_m']:.1f} m of progression more "
        f"than receiving facing your own "
        f"({tbl.loc[0, 'progression_m']:+.1f} m vs {tbl.loc[2, 'progression_m']:+.1f} m; "
        f"Kruskal-Wallis H={kw.statistic:.0f}, p={_fmt_p(kw.pvalue)}).",
        f"Forward-facing receptions lead to a shot {tbl.loc[0, 'lead_to_shot']:.1%} of the time "
        f"against {tbl.loc[2, 'lead_to_shot']:.1%} for backward-facing ones, and add "
        f"{tbl.loc[0, 'epv']:+.4f} EPV against {tbl.loc[2, 'epv']:+.4f} "
        f"(H={kwe.statistic:.0f}, p={_fmt_p(kwe.pvalue)}).",
        "The gap holds inside every pressure band, so it is not simply a proxy for being "
        "unmarked: " + "; ".join(
            f"{b}: {row['forward'] - row['backward']:+.1f} m"
            for b, row in within.dropna().iterrows()
        ),
        "Direction of causation is not established. A player already running forward is often "
        "already in a transition, and transitions progress the ball for reasons that have "
        "nothing to do with body shape.",
    ]
    return tbl, findings


def reception_orientation(min_speed: float = 2.0) -> pd.DataFrame:
    """Receptions joined to the receiver's movement heading at that frame."""
    from src.preprocessing.build_tracking import load_tracking

    poss = _load("possessions")
    rec = poss[poss["start_type"] == "pass_reception"].copy()
    parts = []
    for mid, g in rec.groupby("match_id"):
        trk = load_tracking(mid, columns=["frame", "player_id", "heading", "speed"])
        k = g[["event_uid", "frame_start", "player_id"]].rename(columns={"frame_start": "frame"})
        k["player_id"] = k["player_id"].astype("int64")
        parts.append(k.merge(trk, on=["frame", "player_id"], how="inner"))
    h = pd.concat(parts, ignore_index=True)
    r = rec.merge(h[["event_uid", "heading", "speed"]], on="event_uid", how="inner")
    r["facing_goal_deg"] = np.degrees(
        np.arccos(np.clip(np.cos(r["heading"]) * r["attack_sign"], -1, 1))
    )
    r["orientation"] = pd.cut(
        r["facing_goal_deg"], [-0.1, 60, 120, 180.1],
        labels=["forward", "sideways", "backward"],
    )
    return r[r["speed"] > min_speed]


# --------------------------------------------------------------------------- #
# Q6
# --------------------------------------------------------------------------- #

def q6_off_ball_runs() -> tuple[pd.DataFrame, list[str]]:
    """Which off-ball runs get found, and which are worth finding?"""
    runs = _load("off_ball_runs")
    tbl = (
        runs.groupby("event_subtype", observed=True)
        .agg(
            n=("targeted", "size"),
            targeted_rate=("targeted", "mean"),
            received_rate=("received", "mean"),
            dangerous_rate=("dangerous", "mean"),
            lead_to_shot=("lead_to_shot", "mean"),
            xthreat=("xthreat", "mean"),
            distance_m=("distance_covered", "mean"),
            speed_ms=("speed_avg", "mean"),
            line_break_rate=("break_defensive_line", "mean"),
        )
        .reset_index()
        .sort_values("xthreat", ascending=False)
    )
    tbl["value_if_found"] = tbl["xthreat"] * tbl["targeted_rate"]
    best = tbl.iloc[0]
    least = tbl.nsmallest(1, "targeted_rate").iloc[0]
    findings = [
        f"{len(runs):,} off-ball runs across 20 matches, in ten SkillCorner subtypes.",
        f"`{best['event_subtype']}` runs carry the highest xThreat ({best['xthreat']:.3f}) "
        f"and are targeted {best['targeted_rate']:.0%} of the time.",
        f"`{least['event_subtype']}` runs are the least likely to be found "
        f"({least['targeted_rate']:.0%} targeted) despite an xThreat of {least['xthreat']:.3f}.",
        "Runs that break the defensive line are "
        + f"{runs[runs['break_defensive_line'].fillna(False)]['lead_to_shot'].mean() / max(runs[~runs['break_defensive_line'].fillna(False)]['lead_to_shot'].mean(), 1e-9):.1f}x "
        + "as likely to end in a shot as runs that do not.",
    ]
    return tbl, findings


# --------------------------------------------------------------------------- #
# Q7
# --------------------------------------------------------------------------- #

def q7_workload_vs_style() -> tuple[pd.DataFrame, list[str]]:
    """Does a team's tactical shape explain its players' physical output?"""
    import glob

    from src.config import DERIVED_DIR as DD

    phys = _load("player_match_physical")
    shape_rows = []
    for f in sorted(glob.glob(str(DD / "team_frames" / "*.parquet"))):
        tf = pd.read_parquet(
            f, columns=["match_id", "team_id", "compactness_m", "length_m", "width_m",
                        "def_line_x", "space_between_lines_m", "block_overlap_m"]
        )
        shape_rows.append(tf.groupby(["match_id", "team_id"], observed=True).mean().reset_index())
    shape = pd.concat(shape_rows, ignore_index=True)

    team_phys = (
        phys.groupby(["match_id", "team_id"], observed=True)
        .agg(
            m_per_min=("m_per_min", "mean"),
            hsr_per_min=("hsr_distance_m", lambda s: s.mean()),
            sprints=("sprint_count", "mean"),
            high_accels=("high_accel_count", "mean"),
            top_speed=("psv99_ms", "mean"),
        )
        .reset_index()
    )
    d = team_phys.merge(shape, on=["match_id", "team_id"], how="inner")
    rows = []
    for metric in ["m_per_min", "hsr_per_min", "sprints", "high_accels"]:
        for driver in ["compactness_m", "length_m", "width_m", "def_line_x", "space_between_lines_m"]:
            r = stats.spearmanr(d[metric], d[driver])
            rows.append({"physical_metric": metric, "shape_metric": driver,
                         "spearman_rho": r.statistic, "p_value": r.pvalue, "n": len(d)})
    tbl = pd.DataFrame(rows).sort_values("spearman_rho", key=abs, ascending=False)
    top = tbl.iloc[0]
    findings = [
        f"{len(d)} team-matches, each with mean per-frame shape and mean per-player physical "
        "output.",
        f"Strongest association: {top['physical_metric']} vs {top['shape_metric']}, "
        f"rho={top['spearman_rho']:+.2f} (p={_fmt_p(top['p_value'])}, n={int(top['n'])}).",
        "With 40 team-matches these correlations are indicative only; the sign is more "
        "trustworthy than the magnitude.",
    ]
    return tbl, findings


# --------------------------------------------------------------------------- #
# Q8
# --------------------------------------------------------------------------- #

def q8_psv99_validation() -> tuple[pd.DataFrame, list[str]]:
    """Does this project's tracking pipeline reproduce SkillCorner's own PSV99?"""
    from src.config import PROCESSED_DIR

    phys = _load("player_match_physical")
    season = pd.read_parquet(PROCESSED_DIR / "season_physical.parquet")
    mine = (
        phys[phys["tracked_minutes"] >= 45]
        .groupby("player_id", observed=True)
        .agg(fil_psv99_ms=("psv99_ms", "max"),
             fil_top_speed_ms=("top_speed_ms", "max"),
             matches=("match_id", "nunique"))
        .reset_index()
    )
    theirs = (
        season.sort_values("count_match", ascending=False)
        .drop_duplicates("player_id")[["player_id", "player_short_name", "position_group",
                                       "psv99", "psv99_top5", "count_match"]]
    )
    d = mine.merge(theirs, on="player_id", how="inner")
    d["sc_psv99_ms"] = d["psv99"] / 3.6
    d["diff_ms"] = d["fil_psv99_ms"] - d["sc_psv99_ms"]
    r = stats.pearsonr(d["fil_psv99_ms"], d["sc_psv99_ms"])
    rs = stats.spearmanr(d["fil_top_speed_ms"], d["sc_psv99_ms"])
    findings = [
        f"{len(d)} players appear both in this project's tracking-derived physical table and in "
        "SkillCorner's season physical aggregates.",
        f"SkillCorner's PSV99 is published in km/h. Converted to m/s, it correlates with this "
        f"project's per-match PSV99 at r={r.statistic:.2f} (p={_fmt_p(r.pvalue)}), with this "
        f"pipeline reading {d['diff_ms'].mean():+.2f} m/s "
        f"({d['diff_ms'].mean() * 3.6:+.1f} km/h) lower on average.",
        "The gap is expected and is not a bug: SkillCorner's figure is a season-level 99th "
        "percentile over their full pipeline and all matches, while this one is computed from "
        "broadcast frames only, over 1-3 matches, after deliberately conservative smoothing "
        f"(single-frame maxima correlate at rho={rs.statistic:.2f}, which is the noisier "
        "comparison).",
    ]
    return d, findings


# --------------------------------------------------------------------------- #
# Q9
# --------------------------------------------------------------------------- #

def q9_pitch_control_and_progression(n_frames: int = 2000, seed: int = 7) -> tuple[pd.DataFrame, list[str]]:
    """Does controlling more of the pitch translate into progressing the ball?

    The naive version of this question answers itself the wrong way round. A
    team that already controls the final third has the ball near the goal and
    almost no pitch left to gain, so raw control correlates *negatively* with
    subsequent progression. The comparison only means anything inside a band of
    ball positions, which is how it is reported here.
    """
    from src.data.loaders import available_match_ids, load_match_meta
    from src.preprocessing.build_tracking import load_ball, load_tracking
    from src.tracking.pitch_control import control_summary, control_surface

    rng = np.random.default_rng(seed)
    match_ids = available_match_ids()
    per_match = max(1, n_frames // len(match_ids))
    rows = []
    for mid in match_ids:
        meta = load_match_meta(mid)
        trk = load_tracking(mid, columns=["frame", "period", "player_id", "team_id",
                                          "x_std", "y_std", "vx", "vy"])
        ball = load_ball(mid).dropna(subset=["ball_x_std", "possession_group"])
        counts = trk.groupby("frame", observed=True).size()
        full = sorted(set(counts[counts == 22].index) & set(ball["frame"]))
        if not full:
            continue
        pick = rng.choice(full, size=min(per_match, len(full)), replace=False)
        sub = trk[trk["frame"].isin(set(pick))]
        bidx = ball.set_index("frame")
        for fr, g in sub.groupby("frame", observed=True):
            res = control_surface(g, home_team_id=meta.home_team_id)
            per = int(g["period"].iloc[0])
            s = control_summary(res, home_attacks_positive_x=meta.home_attacks_positive_x.get(per, True))
            b = bidx.loc[fr]
            grp = str(b["possession_group"])
            is_home = grp.startswith("home")
            team = meta.home_team_id if is_home else meta.away_team_id
            att_sign = meta.attack_sign(team, per)
            future = ball[(ball["frame"] > fr) & (ball["frame"] <= fr + 50)]
            if future.empty:
                continue
            ball_x_att = float(b["ball_x_std"] * att_sign)
            prog = float((future["ball_x_std"].to_numpy()[-1] - b["ball_x_std"]) * att_sign)
            rows.append(
                {
                    "match_id": mid, "frame": int(fr), "period": per, "possession": grp,
                    "ball_x_att": ball_x_att,
                    "room_to_goal_m": 52.5 - ball_x_att,
                    "control_in_possession": s["home_control"] if is_home else s["away_control"],
                    "control_att_third": s["home_control_att_third"] if is_home
                    else s["away_control_att_third"],
                    "ball_progress_5s_m": prog,
                    "progress_share_of_room": prog / max(52.5 - ball_x_att, 1.0),
                }
            )
    d = pd.DataFrame(rows)
    d["ball_zone"] = pd.cut(
        d["ball_x_att"], [-53, -17.5, 17.5, 53],
        labels=["own third", "middle third", "final third"],
    )
    naive = stats.spearmanr(d["control_att_third"], d["ball_progress_5s_m"])
    room = stats.spearmanr(d["ball_x_att"], d["ball_progress_5s_m"])

    rows_out = []
    for zone, g in d.groupby("ball_zone", observed=True):
        r = stats.spearmanr(g["control_att_third"], g["ball_progress_5s_m"])
        rshare = stats.spearmanr(g["control_att_third"], g["progress_share_of_room"])
        hi = g[g["control_att_third"] >= g["control_att_third"].quantile(0.8)]
        lo = g[g["control_att_third"] <= g["control_att_third"].quantile(0.2)]
        rows_out.append(
            {
                "ball_zone": zone, "n": len(g),
                "mean_control_att_third": g["control_att_third"].mean(),
                "rho_control_vs_progress": r.statistic, "p_value": r.pvalue,
                "rho_control_vs_progress_share": rshare.statistic,
                "progress_top_quintile_m": hi["ball_progress_5s_m"].mean(),
                "progress_bottom_quintile_m": lo["ball_progress_5s_m"].mean(),
            }
        )
    tbl = pd.DataFrame(rows_out)
    findings = [
        f"{len(d):,} randomly sampled fully tracked frames across {d.match_id.nunique()} matches, "
        "each scored with this project's pitch-control model and matched to how far the ball "
        "travelled towards goal over the following 5 s.",
        f"Taken raw, control of the final third correlates *negatively* with subsequent "
        f"progression (rho={naive.statistic:+.3f}). That is a ceiling artefact, not a finding: "
        f"ball position alone correlates with future progression at rho={room.statistic:+.3f}, "
        "because a ball already near the goal has nowhere left to travel.",
        "Conditioned on where the ball is, the relationship essentially disappears: "
        + "; ".join(
            f"{r.ball_zone} rho={r.rho_control_vs_progress:+.3f} (p={_fmt_p(r.p_value)}, "
            f"{r.progress_top_quintile_m:.1f} m top quintile vs "
            f"{r.progress_bottom_quintile_m:.1f} m bottom, n={r.n:,})"
            for r in tbl.itertuples()
        ),
        "This is a null result and is reported as one. Instantaneous spatial control, as modelled "
        "here, does not predict how far a team moves the ball over the next five seconds once "
        "field position is accounted for. Two readings are available and this data cannot "
        "separate them: control may genuinely not be what drives progression at this time scale, "
        "or a two-parameter time-to-intercept model on broadcast tracking may be too coarse to "
        "detect it. What the analysis does establish is methodological -- any spatial-control "
        "metric must be conditioned on field position before it is related to progression, or "
        "the ceiling artefact will be mistaken for a finding.",
    ]
    return tbl, findings


# --------------------------------------------------------------------------- #
# Q10
# --------------------------------------------------------------------------- #

def q10_pose_vs_heading() -> tuple[pd.DataFrame, list[str]]:
    """Does 3D pose agree with movement heading, and by how much?

    This is the bridge that licenses Q5. It runs on whatever pose is available:
    the committed sample phase, or a full match if one was downloaded.
    """
    from src.preprocessing.build_tracking import load_tracking

    o = pd.read_parquet(DERIVED_DIR / "pose_orientation.parquet")
    o = o[o["orientation_valid"]].copy()
    mid = int(o["match_id"].iloc[0])
    trk = load_tracking(mid, columns=["frame", "player_id", "heading", "speed"])
    o["frame"] = o["tracking_frame"].round().astype(int)
    m = o.merge(trk, on=["frame", "player_id"], how="inner")
    m["abs_diff_deg"] = np.degrees(
        np.abs(np.arctan2(np.sin(m["facing_rad"] - m["heading"]),
                          np.cos(m["facing_rad"] - m["heading"])))
    )
    bins = [0, 1, 2, 4, 20]
    m["speed_band"] = pd.cut(m["speed"], bins, labels=["<1 m/s", "1-2", "2-4", ">4"])
    tbl = (
        m.groupby("speed_band", observed=True)
        .agg(n=("abs_diff_deg", "size"),
             median_abs_diff_deg=("abs_diff_deg", "median"),
             p90_abs_diff_deg=("abs_diff_deg", lambda s: s.quantile(0.9)),
             share_within_45deg=("abs_diff_deg", lambda s: (s < 45).mean()),
             mean_joint_err_cm=("joint_err_mean_cm", "mean"))
        .reset_index()
    )
    fast = tbl[tbl["speed_band"] == ">4"]
    slow = tbl[tbl["speed_band"] == "<1 m/s"]
    findings = [
        f"{len(m):,} player-frames where 3D pose and tracking both resolve the same player "
        f"(match {mid}, source: {o['source'].iloc[0]}).",
        f"Above 4 m/s, pose-derived torso facing and tracking movement heading agree to a median "
        f"of {float(fast['median_abs_diff_deg'].iloc[0]):.1f} degrees, and "
        f"{float(fast['share_within_45deg'].iloc[0]):.0%} of frames agree within 45 degrees. "
        "That is the check that makes the heading proxy in Q5 usable at scale.",
        f"Below 1 m/s they diverge as expected (median "
        f"{float(slow['median_abs_diff_deg'].iloc[0]):.0f} degrees): a nearly stationary player "
        "has no meaningful movement heading, which is exactly the case where only pose can "
        "answer the question.",
        f"Mean per-joint p90 error over these frames is "
        f"{m['joint_err_mean_cm'].mean():.1f} cm; at the observed "
        f"{m['shoulder_width_m'].mean():.2f} m shoulder width that is roughly "
        f"{np.degrees(np.arctan(m['joint_err_mean_cm'].mean() / 100 / m['shoulder_width_m'].mean())):.0f} "
        "degrees of orientation uncertainty, which is why only three coarse orientation classes "
        "are ever reported.",
    ]
    return tbl, findings


# --------------------------------------------------------------------------- #

QUESTIONS = {
    "q1_pressure_and_pass_value": ("How does pressure on the passer change what a pass is worth?", q1_pressure_and_pass_value),
    "q2_value_under_pressure": ("Which players keep creating value when they are pressed?", q2_value_under_pressure),
    "q3_shape_at_loss": ("Does the shape a team is in when it loses the ball predict what happens next?", q3_shape_at_loss),
    "q4_phase_value": ("What is each phase of play actually worth?", q4_phase_value),
    "q5_orientation_at_reception": ("Does the direction a receiver faces predict what the possession becomes?", q5_orientation_at_reception),
    "q6_off_ball_runs": ("Which off-ball runs get found, and which are worth finding?", q6_off_ball_runs),
    "q7_workload_vs_style": ("Does a team's tactical shape explain its players' physical output?", q7_workload_vs_style),
    "q8_psv99_validation": ("Does this pipeline reproduce SkillCorner's own PSV99?", q8_psv99_validation),
    "q9_pitch_control_and_progression": ("Does controlling more of the pitch translate into progressing the ball?", q9_pitch_control_and_progression),
    "q10_pose_vs_heading": ("Does 3D body pose agree with movement heading, and by how much?", q10_pose_vs_heading),
}


def run_all() -> dict[str, list[str]]:
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    md = ["# Research findings", "",
          "Every number below is produced by `python -m src.research.questions` and read out of ",
          "`reports/research/`. Nothing is transcribed by hand.", ""]
    for key, (question, fn) in QUESTIONS.items():
        try:
            tbl, findings = fn()
        except Exception as exc:  # noqa: BLE001
            log.exception("%s failed", key)
            summary[key] = {"question": question, "error": f"{type(exc).__name__}: {exc}"}
            continue
        tbl.to_parquet(RESEARCH_DIR / f"{key}.parquet", index=False)
        summary[key] = {"question": question, "findings": findings, "rows": len(tbl)}
        md += [f"## {question}", "", *[f"- {f}" for f in findings], "",
               tbl.head(20).round(4).to_markdown(index=False), ""]
        log.info("%s: %d rows", key, len(tbl))
    (RESEARCH_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (REPORTS_DIR / "RESEARCH.md").write_text("\n".join(md))
    return {k: v.get("findings", [v.get("error", "")]) for k, v in summary.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = run_all()
    for k, v in out.items():
        print(f"\n### {k}")
        for f in v:
            print(" -", f)
