"""Physical output, from tracking and from SkillCorner's season aggregates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import caveat, empty, metric_row, note, page_header, plot, section
from src.config import SERIES, SPEED_ZONES
from src.visualization.charts import (bar_with_error, distribution, ranked_bar, scatter,
                                      stacked_bars, timeline)

page_header(
    "Physical",
    "Running output computed from the broadcast tracking in this repository, and "
    "SkillCorner's own season physical aggregates. The two answer different questions and "
    "are kept apart on purpose.",
    level="Level 6",
)

phys = D.table("player_match_physical")
sprints = D.table("sprints")
season = D.table("season_physical")

note(
    "**Tracking-derived numbers are distances while visible.** Broadcast tracking sees about "
    "73% of a match; the rest is replays, crowd shots and stoppages. Every row here records "
    "`tracked_minutes` and `coverage` beside the totals, and none of them should be read as a "
    "match total. SkillCorner's season aggregates, on the last tab, are their own full-pipeline "
    "figures filtered to 60+ minute performances — they are per-match averages, not season sums."
)

tabs = st.tabs(["Match workload", "Intensity profile", "Sprints", "Season aggregates",
                "Pipeline cross-check"])

# --------------------------------------------------------------------------- #
with tabs[0]:
    labels = D.match_label_map()
    c1, c2 = st.columns([3, 2])
    label = c1.selectbox("Match", ["All matches"] + list(labels))
    p = phys if label == "All matches" else phys[phys["match_id"] == labels[label]]
    pos = c2.multiselect("Position group", sorted(phys["position_group"].dropna().unique()))
    if pos:
        p = p[p["position_group"].isin(pos)]
    p = p[p["tracked_minutes"] >= 20]

    if p.empty:
        empty("No players with 20+ tracked minutes under those filters.")
    else:
        metric_row(
            [("Player-matches", f"{len(p):,}", "20+ tracked minutes"),
             ("Mean distance", f"{p['total_distance_m'].mean():,.0f} m", "while tracked"),
             ("Mean coverage", f"{p['coverage'].mean():.0%}", "tracked ÷ played minutes"),
             ("Mean m/min", f"{p['m_per_min'].mean():.0f}", "while tracked"),
             ("Mean PSV99", f"{p['psv99_ms'].mean() * 3.6:.1f} km/h",
              "99th percentile speed, robust to a single bad frame")]
        )
        cc = st.columns(2)
        with cc[0]:
            plot(ranked_bar(p, label="player_name", value="total_distance_m", value_fmt=",.0f",
                            unit=" m", title="Distance covered while tracked", top=15),
                 key="ph_d")
        with cc[1]:
            plot(ranked_bar(p, label="player_name", value="hsr_distance_m", value_fmt=",.0f",
                            unit=" m", colour=SERIES[2],
                            title="High-speed running (≥5.5 m/s)", top=15), key="ph_h")

        section("Distance by intensity zone")
        zone_cols = [f"distance_{z[0]}_m" for z in SPEED_ZONES]
        zone_names = ["walk <1.5", "jog 1.5–3.5", "run 3.5–5.5", "HSR 5.5–7", "sprint ≥7"]
        top = p.nlargest(15, "total_distance_m")[["player_name"] + zone_cols]
        plot(stacked_bars(top, index="player_name", columns=zone_cols, names=zone_names,
                          unit=" m", title="Where the distance comes from (m/s bands)"),
             key="ph_z")

        section("Position groups")
        cc = st.columns(2)
        with cc[0]:
            plot(distribution(p, value="m_per_min", group="position_group",
                              title="Metres per tracked minute", y_title="m/min", height=380),
                 key="ph_pg1")
        with cc[1]:
            plot(distribution(p, value="psv99_ms", group="position_group",
                              title="PSV99 by position group", y_title="m/s", height=380),
                 key="ph_pg2")

        st.dataframe(
            p[["player_name", "team_id", "position", "tracked_minutes", "coverage",
               "total_distance_m", "m_per_min", "hsr_distance_m", "sprint_distance_m",
               "sprint_count", "high_accel_count", "high_decel_count", "psv99_ms",
               "top_speed_ms", "detected_frac"]]
            .sort_values("total_distance_m", ascending=False).round(2),
            hide_index=True, width="stretch", height=340,
        )

# --------------------------------------------------------------------------- #
with tabs[1]:
    p = phys[phys["tracked_minutes"] >= 45].copy()
    p["hsr_share"] = p["hsr_distance_m"] / p["total_distance_m"]
    p["tip_share"] = p["distance_tip_m"] / (p["distance_tip_m"] + p["distance_otip_m"])
    cc = st.columns(2)
    with cc[0]:
        plot(
            scatter(p, x="m_per_min", y="hsr_share", label="player_name",
                    colour_col="position_group", x_title="metres per tracked minute",
                    y_title="share of distance at ≥5.5 m/s",
                    title="Volume against intensity", height=470),
            key="pi_1",
        )
    with cc[1]:
        plot(
            scatter(p, x="distance_tip_m", y="distance_otip_m", label="player_name",
                    colour_col="position_group",
                    x_title="distance while the team has the ball (m)",
                    y_title="distance while the team does not (m)",
                    title="In possession against out of possession", height=470),
            key="pi_2",
        )
    note(
        "The right-hand chart is only possible because the tracking file carries a per-frame "
        "possession flag. Most players sit above the diagonal: A-League footballers in this "
        "sample run further without the ball than with it."
    )
    section("Accelerations and decelerations")
    cc = st.columns(2)
    with cc[0]:
        plot(ranked_bar(p, label="player_name", value="high_accel_count", value_fmt=",.0f",
                        title="High accelerations (≥3 m/s², ≥0.3 s)", top=14), key="pi_a")
    with cc[1]:
        plot(ranked_bar(p, label="player_name", value="high_decel_count", value_fmt=",.0f",
                        colour=SERIES[1], title="High decelerations (≤−3 m/s²)", top=14),
             key="pi_d")
    caveat(
        "Acceleration counts are the metric most sensitive to smoothing. These come from the "
        "analytic second derivative of a 1.3 s Savitzky-Golay fit, counted only where the "
        "threshold is held for at least 0.3 s. A shorter window would produce far more of them "
        "and most of the extras would be noise."
    )

# --------------------------------------------------------------------------- #
with tabs[2]:
    metric_row(
        [("Sprints detected", f"{len(sprints):,}", "≥7 m/s held for ≥0.7 s"),
         ("Mean duration", f"{sprints['duration_s'].mean():.1f} s", ""),
         ("Mean peak", f"{sprints['peak_speed'].mean():.1f} m/s",
          f"{sprints['peak_speed'].mean() * 3.6:.1f} km/h"),
         ("Longest", f"{sprints['duration_s'].max():.1f} s", ""),
         ("Fastest", f"{sprints['peak_speed'].max():.1f} m/s",
          f"{sprints['peak_speed'].max() * 3.6:.1f} km/h")]
    )
    s = sprints.merge(
        phys[["match_id", "player_id", "player_name", "position_group", "team_id"]],
        on=["match_id", "player_id"], how="left",
    )
    cc = st.columns(2)
    with cc[0]:
        plot(
            scatter(s.sample(min(len(s), 2500), random_state=2), x="duration_s", y="peak_speed",
                    label="player_name", colour_col="position_group",
                    x_title="sprint duration (s)", y_title="peak speed (m/s)",
                    title="Every detected sprint", height=460, quadrants=False),
            key="sp_1",
        )
    with cc[1]:
        cnt = s.groupby("player_name", observed=True).agg(
            sprints=("duration_s", "size"), peak=("peak_speed", "max")).reset_index()
        plot(ranked_bar(cnt, label="player_name", value="sprints", value_fmt=",.0f",
                        title="Most sprints across the sample", top=15), key="sp_2")
    section("Sprint definition")
    st.markdown(
        "A sprint here is a run where the smoothed speed stays at or above **7.0 m/s** for at "
        "least **0.7 s** (7 consecutive frames at 10 Hz). Both numbers are in `src/config.py`. "
        "Requiring the threshold to be *held* is what stops a single mis-tracked frame becoming "
        "a sprint: without it the count roughly doubles and the extra entries last 0.1 s."
    )

# --------------------------------------------------------------------------- #
with tabs[3]:
    prof = season[season["count_match"] >= 4].copy()
    prof["psv99_kmh"] = prof["psv99"]
    metric = st.selectbox(
        "Metric",
        ["total_metersperminute_full_all", "hsr_distance_full_all", "sprint_distance_full_all",
         "sprint_count_full_all", "highaccel_count_full_all", "highdecel_count_full_all",
         "psv99", "psv99_top5", "explacceltosprint_count_full_all",
         "hsr_distance_full_otip", "hsr_distance_full_tip"],
    )
    d = prof[["player_short_name", "team_name", "position_group", "count_match", metric]].dropna()
    cc = st.columns([2, 3])
    with cc[0]:
        plot(ranked_bar(d, label="player_short_name", value=metric, value_fmt=".1f",
                        title=f"Top 15 — {metric}", top=15), key="sa_1")
    with cc[1]:
        plot(distribution(prof.dropna(subset=[metric]), value=metric, group="position_group",
                          title=f"{metric} by position group", height=470), key="sa_2")
    st.dataframe(d.sort_values(metric, ascending=False).round(2), hide_index=True,
                 width="stretch", height=320)
    st.caption(
        "`_full_tip` / `_full_otip` split each metric by whether the player's team had the "
        "ball. `psv99` is in km/h."
    )

# --------------------------------------------------------------------------- #
with tabs[4]:
    section("Does this pipeline reproduce SkillCorner's own numbers?")
    r = D.research("q8_psv99_validation")
    if r.empty:
        empty("Run `python -m src.research.questions` to generate this comparison.")
    else:
        rr = r.dropna(subset=["fil_psv99_ms", "sc_psv99_ms"])
        metric_row(
            [("Players matched", f"{len(rr)}", "in both tables"),
             ("Correlation", f"{np.corrcoef(rr['fil_psv99_ms'], rr['sc_psv99_ms'])[0, 1]:.2f}",
              "Pearson r on PSV99"),
             ("Mean offset", f"{rr['diff_ms'].mean() * 3.6:+.1f} km/h",
              "this pipeline reads lower")]
        )
        plot(
            scatter(rr, x="sc_psv99_ms", y="fil_psv99_ms", label="player_short_name",
                    colour_col="position_group",
                    x_title="SkillCorner season PSV99 (m/s)",
                    y_title="this pipeline's per-match PSV99 (m/s)",
                    title="Independent pipelines, same players", height=500, quadrants=False),
            key="xc_1",
        )
        note(
            "The offset is expected. SkillCorner's PSV99 is a season-level 99th percentile from "
            "their full pipeline over every match a player featured in; this one is computed "
            "from broadcast frames only, over one to three matches, after a deliberately "
            "conservative 1.3 s smoothing window. The point of the chart is the correlation — "
            "the two pipelines rank players similarly — not the intercept."
        )
