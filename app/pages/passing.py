"""Passing: networks, maps, progression and value."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import caveat, empty, metric_row, note, page_header, plot, section
from src.config import SERIES
from src.features.networks import build_network, first_substitution_frame
from src.visualization.charts import bar_with_error, ranked_bar, scatter
from src.visualization.pitch_plots import arrows, heatmap, passing_network

page_header(
    "Passing",
    "Passing networks placed at players' real average positions rather than at their "
    "touches, plus pass maps filtered by everything the dynamic events know about a pass.",
    level="Levels 1-4",
)

passes = D.table("passes")
labels = D.match_label_map()

tabs = st.tabs(["Network", "Pass maps", "Progression & value", "League passers"])

# --------------------------------------------------------------------------- #
with tabs[0]:
    c1, c2, c3 = st.columns([3, 2, 2])
    label = c1.selectbox("Match", list(labels), key="pass_match")
    match_id = labels[label]
    meta = D.meta(match_id)
    home_name, away_name = meta.home_team_name, meta.away_team_name
    team_pick = c2.selectbox("Team", [home_name, away_name], key="pass_team")
    team_id = meta.home_team_id if team_pick == home_name else meta.away_team_id
    colour = SERIES[0] if team_pick == home_name else SERIES[1]
    window = c3.selectbox("Window", ["Until first substitution", "Full match", "First half",
                                     "Second half"], key="pass_window")

    squads = D.squads()
    p1_start = meta.period_start_frames.get(1, 0)
    p2_start = meta.period_start_frames.get(2, 0)
    p2_end = meta.period_end_frames.get(2, 10**9)
    sub = first_substitution_frame(squads, match_id, team_id)
    frange = {
        "Until first substitution": (p1_start, sub or p2_end),
        "Full match": (p1_start, p2_end),
        "First half": (p1_start, meta.period_end_frames.get(1, p2_start)),
        "Second half": (p2_start, p2_end),
    }[window]

    mp = passes[passes["match_id"] == match_id]
    trk = D.match_tracking(match_id, columns=("frame", "player_id", "team_id", "x_std",
                                              "y_std", "attack_sign", "position"))
    nodes, edges = build_network(mp, trk, team_id=team_id, frame_range=frange)

    if nodes.empty:
        empty("No completed passes in this window.")
    else:
        cc = st.columns([3, 2])
        with cc[0]:
            plot(
                passing_network(nodes, edges, colour=colour, height=560,
                                title=f"{team_pick} — {window.lower()}"),
                key="pn_net",
            )
            st.caption(
                "Nodes sit at each player's mean **tracked** position over the window, not at "
                "the average position of their passes. Node size is total involvement "
                "(passes made plus received); edge width is pass volume between the pair."
            )
        with cc[1]:
            metric_row(
                [("Completed passes", f"{int(edges['weight'].sum()):,}", f"in {window.lower()}"),
                 ("Players linked", f"{len(nodes)}", f"{len(edges)} pairs with ≥2 passes")],
                columns=2,
            )
            top = edges.head(12)[["label", "weight", "epv", "progression"]].copy()
            top.columns = ["pair", "passes", "EPV", "mean prog (m)"]
            st.dataframe(top.round(4), hide_index=True, width="stretch", height=430)

# --------------------------------------------------------------------------- #
with tabs[1]:
    c1, c2, c3, c4 = st.columns(4)
    label2 = c1.selectbox("Match", ["All matches"] + list(labels), key="pm_match")
    pm = passes if label2 == "All matches" else passes[passes["match_id"] == labels[label2]]
    teams = sorted(pm["team_shortname"].dropna().unique())
    team2 = c2.selectbox("Team", ["All"] + teams, key="pm_team")
    if team2 != "All":
        pm = pm[pm["team_shortname"] == team2]
    players = ["All"] + sorted(pm["player_name"].dropna().unique().tolist())
    player = c3.selectbox("Player", players, key="pm_player")
    if player != "All":
        pm = pm[pm["player_name"] == player]
    filt = c4.selectbox("Show", ["All passes", "Completed only", "Progressive", "Line-breaking",
                                 "Under high pressure", "Into the box", "Led to a shot"],
                        key="pm_show")
    if filt == "Completed only":
        pm = pm[pm["completed"]]
    elif filt == "Progressive":
        pm = pm[pm["fil_progressive_pass"].fillna(False)]
    elif filt == "Line-breaking":
        pm = pm[pm["fil_line_breaking"].fillna(False)]
    elif filt == "Under high pressure":
        pm = pm[pm["overall_pressure_start_id"] >= 4]
    elif filt == "Into the box":
        pm = pm[pm["penalty_area_end"].fillna(False)]
    elif filt == "Led to a shot":
        pm = pm[pm["lead_to_shot"].fillna(False)]

    if pm.empty:
        empty("No passes match those filters.")
    else:
        metric_row(
            [("Passes", f"{len(pm):,}", filt.lower()),
             ("Completion", f"{pm['completed'].mean():.1%}", ""),
             ("Mean progression", f"{pm['progression_m'].mean():+.1f} m", ""),
             ("Mean EPV", f"{pm['pass_epv_total'].mean():+.4f}", "SkillCorner pass EPV"),
             ("Line-breaking", f"{pm['fil_line_breaking'].fillna(False).mean():.1%}", "")]
        )
        cc = st.columns([3, 2])
        with cc[0]:
            sample = pm.sample(min(len(pm), 700), random_state=1)
            plot(
                arrows(sample, colour_col="pass_outcome",
                       colour_map={"successful": SERIES[2], "unsuccessful": SERIES[1],
                                   "offside": SERIES[3]},
                       hover=["player_name", "player_targeted_name", "progression_m"],
                       title=f"{len(sample):,} of {len(pm):,} passes", height=520,
                       opacity=0.55, base_width=1.0),
                key="pm_arrows",
            )
        with cc[1]:
            plot(heatmap(pm["x_end_att"], pm["y_end_att"], title="Where these passes arrive",
                         height=520), key="pm_heat")

# --------------------------------------------------------------------------- #
with tabs[2]:
    min_p = st.slider("Minimum passes attempted", 20, 400, 120, step=20, key="pv_min")
    g = (
        passes.groupby(["player_id", "player_name", "team_shortname", "player_position"],
                       observed=True)
        .agg(attempted=("completed", "size"), completion=("completed", "mean"),
             progression=("progression_m", "mean"), epv_total=("pass_epv_total", "sum"),
             epv_per_pass=("pass_epv_total", "mean"),
             line_break_rate=("fil_line_breaking", "mean"),
             forward_share=("pass_direction", lambda s: (s == "forward").mean()),
             mean_pressure=("overall_pressure_start_id", "mean"),
             xpass=("player_targeted_xpass_completion", "mean"))
        .reset_index()
    )
    g = g[g["attempted"] >= min_p]
    g["completion_over_expected"] = g["completion"] - g["xpass"]
    if g.empty:
        empty("Lower the threshold.")
    else:
        cc = st.columns(2)
        with cc[0]:
            plot(ranked_bar(g, label="player_name", value="epv_total", value_fmt="+.3f",
                            title="Total EPV added by passing", top=15), key="pv_r1")
        with cc[1]:
            plot(ranked_bar(g, label="player_name", value="line_break_rate", value_fmt=".1%",
                            title="Share of passes that break a defensive line", top=15,
                            colour=SERIES[2]), key="pv_r2")
        g = g.dropna(subset=["xpass"])
        plot(
            scatter(g, x="xpass", y="completion", label="player_name", size_col="attempted",
                    x_title="expected completion (SkillCorner xPass)",
                    y_title="actual completion",
                    title="Completion against difficulty — above the diagonal is over-performance",
                    annotate_top=0, highlight=list(g.nlargest(4, "completion_over_expected")["player_name"])
                    + list(g.nsmallest(3, "completion_over_expected")["player_name"]),
                    height=480, quadrants=False),
            key="pv_scatter",
        )
        st.caption(
            "SkillCorner supplies an expected-completion value per pass. Plotting realised "
            "completion against it separates safe passers from difficult passers who complete "
            "anyway; the median player sits on the diagonal by construction."
        )
        st.dataframe(g.sort_values("epv_total", ascending=False).round(4), hide_index=True,
                     width="stretch", height=340)

# --------------------------------------------------------------------------- #
with tabs[3]:
    season = D.table("season_passing")
    prof = season[season["performance_included_count"] >= 4].copy()
    metric_row(
        [("Players", f"{len(prof)}", "60+ minute performances, ≥4 matches"),
         ("Source", "season aggregates", "whole 2024/25 league, not just these 20 matches")]
        , columns=2,
    )
    note(
        "These are SkillCorner's own season aggregates. They are **per-match averages**, "
        "filtered to performances of 60+ minutes, and `_p30tip` columns are rates per 30 "
        "minutes of team possession — not raw counts. They are not the sum of the match data "
        "on the other tabs."
    )
    metric = st.selectbox(
        "Metric",
        ["pass_count_linebreak_completed_p30tip", "pass_count_attempted_p30tip",
         "pass_pct_completed", "pass_count_dangerous_completed_p30tip",
         "pass_count_torun_completed_p30tip", "pass_avgxpass_attempted",
         "pass_count_shotwithin10s_p30tip"],
    )
    d = prof[["player_short_name", "team_name", "position_group", metric]].dropna()
    cc = st.columns([2, 3])
    with cc[0]:
        plot(ranked_bar(d, label="player_short_name", value=metric, value_fmt=".2f",
                        title=f"Top 15 — {metric}", top=15), key="ps_rank")
    with cc[1]:
        plot(
            scatter(prof.dropna(subset=["pass_count_attempted_p30tip", "pass_pct_completed"]),
                    x="pass_count_attempted_p30tip", y="pass_pct_completed",
                    label="player_short_name", colour_col="position_group",
                    x_title="passes attempted per 30 min in possession",
                    y_title="completion %", height=470,
                    title="Volume against completion, by position group"),
            key="ps_scatter",
        )
