"""Off-ball movement, pressing and transitions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import caveat, empty, metric_row, note, page_header, plot, section
from src.config import SERIES
from src.visualization.charts import bar_with_error, distribution, ranked_bar, scatter
from src.visualization.pitch_plots import arrows, heatmap

page_header(
    "Off-Ball & Pressing",
    "What happens away from the ball: the runs players make, the pressure they apply, and "
    "the eight seconds after a turnover when both teams' shapes are furthest from where "
    "they want to be.",
    level="Level 5",
)

runs = D.table("off_ball_runs")
pressures = D.table("pressures")
transitions = D.table("transitions")
matches = D.matches()
team_names = pd.concat([
    matches[["home_team_id", "home_team"]].rename(columns={"home_team_id": "team_id", "home_team": "team"}),
    matches[["away_team_id", "away_team"]].rename(columns={"away_team_id": "team_id", "away_team": "team"}),
]).drop_duplicates("team_id")

tabs = st.tabs(["Off-ball runs", "Pressing", "Transitions"])

# --------------------------------------------------------------------------- #
with tabs[0]:
    metric_row(
        [("Runs", f"{len(runs):,}", "SkillCorner off-ball run events, 10 subtypes"),
         ("Targeted", f"{runs['targeted'].mean():.1%}", "the passer aimed at the runner"),
         ("Received", f"{runs['received'].mean():.1%}", "the runner got the ball"),
         ("Line-breaking", f"{runs['break_defensive_line'].fillna(False).mean():.1%}", ""),
         ("Mean distance", f"{runs['distance_covered'].mean():.1f} m",
          f"at {runs['speed_avg'].mean():.1f} m/s")]
    )

    by = (
        runs.groupby("event_subtype", observed=True)
        .agg(n=("targeted", "size"), targeted=("targeted", "mean"),
             received=("received", "mean"), xthreat=("xthreat", "mean"),
             lead_to_shot=("lead_to_shot", "mean"),
             distance=("distance_covered", "mean"), speed=("speed_avg", "mean"),
             line_break=("break_defensive_line", "mean"))
        .reset_index().sort_values("xthreat", ascending=False)
    )
    cc = st.columns(2)
    with cc[0]:
        plot(bar_with_error(by, x="event_subtype", y="xthreat", y_title="mean xThreat",
                            value_fmt=".3f", title="Threat carried by each run type"),
             key="ob_x")
    with cc[1]:
        plot(bar_with_error(by, x="event_subtype", y="targeted", y_title="share targeted",
                            value_fmt=".0%", colour=SERIES[2],
                            title="How often each run type is actually found"), key="ob_t")
    plot(
        scatter(by, x="targeted", y="xthreat", label="event_subtype", size_col="n",
                x_title="share of runs the passer targets", y_title="mean xThreat of the run",
                title="Which runs are worth finding, and which get found",
                annotate_top=10, height=460),
        key="ob_sc",
    )
    note(
        "Top-left is the interesting quadrant: high threat, rarely found. These are "
        "SkillCorner's own run classifications and xThreat values, used as delivered."
    )
    st.dataframe(by.round(4), hide_index=True, width="stretch")

    section("Runners")
    minr = st.slider("Minimum runs", 10, 200, 40, step=10)
    gp = (
        runs.groupby(["player_name", "team_shortname", "player_position"], observed=True)
        .agg(runs=("targeted", "size"), targeted=("targeted", "mean"),
             received=("received", "mean"), xthreat=("xthreat", "mean"),
             line_break=("break_defensive_line", "mean"),
             behind_runs=("event_subtype", lambda s: (s == "behind").mean()),
             distance=("distance_covered", "mean"))
        .reset_index()
    )
    gp = gp[gp["runs"] >= minr]
    if gp.empty:
        empty("Lower the threshold.")
    else:
        cc = st.columns(2)
        with cc[0]:
            plot(ranked_bar(gp, label="player_name", value="runs", value_fmt=",.0f",
                            title="Most off-ball runs", top=14), key="ob_r1")
        with cc[1]:
            plot(ranked_bar(gp, label="player_name", value="xthreat", value_fmt=".3f",
                            title="Highest mean threat per run", top=14, colour=SERIES[2]),
                 key="ob_r2")
        sel = st.selectbox("Run map for", sorted(gp["player_name"]))
        rr = runs[runs["player_name"] == sel]
        cc2 = st.columns([3, 2])
        with cc2[0]:
            plot(arrows(rr, colour_col="event_subtype",
                        hover=["event_subtype", "distance_covered", "received"],
                        title=f"{sel} — {len(rr)} off-ball runs", height=500, base_width=1.2,
                        opacity=0.6), key="ob_map")
        with cc2[1]:
            st.dataframe(
                rr.groupby("event_subtype", observed=True)
                .agg(n=("targeted", "size"), targeted=("targeted", "mean"),
                     received=("received", "mean"), xthreat=("xthreat", "mean"))
                .round(3).reset_index(),
                hide_index=True, width="stretch", height=430,
            )

# --------------------------------------------------------------------------- #
with tabs[1]:
    metric_row(
        [("On-ball engagements", f"{len(pressures):,}", "pressure, pressing, counter-press…"),
         ("Counter-presses", f"{int((pressures['event_subtype'] == 'counter_press').sum()):,}",
          "within 5 s of a loss, by SkillCorner's definition"),
         ("Beaten by possession", f"{pressures['beaten_by_possession'].fillna(False).mean():.1%}",
          "the presser was beaten on the ball"),
         ("Forced backward", f"{pressures['force_backward'].fillna(False).mean():.1%}", ""),
         ("Mean engagement distance", f"{pressures['interplayer_distance_min'].mean():.1f} m",
          "closest approach to the ball carrier")]
    )
    by = (
        pressures.groupby("event_subtype", observed=True)
        .agg(n=("beaten_by_possession", "size"),
             beaten=("beaten_by_possession", "mean"),
             forced_backward=("force_backward", "mean"),
             stopped_danger=("stop_possession_danger", "mean"),
             min_distance=("interplayer_distance_min", "mean"),
             chain_length=("pressing_chain_length", "mean"))
        .reset_index()
    )
    cc = st.columns(2)
    with cc[0]:
        plot(bar_with_error(by, x="event_subtype", y="beaten", y_title="share beaten",
                            value_fmt=".0%", colour=SERIES[1],
                            title="How often each engagement type is beaten"), key="pr_b")
    with cc[1]:
        plot(bar_with_error(by, x="event_subtype", y="forced_backward",
                            y_title="share forcing a backward action", value_fmt=".0%",
                            colour=SERIES[2], title="How often it forces the ball backwards"),
             key="pr_f")
    st.dataframe(by.round(3), hide_index=True, width="stretch")

    section("Pressers")
    minp = st.slider("Minimum engagements", 10, 200, 40, step=10, key="pr_min")
    gp = (
        pressures.groupby(["player_name", "team_shortname", "player_position"], observed=True)
        .agg(engagements=("beaten_by_possession", "size"),
             counter_press_share=("event_subtype", lambda s: (s == "counter_press").mean()),
             beaten=("beaten_by_possession", "mean"),
             forced_backward=("force_backward", "mean"),
             stopped_danger=("stop_possession_danger", "mean"),
             min_distance=("interplayer_distance_min", "mean"))
        .reset_index()
    )
    gp = gp[gp["engagements"] >= minp]
    if not gp.empty:
        plot(
            scatter(gp, x="engagements", y="beaten", label="player_name",
                    x_title="on-ball engagements", y_title="share beaten by the carrier",
                    title="Volume of pressing against how often it is beaten",
                    annotate_top=0, highlight=list(gp.nsmallest(4, "beaten")["player_name"]),
                    height=460),
            key="pr_sc",
        )
        st.dataframe(gp.sort_values("engagements", ascending=False).round(3),
                     hide_index=True, width="stretch", height=320)
    section("Where pressure happens")
    cc = st.columns(2)
    with cc[0]:
        plot(heatmap(pressures["x_start_att"], pressures["y_start_att"],
                     title="All on-ball engagements (presser's attacking frame)", height=420),
             key="pr_h1")
    with cc[1]:
        cp = pressures[pressures["event_subtype"] == "counter_press"]
        plot(heatmap(cp["x_start_att"], cp["y_start_att"],
                     title="Counter-presses only", height=420), key="pr_h2")

# --------------------------------------------------------------------------- #
with tabs[2]:
    tr = transitions.merge(team_names, left_on="winning_team_id", right_on="team_id", how="left")
    metric_row(
        [("Turnovers", f"{len(tr):,}", "changes of team possession across 20 matches"),
         ("Mean progression in 8 s", f"{tr['max_progress_m'].mean():.1f} m",
          "furthest the ball gets towards goal"),
         ("Mean hold time", f"{tr['held_possession_s'].mean():.1f} s",
          "how long the winning team keeps it"),
         ("Recovered in own third",
          f"{(tr['regain_x_att'] < -17.5).mean():.0%}", "of all regains"),
         ("Recovered in final third",
          f"{(tr['regain_x_att'] > 17.5).mean():.0%}", "")]
    )
    q = pd.qcut(tr["loser_compactness_start_m"].dropna(), 5,
                labels=["most compact", "Q2", "Q3", "Q4", "most stretched"])
    g = (
        tr.dropna(subset=["loser_compactness_start_m"]).assign(quintile=q)
        .groupby("quintile", observed=True)
        .agg(n=("max_progress_m", "size"),
             compactness=("loser_compactness_start_m", "mean"),
             opp_progress=("max_progress_m", "mean"),
             opp_hold_s=("held_possession_s", "mean"),
             recovery=("loser_recovery_m", "mean"))
        .reset_index()
    )
    cc = st.columns(2)
    with cc[0]:
        plot(bar_with_error(g, x="quintile", y="opp_progress",
                            y_title="opponent progression in 8 s (m)", value_fmt=".1f",
                            title="What the losing team's shape costs them"), key="tr_1")
    with cc[1]:
        plot(bar_with_error(g, x="quintile", y="opp_hold_s",
                            y_title="opponent hold time (s)", value_fmt=".1f",
                            colour=SERIES[2], title="How long the opponent keeps the ball"),
             key="tr_2")
    note(
        "Teams that lose the ball while stretched concede *less* progression and give the ball "
        "back sooner. A stretched shape here mostly means a team committed high up the pitch, "
        "and the counter-press that follows is what shortens the opponent's possession. This is "
        "an association across 5,147 turnovers in 20 matches, not a licence to leave gaps."
    )
    section("Where turnovers happen and what follows")
    cc = st.columns(2)
    with cc[0]:
        plot(heatmap(tr["regain_x_att"], tr["regain_y_att"],
                     title="Regain locations (winning team's attacking frame)", height=420),
             key="tr_h")
    with cc[1]:
        hi = tr[tr["max_progress_m"] > tr["max_progress_m"].quantile(0.9)]
        plot(heatmap(hi["regain_x_att"], hi["regain_y_att"],
                     title="Regains that led to the top 10% of progressions", height=420),
             key="tr_h2")
    section("By team")
    byteam = (
        tr.groupby("team", observed=True)
        .agg(turnovers_won=("max_progress_m", "size"),
             progression=("max_progress_m", "mean"),
             hold_s=("held_possession_s", "mean"),
             regain_x=("regain_x_att", "mean"))
        .reset_index().dropna()
    )
    cc = st.columns(2)
    with cc[0]:
        plot(ranked_bar(byteam, label="team", value="progression", value_fmt=".1f", unit=" m",
                        title="Mean progression after winning the ball"), key="tr_b1")
    with cc[1]:
        plot(ranked_bar(byteam, label="team", value="regain_x", value_fmt="+.1f", unit=" m",
                        colour=SERIES[2], title="Mean height of the regain"), key="tr_b2")
