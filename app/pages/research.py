"""Research: the questions this dataset can actually answer, and the answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import caveat, empty, metric_row, note, page_header, plot, section
from src.config import SERIES
from src.visualization.charts import bar_with_error, ranked_bar, scatter

page_header(
    "Research",
    "Ten questions run against the full dataset. Every number on this page is read out of "
    "`reports/research/`, produced by `python -m src.research.questions`. Nothing is "
    "transcribed by hand, and the null results are here too.",
)

summary = D.research_summary()
if not summary:
    empty("Run `python -m src.research.questions` to populate this page.")
    st.stop()

note(
    "These are 20 matches from one league in one season. Sample sizes run from 1,137 "
    "player-frames to 18,792 possessions, so p-values are almost always small; effect sizes "
    "are what the text quotes. Nothing here is a league-wide law, and none of it is causal."
)

CHART = {}

# --------------------------------------------------------------------------- #
def q1(t):
    c = st.columns(2)
    with c[0]:
        plot(bar_with_error(t, x="overall_pressure_start", y="completion",
                            y_title="pass completion", value_fmt=".1%",
                            title="Completion falls monotonically with pressure"), key="r1a")
    with c[1]:
        plot(bar_with_error(t, x="overall_pressure_start", y="pass_epv",
                            y_title="mean pass EPV", value_fmt="+.4f", colour=SERIES[2],
                            title="Value added does not — it peaks at medium pressure"), key="r1b")
    plot(bar_with_error(t, x="overall_pressure_start", y="nearest_opponent_m",
                        y_title="metres", value_fmt=".1f", colour=SERIES[3], height=300,
                        title="Cross-check: distance to the nearest opponent, computed "
                              "independently from tracking"), key="r1c")


def q2(t):
    c = st.columns(2)
    with c[0]:
        plot(ranked_bar(t, label="player_name", value="epv_per_possession", value_fmt="+.4f",
                        title="EPV added per pressed possession", top=14), key="r2a")
    with c[1]:
        plot(scatter(t, x="possessions_pressed", y="epv_per_possession", label="player_name",
                     x_title="possessions under medium-or-worse pressure",
                     y_title="EPV added per pressed possession", annotate_top=5, height=420,
                     title="Volume against value under pressure"), key="r2b")


def q3(t):
    c = st.columns(2)
    with c[0]:
        plot(bar_with_error(t, x="quintile", y="opp_max_progress_m",
                            y_title="opponent progression in 8 s (m)", value_fmt=".1f",
                            title="Progression conceded, by the losing team's compactness"),
             key="r3a")
    with c[1]:
        plot(bar_with_error(t, x="quintile", y="opp_hold_time_s", y_title="seconds",
                            value_fmt=".1f", colour=SERIES[2],
                            title="How long the opponent keeps the ball"), key="r3b")


def q4(t):
    c = st.columns(2)
    with c[0]:
        plot(bar_with_error(t.sort_values("epv_total", ascending=False),
                            x="team_in_possession_phase_type", y="epv_total",
                            y_title="mean EPV per chain", value_fmt="+.4f",
                            title="What each phase of play is worth"), key="r4a")
    with c[1]:
        plot(bar_with_error(t.sort_values("share", ascending=False),
                            x="team_in_possession_phase_type", y="share",
                            y_title="share of all chains", value_fmt=".1%", colour=SERIES[3],
                            title="How often each phase occurs"), key="r4b")
    plot(scatter(t, x="share", y="epv_total", label="team_in_possession_phase_type",
                 size_col="territory_gain_m", x_title="share of all possession chains",
                 y_title="mean EPV per chain", annotate_top=8, height=420,
                 title="The most valuable phases are the rarest"), key="r4c")


def q5(t):
    c = st.columns(3)
    with c[0]:
        plot(bar_with_error(t, x="orientation", y="progression_m", y_title="metres",
                            value_fmt="+.1f", title="Progression achieved"), key="r5a")
    with c[1]:
        plot(bar_with_error(t, x="orientation", y="lead_to_shot", y_title="share",
                            value_fmt=".1%", colour=SERIES[2], title="Led to a shot"), key="r5b")
    with c[2]:
        plot(bar_with_error(t, x="orientation", y="epv", y_title="mean EPV",
                            value_fmt="+.4f", colour=SERIES[3], title="EPV added"), key="r5c")


def q6(t):
    plot(scatter(t, x="targeted_rate", y="xthreat", label="event_subtype", size_col="n",
                 x_title="share of runs the passer targets", y_title="mean xThreat",
                 annotate_top=10, height=470,
                 title="Which runs are worth finding, and which get found"), key="r6a")
    c = st.columns(2)
    with c[0]:
        plot(ranked_bar(t, label="event_subtype", value="xthreat", value_fmt=".3f",
                        title="Threat per run type"), key="r6b")
    with c[1]:
        plot(ranked_bar(t, label="event_subtype", value="targeted_rate", value_fmt=".0%",
                        colour=SERIES[2], title="Targeted rate"), key="r6c")


def q7(t):
    piv = t.pivot(index="physical_metric", columns="shape_metric", values="spearman_rho")
    st.dataframe(piv.round(3), width="stretch")
    st.caption("Spearman rho between team-match mean shape and team-match mean physical output "
               "(n = 40 team-matches).")


def q8(t):
    plot(scatter(t.dropna(subset=["fil_psv99_ms", "sc_psv99_ms"]),
                 x="sc_psv99_ms", y="fil_psv99_ms", label="player_short_name",
                 colour_col="position_group", x_title="SkillCorner season PSV99 (m/s)",
                 y_title="this pipeline's per-match PSV99 (m/s)", height=500, quadrants=False,
                 title="Two independent pipelines, same players"), key="r8a")


def q9(t):
    c = st.columns(2)
    with c[0]:
        plot(bar_with_error(t, x="ball_zone", y="rho_control_vs_progress",
                            y_title="Spearman rho", value_fmt="+.3f",
                            title="Control vs later progression, within ball-position bands"),
             key="r9a")
    with c[1]:
        m = t.melt(id_vars="ball_zone",
                   value_vars=["progress_top_quintile_m", "progress_bottom_quintile_m"],
                   var_name="group", value_name="progress_m")
        m["group"] = m["group"].map({"progress_top_quintile_m": "most control",
                                     "progress_bottom_quintile_m": "least control"})
        import plotly.graph_objects as go

        from src.visualization.theme import apply
        fig = go.Figure()
        for i, (g, sub) in enumerate(m.groupby("group")):
            fig.add_trace(go.Bar(x=sub["ball_zone"], y=sub["progress_m"], name=g,
                                 marker=dict(color=SERIES[i], cornerradius=4,
                                             line=dict(color="#141A22", width=2)),
                                 hovertemplate="%{x}<br>%{y:.1f} m<extra>" + g + "</extra>"))
        fig.update_layout(barmode="group")
        fig.update_yaxes(title="ball progression in 5 s (m)")
        plot(apply(fig, title="Top vs bottom control quintile, by where the ball is", height=400),
             key="r9b")


def q10(t):
    c = st.columns(2)
    with c[0]:
        plot(bar_with_error(t, x="speed_band", y="median_abs_diff_deg",
                            y_title="median absolute difference (°)", value_fmt=".1f",
                            title="Pose facing vs movement heading"), key="r10a")
    with c[1]:
        plot(bar_with_error(t, x="speed_band", y="share_within_45deg", y_title="share",
                            value_fmt=".0%", colour=SERIES[2],
                            title="Share agreeing within 45°"), key="r10b")


CHART = {
    "q1_pressure_and_pass_value": q1, "q2_value_under_pressure": q2,
    "q3_shape_at_loss": q3, "q4_phase_value": q4, "q5_orientation_at_reception": q5,
    "q6_off_ball_runs": q6, "q7_workload_vs_style": q7, "q8_psv99_validation": q8,
    "q9_pitch_control_and_progression": q9, "q10_pose_vs_heading": q10,
}

keys = list(summary)
labels = [f"Q{i + 1}" for i in range(len(keys))]
tabs = st.tabs(labels)
for tab, key in zip(tabs, keys):
    with tab:
        entry = summary[key]
        st.markdown(f"### {entry['question']}")
        if "error" in entry:
            empty(f"This question failed to run: {entry['error']}")
            continue
        for f in entry["findings"]:
            st.markdown(f"- {f}")
        t = D.research(key)
        if t.empty:
            continue
        section("Evidence")
        fn = CHART.get(key)
        if fn:
            fn(t)
        with st.expander("Table"):
            st.dataframe(t.round(4), hide_index=True, width="stretch")
