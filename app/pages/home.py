"""Overview: what the dataset is, and the question ladder the project climbs."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import metric_row, note, page_header, plot, section
from src.visualization.charts import ranked_bar, scatter

page_header(
    "Football Intelligence Lab",
    "What can you learn about football when event data, broadcast tracking, physical "
    "output and 3D body pose describe the same moment? This platform puts all four on "
    "one clock and works up from what happened to what the players' bodies were doing.",
)

matches = D.matches()
squads = D.squads()
poss = D.table("possessions")
runs = D.table("off_ball_runs")
phys = D.table("player_match_physical")
tp = D.table("team_possessions")

metric_row(
    [
        ("Matches", f"{len(matches)}", "A-League 2024/25, full broadcast tracking"),
        ("Player-frames", "19.3M", "10 Hz tracking, smoothed and differentiated"),
        ("Dynamic events", f"{94517:,}", "possessions, passing options, runs, pressures"),
        ("Possessions", f"{len(poss):,}", f"in {len(tp):,} possession chains"),
        ("Players", f"{squads.player_id.nunique()}", "across 13 clubs"),
        ("Pose joints", f"{143637:,}", "29 joints x 3D, one phase of play"),
    ]
)

section("The ladder", "each page in this app answers one rung, using data the rung below cannot see")

LADDER = [
    ("L1", "What happened?", "Dynamic events: 18,792 player possessions, 16,334 passes, "
     "9,894 off-ball runs, 17,400 on-ball engagements.", "Possession & EPV"),
    ("L2", "Where did it happen?", "Every event carries pitch coordinates — once you know "
     "they are stored in each team's own attacking frame, not the broadcast's.", "Match Explorer"),
    ("L3", "Who else was there?", "22 players at 10 Hz. The event tells you a pass was made; "
     "tracking tells you who could have received it.", "Match Explorer"),
    ("L4", "Why did it happen?", "Pressure, space, defensive lines and a pitch-control model "
     "turn a location into a situation.", "Tactical Analysis"),
    ("L5", "What happened away from the ball?", "Off-ball runs, team shape, compactness and "
     "what changes in the eight seconds after a turnover.", "Off-Ball & Pressing"),
    ("L6", "What did it cost?", "Distance, high-speed running and accelerations, computed from "
     "tracking and cross-checked against SkillCorner's season aggregates.", "Physical"),
    ("L7", "What was the body doing?", "29 3D joints per player at 25 Hz: which way a receiver "
     "was actually facing, not just which way they were moving.", "3D Body Pose"),
]
for lvl, q, a, page in LADDER:
    st.markdown(
        f'<div class="fil-story"><div class="lvl">{lvl}</div>'
        f'<div><div class="q">{q}</div><div class="a">{a} '
        f'<em style="color:#3987e5">→ {page}</em></div></div></div>',
        unsafe_allow_html=True,
    )

section("Matches in the dataset")
show = matches[["date", "round", "home_team", "home_score", "away_score", "away_team",
                "stadium", "pitch_length", "has_bodypose"]].copy()
show.columns = ["Date", "Round", "Home", "H", "A", "Away", "Stadium", "Pitch length (m)", "Body pose"]
st.dataframe(show, width="stretch", hide_index=True, height=340)
note(
    "Pitch length varies between 104 m and 106 m across these fixtures. Every coordinate in "
    "the processed tables is therefore also stored rescaled to a standard 105 × 68 pitch, so "
    "that a cross-match comparison is comparing the same thing."
)

section("A first look at the season")
c1, c2 = st.columns(2)
with c1:
    team_epv = (
        tp.groupby("team", observed=True)
        .agg(possessions=("epv_total", "size"), epv=("epv_total", "sum"),
             shot_rate=("lead_to_shot", "mean"))
        .reset_index()
    )
    plot(
        ranked_bar(team_epv, label="team", value="epv", value_fmt=".2f",
                   title="Total EPV created, all matches in this sample"),
        key="home_team_epv",
    )
with c2:
    top = (
        phys.groupby("player_name", observed=True)
        .agg(distance=("total_distance_m", "mean"), hsr=("hsr_distance_m", "mean"),
             minutes=("tracked_minutes", "sum"))
        .reset_index()
    )
    top = top[top["minutes"] >= 60]
    plot(
        scatter(top, x="distance", y="hsr", label="player_name",
                x_title="mean distance while tracked (m)",
                y_title="mean high-speed running (m)",
                title="Volume against intensity, per match tracked",
                annotate_top=5, height=430),
        key="home_phys_scatter",
    )
note(
    "Distances here are computed over the frames the broadcast actually shows, which is about "
    "73% of a match. They are not match totals, and the Physical page explains the difference."
)
