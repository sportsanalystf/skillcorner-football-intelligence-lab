"""Match Explorer: a moment in a match, from four data sources at once."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import caveat, empty, metric_row, note, page_header, plot, section
from src.config import SERIES
from src.tracking.pitch_control import control_summary, control_surface
from src.visualization.charts import half_time_marker, timeline
from src.visualization.pitch_plots import control_map, tracking_snapshot, trajectory

page_header(
    "Match Explorer",
    "Pick a moment and see everything the four data sources say about it: where all 22 "
    "players and the ball were, which phase of play both teams were in, what the event "
    "record called it, and how the pitch was divided.",
    level="Levels 1-4",
)

labels = D.match_label_map()
c1, c2 = st.columns([3, 2])
label = c1.selectbox("Match", list(labels), index=0, key="me_match")
match_id = labels[label]
meta = D.meta(match_id)
home_name, away_name = meta.home_team_name, meta.away_team_name

poss = D.match_possessions(match_id)
phases = D.match_phases(match_id)
ball = D.match_ball(match_id)
events = D.match_events(match_id)

mode = c2.radio("Jump to", ["Possession", "Free scrub"], horizontal=True, key="me_mode")

# --------------------------------------------------------------------------- #
# Moment selection
# --------------------------------------------------------------------------- #
if mode == "Possession":
    f1, f2, f3 = st.columns([2, 2, 3])
    team_pick = f1.selectbox("Team", [home_name, away_name], key="me_team")
    team_id = meta.home_team_id if team_pick == home_name else meta.away_team_id
    kinds = ["all", "led to a shot", "progressive pass", "under high pressure",
             "line-breaking pass", "highest EPV"]
    kind = f2.selectbox("Filter", kinds, key="me_filter")

    sel = poss[poss["team_id"] == team_id].copy()
    if kind == "led to a shot":
        sel = sel[sel["lead_to_shot"].fillna(False)]
    elif kind == "progressive pass":
        sel = sel[sel["fil_is_progressive"].fillna(False)]
    elif kind == "under high pressure":
        sel = sel[sel["overall_pressure_start_id"] >= 4]
    elif kind == "line-breaking pass":
        sel = sel[sel[["first_line_break", "second_last_line_break", "last_line_break"]]
                  .fillna(False).any(axis=1)]
    elif kind == "highest EPV":
        sel = sel.nlargest(60, "possession_epv_total")
    sel = sel.sort_values("frame_start")

    if sel.empty:
        empty("No possessions match that filter in this match.")
        st.stop()

    opts = {
        f"{D.clock_label(r.match_clock_s)}  {r.player_name}  ·  {r.end_type}"
        f"{'  ·  shot' if r.lead_to_shot else ''}": int(r.frame_start)
        for r in sel.itertuples()
    }
    # The key includes the filter so that changing the filter resets the
    # selection instead of leaving a stale value pointing at a vanished option.
    chosen = f3.selectbox(f"{len(opts)} possessions", list(opts),
                          key=f"me_poss::{match_id}::{team_pick}::{kind}")
    frame = opts[chosen]
    row = sel[sel["frame_start"] == frame].iloc[0]
else:
    tracked = ball.dropna(subset=["ball_x"])
    lo, hi = int(tracked["frame"].min()), int(tracked["frame"].max())
    frame = st.slider("Frame", lo, hi, value=int(np.percentile(tracked["frame"], 40)),
                      step=10, key="me_frame")
    row = None

fr_ball = ball[ball["frame"] == frame]
fr_ball = fr_ball.iloc[0] if len(fr_ball) else None
clock = D.clock_label(float(fr_ball["match_clock_s"])) if fr_ball is not None else "--:--"

players = D.frame_players(match_id, int(frame))
if players.empty:
    empty(
        f"Frame {frame} carries no tracking. Roughly 27% of frames in a broadcast file are "
        "replays, crowd shots or stoppages — SkillCorner records the frame but no positions."
    )
    st.stop()

phase = phases[(phases["frame_start"] <= frame) & (phases["frame_end"] >= frame)]
phase = phase.iloc[0] if len(phase) else None

# --------------------------------------------------------------------------- #
# Context strip
# --------------------------------------------------------------------------- #
poss_group = str(fr_ball["possession_group"]) if fr_ball is not None else "—"
in_poss = home_name if poss_group.startswith("home") else away_name if poss_group.startswith("away") else "—"
metric_row(
    [
        ("Clock", clock, f"period {int(players['period'].iloc[0])}, frame {frame}"),
        ("In possession", in_poss, poss_group),
        ("Phase (in poss.)",
         str(phase["team_in_possession_phase_type"]) if phase is not None else "ball out of play",
         f"{phase['team_in_possession_shortname']}" if phase is not None else "phases only exist in play"),
        ("Phase (out of poss.)",
         str(phase["team_out_of_possession_phase_type"]) if phase is not None else "—",
         "the concurrent defensive state"),
        ("Detected players", f"{int(players['is_detected'].sum())}/22",
         "the rest are SkillCorner extrapolations"),
    ]
)

# --------------------------------------------------------------------------- #
# The pitch
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])

with left:
    opt1, opt2, opt3 = st.columns(3)
    show_control = opt1.toggle("Pitch control", value=True)
    show_vel = opt2.toggle("Velocity vectors", value=True)
    orient = opt3.selectbox("Orientation", ["Broadcast", f"{home_name} → right", f"{away_name} → right"])

    pl = players.copy()
    flip = 1
    if orient.startswith(home_name):
        flip = int(pl["home_attack_sign"].iloc[0])
    elif orient.startswith(away_name):
        flip = -int(pl["home_attack_sign"].iloc[0])
    for c in ("x_std", "y_std", "vx", "vy"):
        pl[c] = pl[c] * flip
    bx = fr_ball.copy() if fr_ball is not None else None
    if bx is not None:
        bx["ball_x_std"] = bx["ball_x_std"] * flip
        bx["ball_y_std"] = bx["ball_y_std"] * flip

    control = None
    if show_control:
        control = control_surface(pl, home_team_id=meta.home_team_id)

    highlight = set()
    if row is not None:
        highlight = {int(row["player_id"])}
        if pd.notna(row.get("player_targeted_id")):
            highlight.add(int(row["player_targeted_id"]))

    plot(
        tracking_snapshot(
            pl, bx, home_team_id=meta.home_team_id,
            home_name=home_name, away_name=away_name,
            show_velocity=show_vel, highlight=highlight, control=control,
            title=(f"{clock} — {in_poss} in possession" if in_poss != "—"
                   else f"{clock} — possession unassigned in this frame"),
            height=560,
        ),
        key="me_snapshot",
    )
    if show_control:
        s = control_summary(
            control, home_attacks_positive_x=meta.home_attacks_positive_x.get(
                int(players["period"].iloc[0]), True)
        )
        st.caption(
            f"Pitch control — {home_name} {s['home_control']:.0%} of the pitch "
            f"({s['home_control_att_third']:.0%} of their attacking third) · "
            f"{away_name} {s['away_control']:.0%} ({s['away_control_att_third']:.0%}). "
            "Model and parameters: docs/METHODOLOGY.md."
        )

with right:
    if row is not None:
        section("This possession")
        st.markdown(
            f"**{row['player_name']}** ({row['player_position']}, {row['team_shortname']}) · "
            f"started by *{row['start_type']}*, ended in *{row['end_type']}*"
        )
        metric_row(
            [
                ("Progression", f"{row['progression_m']:+.1f} m", "towards the opposition goal"),
                ("EPV", f"{row['possession_epv_total']:+.4f}"
                 if pd.notna(row["possession_epv_total"]) else "—", "SkillCorner possession EPV"),
            ],
            columns=2,
        )
        metric_row(
            [
                ("Pressure", str(row["overall_pressure_start"]) if pd.notna(row["overall_pressure_start"]) else "—",
                 f"nearest opponent {row['fil_nearest_opponent_m']:.1f} m"
                 if pd.notna(row.get("fil_nearest_opponent_m")) else "tracking-derived"),
                ("Passing options", f"{row['n_passing_options']:.0f}"
                 if pd.notna(row["n_passing_options"]) else "—",
                 f"{row['n_passing_options_ahead']:.0f} ahead of the ball"
                 if pd.notna(row["n_passing_options_ahead"]) else ""),
            ],
            columns=2,
        )
        det = {
            "Receiver": row.get("player_targeted_name"),
            "Pass outcome": row.get("pass_outcome"),
            "Pass direction": row.get("pass_direction"),
            "Line break": row.get("furthest_line_break"),
            "Opponents bypassed": row.get("n_opponents_bypassed"),
            "Reception difficulty": row.get("reception_difficulty_start"),
            "Space constraint": row.get("space_constraint_start"),
            "Time to impact": row.get("time_to_impact_start"),
            "Teammates within 10 m": row.get("fil_teammates_within_10m"),
            "Opponents within 10 m": row.get("fil_opponents_within_10m"),
            "Opponents ahead in corridor": row.get("fil_opponents_ahead_in_corridor"),
        }
        st.dataframe(
            pd.DataFrame(
                {"": list(det),
                 "value": [("—" if v is None or (isinstance(v, float) and pd.isna(v))
                            else str(v)) for v in det.values()]}
            ),
            hide_index=True, width="stretch", height=min(420, 36 * len(det)),
        )
    else:
        section("Events in this second")
        near = events[(events["frame_start"] - frame).abs() <= 10]
        if near.empty:
            empty("No dynamic events within one second of this frame.")
        else:
            st.dataframe(
                near[["event_type", "event_subtype", "player_name", "team_shortname",
                      "x_start_att", "y_start_att"]]
                .rename(columns={"x_start_att": "x (att)", "y_start_att": "y (att)"})
                .round(1),
                hide_index=True, width="stretch", height=340,
            )

# --------------------------------------------------------------------------- #
# Around the moment
# --------------------------------------------------------------------------- #
section("The ten seconds around this moment", "ball trajectory and the run-up on the clock")
w1, w2 = st.columns([2, 3])

win = D.tracking_window(match_id, frame - 50, frame + 50)
bwin = ball[(ball["frame"] >= frame - 50) & (ball["frame"] <= frame + 50)].dropna(subset=["ball_x_std"])
with w1:
    if bwin.empty:
        empty("The ball is not tracked through this window.")
    else:
        path = bwin.rename(columns={"ball_x_std": "x_std", "ball_y_std": "y_std",
                                    "ball_speed": "speed"})
        plot(trajectory(path, colour_by="speed", title="Ball path, ±5 s", height=380),
             key="me_balltraj")
with w2:
    if win.empty:
        empty("No tracking in this window.")
    else:
        sp = (
            win.groupby(["match_clock_s", "side"], observed=True)["speed"].mean()
            .unstack().reset_index()
        )
        sp["minute"] = sp["match_clock_s"] / 60
        cols = {c: (home_name if c == "home" else away_name) for c in sp.columns if c in ("home", "away")}
        plot(
            timeline(sp, x="minute", series=cols, y_title="mean team speed (m/s)",
                     title="Mean team speed through the window", height=380, hover_fmt=".2f"),
            key="me_speed",
        )

section("Match timeline", "EPV added per possession, and where the ball was")
tp = D.match_team_possessions(match_id)
if not tp.empty:
    t = tp.copy()
    t["minute"] = t["match_clock_s"] / 60
    t["home_epv"] = np.where(t["is_home"], t["epv_total"], np.nan)
    t["away_epv"] = np.where(~t["is_home"], t["epv_total"], np.nan)
    t = t.sort_values("minute")
    t["home_cum"] = t["home_epv"].fillna(0).cumsum()
    t["away_cum"] = t["away_epv"].fillna(0).cumsum()
    fig = timeline(t, x="minute", series={"home_cum": home_name, "away_cum": away_name},
                   y_title="cumulative EPV", title="Cumulative EPV created", height=300)
    plot(half_time_marker(fig), key="me_epv_timeline")
    caveat(
        "EPV is SkillCorner's own possession value model, populated on player-possession rows "
        "only (95% of them). It is used as delivered and not recomputed."
    )
