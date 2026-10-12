"""Tactical Analysis: team shape, compactness, lines and pitch control over time."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import caveat, empty, metric_row, note, page_header, plot, section
from src.config import SERIES
from src.tracking.pitch_control import control_summary, control_surface
from src.visualization.charts import distribution, half_time_marker, ranked_bar, timeline
from src.visualization.pitch_plots import control_map, heatmap, team_shape_snapshot

page_header(
    "Tactical Analysis",
    "Team shape read frame by frame off the tracking: how high the line sits, how compact "
    "the block is, how much of the pitch each side controls, and how all of that moves "
    "through a match.",
    level="Levels 4-5",
)

labels = D.match_label_map()
c1, c2, c3 = st.columns([3, 2, 2])
label = c1.selectbox("Match", list(labels), key="tac_match")
match_id = labels[label]
meta = D.meta(match_id)
home_name, away_name = meta.home_team_name, meta.away_team_name
team_pick = c2.selectbox("Team", [home_name, away_name], key="tac_team")
team_id = meta.home_team_id if team_pick == home_name else meta.away_team_id
opp_id = meta.away_team_id if team_pick == home_name else meta.home_team_id
colour = SERIES[0] if team_pick == home_name else SERIES[1]
smooth_s = c3.slider("Smoothing (s)", 1, 60, 15)

tf = D.match_team_frames(match_id, stride=5)
if tf.empty:
    empty("Team-shape features have not been built for this match.")
    st.stop()

mine = tf[tf["team_id"] == team_id].sort_values("frame").copy()
theirs = tf[tf["team_id"] == opp_id].sort_values("frame").copy()
mine["minute"] = mine["match_clock_s"] / 60

metric_row(
    [
        ("Defensive line", f"{mine['def_line_x'].mean():+.1f} m",
         "mean x of the 4 deepest outfielders, own goal at −52.5"),
        ("Block depth", f"{mine['vertical_compactness_m'].mean():.1f} m",
         "attacking line minus defensive line"),
        ("Width", f"{mine['width_m'].mean():.1f} m", "widest to widest, goalkeeper excluded"),
        ("Compactness", f"{mine['compactness_m'].mean():.1f} m",
         "mean distance from a player to the team centroid"),
        ("Shape area", f"{mine['hull_area_m2'].mean():,.0f} m²",
         "convex hull of the 10 outfield players"),
        ("Space between lines", f"{mine['space_between_lines_m'].mean():+.1f} m",
         "gap from our attacking line to their defensive line"),
    ]
)
note(
    "Every number on this page is in the selected team's own attacking frame, on a standard "
    "105 × 68 pitch, with the goalkeeper excluded. A keeper 40 m behind the block turns team "
    "length into a measure of nothing."
)

tabs = st.tabs(["Shape over time", "Snapshot", "Pitch control", "Territory", "Both teams"])

# --------------------------------------------------------------------------- #
with tabs[0]:
    win = max(1, int(smooth_s * 2))   # team frames are stored at 2 Hz after stride
    sm = mine.set_index("minute")[
        ["def_line_x", "att_line_x", "mid_line_x", "width_m", "length_m",
         "compactness_m", "hull_area_m2", "space_between_lines_m", "block_overlap_m"]
    ].rolling(win, min_periods=1).mean().reset_index()

    fig = timeline(
        sm, x="minute",
        series={"def_line_x": "defensive line", "mid_line_x": "midfield line",
                "att_line_x": "attacking line"},
        y_title="x, own goal at −52.5 (m)", title=f"{team_pick} — unit lines through the match",
        height=320, hover_fmt=".1f",
    )
    plot(half_time_marker(fig), key="tac_lines")

    c = st.columns(2)
    with c[0]:
        fig = timeline(sm, x="minute",
                       series={"width_m": "width", "length_m": "length"},
                       y_title="metres", title="Width and length", height=300, hover_fmt=".1f")
        plot(half_time_marker(fig), key="tac_wl")
    with c[1]:
        fig = timeline(sm, x="minute",
                       series={"compactness_m": "compactness",
                               "space_between_lines_m": "space to their last line"},
                       y_title="metres", title="Compactness and the gap in front of their line",
                       height=300, hover_fmt=".1f")
        plot(half_time_marker(fig), key="tac_comp")

    section("By phase of play", "shape while each in-possession phase is running")
    phases = D.match_phases(match_id)
    ph = phases[phases["team_in_possession_id"] == team_id]
    if ph.empty:
        empty("No phases for this team.")
    else:
        rows = []
        for r in ph.itertuples():
            seg = mine[(mine["frame"] >= r.frame_start) & (mine["frame"] <= r.frame_end)]
            if seg.empty:
                continue
            rows.append({"phase": r.team_in_possession_phase_type,
                         "compactness_m": seg["compactness_m"].mean(),
                         "width_m": seg["width_m"].mean(),
                         "def_line_x": seg["def_line_x"].mean(),
                         "length_m": seg["length_m"].mean()})
        byphase = pd.DataFrame(rows)
        if byphase.empty:
            empty("No overlap between phases and tracked frames.")
        else:
            g = byphase.groupby("phase").mean(numeric_only=True).reset_index()
            cc = st.columns(2)
            with cc[0]:
                plot(ranked_bar(g, label="phase", value="width_m", value_fmt=".1f", unit=" m",
                                colour=colour, title="Mean width by phase"), key="tac_pw")
            with cc[1]:
                plot(ranked_bar(g, label="phase", value="def_line_x", value_fmt="+.1f", unit=" m",
                                colour=colour, title="Mean defensive line height by phase"),
                     key="tac_pd")

# --------------------------------------------------------------------------- #
with tabs[1]:
    tracked = mine[mine["n_players"] >= 9]
    if tracked.empty:
        empty("No fully tracked frames for this team.")
    else:
        f = st.select_slider(
            "Frame", options=tracked["frame"].tolist(),
            value=int(tracked["frame"].iloc[len(tracked) // 3]),
            format_func=lambda v: D.clock_label(
                float(tracked.loc[tracked["frame"] == v, "match_clock_s"].iloc[0])
            ),
        )
        pl = D.frame_players(match_id, int(f))
        pl = pl[(pl["team_id"] == team_id) & (pl["position"].astype(str) != "GK")].copy()
        pl["xa"] = pl["x_std"] * pl["attack_sign"]
        pl["ya"] = pl["y_std"] * pl["attack_sign"]
        row = tracked[tracked["frame"] == f].iloc[0]
        cc = st.columns([3, 2])
        with cc[0]:
            plot(
                team_shape_snapshot(pl, colour=colour, height=520,
                                    title=f"{team_pick} shape at {D.clock_label(row['match_clock_s'])}"),
                key="tac_shape",
            )
        with cc[1]:
            metric_row([("Width", f"{row['width_m']:.1f} m", ""),
                        ("Length", f"{row['length_m']:.1f} m", "")], columns=2)
            metric_row([("Hull area", f"{row['hull_area_m2']:,.0f} m²", ""),
                        ("Compactness", f"{row['compactness_m']:.1f} m", "")], columns=2)
            metric_row([("Def line", f"{row['def_line_x']:+.1f} m", ""),
                        ("Att line", f"{row['att_line_x']:+.1f} m", "")], columns=2)
            st.caption(
                f"Superiority in the attacking third: {row['superiority_att_third']:+.0f} · "
                f"in the defensive third: {row['superiority_def_third']:+.0f} · "
                f"players tracked: {int(row['n_players'])} · "
                f"detected (not extrapolated): {row['detected_frac']:.0%}"
            )

# --------------------------------------------------------------------------- #
with tabs[2]:
    st.caption(
        "This pitch-control model is **this project's**, not a SkillCorner metric. Each player "
        "continues at their current velocity for a 0.7 s reaction, then runs straight at 8 m/s; "
        "control of a cell is a softmax over the two teams' best arrival times. Full derivation "
        "in docs/METHODOLOGY.md."
    )
    ball = D.match_ball(match_id).dropna(subset=["ball_x"])
    opts = ball["frame"].iloc[::50].tolist()
    f = st.select_slider(
        "Frame", options=opts, value=opts[len(opts) // 3],
        format_func=lambda v: D.clock_label(float(ball.loc[ball["frame"] == v, "match_clock_s"].iloc[0])),
        key="tac_pc_frame",
    )
    pl = D.frame_players(match_id, int(f))
    if pl.empty:
        empty("No tracking at this frame.")
    else:
        res = control_surface(pl, home_team_id=meta.home_team_id)
        per = int(pl["period"].iloc[0])
        s = control_summary(res, home_attacks_positive_x=meta.home_attacks_positive_x.get(per, True))
        cc = st.columns([3, 2])
        with cc[0]:
            plot(control_map(res, home_name=home_name, away_name=away_name, height=500,
                             title=f"Pitch control at {D.clock_label(float(ball.loc[ball['frame'] == f, 'match_clock_s'].iloc[0]))}"),
                 key="tac_pc")
        with cc[1]:
            metric_row([(home_name, f"{s['home_control']:.0%}", "share of the pitch"),
                        (away_name, f"{s['away_control']:.0%}", "share of the pitch")], columns=2)
            metric_row([(f"{home_name} att. third", f"{s['home_control_att_third']:.0%}", ""),
                        (f"{away_name} att. third", f"{s['away_control_att_third']:.0%}", "")],
                       columns=2)
            st.caption(
                "The research page shows what this model does and does not predict: once ball "
                "position is controlled for, instantaneous control does not forecast how far a "
                "team moves the ball over the following five seconds."
            )

# --------------------------------------------------------------------------- #
with tabs[3]:
    trk = D.match_tracking(match_id, columns=("team_id", "x_std", "y_std", "attack_sign", "position"))
    sub = trk[(trk["team_id"] == team_id) & (trk["position"].astype(str) != "GK")]
    cc = st.columns(2)
    with cc[0]:
        plot(
            heatmap(sub["x_std"] * sub["attack_sign"], sub["y_std"] * sub["attack_sign"],
                    title=f"{team_pick} — where the outfield ten spent the match", height=420),
            key="tac_heat",
        )
    with cc[1]:
        tp = D.match_team_possessions(match_id)
        t = tp[tp["team_id"] == team_id]
        if t.empty:
            empty("No possessions for this team.")
        else:
            plot(
                heatmap(t["x_end_att"], t["y_end_att"],
                        title=f"{team_pick} — where possessions ended", height=420),
                key="tac_heat2",
            )
    caveat(
        "Occupancy maps are built from tracked frames only. Because the broadcast follows the "
        "ball, time near the ball is over-represented relative to a full-pitch optical system."
    )

# --------------------------------------------------------------------------- #
with tabs[4]:
    both = tf.copy()
    both["team"] = np.where(both["team_id"] == meta.home_team_id, home_name, away_name)
    cc = st.columns(2)
    with cc[0]:
        plot(distribution(both, value="compactness_m", group="team",
                          title="Compactness distribution", y_title="metres", height=360),
             key="tac_d1")
    with cc[1]:
        plot(distribution(both, value="def_line_x", group="team",
                          title="Defensive line height distribution",
                          y_title="x in own attacking frame (m)", height=360),
             key="tac_d2")
    summ = (
        both.groupby("team")
        .agg(**{
            "frames": ("frame", "size"),
            "width (m)": ("width_m", "mean"),
            "length (m)": ("length_m", "mean"),
            "compactness (m)": ("compactness_m", "mean"),
            "hull area (m²)": ("hull_area_m2", "mean"),
            "def line (m)": ("def_line_x", "mean"),
            "att line (m)": ("att_line_x", "mean"),
            "space to their line (m)": ("space_between_lines_m", "mean"),
        })
        .round(1).reset_index()
    )
    st.dataframe(summ, hide_index=True, width="stretch")
