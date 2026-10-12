"""3D body pose: skeletons, orientation and what pose adds over tracking."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import caveat, empty, metric_row, note, page_header, plot, section
from src.config import SERIES
from src.data.pose_loading import available_sources
from src.visualization.charts import bar_with_error, distribution, timeline
from src.visualization.pitch_plots import trajectory
from src.visualization.pose_plots import orientation_on_pitch, orientation_rose, skeleton_3d

page_header(
    "3D Body Pose",
    "29 joints per player at 25 Hz, in three dimensions. This is the only layer that can say "
    "which way a player was actually facing, as opposed to which way they were moving.",
    level="Level 7",
)

src = available_sources()
joints = D.table("pose_joints")
orient = D.table("pose_orientation")

if joints.empty:
    empty("No pose tables built. Run `python -m src.features.pose`.")
    st.stop()

match_id = int(orient["match_id"].iloc[0])
meta = D.meta(match_id)
source = str(orient["source"].iloc[0])

if source == "sample_phase":
    note(
        "**Running on the committed sample phase.** SkillCorner hosts the two full pose matches "
        "(≈600 MB zipped, 3.3 GB of JSON each) on the Hugging Face Hub, which the network this "
        "project was built on blocks. Everything below therefore runs on the one phase of play "
        "committed to the SkillCorner repository: a 12.3-second Brisbane Roar quick break from "
        "the second half of Brisbane Roar v Perth Glory, 309 pose frames, 143,637 joint "
        "observations. `python -m src.data.download --bodypose` fetches the full matches on a "
        "network that allows huggingface.co, and every table and chart on this page is built by "
        "the same code path from whichever source is present.",
        "warn",
    )

phase_start = int(orient["tracking_frame"].min())
phase_end = int(orient["tracking_frame"].max())
poss = D.match_possessions(match_id)
in_phase = poss[poss["frame_start"].between(phase_start - 20, phase_end + 20)]

metric_row(
    [
        ("Source", "sample phase" if source == "sample_phase" else "full match",
         meta.label),
        ("Pose frames", f"{orient['pose_frame'].nunique():,}", "25 Hz"),
        ("Player-frames with joints", f"{len(orient):,}",
         f"{orient['player_id'].nunique()} players"),
        ("Joint observations", f"{len(joints):,}", "29 joints × 3D"),
        ("Mean joint error", f"{orient['joint_err_mean_cm'].mean():.1f} cm",
         "SkillCorner's own p90 radius"),
        ("Tracking frames covered", f"{phase_start}–{phase_end}",
         f"{(phase_end - phase_start) / 10:.1f} s"),
    ]
)

tabs = st.tabs(["3D skeleton", "Body orientation", "Orientation vs movement",
                "The phase, end to end"])

# --------------------------------------------------------------------------- #
with tabs[0]:
    names = (
        orient[["player_id", "display_name", "side", "position"]]
        .drop_duplicates().dropna(subset=["display_name"]).sort_values("display_name")
    )
    c1, c2 = st.columns([2, 3])
    who = c1.selectbox("Player", names["display_name"].tolist())
    pid = int(names[names["display_name"] == who]["player_id"].iloc[0])
    side = str(names[names["display_name"] == who]["side"].iloc[0])
    frames = sorted(orient[orient["player_id"] == pid]["pose_frame"].unique())
    f = c2.select_slider(
        "Pose frame", options=frames, value=frames[len(frames) // 2],
        format_func=lambda v: f"{v}  ({(v / 25 - frames[0] / 25):+.2f} s into the phase)",
    )
    jf = joints[(joints["player_id"] == pid) & (joints["pose_frame"] == f)]
    orow = orient[(orient["player_id"] == pid) & (orient["pose_frame"] == f)]

    cc = st.columns([3, 2])
    with cc[0]:
        plot(
            skeleton_3d(jf, colour=SERIES[0] if side == "home" else SERIES[1],
                        title=f"{who} — pose frame {f}", height=620),
            key="bp_skel",
        )
        st.caption(
            "Marker size is that joint's own p90 error radius: the bigger the dot, the less "
            "certain the estimate. Axes are metres relative to the player's own centroid in x "
            "and y; z is as delivered."
        )
    with cc[1]:
        if not orow.empty:
            r = orow.iloc[0]
            metric_row([("Facing", f"{r['facing_goal_deg']:.0f}°",
                         f"from the goal they attack — {r['body_orientation_class']}"),
                        ("Openness", f"{r['openness']:+.2f}", "cosine of that angle")], columns=2)
            metric_row([("Shoulder width", f"{r['shoulder_width_m']:.2f} m", "biacromial"),
                        ("Hip width", f"{r['hip_width_m']:.2f} m", "")], columns=2)
            metric_row([("Trunk lean", f"{r['trunk_lean_deg']:.0f}°", "within the pose"),
                        ("Shoulder–hip twist", f"{r['shoulder_hip_twist_deg']:.0f}°", "")],
                       columns=2)
            metric_row([("Stance width", f"{r['stance_width_m']:.2f} m", "ankle to ankle"),
                        ("Ball distance", f"{r['ball_distance_m']:.1f} m",
                         f"{r['ball_bearing_deg']:.0f}° off the facing direction")], columns=2)
            st.dataframe(
                jf[["joint", "jx", "jy", "jz", "p90_mae_cm"]].round(3),
                hide_index=True, width="stretch", height=300,
            )
    caveat(
        "SkillCorner state that z is accurate **relative to the player's centroid**, not in the "
        "pitch coordinate frame. Trunk lean is therefore reported as an angle within the pose "
        "and never as an absolute posture relative to the ground. Nothing on this page claims "
        "biomechanical accuracy."
    )

# --------------------------------------------------------------------------- #
with tabs[1]:
    cc = st.columns([2, 3])
    with cc[0]:
        plot(orientation_rose(orient, title="Which way bodies point, relative to the goal "
                                            "each player attacks"), key="bp_rose")
        st.dataframe(
            orient["body_orientation_class"].value_counts().rename("player-frames")
            .reset_index(),
            hide_index=True, width="stretch",
        )
    with cc[1]:
        pf = sorted(orient["pose_frame"].unique())
        f2 = st.select_slider("Pose frame", options=pf, value=pf[len(pf) // 2], key="bp_f2")
        snap = orient[orient["pose_frame"] == f2]
        plot(
            orientation_on_pitch(
                snap, title=f"Body orientation of every posed player, frame {f2}", height=470
            ),
            key="bp_pitch",
        )
        st.caption(
            "Each line is 3 m long and points where that player's torso is square to. Only "
            "detected players get pose, so the count changes frame to frame."
        )

    section("Orientation by role")
    g = (
        orient.groupby("position", observed=True)
        .agg(frames=("facing_goal_deg", "size"),
             mean_facing_deg=("facing_goal_deg", "mean"),
             openness=("openness", "mean"),
             trunk_lean_deg=("trunk_lean_deg", "mean"),
             twist_deg=("shoulder_hip_twist_deg", "mean"))
        .reset_index().sort_values("openness", ascending=False)
    )
    plot(bar_with_error(g, x="position", y="openness", y_title="mean openness (cos of angle to goal)",
                        value_fmt="+.2f", title="How open each position's body is, in this phase"),
         key="bp_role")
    st.dataframe(g.round(2), hide_index=True, width="stretch")
    caveat(
        f"n = {len(orient):,} player-frames from a single {(phase_end - phase_start) / 10:.1f}-second "
        "phase. Read this as a demonstration of the measurement, not as a positional norm."
    )

# --------------------------------------------------------------------------- #
with tabs[2]:
    r = D.research("q10_pose_vs_heading")
    if r.empty:
        empty("Run `python -m src.research.questions` to generate this comparison.")
    else:
        st.markdown(
            "Pose is the only source that knows body orientation. Tracking knows movement "
            "heading. Where a player is moving quickly the two should agree, and where a "
            "player is nearly still they should not — that is the whole reason pose is worth "
            "having. This is the check that licenses the large-scale orientation analysis on "
            "the Research page."
        )
        cc = st.columns([2, 3])
        with cc[0]:
            st.dataframe(r.round(2), hide_index=True, width="stretch")
        with cc[1]:
            plot(bar_with_error(r, x="speed_band", y="median_abs_diff_deg",
                                y_title="median |pose facing − movement heading| (°)",
                                value_fmt=".1f",
                                title="Agreement between pose and tracking, by speed"),
                 key="bp_agree")
        note(
            f"Above 4 m/s the two agree to a median of "
            f"{float(r[r['speed_band'] == '>4']['median_abs_diff_deg'].iloc[0]):.1f}°. Below "
            f"1 m/s they diverge to "
            f"{float(r[r['speed_band'] == '<1 m/s']['median_abs_diff_deg'].iloc[0]):.0f}° — a "
            "standing player has no meaningful heading, and that is exactly the situation where "
            "only pose can answer the question."
        )
        st.caption(
            "This also confirms the sign convention of the construction: if the shoulder-line "
            "rotation were inverted, the fast-moving bands would sit near 180°, not near 7°."
        )

# --------------------------------------------------------------------------- #
with tabs[3]:
    st.markdown(
        f"The sample phase is one **quick break**: Brisbane Roar intercept on halfway and shoot "
        f"{(phase_end - phase_start) / 10:.1f} seconds later. Every layer of the dataset "
        "describes it at once, which is the whole argument of this project in twelve seconds."
    )
    if in_phase.empty:
        empty("No possessions overlap this phase.")
    else:
        show = in_phase[["match_clock_s", "player_name", "team_shortname", "start_type",
                         "end_type", "x_start_att", "progression_m"]].copy()
        show["match_clock_s"] = show["match_clock_s"].map(D.clock_label)
        show.columns = ["clock", "player", "team", "started", "ended", "x (att, m)", "prog (m)"]
        st.dataframe(show.round(1), hide_index=True, width="stretch")

    cc = st.columns(2)
    with cc[0]:
        ball = D.match_ball(match_id)
        seg = ball[(ball["frame"] >= phase_start) & (ball["frame"] <= phase_end)].dropna(
            subset=["ball_x_std"])
        if not seg.empty:
            path = seg.rename(columns={"ball_x_std": "x_std", "ball_y_std": "y_std",
                                       "ball_speed": "speed"})
            plot(trajectory(path, colour_by="speed", title="Ball path through the break",
                            height=420), key="bp_ball")
    with cc[1]:
        o = orient.copy()
        o["second"] = (o["tracking_frame"] - phase_start) / 10
        byteam = (
            o.groupby(["second", "side"], observed=True)["openness"].mean().unstack().reset_index()
        )
        cols = {c: (meta.home_team_name if c == "home" else meta.away_team_name)
                for c in byteam.columns if c in ("home", "away")}
        if cols:
            plot(
                timeline(byteam, x="second", series=cols, x_title="seconds into the phase",
                         y_title="mean openness (cos angle to attacked goal)",
                         title="How both teams' bodies turn through the break", height=420,
                         hover_fmt=".2f"),
                key="bp_open",
            )
    note(
        "Openness is the cosine of the angle between a player's torso facing direction and the "
        "goal that player's team is attacking: +1 is square-on to the opposition goal, −1 is "
        "facing your own. Averaged per team, it is a compact reading of who is turned towards "
        "the game."
    )
