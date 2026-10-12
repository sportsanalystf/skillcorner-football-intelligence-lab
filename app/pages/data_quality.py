"""Data quality: what is measured, what is missing, and what that costs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import caveat, empty, metric_row, note, page_header, plot, section
from src.config import DOCS_DIR, SERIES
from src.visualization.charts import distribution, ranked_bar, scatter

page_header(
    "Data Quality",
    "SkillCorner's README says about 97% of player identities are accurate and that speed "
    "and acceleration need smoothing. This page measures both, and everything else that can "
    "quietly poison a tracking analysis.",
)

v = D.validation()
if v.empty:
    empty("Run `python -m src.preprocessing.validate` to populate this page.")
    st.stop()

piv = v.pivot_table(index="match_id", columns="name", values="value", aggfunc="first")


def med(name: str, default=np.nan) -> float:
    return float(piv[name].median()) if name in piv else default


metric_row(
    [
        ("Matches validated", f"{v['match_id'].nunique()}", "every match with tracking"),
        ("Duplicate frames", f"{int(piv['duplicate_frames'].sum())}", "across all matches"),
        ("Frame sequence breaks", f"{int(piv['frame_sequence_breaks'].sum())}",
         "frames increment by exactly 1"),
        ("Clock reconstruction error", f"{piv['clock_max_abs_error'].max():.3g} s",
         "derived match clock vs the file's own"),
        ("Event↔tracking offset", f"{piv['event_sync_median_error_m'].median():.3g} m",
         "median, after correcting event orientation"),
    ]
)
note(
    "The event-to-tracking offset is exactly zero because dynamic events store coordinates in "
    "each team's **attacking** frame while tracking stores the broadcast frame. Before that was "
    "established the same measurement read a median of 9.2 m and a 90th percentile of 80 m — "
    "half the events were mirrored. It is the single most consequential thing discovered during "
    "reconnaissance and it is asserted in the test suite."
)

tabs = st.tabs(["Coverage", "Kinematics", "Identity & geometry", "Events & phases", "All checks"])

# --------------------------------------------------------------------------- #
with tabs[0]:
    cov = piv[["ball_frames", "frames_without_tracking", "frames_with_22_players",
               "frames_with_other_counts", "detected_fraction", "ball_detected_fraction",
               "frames_with_possession_group", "frames_with_possession_player"]].copy()
    cov["untracked_share"] = cov["frames_without_tracking"] / cov["ball_frames"]
    metric_row(
        [("Frames with no tracking", f"{cov['untracked_share'].median():.0%}",
          "replays, crowd shots, stoppages"),
         ("Player positions detected", f"{med('detected_fraction'):.0%}",
          "the rest are SkillCorner extrapolations"),
         ("Ball detected when tracked", f"{med('ball_detected_fraction'):.0%}", ""),
         ("Frames with a possession team", f"{med('frames_with_possession_group') / med('ball_frames'):.0%}",
          "of all frames"),
         ("Frames with a possession player",
          f"{med('frames_with_possession_player') / med('ball_frames'):.0%}", "")]
    )
    d = cov.reset_index()
    d["match_id"] = d["match_id"].astype(str)
    c = st.columns(2)
    with c[0]:
        plot(ranked_bar(d, label="match_id", value="untracked_share", value_fmt=".0%",
                        title="Share of frames with no tracking, by match", top=20,
                        colour=SERIES[1]), key="dq_1")
    with c[1]:
        plot(ranked_bar(d, label="match_id", value="detected_fraction", value_fmt=".0%",
                        title="Share of player positions actually detected", top=20),
             key="dq_2")
    caveat(
        "Extrapolated positions are plausible, smooth and not observations. They inflate nothing "
        "obviously, but they are why this project reports `detected_frac` on every physical row "
        "and never presents a tracked distance as a match total."
    )

# --------------------------------------------------------------------------- #
with tabs[1]:
    metric_row(
        [("Max speed observed", f"{piv['max_speed_ms'].max():.1f} m/s",
          f"{piv['max_speed_ms'].max() * 3.6:.0f} km/h — a single frame"),
         ("99.9th percentile speed", f"{med('p999_speed_ms'):.1f} m/s",
          "the robust ceiling"),
         ("Speeds masked as implausible", f"{int(piv['speed_masked_implausible'].sum()):,}",
          f"{piv['speed_masked_implausible'].sum() / piv['player_frames'].sum():.4%} of player-frames"),
         ("Max |acceleration|", f"{piv['max_abs_accel_ms2'].max():.1f} m/s²", "after smoothing"),
         ("Ball max speed", f"{piv['ball_max_speed_ms'].max():.0f} m/s",
          "gated separately from players")]
    )
    st.markdown(
        "Speeds above **12 m/s** or accelerations beyond **8 m/s²** are **masked, not clipped**. "
        "Clipping would quietly add a plausible-looking 12 m/s to a distance total; masking "
        "leaves a hole that the totals skip. The masked share is under 0.02% of player-frames, "
        "which is the honest measure of how much of the smoothed signal is unusable."
    )
    trk = D.match_tracking(int(piv.index[0]), columns=("speed", "is_detected", "position"))
    sub = trk.dropna(subset=["speed"]).sample(min(len(trk), 40000), random_state=3)
    sub["detected"] = np.where(sub["is_detected"], "detected", "extrapolated")
    plot(distribution(sub, value="speed", group="detected",
                      title=f"Speed distribution, match {piv.index[0]} — detected vs extrapolated",
                      y_title="m/s", height=400), key="dq_sp")
    st.caption(
        "Extrapolated frames carry a visibly narrower speed distribution: SkillCorner's "
        "extrapolation is smooth by construction, so it under-represents the extremes. Physical "
        "metrics computed over a heavily extrapolated window will read low, not high."
    )

# --------------------------------------------------------------------------- #
with tabs[2]:
    metric_row(
        [("Single-frame jumps > 2.5 m", f"{int(piv['teleport_steps'].sum())}",
          "identity swaps and homography glitches"),
         ("Largest single-frame step", f"{piv['max_single_frame_step'].max():.2f} m",
          "at 10 Hz, i.e. the implied speed cap"),
         ("Positions outside the pitch + 5 m", f"{int(piv['players_out_of_bounds'].sum()):,}",
          f"{piv['players_out_of_bounds'].sum() / piv['player_frames'].sum():.4%} of rows"),
         ("Events with an unknown player", f"{int(piv['events_with_unknown_player'].sum())}",
          "every event id resolves to a squad member"),
         ("Frames with a partial squad", f"{int(piv['frames_with_other_counts'].sum())}",
          "player_data is 22 players or empty, never partial")]
    )
    note(
        "No teleports at all is initially surprising given SkillCorner's 97% identity figure. "
        "The explanation is that the published file is *extrapolated*: identity errors are "
        "smoothed into a continuous track rather than left as a visible jump. The 3% is real "
        "and simply not detectable from position continuity — which is why this project reports "
        "it as a known limitation rather than claiming to have measured it away."
    )
    d = piv[["players_out_of_bounds", "max_single_frame_step", "ball_z_min", "ball_z_max"]].reset_index()
    d["match_id"] = d["match_id"].astype(str)
    st.dataframe(d.round(3), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
with tabs[3]:
    metric_row(
        [("Events", f"{int(piv['events'].sum()):,}", "across all matches"),
         ("Possessions with EPV", f"{med('possession_epv_populated'):.0%}", ""),
         ("Possessions with pressure", f"{med('pressure_populated'):.0%}", ""),
         ("Fully extrapolated events", f"{int(piv['fully_extrapolated_events'].sum()):,}",
          "the whole event window is extrapolated tracking"),
         ("Phase coverage of tracked frames", f"{med('phase_coverage_of_tracked_frames'):.0%}",
          "phases only exist while the ball is in play")]
    )
    ev_cols = [c for c in piv.columns if c.startswith("events_") and c != "events_with_unknown_player"]
    counts = piv[ev_cols].sum().rename("events").reset_index()
    counts["index"] = counts["name" if "name" in counts else "index"].str.replace("events_", "", regex=False)
    plot(ranked_bar(counts.rename(columns={"index": "event_type"}), label="event_type",
                    value="events", value_fmt=",.0f", title="Events by type, all matches"),
         key="dq_ev")
    st.dataframe(piv[["phases", "phase_overlaps", "phase_coverage_of_tracked_frames",
                      "event_sync_matched", "event_sync_median_error_m",
                      "event_sync_p90_error_m"]].round(3), width="stretch")

# --------------------------------------------------------------------------- #
with tabs[4]:
    q = st.text_input("Filter checks", "")
    show = v.copy()
    if q:
        show = show[show["name"].str.contains(q, case=False)]
    agg = (
        show.groupby(["name", "unit", "note"], dropna=False)["value"]
        .agg(["min", "median", "max"]).reset_index().round(4)
    )
    st.dataframe(agg, hide_index=True, width="stretch", height=560)
    lim = DOCS_DIR / "LIMITATIONS.md"
    if lim.exists():
        with st.expander("docs/LIMITATIONS.md"):
            st.markdown(lim.read_text())
