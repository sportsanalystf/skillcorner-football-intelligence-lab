"""Possession chains and Expected Possession Value."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import caveat, empty, metric_row, note, page_header, plot, section
from src.config import SERIES
from src.visualization.charts import (bar_with_error, distribution, half_time_marker,
                                      ranked_bar, scatter, timeline)
from src.visualization.pitch_plots import arrows, heatmap

page_header(
    "Possession & EPV",
    "Player possessions stitched into chains, and SkillCorner's Expected Possession Value "
    "used as delivered: how much a possession shifts the chance of scoring, and of conceding.",
    level="Levels 1 & 4",
)

poss = D.table("possessions")
tp = D.table("team_possessions")
matches = D.matches()

scope = st.radio("Scope", ["All matches", "One match"], horizontal=True, key="pe_scope")
if scope == "One match":
    labels = D.match_label_map()
    label = st.selectbox("Match", list(labels), key="pe_match")
    mid = labels[label]
    poss = poss[poss["match_id"] == mid]
    tp = tp[tp["match_id"] == mid]

epv_pop = poss["possession_epv_total"].notna().mean()
metric_row(
    [
        ("Player possessions", f"{len(poss):,}", f"{epv_pop:.0%} carry EPV"),
        ("Possession chains", f"{len(tp):,}", "one per phase of play"),
        ("Mean chain length", f"{tp['n_player_possessions'].mean():.1f}",
         f"{tp['duration_s'].mean():.1f} s on average"),
        ("Chains ending in a shot", f"{tp['lead_to_shot'].mean():.1%}",
         f"{int(tp['lead_to_shot'].sum()):,} shots"),
        ("Mean EPV per chain", f"{tp['epv_total'].mean():+.4f}", "net of EPV conceded"),
    ]
)

_cov = poss.groupby("match_id")["possession_epv_total"].apply(lambda s: s.notna().mean())
_missing = sorted(_cov[_cov == 0].index)
if _missing:
    note(
        f"**EPV is entirely absent from {len(_missing)} of the matches in scope** "
        f"({', '.join(str(m) for m in _missing)}): those files carry no EPV, no pressure band "
        "and no reception difficulty on any possession row. That is "
        f"{poss['match_id'].isin(_missing).sum():,} of {len(poss):,} possessions here. Averages "
        "on this page are over the possessions that do carry a value, never over zeros.",
        "warn",
    )

tabs = st.tabs(["Chain explorer", "EPV rankings", "EPV by situation", "Pressure", "Where value is created"])

# --------------------------------------------------------------------------- #
with tabs[0]:
    if scope == "All matches":
        note("Pick a single match above to explore individual possession chains.")
    else:
        c1, c2 = st.columns([2, 3])
        team = c1.selectbox("Team", sorted(tp["team"].dropna().unique()))
        t = tp[tp["team"] == team].sort_values("epv_total", ascending=False)
        sort_by = c2.selectbox("Rank by", ["EPV created", "Territory gained", "Duration",
                                           "Passes", "Ended in a shot"])
        key = {"EPV created": "epv_total", "Territory gained": "territory_gain_m",
               "Duration": "duration_s", "Passes": "n_passes",
               "Ended in a shot": "lead_to_shot"}[sort_by]
        t = t.sort_values(key, ascending=False)
        opts = {
            f"{D.clock_label(r.match_clock_s)}  ·  {r.team_in_possession_phase_type}  ·  "
            f"{r.n_passes:.0f} passes  ·  {r.territory_gain_m:+.0f} m  ·  EPV {r.epv_total:+.4f}"
            f"{'  ·  SHOT' if r.lead_to_shot else ''}": int(r.phase_index)
            for r in t.head(80).itertuples()
        }
        if not opts:
            empty("No chains for this team.")
        else:
            pick = st.selectbox("Chain", list(opts))
            pidx = opts[pick]
            chain = poss[poss["phase_index"] == pidx].sort_values("frame_start")
            head = t[t["phase_index"] == pidx].iloc[0]

            metric_row(
                [
                    ("Duration", f"{head['duration_s']:.1f} s", f"{head['n_passes']:.0f} passes"),
                    ("Territory", f"{head['territory_gain_m']:+.1f} m",
                     f"{head['speed_of_progression_mps']:+.1f} m/s"),
                    ("Players", f"{head['n_players_involved']:.0f}", str(head["players"])[:44]),
                    ("EPV", f"{head['epv_total']:+.4f}",
                     f"for {head['epv_delta_for']:+.4f} / against {head['epv_delta_against']:+.4f}"),
                    ("Ended", str(head["end_type"]), f"phase: {head['team_in_possession_phase_type']}"),
                ]
            )
            cc = st.columns([3, 2])
            with cc[0]:
                fig = arrows(
                    chain, colour_col=None, single_colour=SERIES[0],
                    hover=["player_name", "end_type", "progression_m"],
                    title="The chain, in the attacking direction", height=440,
                )
                plot(fig, key="pe_chain")
            with cc[1]:
                show = chain[["match_clock_s", "player_name", "start_type", "end_type",
                              "progression_m", "overall_pressure_start",
                              "possession_epv_total"]].copy()
                show["match_clock_s"] = show["match_clock_s"].map(D.clock_label)
                show.columns = ["clock", "player", "started", "ended", "prog (m)",
                                "pressure", "EPV"]
                st.dataframe(show.round(4), hide_index=True, width="stretch", height=440)

# --------------------------------------------------------------------------- #
with tabs[1]:
    min_p = st.slider("Minimum possessions", 20, 400, 100, step=20)
    pl = (
        poss.groupby(["player_id", "player_name", "team_shortname", "player_position"],
                     observed=True)
        .agg(possessions=("possession_epv_total", "size"),
             epv_total=("possession_epv_total", "sum"),
             epv_per_possession=("possession_epv_total", "mean"),
             progression=("progression_m", "mean"),
             retention=("end_type", lambda s: (s != "possession_loss").mean()))
        .reset_index()
    )
    pl = pl[pl["possessions"] >= min_p]
    if pl.empty:
        empty("No players meet that threshold. Lower the minimum.")
    else:
        cc = st.columns(2)
        with cc[0]:
            plot(ranked_bar(pl, label="player_name", value="epv_total", value_fmt="+.3f",
                            title="Total EPV added", top=15), key="pe_r1")
        with cc[1]:
            plot(ranked_bar(pl, label="player_name", value="epv_per_possession",
                            value_fmt="+.4f", title="EPV added per possession", top=15,
                            colour=SERIES[2]), key="pe_r2")
        plot(
            scatter(pl, x="possessions", y="epv_per_possession", label="player_name",
                    size_col="epv_total", x_title="possessions",
                    y_title="EPV added per possession", annotate_top=6, height=460,
                    title="Volume against efficiency"),
            key="pe_scatter",
        )
        st.dataframe(
            pl.sort_values("epv_total", ascending=False).round(4),
            hide_index=True, width="stretch", height=320,
        )

    section("Teams")
    teams = (
        tp.groupby("team", observed=True)
        .agg(chains=("epv_total", "size"), epv_created=("epv_delta_for", "sum"),
             epv_conceded=("epv_delta_against", "sum"), epv_net=("epv_total", "sum"),
             epv_per_chain=("epv_total", "mean"), shot_rate=("lead_to_shot", "mean"),
             territory=("territory_gain_m", "mean"))
        .reset_index()
    )
    cc = st.columns(2)
    with cc[0]:
        plot(ranked_bar(teams, label="team", value="epv_net", value_fmt="+.2f",
                        title="Net EPV"), key="pe_t1")
    with cc[1]:
        plot(ranked_bar(teams, label="team", value="epv_per_chain", value_fmt="+.4f",
                        title="Net EPV per possession chain", colour=SERIES[2]), key="pe_t2")

# --------------------------------------------------------------------------- #
with tabs[2]:
    byphase = (
        tp.groupby("team_in_possession_phase_type", observed=True)
        .agg(n=("epv_total", "size"), epv=("epv_total", "mean"),
             territory=("territory_gain_m", "mean"), shot=("lead_to_shot", "mean"),
             duration=("duration_s", "mean"))
        .reset_index().sort_values("epv", ascending=False)
    )
    cc = st.columns(2)
    with cc[0]:
        plot(bar_with_error(byphase, x="team_in_possession_phase_type", y="epv",
                            y_title="mean EPV per chain", value_fmt="+.4f",
                            title="What each phase of play is worth"), key="pe_ph1")
    with cc[1]:
        plot(bar_with_error(byphase, x="team_in_possession_phase_type", y="shot",
                            y_title="share ending in a shot", value_fmt=".1%",
                            colour=SERIES[3], title="Shot rate by phase"), key="pe_ph2")
    st.dataframe(byphase.round(4), hide_index=True, width="stretch")
    note(
        "Transition and quick-break possessions are rare and enormously valuable: together they "
        "are under 4% of chains but carry the highest EPV and shot rate of any phase. Build-up, "
        "the most common phase, gains essentially no territory on average."
    )

    section("Starting situation")
    byst = (
        poss.groupby("start_type", observed=True)
        .agg(n=("possession_epv_total", "size"), epv=("possession_epv_total", "mean"),
             progression=("progression_m", "mean"))
        .reset_index().sort_values("epv", ascending=False)
    )
    plot(bar_with_error(byst, x="start_type", y="epv", y_title="mean EPV",
                        value_fmt="+.4f", title="EPV by how the possession started",
                        height=360), key="pe_start")

# --------------------------------------------------------------------------- #
with tabs[3]:
    p = poss[poss["overall_pressure_start_id"].notna()]
    order = ["no_pressure", "low_pressure", "medium_pressure", "high_pressure", "very_high_pressure"]
    g = (
        p.groupby("overall_pressure_start", observed=True)
        .agg(n=("possession_epv_total", "size"),
             epv=("possession_epv_total", "mean"),
             retention=("end_type", lambda s: (s != "possession_loss").mean()),
             progression=("progression_m", "mean"),
             nearest_opponent=("fil_nearest_opponent_m", "mean"))
        .reindex(order).reset_index()
    )
    cc = st.columns(2)
    with cc[0]:
        plot(bar_with_error(g, x="overall_pressure_start", y="epv", y_title="mean EPV",
                            value_fmt="+.4f", title="EPV by pressure on the ball carrier"),
             key="pe_pr1")
    with cc[1]:
        plot(bar_with_error(g, x="overall_pressure_start", y="nearest_opponent",
                            y_title="metres", value_fmt=".1f", colour=SERIES[2],
                            title="Distance to the nearest opponent, from tracking"),
             key="pe_pr2")
    note(
        "The right-hand chart is a cross-check, not a metric: SkillCorner's pressure bands were "
        "produced by their model, and the distances were computed independently from the "
        "tracking file. They line up monotonically, 11.8 m down to 1.3 m."
    )
    st.dataframe(g.round(4), hide_index=True, width="stretch")

    section("Who creates value under pressure")
    pressed = poss[poss["overall_pressure_start_id"] >= 3]
    gp = (
        pressed.groupby(["player_name", "team_shortname"], observed=True)
        .agg(pressed_possessions=("possession_epv_total", "size"),
             epv_per_possession=("possession_epv_total", "mean"),
             retention=("end_type", lambda s: (s != "possession_loss").mean()))
        .reset_index()
    )
    gp = gp[gp["pressed_possessions"] >= (20 if scope == "One match" else 60)]
    if gp.empty:
        empty("Not enough pressed possessions in this scope.")
    else:
        plot(ranked_bar(gp, label="player_name", value="epv_per_possession", value_fmt="+.4f",
                        title="EPV added per possession under medium-or-worse pressure", top=14),
             key="pe_pr3")

# --------------------------------------------------------------------------- #
with tabs[4]:
    cc = st.columns(2)
    hi = poss[poss["possession_epv_total"] > poss["possession_epv_total"].quantile(0.95)]
    lo = poss[poss["possession_epv_total"] < poss["possession_epv_total"].quantile(0.05)]
    with cc[0]:
        plot(heatmap(hi["x_start_att"], hi["y_start_att"],
                     title="Where the top 5% EPV possessions start", height=420), key="pe_h1")
    with cc[1]:
        plot(heatmap(lo["x_start_att"], lo["y_start_att"],
                     title="Where the bottom 5% EPV possessions start", height=420), key="pe_h2")
    zones = poss.dropna(subset=["third_start", "channel_start", "possession_epv_total"])
    piv = zones.pivot_table(index="channel_start", columns="third_start",
                            values="possession_epv_total", aggfunc="mean").round(4)
    st.dataframe(piv, width="stretch")
    st.caption("Mean EPV added per possession, by the zone the possession started in.")
