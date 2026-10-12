"""Player Explorer: profile card, percentiles and statistical similarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.components import data as D
from app.components.ui import caveat, empty, metric_row, note, page_header, plot, section
from src.config import SERIES
from src.visualization.charts import percentile_rows, ranked_bar, scatter
from src.visualization.pitch_plots import arrows, heatmap

page_header(
    "Player Explorer",
    "One player at a time: season profile against positional peers, what the 20 tracked "
    "matches say about them, and who else in the league produces a similar statistical "
    "profile.",
    level="Levels 1-6",
)

profile = D.table("player_season_profile")
pct_pos = D.table("player_percentiles_position")
sim = D.table("player_similarity")
emb = D.table("player_embedding")
poss = D.table("possessions")
passes = D.table("passes")
runs = D.table("off_ball_runs")
pressures = D.table("pressures")
phys = D.table("player_match_physical")

profile = profile.copy()
profile["display"] = (
    profile["player_short_name"].fillna(profile["player_name"])
    + "  ·  " + profile["team_name"].str.replace(" Football Club", "", regex=False)
    + "  ·  " + profile["position_group"]
)

c1, c2 = st.columns([3, 2])
pick = c1.selectbox("Player", profile.sort_values("display")["display"].tolist())
row = profile[profile["display"] == pick].iloc[0]
pid = int(row["player_id"])
pidx = row["profile_id"]

metric_row(
    [
        ("Player", str(row["player_short_name"] or row["player_name"]), str(row["team_name"])[:28]),
        ("Role", str(row["position_group"]), f"{int(row['count_match'])} included performances"),
        ("Minutes / match", f"{row['minutes']:.0f}", "60+ minute performances only"),
        ("Top speed", f"{row['psv99_kmh']:.1f} km/h", "SkillCorner season PSV99"),
        ("Distance / 90", f"{row['total_distance_per90']:,.0f} m", ""),
        ("m / min", f"{row['total_metersperminute_full_all']:.0f}", ""),
    ]
)

tabs = st.tabs(["Profile card", "In the tracked matches", "Similar players", "Style map"])

CARD = {
    "Physical": [
        ("Distance per 90", "total_distance_per90", "{:,.0f} m"),
        ("Metres per minute", "total_metersperminute_full_all", "{:.0f}"),
        ("High-speed running per 90", "hsr_distance_per90", "{:,.0f} m"),
        ("Sprint distance per 90", "sprint_distance_per90", "{:,.0f} m"),
        ("Sprints per 90", "sprint_count_per90", "{:.1f}"),
        ("High accelerations per 90", "highaccel_count_per90", "{:.1f}"),
        ("High decelerations per 90", "highdecel_count_per90", "{:.1f}"),
        ("Peak speed (PSV99)", "psv99_kmh", "{:.1f} km/h"),
    ],
    "Passing": [
        ("Passes per 30 TIP", "pass_count_attempted_p30tip", "{:.1f}"),
        ("Completion", "pass_pct_completed", "{:.1%}"),
        ("Expected completion", "pass_avgxpass_attempted", "{:.2f}"),
        ("Line-breaking passes per 30 TIP", "pass_count_linebreak_completed_p30tip", "{:.1f}"),
        ("Dangerous passes per 30 TIP", "pass_count_dangerous_completed_p30tip", "{:.2f}"),
        ("Passes to runs per 30 TIP", "pass_count_torun_completed_p30tip", "{:.1f}"),
        ("Passes leading to a shot per 30 TIP", "pass_count_shotwithin10s_p30tip", "{:.2f}"),
        ("Mean pass distance", "pass_avgdistance", "{:.1f} m"),
    ],
    "Off-ball movement": [
        ("Runs per 30 TIP", "offballrun_count_p30tip", "{:.1f}"),
        ("Runs in behind per 30 TIP", "behindrun_count_p30tip", "{:.2f}"),
        ("Support runs per 30 TIP", "supportrun_count_p30tip", "{:.2f}"),
        ("Runs ahead of the ball per 30 TIP", "aheadoftheballrun_count_p30tip", "{:.2f}"),
        ("Cross-receiver runs per 30 TIP", "crossreceiverrun_count_p30tip", "{:.2f}"),
        ("Runs targeted per 30 TIP", "offballrun_count_targeted_p30tip", "{:.2f}"),
        ("Runs received per 30 TIP", "offballrun_count_received_p30tip", "{:.2f}"),
    ],
}

# --------------------------------------------------------------------------- #
with tabs[0]:
    pr = pct_pos[pct_pos["profile_id"] == pidx]
    if pr.empty:
        empty("No percentile row for this player profile.")
    else:
        prow = pr.iloc[0]
        st.caption(
            f"Percentiles are within **{row['position_group']}** "
            f"({int((pct_pos['position_group'] == row['position_group']).sum())} player-profiles "
            "in the league with 4+ included performances). Bars show the percentile; the number "
            "beside each bar is the raw value."
        )
        cols = st.columns(len(CARD))
        for col, (group, items) in zip(cols, CARD.items()):
            avail = [(lab, k, f) for lab, k, f in items if k in prow.index and pd.notna(row.get(k))]
            with col:
                plot(
                    percentile_rows(
                        [(lab, float(prow[k]), f.format(row[k])) for lab, k, f in avail],
                        title=group, height=max(240, 34 * len(avail) + 70),
                    ),
                    key=f"pc_{group}",
                )
    caveat(
        "These percentiles come from SkillCorner's season aggregates for the whole 2024/25 "
        "A-League, at the grain of (player, team, position group). A player who appeared in two "
        "roles has two profiles and each is ranked against its own peer group."
    )

# --------------------------------------------------------------------------- #
with tabs[1]:
    pp = poss[poss["player_id"] == pid]
    pa = passes[passes["player_id"] == pid]
    ru = runs[runs["player_id"] == pid]
    pe = pressures[pressures["player_id"] == pid]
    ph = phys[phys["player_id"] == pid]
    if pp.empty and ph.empty:
        empty(
            "This player does not appear in the 20 matches with tracking. The profile card "
            "still works — it is built from the season-wide aggregates."
        )
    else:
        metric_row(
            [("Matches tracked", f"{ph['match_id'].nunique() if len(ph) else 0}", ""),
             ("Possessions", f"{len(pp):,}", f"{pp['possession_epv_total'].sum():+.3f} EPV"),
             ("Passes", f"{len(pa):,}",
              f"{pa['completed'].mean():.0%} completed" if len(pa) else ""),
             ("Off-ball runs", f"{len(ru):,}",
              f"{ru['targeted'].mean():.0%} targeted" if len(ru) else ""),
             ("Engagements", f"{len(pe):,}",
              f"{pe['beaten_by_possession'].fillna(False).mean():.0%} beaten" if len(pe) else "")]
        )
        cc = st.columns(2)
        with cc[0]:
            trk_pos = pp[["x_start_att", "y_start_att"]].dropna()
            if len(trk_pos) > 10:
                plot(heatmap(trk_pos["x_start_att"], trk_pos["y_start_att"],
                             title="Where this player takes possession", height=430),
                     key="pex_h")
            else:
                empty("Too few possessions to map.")
        with cc[1]:
            if len(pa):
                plot(arrows(pa, colour_col="pass_outcome",
                            colour_map={"successful": SERIES[2], "unsuccessful": SERIES[1],
                                        "offside": SERIES[3]},
                            hover=["player_targeted_name", "progression_m"],
                            title=f"{len(pa)} passes", height=430, base_width=1.1, opacity=0.6),
                     key="pex_p")
            else:
                empty("No passes recorded.")
        if len(ph):
            section("Per match")
            show = ph[["match_id", "position", "tracked_minutes", "coverage",
                       "total_distance_m", "m_per_min", "hsr_distance_m", "sprint_count",
                       "psv99_ms", "high_accel_count"]].round(2)
            st.dataframe(show, hide_index=True, width="stretch")
        if len(pp):
            section("Under pressure")
            g = (
                pp.groupby("overall_pressure_start", observed=True)
                .agg(n=("possession_epv_total", "size"), epv=("possession_epv_total", "mean"),
                     retention=("end_type", lambda s: (s != "possession_loss").mean()),
                     progression=("progression_m", "mean"))
                .reset_index()
            )
            st.dataframe(g.round(4), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
with tabs[2]:
    s = sim[sim["profile_id"] == pidx]
    if s.empty:
        empty("No similarity row for this profile.")
    else:
        st.caption(
            "Cosine similarity over 154 rate-based features (physical per-90, passing per-30 "
            "in-possession, off-ball runs per-30), each rank-transformed to a league percentile "
            "and standardised. **This is a description of statistical profile, not a scouting "
            "recommendation** — it knows nothing about level, age, contract or role in a system."
        )
        show = s[["rank", "similar_player_name", "similar_team_name",
                  "similar_position_group", "similarity"]].copy()
        show.columns = ["#", "player", "team", "role", "similarity"]
        cc = st.columns([2, 3])
        with cc[0]:
            st.dataframe(show.round(3), hide_index=True, width="stretch", height=430)
        with cc[1]:
            plot(ranked_bar(show, label="player", value="similarity", value_fmt=".3f",
                            title="Most similar profiles", top=12), key="sim_bar")

        section("Head to head")
        other = st.selectbox("Compare with", show["player"].tolist())
        orow = profile[profile["profile_id"] ==
                       s[s["similar_player_name"] == other]["similar_profile_id"].iloc[0]]
        if not orow.empty:
            orow = orow.iloc[0]
            prow = pct_pos[pct_pos["profile_id"] == pidx]
            orow_p = pct_pos[pct_pos["profile_id"] == orow["profile_id"]]
            keys = [k for _, k, _ in CARD["Physical"] + CARD["Passing"] if k in profile.columns]
            comp = pd.DataFrame({
                "metric": keys,
                pick.split("  ·  ")[0]: [row.get(k) for k in keys],
                other: [orow.get(k) for k in keys],
            })
            st.dataframe(comp.round(3), hide_index=True, width="stretch", height=420)

# --------------------------------------------------------------------------- #
with tabs[3]:
    e = emb.copy()
    e["display"] = e["player_short_name"].fillna(e["player_name"])
    v1 = float(e["pc1_var"].iloc[0]) if len(e) else 0
    v2 = float(e["pc2_var"].iloc[0]) if len(e) else 0
    k90 = int(e["n_components_for_90pct"].iloc[0]) if len(e) else 0
    plot(
        scatter(e, x="pc1", y="pc2", label="display", colour_col="position_group",
                x_title=f"PC1 ({v1:.0%} of variance)", y_title=f"PC2 ({v2:.0%})",
                title="Style map — PCA of the standardised season profile",
                highlight=[str(row["player_short_name"] or row["player_name"])],
                height=560, quadrants=False),
        key="emb",
    )
    st.caption(
        f"Two components carry {v1 + v2:.0%} of the variance; {k90} are needed for 90%, which "
        "is why similarity is computed in the full standardised space and the map is only used "
        "for orientation. Position groups separate cleanly along PC1 without ever being given "
        "to the model."
    )
