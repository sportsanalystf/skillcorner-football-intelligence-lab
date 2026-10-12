"""Generate and execute the exploratory notebooks.

The notebooks are written here rather than by hand so they cannot drift from
the library, and they are executed so the committed copies carry real output.

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import nbformat as nbf
from nbclient import NotebookClient

OUT = ROOT / "notebooks"

HEAD = """import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.config import DERIVED_DIR, PROCESSED_DIR, PALETTE
plt.style.use("dark_background")
plt.rcParams.update({
    "figure.facecolor": PALETTE["surface"], "axes.facecolor": PALETTE["surface"],
    "savefig.facecolor": PALETTE["surface"], "grid.color": PALETTE["grid"],
    "axes.edgecolor": PALETTE["line"], "text.color": PALETTE["text"],
    "axes.labelcolor": PALETTE["text_secondary"], "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"], "font.size": 9, "figure.dpi": 110,
})
pd.set_option("display.width", 140, "display.max_columns", 40)
"""

NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_data_exploration": [
        ("md", """# 01 · Data exploration

What is actually in the SkillCorner Open Data drop, established by reading the
files rather than the documentation. Everything found here is written up in
`docs/DATA_DICTIONARY.md`."""),
        ("code", HEAD),
        ("md", "## The 20 matches\n\nThe repository README says 10; it ships 20."),
        ("code", """matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet")
print(f"{len(matches)} matches, {pd.concat([matches.home_team, matches.away_team]).nunique()} teams")
matches[["date", "round", "home_team", "home_score", "away_score", "away_team",
         "pitch_length", "pitch_width", "has_bodypose"]]"""),
        ("md", "Pitch size is not constant, which is why every table also carries "
               "coordinates rescaled to a standard 105 x 68 pitch."),
        ("code", """matches.groupby(["pitch_length", "pitch_width"]).size().rename("matches").reset_index()"""),
        ("md", "## The tracking file\n\nOne JSON object per video frame at 10 fps."),
        ("code", """from src.data.loaders import iter_tracking
import itertools, json
rec = next(itertools.islice(iter_tracking(1874553), 300, 301))
rec["player_data"] = rec["player_data"][:2]
print(json.dumps(rec, indent=1)[:1200])"""),
        ("code", """from src.preprocessing.build_tracking import load_ball, load_tracking
ball = load_ball(1874553)
trk = load_tracking(1874553)
print(f"frames            {len(ball):,}  ({ball.frame.min()}-{ball.frame.max()}, consecutive: "
      f"{bool((np.diff(np.sort(ball.frame)) == 1).all())})")
print(f"no tracking       {ball.ball_x.isna().sum():,}  ({ball.ball_x.isna().mean():.0%})")
print(f"player-frames     {len(trk):,}")
print(f"detected          {trk.is_detected.mean():.0%}  (the rest are extrapolations)")
print(f"players per frame {sorted(trk.groupby('frame').size().unique())}")"""),
        ("md", "## The event taxonomy\n\nFour event types share one 322-column table. There is "
               "**no** shot, tackle, foul or set-piece event type."),
        ("code", """ev = pd.read_parquet(PROCESSED_DIR / "events.parquet",
                     columns=["event_type", "event_subtype"])
ev.value_counts(dropna=False).rename("rows").reset_index()"""),
        ("md", "## Which columns belong to which event type\n\nMost of the 322 columns are "
               "empty for most rows, and which ones depends entirely on the event type."),
        ("code", """full = pd.read_parquet(PROCESSED_DIR / "events.parquet")
probe = ["x_start", "possession_epv_total", "overall_pressure_start", "pass_outcome",
         "first_line_break", "event_subtype", "pressing_chain", "player_targeted_id",
         "xthreat", "xpass_completion", "player_targeted_xpass_completion"]
(full.groupby("event_type")[probe].apply(lambda g: g.notna().mean()).round(2))"""),
        ("md", """`xthreat` and `xpass_completion` sit on `passing_option` and `off_ball_run`
rows and are **never** populated on `player_possession`. The possession
equivalents are the `player_targeted_*` columns. Getting this wrong silently
produces an all-NaN column."""),
        ("md", "## Coverage gaps in the source\n\nFour matches carry no EPV, pressure or "
               "reception difficulty at all."),
        ("code", """poss = pd.read_parquet(DERIVED_DIR / "possessions.parquet",
                       columns=["match_id", "possession_epv_total", "overall_pressure_start",
                                "reception_difficulty_start"])
cov = poss.groupby("match_id").apply(
    lambda g: pd.Series({"possessions": len(g),
                         "epv": g.possession_epv_total.notna().mean(),
                         "pressure": g.overall_pressure_start.notna().mean(),
                         "reception_difficulty": g.reception_difficulty_start.notna().mean()}),
    include_groups=False).round(3)
cov"""),
        ("code", """missing = cov[cov.epv == 0].index.tolist()
print("matches with no EPV at all:", missing)
print(f"{poss.match_id.isin(missing).sum():,} of {len(poss):,} possessions "
      f"({poss.match_id.isin(missing).mean():.0%})")"""),
        ("md", "## Phases of play"),
        ("code", """ph = pd.read_parquet(PROCESSED_DIR / "phases.parquet")
print("in possession :", sorted(ph.team_in_possession_phase_type.dropna().unique()))
print("out of poss.  :", sorted(ph.team_out_of_possession_phase_type.dropna().unique()))
ph.groupby("team_in_possession_phase_type").agg(
    n=("duration", "size"), mean_duration_s=("duration", "mean")).round(1)"""),
    ],

    "02_tracking_validation": [
        ("md", """# 02 · Tracking validation

Two things are established here, both load-bearing for everything downstream:
the match clock, and the fact that **event coordinates are stored in each
team's attacking frame while tracking is in broadcast orientation**."""),
        ("code", HEAD),
        ("md", "## The clock\n\nOne formula reproduces the timestamp string in the tracking file."),
        ("code", """from src.data.loaders import iter_tracking, load_match_meta
meta = load_match_meta(1874553)
print("period start frames:", meta.period_start_frames)
rows = []
for rec in iter_tracking(1874553):
    if rec["timestamp"] is None or rec["frame"] % 6000:
        continue
    h, m, s = rec["timestamp"].split(":")
    rows.append({"frame": rec["frame"], "period": rec["period"], "file": rec["timestamp"],
                 "file_s": int(h)*3600 + int(m)*60 + float(s),
                 "derived_s": float(meta.clock_seconds([rec["frame"]], [rec["period"]])[0])})
chk = pd.DataFrame(rows)
chk["error_s"] = (chk.derived_s - chk.file_s).abs()
chk"""),
        ("md", "## Orientation\n\nJoin every player-possession event to the tracked position "
               "of its own player at `frame_start`, and measure the distance two ways."),
        ("code", """from src.preprocessing.build_tracking import load_tracking
ev = pd.read_parquet(PROCESSED_DIR / "events.parquet", filters=[("match_id", "==", 1874553)])
trk = load_tracking(1874553)
pp = ev[(ev.event_type == "player_possession") & ev.player_id.notna()]
key = pp[["frame_start", "player_id", "x_start", "y_start", "attack_sign"]].rename(
    columns={"frame_start": "frame"})
key["player_id"] = key.player_id.astype("int64")
m = key.merge(trk[["frame", "player_id", "x", "y"]], on=["frame", "player_id"])
m["raw_error_m"] = np.hypot(m.x_start - m.x, m.y_start - m.y)
m["flipped_error_m"] = np.hypot(m.x_start * m.attack_sign - m.x,
                                m.y_start * m.attack_sign - m.y)
m.groupby("attack_sign")[["raw_error_m", "flipped_error_m"]].median().round(4)"""),
        ("md", """For teams attacking +x the raw coordinates already match. For teams
attacking -x the raw coordinates are the mirror image, 57 m away, and only
match after the flip. This is the single most consequential fact in the
dataset and it is not documented upstream."""),
        ("code", """fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for a, col, title in zip(ax, ["raw_error_m", "flipped_error_m"],
                         ["as delivered", "x attack_sign"]):
    a.hist(m[col], bins=60, color=PALETTE["accent"] if "flip" in col else PALETTE["accent2"])
    a.set_title(f"event vs tracking position: {title}"); a.set_xlabel("error (m)")
ax[0].set_ylabel("events"); plt.tight_layout()"""),
        ("md", "## Why smoothing is not optional"),
        ("code", """from src.tracking.kinematics import smooth_and_differentiate
from src.config import DT
p1 = trk[(trk.period == 1) & (trk.position != "GK")]
busiest = p1.player_id.value_counts().index[0]
one = p1[p1.player_id == busiest].sort_values("frame").head(600)
raw_speed = np.hypot(one.x.diff(), one.y.diff()) / DT
k = smooth_and_differentiate(one.x.to_numpy("float64"), one.y.to_numpy("float64"),
                             np.ones(len(one), bool))
t = np.arange(len(one)) * DT
plt.figure(figsize=(10, 3))
plt.plot(t, raw_speed, lw=0.7, color=PALETTE["accent2"], label="raw first difference")
plt.plot(t, k["speed"], lw=1.6, color=PALETTE["accent"], label="Savitzky-Golay, 1.3 s")
plt.axhline(7.0, ls=":", lw=1, color=PALETTE["muted"])
plt.text(0.5, 7.2, "sprint threshold", color=PALETTE["muted"], fontsize=8)
plt.ylabel("speed (m/s)"); plt.xlabel("seconds"); plt.legend(frameon=False); plt.tight_layout()
print(f"raw:      mean {np.nanmean(raw_speed):.2f}  sd {np.nanstd(raw_speed):.2f}  "
      f"max {np.nanmax(raw_speed):.1f} m/s")
print(f"smoothed: mean {np.nanmean(k['speed']):.2f}  sd {np.nanstd(k['speed']):.2f}  "
      f"max {np.nanmax(k['speed']):.1f} m/s")"""),
        ("md", "Raw differencing produces speeds that no human reaches. The filter recovers a "
               "constant 6 m/s test signal to 6.001 +/- 0.043 m/s (`tests/test_kinematics.py`)."),
        ("md", "## The full validation report"),
        ("code", """from src.config import REPORTS_DIR
v = pd.read_parquet(REPORTS_DIR / "validation.parquet")
v.groupby(["name", "unit"]).value.agg(["min", "median", "max"]).round(4).head(45)"""),
    ],

    "03_possession_and_epv": [
        ("md", "# 03 · Possession chains and EPV\n\nWhat a possession is worth, and what makes "
               "it worth that."),
        ("code", HEAD),
        ("code", """poss = pd.read_parquet(DERIVED_DIR / "possessions.parquet")
tp = pd.read_parquet(DERIVED_DIR / "team_possessions.parquet")
print(f"{len(poss):,} player possessions in {len(tp):,} chains")
print(f"EPV populated on {poss.possession_epv_total.notna().mean():.0%} of possessions")
tp[["team", "duration_s", "n_player_possessions", "n_passes", "territory_gain_m",
    "epv_total", "lead_to_shot"]].describe().round(3)"""),
        ("md", "## Phases of play are not equally valuable"),
        ("code", """by = tp.groupby("team_in_possession_phase_type").agg(
    n=("epv_total", "size"), share=("epv_total", lambda s: len(s) / len(tp)),
    duration_s=("duration_s", "mean"), territory_m=("territory_gain_m", "mean"),
    shot_rate=("lead_to_shot", "mean"), epv=("epv_total", "mean")).sort_values("epv", ascending=False)
by.round(4)"""),
        ("code", """fig, ax = plt.subplots(figsize=(8, 3.4))
b = by.sort_values("epv")
ax.barh(b.index, b.epv, color=PALETTE["accent"])
for i, (v, s) in enumerate(zip(b.epv, b.share)):
    ax.text(v + 0.0008, i, f"{v:+.4f}   ({s:.0%} of chains)", va="center", fontsize=8,
            color=PALETTE["text_secondary"])
ax.set_xlabel("mean EPV per possession chain"); ax.set_xlim(right=b.epv.max() * 1.9)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout()"""),
        ("md", "Transitions and quick breaks are worth several times a build-up possession and "
               "make up under 4% of chains."),
        ("md", "## Pressure"),
        ("code", """order = ["no_pressure", "low_pressure", "medium_pressure", "high_pressure",
         "very_high_pressure"]
p = poss[poss.end_type == "pass"]
g = p.assign(ok=p.pass_outcome == "successful").groupby("overall_pressure_start").agg(
    n=("ok", "size"), completion=("ok", "mean"), pass_epv=("pass_epv_total", "mean"),
    nearest_opponent_m=("fil_nearest_opponent_m", "mean")).reindex(order)
g.round(4)"""),
        ("md", """`nearest_opponent_m` is computed here from tracking and never enters
SkillCorner's pressure model, so the monotone 11.8 m -> 1.3 m gradient is an
independent confirmation that the bands mean what they say."""),
        ("code", """fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
ax[0].bar(range(5), g.completion, color=PALETTE["accent"])
ax[0].set_xticks(range(5)); ax[0].set_xticklabels([o.replace("_", "\\n") for o in order], fontsize=7)
ax[0].set_ylabel("pass completion"); ax[0].set_ylim(0.5, 1.0)
ax[1].bar(range(5), g.pass_epv, color=PALETTE["good"])
ax[1].set_xticks(range(5)); ax[1].set_xticklabels([o.replace("_", "\\n") for o in order], fontsize=7)
ax[1].set_ylabel("mean pass EPV"); ax[1].axhline(0, color=PALETTE["line"], lw=1)
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()"""),
        ("md", "Completion falls monotonically. Value added does not: it peaks at *medium* "
               "pressure and only turns negative at very high pressure."),
    ],

    "04_team_shape_and_transitions": [
        ("md", "# 04 · Team shape and transitions\n\nWhat the tracking says about structure, "
               "and what changes in the eight seconds after a turnover."),
        ("code", HEAD),
        ("code", """tf = pd.read_parquet(DERIVED_DIR / "team_frames" / "1874553.parquet")
matches = pd.read_parquet(PROCESSED_DIR / "matches.parquet").set_index("match_id")
names = matches.loc[1874553]
tf["team"] = np.where(tf.team_id == names.home_team_id, names.home_team, names.away_team)
tf.groupby("team")[["width_m", "length_m", "compactness_m", "hull_area_m2",
                    "def_line_x", "att_line_x", "space_between_lines_m"]].mean().round(1)"""),
        ("md", "## Lines through the match"),
        ("code", """home = tf[tf.team == names.home_team].sort_values("frame")
sm = home.set_index(home.match_clock_s / 60)[["def_line_x", "mid_line_x", "att_line_x"]] \\
        .rolling(150, min_periods=1).mean()
plt.figure(figsize=(11, 3.2))
for c, col in zip(sm.columns, [PALETTE["accent"], PALETTE["accent3"], PALETTE["accent2"]]):
    plt.plot(sm.index, sm[c], lw=1.4, color=col, label=c.replace("_", " "))
plt.axvline(45, ls=":", color=PALETTE["line"]); plt.axhline(0, lw=0.8, color=PALETTE["line"])
plt.ylabel("x, own goal at -52.5 (m)"); plt.xlabel("match minute")
plt.title(f"{names.home_team} unit lines", loc="left"); plt.legend(frameon=False, ncol=3)
plt.tight_layout()"""),
        ("md", "## Shape when the ball is lost"),
        ("code", """tr = pd.read_parquet(DERIVED_DIR / "transitions.parquet").dropna(
    subset=["loser_compactness_start_m"])
q = pd.qcut(tr.loser_compactness_start_m, 5,
            labels=["most compact", "Q2", "Q3", "Q4", "most stretched"])
out = tr.assign(quintile=q).groupby("quintile", observed=True).agg(
    n=("max_progress_m", "size"), compactness_m=("loser_compactness_start_m", "mean"),
    opp_progress_m=("max_progress_m", "mean"), opp_hold_s=("held_possession_s", "mean"))
out.round(2)"""),
        ("code", """fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].bar(range(5), out.opp_progress_m, color=PALETTE["accent"])
ax[0].set_ylabel("opponent progression in 8 s (m)")
ax[1].bar(range(5), out.opp_hold_s, color=PALETTE["good"])
ax[1].set_ylabel("opponent hold time (s)")
for a in ax:
    a.set_xticks(range(5)); a.set_xticklabels(out.index, rotation=25, fontsize=7)
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()"""),
        ("md", """Teams that lose the ball while stretched concede **less** progression and get
it back sooner. A stretched shape is mostly a team committed high up the pitch,
and the counter-press that follows shortens the opponent's possession. This is
an association across 5,147 turnovers, not a licence to leave gaps."""),
    ],

    "05_physical": [
        ("md", "# 05 · Physical output\n\nWhat the tracking says about running, and whether it "
               "agrees with SkillCorner's own pipeline."),
        ("code", HEAD),
        ("code", """phys = pd.read_parquet(DERIVED_DIR / "player_match_physical.parquet")
out = phys[(phys.tracked_minutes >= 45) & (phys.position != "GK")]
out[["tracked_minutes", "coverage", "total_distance_m", "m_per_min", "hsr_distance_m",
     "sprint_count", "psv99_ms", "top_speed_ms", "detected_frac"]].describe().round(2)"""),
        ("md", "Coverage is the number to keep in view: the broadcast shows roughly 73% of the "
               "minutes a player was on the pitch, so these are distances *while visible*."),
        ("code", """fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
for grp, sub in out.groupby("position_group", observed=True):
    ax[0].scatter(sub.m_per_min, sub.hsr_distance_m / sub.total_distance_m, s=14,
                  alpha=0.8, label=grp)
ax[0].set_xlabel("metres per tracked minute"); ax[0].set_ylabel("share of distance at >=5.5 m/s")
ax[0].legend(fontsize=6, frameon=False)
ax[1].scatter(out.distance_tip_m, out.distance_otip_m, s=14, alpha=0.8, color=PALETTE["accent"])
lim = [0, max(out.distance_tip_m.max(), out.distance_otip_m.max())]
ax[1].plot(lim, lim, ls=":", color=PALETTE["muted"], lw=1)
ax[1].set_xlabel("distance in possession (m)"); ax[1].set_ylabel("distance out of possession (m)")
for a in ax: a.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
print(f"{(out.distance_otip_m > out.distance_tip_m).mean():.0%} of player-matches cover more "
      "ground without the ball than with it")"""),
        ("md", "## Cross-check against SkillCorner's own PSV99"),
        ("code", """from src.config import REPORTS_DIR
r = pd.read_parquet(REPORTS_DIR / "research" / "q8_psv99_validation.parquet").dropna(
    subset=["fil_psv99_ms", "sc_psv99_ms"])
print(f"n = {len(r)} players in both tables")
print(f"Pearson r = {np.corrcoef(r.fil_psv99_ms, r.sc_psv99_ms)[0, 1]:.2f}")
print(f"mean offset = {r.diff_ms.mean() * 3.6:+.1f} km/h")
plt.figure(figsize=(4.6, 4.4))
plt.scatter(r.sc_psv99_ms, r.fil_psv99_ms, s=16, alpha=0.8, color=PALETTE["accent"])
lim = [min(r.sc_psv99_ms.min(), r.fil_psv99_ms.min()) - 0.3,
       max(r.sc_psv99_ms.max(), r.fil_psv99_ms.max()) + 0.3]
plt.plot(lim, lim, ls=":", color=PALETTE["muted"], lw=1)
plt.xlabel("SkillCorner season PSV99 (m/s)"); plt.ylabel("this pipeline, per match (m/s)")
plt.gca().spines[["top", "right"]].set_visible(False); plt.tight_layout()"""),
        ("md", """The offset is expected: SkillCorner's figure is a season-level 99th percentile
from their full pipeline; this one covers one to three matches of broadcast
frames after deliberately conservative smoothing. The correlation is the
check, not the intercept."""),
    ],

    "06_body_pose": [
        ("md", """# 06 · 3D body pose

29 joints per player at 25 Hz. The full matches are on the Hugging Face Hub and
could not be downloaded on the network this was built on, so this runs on the
phase of play committed to the SkillCorner repository. The same code reads a
full match archive if one is present."""),
        ("code", HEAD),
        ("code", """from src.data.pose_loading import JOINTS, SKELETON, available_sources
print(available_sources())
joints = pd.read_parquet(DERIVED_DIR / "pose_joints.parquet")
orient = pd.read_parquet(DERIVED_DIR / "pose_orientation.parquet")
print(f"{len(joints):,} joint observations, {len(orient):,} player-frames with a pose")
print(f"{orient.player_id.nunique()} players, pose frames "
      f"{orient.pose_frame.min()}-{orient.pose_frame.max()}")
joints.groupby("joint", observed=True).p90_mae_cm.mean().sort_values().round(1)"""),
        ("md", "Error is smallest at the head and largest at the extremities, which is what a "
               "monocular pose estimator should do."),
        ("md", "## One skeleton"),
        ("code", """f = orient.pose_frame.iloc[len(orient) // 2]
pid = orient[orient.pose_frame == f].player_id.iloc[0]
jf = joints[(joints.pose_frame == f) & (joints.player_id == pid)].set_index("joint")
fig = plt.figure(figsize=(5, 5.4)); ax = fig.add_subplot(111, projection="3d")
ox, oy = jf.jx.mean(), jf.jy.mean()
for a, b in SKELETON:
    if a in jf.index and b in jf.index:
        ax.plot([jf.loc[a, "jx"] - ox, jf.loc[b, "jx"] - ox],
                [jf.loc[a, "jy"] - oy, jf.loc[b, "jy"] - oy],
                [jf.loc[a, "jz"], jf.loc[b, "jz"]], color=PALETTE["accent"], lw=2)
ax.scatter(jf.jx - ox, jf.jy - oy, jf.jz, s=12, color=PALETTE["text"])
ax.set_xlim(-0.9, 0.9); ax.set_ylim(-0.9, 0.9); ax.set_zlim(0, 2.1)
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
ax.set_title(f"player {pid}, pose frame {f}", fontsize=9); plt.tight_layout()"""),
        ("md", """## Does the orientation construction hold up?

The facing vector is the shoulder line rotated 90 degrees. If the sign were
inverted the check below would sit near 180 degrees instead of near 9."""),
        ("code", """from src.preprocessing.build_tracking import load_tracking
o = orient[orient.orientation_valid].copy()
o["frame"] = o.tracking_frame.round().astype(int)
trk = load_tracking(int(o.match_id.iloc[0]), columns=["frame", "player_id", "heading", "speed"])
m = o.merge(trk, on=["frame", "player_id"])
m["abs_diff_deg"] = np.degrees(np.abs(np.arctan2(np.sin(m.facing_rad - m.heading),
                                                 np.cos(m.facing_rad - m.heading))))
m["band"] = pd.cut(m.speed, [0, 1, 2, 4, 20], labels=["<1", "1-2", "2-4", ">4"])
m.groupby("band", observed=True).agg(
    n=("abs_diff_deg", "size"), median_deg=("abs_diff_deg", "median"),
    within_45=("abs_diff_deg", lambda s: (s < 45).mean())).round(2)"""),
        ("code", """fig, ax = plt.subplots(figsize=(7, 3))
for band, sub in m.groupby("band", observed=True):
    ax.hist(sub.abs_diff_deg, bins=36, histtype="step", lw=1.6, label=f"{band} m/s")
ax.set_xlabel("|pose facing - movement heading| (degrees)"); ax.set_ylabel("player-frames")
ax.legend(frameon=False, title="speed", title_fontsize=7, fontsize=7)
ax.spines[["top", "right"]].set_visible(False); plt.tight_layout()"""),
        ("md", """The agreement is monotone in speed: 9 degrees above 4 m/s, 69 degrees below
1 m/s. That is exactly the expected pattern -- a standing player has no
meaningful movement heading -- and it is what licenses using heading as a
proxy for orientation across all 20 matches, restricted to moving receivers."""),
        ("md", "## Orientation in this phase"),
        ("code", """orient.groupby("body_orientation_class", observed=True).agg(
    frames=("openness", "size"), mean_openness=("openness", "mean"),
    trunk_lean_deg=("trunk_lean_deg", "mean"),
    shoulder_hip_twist_deg=("shoulder_hip_twist_deg", "mean")).round(2)"""),
        ("md", "n is one 12.3-second phase of play. This is a demonstration of the measurement, "
               "not a positional norm."),
    ],
}


def build(name: str, cells: list[tuple[str, str]]) -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src)
        for kind, src in cells
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    path = OUT / f"{name}.ipynb"
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}).execute()
    nbf.write(nb, path)
    return path


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    only = sys.argv[1:] or list(NOTEBOOKS)
    for name in only:
        try:
            p = build(name, NOTEBOOKS[name])
            print(f"executed {p.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED {name}: {type(exc).__name__}: {exc}")
