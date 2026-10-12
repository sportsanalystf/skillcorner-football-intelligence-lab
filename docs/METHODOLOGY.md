# Methodology

Every derived metric in this project is defined here: what it is, how it is
computed, what goes into it, what it assumes, and where it breaks. Metrics that
come from SkillCorner are used as delivered and are marked as such; metrics
invented here carry a `fil_` prefix in the data and are never presented as
SkillCorner measurements.

Constants referred to below all live in `src/config.py`, so any number in this
document can be traced to one line of code.

---

## 1. The match clock

**Definition.** A single continuous clock, in seconds from kick-off, shared by
tracking, dynamic events and phases of play.

```
period 1:  match_clock_s = (frame - period_1_start_frame) / 10
period 2:  match_clock_s = 2700 + (frame - period_2_start_frame) / 10
```

**Inputs.** `frame`, `period`, and `match_periods[].start_frame` from the match
JSON.

**Validation.** Reconstructs the tracking file's own `timestamp` string and the
event/phase `time_start` columns exactly (`tests/test_clock_and_orientation.py`).
The single exception is `on_ball_engagement` rows, whose `time_start` runs up to
4 s ahead of their own `frame_start` in the source; the project always uses
`frame_start`, which tracking agrees with to 5e-7 m.

**Assumption.** Period 2 is treated as starting at 45:00 regardless of how long
period 1 actually ran, which is what SkillCorner's own files do.

---

## 2. Coordinate systems

**Definition.** Four spaces, described in full in `docs/DATA_DICTIONARY.md` §3:

| space | orientation | pitch | suffix |
|---|---|---|---|
| raw tracking | broadcast, flips at half time | that match's metres | none |
| raw events / phases | **the row team's attacking direction** | that match's metres | none |
| standard | broadcast | 105 × 68 | `_std` |
| attacking | the team attacks +x | 105 × 68 | `_att` |
| tracking-joinable | broadcast | that match's metres | `_trk` |

**Method.** Rescale by `105 / pitch_length` and `68 / pitch_width`; flip by
`attack_sign ∈ {+1, −1}`, derived from `home_team_side[period]`.

**Validation.** Goalkeepers sit at x < −20 in their own team's attacking frame in
every period of every match. Event coordinates multiplied by `attack_sign`
reproduce the tracked position of the event's own player at `frame_start` to
5e-7 m; without the flip the median error for the half of events belonging to
the team attacking −x is 57 m.

**Limitation.** The 105/104 rescale changes distances by up to 1%. Speeds are
computed on native coordinates before rescaling, so they are unaffected.

---

## 3. Smoothing, velocity and acceleration

**Definition.** Player velocity, speed, heading and tangential acceleration
from broadcast tracking positions at 10 Hz.

**Method.**

1. Each player's track is placed on the full frame grid of their period, with
   missing frames as NaN.
2. Gaps of ≤ 10 frames (1.0 s) are linearly interpolated; longer gaps split the
   track into independent segments. Periods are always hard boundaries.
3. A Savitzky–Golay filter of order 2 over a 13-frame (1.3 s) window is fitted
   to x and y inside each segment. Position, velocity and acceleration are the
   filter's analytic 0th, 1st and 2nd derivatives, not finite differences of the
   smoothed series.
4. Speed is `|v|`. **Tangential** acceleration is `(v · a) / |v|`, i.e. the
   component of `dv/dt` along the direction of travel, so a hard turn at
   constant pace is not counted as an acceleration. `accel_mag = |a|` is kept
   separately.
5. Speeds > 12 m/s or |accel| > 8 m/s² are **masked to NaN, not clipped**.
   Clipping would add a plausible-looking 12 m/s to a distance total; masking
   leaves a hole the totals skip.
6. Step distance is computed between consecutive *smoothed* positions and
   discarded if the frames are not adjacent or the step exceeds 1.6 m.

**Parameters.** `SG_WINDOW_FRAMES = 13`, `SG_POLYORDER = 2`,
`MAX_PLAUSIBLE_SPEED_MS = 12.0`, `MAX_PLAUSIBLE_ACCEL_MS2 = 8.0`,
`MAX_PLAUSIBLE_STEP_M = 1.6`.

**Why these values.** 1.3 s is long enough to remove the sub-metre homography
jitter that dominates the raw signal and short enough to preserve a real
acceleration burst, which lasts about 1–2 s. On a synthetic 6 m/s track with
5 cm of positional noise, this recovers 6.001 m/s with a standard deviation of
0.043 m/s; raw first differencing on the same signal has a standard deviation
around 0.7 m/s. On a synthetic constant 2.5 m/s² ramp it recovers 2.50 m/s².
Both are asserted in `tests/test_kinematics.py`.

**Limitations.** A 1.3 s window attenuates accelerations shorter than about
0.6 s. Extrapolated positions (44% of player-frames) are smooth by
construction, so their speed distribution is visibly narrower than that of
detected positions — physical metrics over a heavily extrapolated window read
low, not high. The ball uses the same machinery with a 0.7 s window and much
wider gates (45 m/s, 400 m/s²), because a struck ball routinely exceeds every
human bound.

---

## 4. Physical output

**Definition.** Per player per match, over the frames in which that player was
tracked.

| metric | definition |
|---|---|
| `total_distance_m` | sum of gated step distances |
| `distance_<zone>_m` | the same, restricted to a speed band |
| `hsr_distance_m` | distance at ≥ 5.5 m/s |
| `sprint_distance_m` | distance at ≥ 7.0 m/s |
| `sprint_count` | runs holding ≥ 7.0 m/s for ≥ 0.7 s |
| `hsr_count` | runs holding ≥ 5.5 m/s for ≥ 0.7 s |
| `high_accel_count` | bouts of ≥ 3 m/s² held for ≥ 0.3 s |
| `psv99_ms` | 99th percentile of the smoothed speed |
| `top_speed_ms` | maximum smoothed speed (a single frame) |
| `m_per_min` | `total_distance_m / tracked_minutes` |
| `coverage` | `tracked_minutes / minutes_played` |
| `distance_tip_m` / `distance_otip_m` | split by the per-frame possession flag |

**Zone boundaries.** 1.5 / 3.5 / 5.5 / 7.0 m/s, chosen to match SkillCorner's
published physical glossary bands so per-match numbers are comparable with the
season aggregates.

**Assumption and limitation.** These are distances *while visible*. Broadcast
tracking covers about 73% of a match, so no number here is a match total, and
every row carries `tracked_minutes`, `coverage` and `detected_frac` so it can
be read correctly. Requiring an effort to be held for 0.7 s roughly halves the
sprint count relative to a single-frame threshold, and the removed entries all
last 0.1 s.

**Cross-check.** Against SkillCorner's own season PSV99 for the 186 players in
both tables: Pearson r = 0.58, with this pipeline reading 5.1 km/h lower on
average. The offset is expected (their figure is a season-level percentile from
their full pipeline; this one covers 1–3 matches of broadcast frames after
conservative smoothing). The correlation, not the intercept, is the check.

---

## 5. Team shape

**Definition.** Per frame per team, in that team's own attacking frame on the
standard pitch, **excluding the goalkeeper**.

| metric | definition |
|---|---|
| `centroid_x`, `centroid_y` | mean outfield position |
| `width_m`, `length_m` | max − min of y and x |
| `hull_area_m2` | area of the convex hull of the outfield players |
| `compactness_m` | mean distance from a player to the centroid |
| `def_line_x` | mean x of the 4 deepest outfielders |
| `att_line_x` | mean x of the 4 highest outfielders |
| `mid_line_x` | median outfield x |
| `vertical_compactness_m` | `att_line_x − def_line_x` |
| `space_between_lines_m` | opponent's defensive line minus our attacking line |
| `block_overlap_m` | our centroid minus theirs, in our frame. Negative = the two blocks sit apart, each in its own half; positive = they overlap toward one goal |
| `superiority_att_third` | our players in our attacking third minus theirs |

**Why the goalkeeper is excluded.** A keeper 40 m behind the block makes team
length a measure of nothing. Every shape metric here would be dominated by the
keeper's position if included.

**Frame of reference.** Both teams' metrics are stored in their *own* attacking
frames. Converting an opponent's line into our frame is a sign flip, which
`add_opponent_context` performs and `tests/test_features.py` asserts.

**Limitations.** Only frames with ≥ 6 tracked outfielders produce a row. The
convex hull is sensitive to a single mis-tracked player; `compactness_m` (mean
distance to centroid) is the robust alternative and both are stored.

---

## 6. Pitch control — **this project's model, not SkillCorner's**

**Definition.** For each cell of a 60 × 40 grid, the probability that the home
team reaches it first.

**Method.** Each player continues at their current velocity for a reaction
period, then runs in a straight line at a fixed maximum speed:

```
t(p, c) = reaction + |c − (x_p + v_p · reaction)| / v_max
P_home(c) = sigmoid( (min_away t − min_home t) / σ )
```

**Parameters.** `reaction = 0.7 s`, `v_max = 8.0 m/s`, `σ = 0.45 s`,
grid 60 × 40 (1.75 m × 1.7 m cells).

**Why so simple.** Broadcast tracking extrapolates 44% of positions and gives
no acceleration trustworthy at 0.1 s. A model with per-player speed profiles
fitted to that data would be more precise about numbers that are not there.
This version has two physical parameters, both stated, and one failure mode
that is easy to describe: it underestimates control for a player decelerating
into a cell.

**Validation.** Exactly 0.5 everywhere on a mirrored configuration;
antisymmetric under swapping the teams; > 0.99 in the cell a lone player stands
in; velocity moves control ahead of the player. All asserted in
`tests/test_pitch_control.py`.

**What it does not do.** Once ball position is controlled for, this model's
final-third control does **not** predict how far a team moves the ball over the
next five seconds (ρ between −0.08 and +0.05 within each third of the pitch,
n = 1,992). That null result is reported on the Research page rather than
buried. Two readings are available and this data cannot separate them: control
may genuinely not drive progression at this time scale, or a two-parameter
model on broadcast tracking may be too coarse to detect it.

---

## 7. Possessions, chains and transitions

**Player possession** is SkillCorner's `player_possession` event, used as
delivered. This project adds tracking-derived context at `frame_start`, all
prefixed `fil_`:

| metric | definition |
|---|---|
| `fil_nearest_opponent_m` | distance to the closest outfield opponent |
| `fil_opponents_within_3m/5m/10m` | counts inside those radii |
| `fil_opponents_ahead` | opponents between the carrier and the attacked goal |
| `fil_opponents_ahead_in_corridor` | the same, within ±10 m laterally |
| `fil_teammates_within_10m`, `fil_teammates_ahead` | the mirror images |
| `fil_local_superiority_10m` | teammates minus opponents within 10 m |
| `fil_is_progressive` | a pass that gains ≥ 5 m towards goal |
| `fil_under_pressure` | SkillCorner pressure band ≥ 4 (high or very high) |

Goalkeepers are excluded from the opponent counts. 0.3% of possessions fall in
an untracked frame and get NaN rather than a guess.

**Possession chain** = all player possessions sharing a `phase_index`. Duration,
territory gained, passes, players involved, line breaks, EPV and pressure are
aggregated over the chain; `speed_of_progression_mps` is territory gained
divided by duration.

**Transition** = a change in the per-frame `possession.group` flag. For each,
the ball's progress towards the winning team's goal is measured over the
following 8 s, together with both teams' shape at the moment of the turnover
and 8 s later.

**Limitation.** Transitions are defined from SkillCorner's possession flag,
which is populated on about 55% of frames. A turnover during an untracked
passage is invisible.

---

## 8. EPV and pressure

Both are **SkillCorner's**, used exactly as delivered and never recomputed.
EPV appears on `player_possession` rows only. Four of the twenty matches
(1886347, 1899585, 1925299, 2017461) carry no EPV, pressure or reception
difficulty at all — 3,829 of 18,792 possessions. Every average is taken over
the rows that carry a value; the app states the coverage on the page.

An independent check that the pressure bands mean what they say: the
tracking-derived distance to the nearest opponent, which never enters
SkillCorner's model, runs monotonically 11.8 m → 7.1 m → 4.9 m → 2.8 m → 1.3 m
across `no_pressure` → `very_high_pressure`.

---

## 9. Passing networks

Nodes sit at each player's **mean tracked position over the window**, not at
the average position of their passes, which is what most passing networks use
and which biases every node towards where that player happened to touch the
ball. Node size is total involvement (passes made plus received); edge width is
pass volume; edges below 2 passes are dropped. The default window ends at the
team's first substitution, because a network spanning a formation change is a
picture of two different teams.

---

## 10. Body orientation from 3D pose

**Definition.** The direction the torso is square to, in the pitch plane.

**Method.** Let `s = lShoulder − rShoulder` in (x, y). The facing vector is `s`
rotated 90° anticlockwise: for a player facing +x the left shoulder sits at +y,
so `s` points at +y and `R(90°)·ŝ = (+1, 0)`. The hips give a second, entirely
independent estimate from `lHip − rHip`. The two are averaged when they agree
to within 90° and the shoulder estimate is used otherwise, because a twisted
torso is real and the shoulders are what a receiving player turns first.

`facing_goal_deg` is the angle between that vector and the goal the player's
team is attacking; `openness = cos(facing_goal_deg)`; the three reported classes
are forward (< 60°), sideways (60–120°) and backward (> 120°).

**Validation.** Two independent checks.

1. *Synthetic.* Six constructed poses at known angles are recovered to 1e-6°,
   which pins the sign convention (`tests/test_features.py`).
2. *Empirical.* Against tracking movement heading on 4,953 real player-frames:
   median disagreement 9.0° above 4 m/s (95% within 45°), 13.1° at 2–4 m/s,
   25.1° at 1–2 m/s and 68.7° below 1 m/s. The gradient is monotonic in speed,
   which is what the construction predicts; an inverted rotation would put the
   fast bands near 180°.

**Error budget.** Mean per-joint p90 error over these frames is 12.2 cm at an
observed 0.37 m shoulder width, i.e. roughly 18° of orientation uncertainty.
That is why only three coarse classes are ever reported and why no claim is
made about differences of 10°.

**Limitations.** Pose exists only for detected players, and only for two
matches (one phase of which is committed to the repository — see
`docs/LIMITATIONS.md`). SkillCorner state that z is accurate relative to the
player's centroid and not in the pitch frame, so trunk lean is reported as an
angle *within the pose* and never as absolute posture. Orientation says nothing
about head or gaze direction.

---

## 11. Orientation at reception, at scale

Because pose covers one phase of play, the large-sample version of the
orientation question uses **movement heading** from smoothed tracking, and says
so everywhere it appears. The sample is restricted to receivers moving faster
than 2 m/s, which is the regime where §10's empirical check shows heading
approximates torso facing to a median of 9–13°. Below that speed the proxy is
measured to fail (25° at 1–2 m/s, 69° below 1 m/s) and those receptions are
excluded rather than assumed.

The result (6,007 receptions): forward-facing receptions gain 6.5 m and lead to
a shot 16.3% of the time; backward-facing receptions lose 1.9 m and lead to a
shot 2.7% of the time. The gap holds inside every pressure band, so it is not
simply a proxy for being unmarked. **Causation is not established**: a player
already running forward is often already in a transition, and transitions
progress the ball for reasons that have nothing to do with body shape.

---

## 12. Player similarity

**Grain.** (player, team, position group) — 406 rows for 268 players, because a
player who appeared in two roles has a genuinely different profile in each.
Filtered to profiles with ≥ 4 included performances, leaving 237.

**Features.** 154 rate-based columns only: physical per-90, passing per-30
minutes in possession, off-ball runs per-30. Raw totals are excluded because
they would make the model a minutes-played detector.

**Transform.** Median-impute (after dropping features missing for > 20% of
players) → rank to a league percentile → standardise. Ranking removes the long
right tails several of these metrics have.

**Similarity.** Cosine similarity in the standardised space — not in PCA space,
so no information is dropped. PCA is used only for the two-dimensional style
map; 40 components are needed for 90% of the variance, which is itself the
argument for not computing similarity in two of them.

**What this is not.** A description of statistical profile, not a scouting
recommendation. It knows nothing about level of opposition, age, contract,
injury history, or whether a player's numbers are a product of their team's
style.

---

## 13. Statistical conventions

* Rank-based tests throughout (Spearman, Kruskal–Wallis): EPV deltas,
  progression and compactness are heavy-tailed and often zero-inflated.
* Sample sizes run from 1,137 player-frames to 18,792 possessions, so p-values
  are almost always tiny. **Effect sizes are what is quoted**; p-values are
  reported for completeness and never as the headline.
* Quintile tables are shown alongside correlations so the shape of a
  relationship is visible, not just its sign.
* Null results are reported as null results.
* All of it is 20 matches from one league in one season. Nothing here is a
  league-wide law, and none of it is causal.
