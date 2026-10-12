# Limitations

Written so that someone can tell, for any number in this project, whether it
will support the claim they want to make. Every figure below was measured by
`python -m src.preprocessing.validate` or by the test suite; none is an
estimate.

---

## 1. What broadcast tracking is

SkillCorner's open data is derived from **broadcast video**, not from a
stadium-mounted optical system. Three consequences run through everything here.

**The camera follows the ball.** About **27%** of frames in a match file carry
no tracking at all — replays, crowd shots, stoppages. Players are tracked for
roughly **73%** of the minutes they played (`coverage` on every physical row).

**Off-camera players are extrapolated.** **44%** of player-frames have
`is_detected = false`. Those positions are plausible and smooth; they are not
observations. Their speed distribution is measurably narrower than that of
detected frames, so physical metrics over a heavily extrapolated window read
**low**, not high.

**Occupancy maps over-represent the ball.** Any heat map built from tracked
frames is biased towards where the ball was, because that is where the camera
was pointing.

### What this means in practice

| you can | you cannot |
|---|---|
| compare players' distance *while visible* within a match | quote a match distance total |
| rank players by PSV99 | quote a peak speed as a personal best |
| compare team shape between phases | claim an average shape over 90 minutes |
| count sprints from tracked frames | say a player made N sprints in the match |

---

## 2. Player identity

SkillCorner's README states that **about 97% of player identities are
accurate**. This project cannot verify that figure and does not try to.

The obvious test — looking for teleports, where a player's position jumps
implausibly between frames — finds **zero** jumps over 2.5 m in a single 0.1 s
frame across all 20 matches, and a maximum single-frame step of 1.76 m. That is
*not* evidence the identities are right. The published file is extrapolated, so
an identity error is smoothed into a continuous track rather than left as a
visible jump. The 3% is real and simply not detectable from position
continuity.

**Consequence.** Any per-player number in this project carries an unknown
identity-error component of roughly that size. Team-level aggregates are
unaffected, because a swap between two players on the same team cancels.

---

## 3. Coverage gaps in the source

### EPV, pressure and reception difficulty are missing from four matches

Matches **1886347, 1899585, 1925299 and 2017461** carry **no** EPV, **no**
pressure band and **no** reception difficulty on any player-possession row.
That is **3,829 of 18,792 possessions (20%)**.

This is undocumented upstream and easy to miss, because the columns exist and
are simply empty. Anything averaging EPV must be aware of it. The app states
the coverage on the Possession page, and `tests/test_pipeline.py` pins the
exact set of matches so a change upstream cannot pass unnoticed.

One of the two body-pose matches (1925299) is in that set, so the pose case
study has no EPV attached to it.

### Other source quirks

| quirk | scale |
|---|---|
| `xthreat` / `xpass_completion` exist on `passing_option` and `off_ball_run` rows, never on `player_possession` (whose equivalents are `player_targeted_*`) | 48,431 vs 18,792 rows |
| `on_ball_engagement` rows carry a `time_start` up to ~4 s ahead of their own `frame_start` | 310 of 4,915 rows in the reference match |
| One event player is absent from the match team sheet (player 973077, match 1953632) | 35 of 94,517 event rows |
| `possession.player_id` is populated on only ~19% of frames (`possession.group` on ~55%) | all matches |
| Phases of play cover ~73% of tracked frames; they do not exist while the ball is out of play | all matches |
| Positions more than 5 m outside the touchline | ~49 rows per match, < 0.01% |
| Ball z reaches small negative values | reconstruction noise |

---

## 4. Coordinate orientation

**Dynamic event and phase coordinates are stored in the attacking frame of the
row's own team; tracking is in broadcast orientation.** This is not stated in
the SkillCorner README, and it is the single most consequential thing
discovered while building this project.

Before it was established, joining events to tracking produced a median
position error of 9.2 m and a 90th percentile of 80 m — half of every event
overlay was a mirror image, and it looked plausible enough on a pitch plot to
be missed. After the correction the same measurement is exactly **0.00 m**.

It is asserted in `tests/test_clock_and_orientation.py` in both directions: the
flipped coordinates must match tracking to 1 cm, *and* the unflipped ones must
be more than 20 m away for the half of events belonging to the team attacking
−x.

---

## 5. Kinematics

Speed and acceleration are model outputs, not measurements. They depend on the
smoothing window, and a different window would give different numbers.

* The 1.3 s Savitzky–Golay window attenuates accelerations shorter than about
  0.6 s. Acceleration counts are the metric most sensitive to this choice.
* Speeds above 12 m/s and accelerations beyond 8 m/s² are masked, which removes
  **under 0.02%** of player-frames. Masked frames contribute distance to the
  total but belong to no speed zone, so the zone sums are a lower bound on the
  total (median gap under 0.1%, worst case 1.4% for a player with very few
  tracked minutes).
* `top_speed_ms` is a single-frame maximum and is noisy by construction.
  `psv99_ms` is the number to use; both are stored so the difference is visible.
* Pitch rescaling to 105 × 68 changes distances by up to 1%. Speeds are computed
  before rescaling.

---

## 6. Pitch control

The pitch-control model in `src/tracking/pitch_control.py` is **this project's
own** and is not a SkillCorner metric. SkillCorner publishes pressure and
space-constraint bands; nothing in their open data is a control surface.

* Two parameters (0.7 s reaction, 8 m/s top speed) applied identically to every
  player, including goalkeepers.
* No acceleration profile, no ball flight time, no fatigue, no positional role.
* Known failure mode: it underestimates control for a player decelerating into
  a cell.
* It is computed on frames where all 22 players are tracked, but 44% of those
  positions are extrapolated.

**It also does not predict what one might hope.** Conditioned on where the ball
is, final-third control has no detectable relationship with how far the team
moves the ball over the next five seconds (|ρ| < 0.08, n = 1,992). Reported as
a null result on the Research page.

---

## 7. Body pose

### The full matches could not be downloaded

SkillCorner hosts the two full pose matches (~600 MB zipped, 3.3 GB of JSON
each) on the Hugging Face Hub. **The network this project was built on blocks
`huggingface.co`**, on both the build machine and the workstation, so the pose
work here runs on the **one phase of play committed to the SkillCorner
repository**: 12.3 seconds of Brisbane Roar v Perth Glory, 309 pose frames,
4,953 player-frames with joints, 143,637 joint observations.

`python -m src.data.download --bodypose` fetches the full matches on any
network that allows huggingface.co, and **every table, chart and test on the
pose path is built by the same code from whichever source is present** — the
loader picks the full archive when it exists and streams it out of the zip
without extracting 3.3 GB. Nothing needs changing to scale up.

### What that costs

* Positional and role breakdowns of orientation come from a single 12-second
  phase and are demonstrations of the measurement, not norms. They are labelled
  as such in the app.
* The orientation-at-reception study therefore runs on movement heading across
  all 20 matches, with pose used to establish how good that proxy is
  (§11 of `docs/METHODOLOGY.md`).

### What pose cannot tell you even at full scale

* **z is relative to the player's centroid**, per SkillCorner, not in the pitch
  frame. Trunk lean is reported as an angle within the pose and never as
  absolute posture.
* Mean per-joint p90 error here is **12.2 cm**, which at a 0.37 m shoulder width
  is about **18°** of orientation uncertainty. Only three coarse classes
  (forward / sideways / backward) are ever reported.
* Pose exists only for **detected** players, so it inherits the camera bias in
  a stronger form than tracking does.
* Torso orientation is not head or gaze direction. Nothing here says what a
  player could see.

---

## 8. Season aggregates

The three CSVs in `data/raw/aggregates/` are **per-match averages, not season
totals** — `minutes_full_all` averages 86 and tops out at 105. They are
filtered by SkillCorner to performances of 60+ minutes, so they are not the sum
of the match data elsewhere in this project and the two should never be added
together.

* Grain is (player, team, position group): 406 rows for 268 players.
* `psv99` is in **km/h**; everything computed from tracking here is in m/s.
* `_p30tip` columns are rates per 30 minutes of team possession, not counts.
* They cover the whole 2024/25 A-League, not only the 20 matches with tracking.

---

## 9. Statistical scope

* **20 matches, one league, one season.** Nothing here generalises without
  further evidence.
* Sample sizes are large enough that almost any effect is "significant". Effect
  sizes are what the text quotes.
* **Nothing in this project is causal.** The clearest example is orientation at
  reception: forward-facing receptions progress the ball 8.4 m further, but a
  player already running forward is often already in a transition, and
  transitions progress the ball for reasons unrelated to body shape.
* Team-level correlations (Q7) rest on 40 team-matches. The sign is more
  trustworthy than the magnitude.
* The player-similarity model describes statistical profile only. It is not a
  recruitment tool and says nothing about level, age or contract.

---

## 10. Engineering

* `data/processed/` is about 1 GB and is rebuilt from `data/raw/`; neither is
  committed.
* Pitch control is computed on demand rather than precomputed. A frame takes
  about 3 ms, so a single surface is instant, but an animation of 300 frames is
  not free.
* The Streamlit app caches per match. Switching matches re-reads a 44 MB
  Parquet file, which takes about a second the first time.
* Team-shape features are stored at the full 10 Hz (1.76 M rows across the
  dataset) and downsampled to 2 Hz in the app for charting.
