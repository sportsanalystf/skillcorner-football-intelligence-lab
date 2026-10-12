# Analytics reference

What is computed, where it lives, and which page of the application uses it.
Definitions and derivations are in `docs/METHODOLOGY.md`; the source schema is
in `docs/DATA_DICTIONARY.md`.

## Analytical tables

| table | grain | rows | contents |
|---|---|---:|---|
| `matches.parquet` | match | 20 | Fixture, score, pitch size, period frames, orientation |
| `squads.parquet` | match x player | 719 | Team sheets, roles, minutes, start/end frames |
| `events.parquet` | match x dynamic event | 94,517 | All 322 source columns plus clock, four coordinate spaces, typed booleans, global uids |
| `phases.parquet` | match x phase | 8,874 | Phase pairs with clock and normalised coordinates |
| `season_physical/passing/obr.parquet` | player x team x role | 406 | SkillCorner season aggregates, unchanged |
| `tracking/match_id=*/tracking.parquet` | match x frame x player | — | Smoothed position, velocity, speed, tangential acceleration, heading, step distance, speed zone, standard coordinates, attack sign |
| `ball/match_id=*/ball.parquet` | match x frame | — | Ball position and speed, possession flags, visible pitch area |
| `derived/team_frames/*.parquet` | match x frame x team | — | Centroid, width, length, hull area, compactness, unit lines, opponent context, third occupation |
| `derived/player_match_physical.parquet` | match x player | 623 | Distance, zones, sprints, accelerations, PSV99, coverage, in/out of possession split |
| `derived/sprints.parquet` | sprint | 3,784 | Every detected sprint with duration and peak speed |
| `derived/possessions.parquet` | match x player possession | 18,792 | SkillCorner possession columns plus tracking-derived fil_* spatial context |
| `derived/team_possessions.parquet` | match x phase | 7,523 | Possession chains: duration, passes, territory, EPV, pressure, outcome |
| `derived/passes.parquet` | pass | 16,334 | Every attempted pass with receiver, outcome, EPV, line breaks |
| `derived/off_ball_runs.parquet` | run | 9,894 | SkillCorner run events with subtype, xThreat, separation |
| `derived/pressures.parquet` | engagement | 17,400 | On-ball engagements with pressing chains |
| `derived/transitions.parquet` | turnover | 5,147 | Regain location, 8 s ball progress, both teams' shape response |
| `derived/player_season_profile.parquet` | player x team x role | 237 | 154 rate features joined across the three aggregate files |
| `derived/player_percentiles_position.parquet` | player profile | 237 | Every feature as a percentile within position group |
| `derived/player_similarity.parquet` | player profile x neighbour | 2,844 | Top-12 cosine neighbours |
| `derived/player_embedding.parquet` | player profile | 237 | Two-component PCA for the style map |
| `derived/pose_joints.parquet` | pose frame x player x joint | 143,637 | 29 joints in 3D with per-joint error |
| `derived/pose_orientation.parquet` | pose frame x player | 1,137 | Facing vector, openness, trunk lean, twist, stance, ball bearing |

Row counts marked — are partitioned per match. Across the dataset: **19,334,986**
player-frames of tracking, **1,244,814** ball frames and **1,757,726** team-frames.

## Metric families

| family | metrics | source | page |
|---|---|---|---|
| Kinematics | speed, tangential acceleration, heading, step distance, speed zone | derived from tracking | Match Explorer, Physical |
| Physical load | distance, zone distances, HSR, sprints, accel/decel counts, PSV99, m/min, in/out of possession split | derived from tracking | Physical |
| Team shape | centroid, width, length, hull area, compactness, defensive/midfield/attacking lines, space between lines, block overlap, third superiority | derived from tracking | Tactical Analysis |
| Pitch control | control surface, share of pitch, share of each third | **this project's model** | Tactical Analysis, Match Explorer |
| Possession value | EPV for/against, deltas, totals, pass EPV | SkillCorner, as delivered | Possession & EPV |
| Pressure and difficulty | overall pressure, reception difficulty, space constraint, time to impact, passing-option ease | SkillCorner, as delivered | Possession & EPV, Passing |
| Spatial context | nearest opponent, opponents/teammates within radii, opponents ahead in corridor, local superiority | derived from tracking (`fil_*`) | Match Explorer, Possession & EPV |
| Passing | volume, completion, expected completion, progression, line breaks, EPV added, networks | SkillCorner + derived | Passing |
| Off-ball movement | ten run subtypes, targeted/received rates, xThreat, separation gain, line breaks | SkillCorner, as delivered | Off-Ball & Pressing |
| Pressing | engagement subtypes, pressing chains, beaten rates, interplayer distance | SkillCorner, as delivered | Off-Ball & Pressing |
| Transitions | regain location, 8 s ball progress, hold time, shape recovery | derived from tracking | Off-Ball & Pressing |
| Player profile | 154 rate features, percentiles within role, cosine similarity, PCA embedding | SkillCorner season aggregates | Player Explorer |
| Body pose | 29 3D joints, facing vector, openness, trunk lean, shoulder–hip twist, stance width, ball bearing | derived from SkillCorner pose | 3D Body Pose |

## Application pages

| page | question it answers | data it needs that the page below cannot see |
|---|---|---|
| Overview | what is in the dataset | — |
| Match Explorer | what was happening at this exact moment | tracking + events + phases + control, on one clock |
| Tactical Analysis | what shape were they in, and who controlled the space | per-frame team geometry |
| Possession & EPV | which possessions were worth something | SkillCorner EPV and pressure |
| Passing | who moves the ball, and how hard were the passes | networks placed on tracked positions |
| Off-Ball & Pressing | what happened away from the ball | run and engagement events, turnover windows |
| Physical | what did it cost, and does the pipeline agree with SkillCorner | tracking kinematics + season aggregates |
| Player Explorer | how does this player compare, and who is like them | season aggregates, percentiles, similarity |
| 3D Body Pose | which way was the body actually facing | 29 joints in 3D |
| Research | what does the data actually support | all of the above |
| Data Quality | how much of it can be trusted | the validation report |

## Research questions

### Q1. How does pressure on the passer change what a pass is worth?

- Completion falls monotonically with pressure, from 93.6% unpressured to 64.3% under very high pressure (Spearman rho=-0.220, p=<1e-16, n=11,461).
- Pass EPV added does not fall monotonically: it peaks at medium pressure (+0.0028) and only turns negative at very high pressure (-0.0008). Overall rho=-0.062, p=7.12e-11.
- SkillCorner's pressure bands line up with tracking geometry that never saw them: mean distance to the nearest opponent runs 11.8 m -> 1.3 m across the five bands.

Evidence: `reports/research/q1_pressure_and_pass_value.parquet`

### Q2. Which players keep creating value when they are pressed?

- 18 players had at least 60 possessions under medium-or-worse pressure across the 20 matches.
- Highest EPV added per pressed possession: D. Arzani (Melbourne V FC) +0.0114; N. Velupillay (Melbourne V FC) +0.0095; D. Pierias (Adelaide United) +0.0046; D. Leonard (Western United) +0.0034; G. May (Auckland FC) +0.0033
- Best pressure resilience (pressed EPV minus overall EPV): D. Arzani +0.0025; K. Bos +0.0019; D. Leonard +0.0019

Evidence: `reports/research/q2_value_under_pressure.parquet`

### Q3. Does the shape a team is in when it loses the ball predict what happens next?

- 5,146 changes of possession analysed across 20 matches.
- Teams that lose the ball while *stretched* (compactness 19.5 m) concede 12.6 m of progression in the next 8 s, versus 15.8 m for the most compact quintile (10.3 m). Spearman rho=-0.075, p=6.17e-08.
- The opponent also holds the ball for less time after a loss by a stretched team (7.2 s vs 10.4 s; rho=-0.169). A stretched shape is mostly a team committed high up the pitch, and the counter-press that follows shortens the opponent's possession -- this is an association in one league's 20 matches, not a demonstration that stretching out is safe.

Evidence: `reports/research/q3_shape_at_loss.parquet`

### Q4. What is each phase of play actually worth?

- 7,523 team possessions, grouped by SkillCorner's in-possession phase type.
- `transition` possessions are worth 7x a build-up possession in EPV (+0.0309 vs +0.0045) and end in a shot 23% of the time, but make up only 2.5% of all possessions.
- Transition and quick-break possessions gain 48 m and 23 m of territory on average, against -0.1 m for build-up.

Evidence: `reports/research/q4_phase_value.parquet`

### Q5. Does the direction a receiver faces predict what the possession becomes?

- 6,007 pass receptions by a player moving faster than 2.0 m/s.
- Receiving while facing the opposition goal is worth 8.4 m of progression more than receiving facing your own (+6.5 m vs -1.9 m; Kruskal-Wallis H=2650, p=<1e-16).
- Forward-facing receptions lead to a shot 16.3% of the time against 2.7% for backward-facing ones, and add +0.0035 EPV against -0.0008 (H=88, p=<1e-16).
- The gap holds inside every pressure band, so it is not simply a proxy for being unmarked: high_pressure: +7.1 m; low_pressure: +10.3 m; medium_pressure: +8.6 m; no_pressure: +9.5 m; very_high_pressure: +7.6 m
- Direction of causation is not established. A player already running forward is often already in a transition, and transitions progress the ball for reasons that have nothing to do with body shape.

Evidence: `reports/research/q5_orientation_at_reception.parquet`

### Q6. Which off-ball runs get found, and which are worth finding?

- 9,894 off-ball runs across 20 matches, in ten SkillCorner subtypes.
- `cross_receiver` runs carry the highest xThreat (0.104) and are targeted 31% of the time.
- `support` runs are the least likely to be found (28% targeted) despite an xThreat of 0.018.
- Runs that break the defensive line are 1.9x as likely to end in a shot as runs that do not.

Evidence: `reports/research/q6_off_ball_runs.parquet`

### Q7. Does a team's tactical shape explain its players' physical output?

- 40 team-matches, each with mean per-frame shape and mean per-player physical output.
- Strongest association: sprints vs width_m, rho=-0.35 (p=2.86e-02, n=40).
- With 40 team-matches these correlations are indicative only; the sign is more trustworthy than the magnitude.

Evidence: `reports/research/q7_workload_vs_style.parquet`

### Q8. Does this pipeline reproduce SkillCorner's own PSV99?

- 186 players appear both in this project's tracking-derived physical table and in SkillCorner's season physical aggregates.
- SkillCorner's PSV99 is published in km/h. Converted to m/s, it correlates with this project's per-match PSV99 at r=0.58 (p=<1e-16), with this pipeline reading -1.43 m/s (-5.1 km/h) lower on average.
- The gap is expected and is not a bug: SkillCorner's figure is a season-level 99th percentile over their full pipeline and all matches, while this one is computed from broadcast frames only, over 1-3 matches, after deliberately conservative smoothing (single-frame maxima correlate at rho=0.41, which is the noisier comparison).

Evidence: `reports/research/q8_psv99_validation.parquet`

### Q9. Does controlling more of the pitch translate into progressing the ball?

- 1,992 randomly sampled fully tracked frames across 20 matches, each scored with this project's pitch-control model and matched to how far the ball travelled towards goal over the following 5 s.
- Taken raw, control of the final third correlates *negatively* with subsequent progression (rho=-0.212). That is a ceiling artefact, not a finding: ball position alone correlates with future progression at rho=-0.294, because a ball already near the goal has nowhere left to travel.
- Conditioned on where the ball is, the relationship essentially disappears: own third rho=+0.048 (p=2.56e-01, 10.2 m top quintile vs 10.0 m bottom, n=570); middle third rho=-0.019 (p=5.71e-01, 2.6 m top quintile vs 4.3 m bottom, n=884); final third rho=-0.079 (p=6.90e-02, -4.6 m top quintile vs -3.3 m bottom, n=537)
- This is a null result and is reported as one. Instantaneous spatial control, as modelled here, does not predict how far a team moves the ball over the next five seconds once field position is accounted for. Two readings are available and this data cannot separate them: control may genuinely not be what drives progression at this time scale, or a two-parameter time-to-intercept model on broadcast tracking may be too coarse to detect it. What the analysis does establish is methodological -- any spatial-control metric must be conditioned on field position before it is related to progression, or the ceiling artefact will be mistaken for a finding.

Evidence: `reports/research/q9_pitch_control_and_progression.parquet`

### Q10. Does 3D body pose agree with movement heading, and by how much?

- 4,953 player-frames where 3D pose and tracking both resolve the same player (match 1925299, source: sample_phase).
- Above 4 m/s, pose-derived torso facing and tracking movement heading agree to a median of 9.0 degrees, and 95% of frames agree within 45 degrees. That is the check that makes the heading proxy in Q5 usable at scale.
- Below 1 m/s they diverge as expected (median 69 degrees): a nearly stationary player has no meaningful movement heading, which is exactly the case where only pose can answer the question.
- Mean per-joint p90 error over these frames is 12.2 cm; at the observed 0.37 m shoulder width that is roughly 18 degrees of orientation uncertainty, which is why only three coarse orientation classes are ever reported.

Evidence: `reports/research/q10_pose_vs_heading.parquet`

