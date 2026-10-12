# Research findings

Every number below is produced by `python -m src.research.questions` and read out of 
`reports/research/`. Nothing is transcribed by hand.

## How does pressure on the passer change what a pass is worth?

- Completion falls monotonically with pressure, from 93.6% unpressured to 64.3% under very high pressure (Spearman rho=-0.220, p=<1e-16, n=11,461).
- Pass EPV added does not fall monotonically: it peaks at medium pressure (+0.0028) and only turns negative at very high pressure (-0.0008). Overall rho=-0.062, p=7.12e-11.
- SkillCorner's pressure bands line up with tracking geometry that never saw them: mean distance to the nearest opponent runs 11.8 m -> 1.3 m across the five bands.

| overall_pressure_start   |    n |   completion |   pass_epv |   progression_m |   forward_share |   nearest_opponent_m |
|:-------------------------|-----:|-------------:|-----------:|----------------:|----------------:|---------------------:|
| no_pressure              | 3779 |       0.936  |     0.002  |          1.9165 |          0.2355 |              11.8252 |
| low_pressure             | 2617 |       0.8938 |     0.0024 |          1.044  |          0.206  |               7.1272 |
| medium_pressure          | 3002 |       0.8241 |     0.0028 |          1.354  |          0.2139 |               4.963  |
| high_pressure            | 1839 |       0.7232 |     0.0009 |          1.3746 |          0.1936 |               2.8054 |
| very_high_pressure       |  224 |       0.6429 |    -0.0008 |          1.6212 |          0.1295 |               1.304  |

## Which players keep creating value when they are pressed?

- 18 players had at least 60 possessions under medium-or-worse pressure across the 20 matches.
- Highest EPV added per pressed possession: D. Arzani (Melbourne V FC) +0.0114; N. Velupillay (Melbourne V FC) +0.0095; D. Pierias (Adelaide United) +0.0046; D. Leonard (Western United) +0.0034; G. May (Auckland FC) +0.0033
- Best pressure resilience (pressed EPV minus overall EPV): D. Arzani +0.0025; K. Bos +0.0019; D. Leonard +0.0019

|   player_id | player_name   | team_shortname   | player_position   |   possessions_pressed |   epv_total |   epv_per_possession |   retention |   progression_m |   mean_pressure |   epv_per_possession_all |   pressure_resilience |
|------------:|:--------------|:-----------------|:------------------|----------------------:|------------:|---------------------:|------------:|----------------:|----------------:|-------------------------:|----------------------:|
|       11885 | D. Arzani     | Melbourne V FC   | RW                |                    89 |      1.0058 |               0.0114 |      0.8539 |          2.1221 |          3.5281 |                   0.0089 |                0.0025 |
|       50955 | N. Velupillay | Melbourne V FC   | LW                |                    65 |      0.6172 |               0.0095 |      0.7846 |          4.2915 |          3.7077 |                   0.0086 |                0.0008 |
|       51005 | D. Pierias    | Adelaide United  | RB                |                   102 |      0.4652 |               0.0046 |      0.8627 |          4.1539 |          3.3922 |                   0.0038 |                0.0008 |
|      957737 | D. Leonard    | Western United   | LCB               |                    63 |      0.2131 |               0.0034 |      0.9206 |         -0.2781 |          3.5238 |                   0.0016 |                0.0019 |
|       38673 | G. May        | Auckland FC      | CF                |                   125 |      0.4122 |               0.0033 |      0.752  |          2.3087 |          3.688  |                   0.003  |                0.0003 |
|       11891 | A. Behich     | Melbourne City   | LB                |                    62 |      0.1956 |               0.0032 |      0.871  |          1.7795 |          3.3871 |                   0.0017 |                0.0014 |
|      811820 | K. Bos        | Melbourne V FC   | LB                |                    61 |      0.1735 |               0.0028 |      0.8689 |          1.0735 |          3.5082 |                   0.0009 |                0.0019 |
|      560917 | R. Bozinovski | Western United   | RM                |                    71 |      0.1904 |               0.0028 |      0.8451 |          1.1472 |          3.493  |                   0.0018 |                0.0009 |
|      795509 | C. Theoharous | CC Mariners      | LW                |                    65 |      0.1575 |               0.0024 |      0.7538 |          2.5237 |          3.6923 |                   0.0042 |               -0.0018 |
|       23922 | B. Garuccio   | Western United   | LB                |                   104 |      0.2087 |               0.002  |      0.8846 |          2.4443 |          3.3462 |                   0.0013 |                0.0007 |
|      133498 | F. De Vries   | Auckland FC      | LB                |                    90 |      0.1594 |               0.0018 |      0.9222 |          0.2835 |          3.5    |                   0.0012 |                0.0006 |
|      560916 | N. Botić      | Western United   | CF                |                    66 |      0.1133 |               0.0018 |      0.8636 |          2.251  |          3.8182 |                   0.0021 |               -0.0003 |
|       51675 | A. Thurgate   | Western United   | LM                |                    79 |      0.084  |               0.0011 |      0.9367 |          1.3034 |          3.4937 |                   0.0004 |                0.0007 |
|       43829 | N. Moreno     | Auckland FC      | LW                |                    66 |      0.0124 |               0.0002 |      0.803  |          2.1559 |          3.6818 |                   0.0005 |               -0.0003 |
|       14736 | L. Verstraete | Auckland FC      | RM                |                    78 |     -0.0696 |              -0.0009 |      0.859  |          1.1169 |          3.641  |                  -0.0002 |               -0.0007 |
|       50959 | S. Mauk       | Adelaide United  | AM                |                    62 |     -0.0919 |              -0.0015 |      0.7903 |          0.7735 |          3.6935 |                  -0.0011 |               -0.0004 |
|      133501 | L. Rogerson   | Auckland FC      | RW                |                    62 |     -0.1239 |              -0.002  |      0.8387 |          0.9283 |          3.6129 |                  -0.0002 |               -0.0018 |
|      560893 | E. Alagich    | Adelaide United  | LDM               |                    93 |     -0.2658 |              -0.0029 |      0.9355 |          0.3685 |          3.5054 |                  -0.0012 |               -0.0017 |

## Does the shape a team is in when it loses the ball predict what happens next?

- 5,146 changes of possession analysed across 20 matches.
- Teams that lose the ball while *stretched* (compactness 19.5 m) concede 12.6 m of progression in the next 8 s, versus 15.8 m for the most compact quintile (10.3 m). Spearman rho=-0.075, p=6.17e-08.
- The opponent also holds the ball for less time after a loss by a stretched team (7.2 s vs 10.4 s; rho=-0.169). A stretched shape is mostly a team committed high up the pitch, and the counter-press that follows shortens the opponent's possession -- this is an association in one league's 20 matches, not a demonstration that stretching out is safe.

| quintile   |    n |   compactness_m |   opp_max_progress_m |   opp_progress_8s_m |   opp_hold_time_s |   loser_recovery_m |
|:-----------|-----:|----------------:|---------------------:|--------------------:|------------------:|-------------------:|
| Q1         | 1030 |         10.2766 |              15.7917 |              5.606  |           10.3914 |            -2.0633 |
| Q2         | 1029 |         13.2433 |              15.9308 |              7.784  |           10.9868 |            -0.898  |
| Q3         | 1029 |         14.999  |              15.7334 |              7.4423 |           10.5499 |             0.0188 |
| Q4         | 1029 |         16.8476 |              15.0288 |              6.163  |            8.6637 |             1.0448 |
| Q5         | 1029 |         19.4977 |              12.6215 |              0.7626 |            7.1513 |             2.6444 |

## What is each phase of play actually worth?

- 7,523 team possessions, grouped by SkillCorner's in-possession phase type.
- `transition` possessions are worth 7x a build-up possession in EPV (+0.0309 vs +0.0045) and end in a shot 23% of the time, but make up only 2.5% of all possessions.
- Transition and quick-break possessions gain 48 m and 23 m of territory on average, against -0.1 m for build-up.

| team_in_possession_phase_type   |    n |   share |   duration_s |   territory_gain_m |   progression_speed_mps |   shot_rate |   goal_rate |   epv_total |   passes |
|:--------------------------------|-----:|--------:|-------------:|-------------------:|------------------------:|------------:|------------:|------------:|---------:|
| transition                      |  187 |  0.0249 |      10.4193 |            48.184  |                  4.8218 |      0.2299 |      0.0428 |      0.0309 |   2.4492 |
| quick_break                     |  105 |  0.014  |       7.3029 |            22.5791 |                  3.2107 |      0.4476 |      0.0476 |      0.0202 |   1.9905 |
| finish                          | 1507 |  0.2003 |       8.3721 |             6.5683 |                  0.9942 |      0.1958 |      0.0239 |      0.0046 |   2.6158 |
| build_up                        | 1073 |  0.1426 |       6.7284 |            -0.1462 |                 -0.0225 |      0.0075 |      0.0019 |      0.0045 |   2.4194 |
| chaotic                         | 1632 |  0.2169 |       1.5823 |             2.377  |                  1.149  |      0.0018 |      0      |      0.0029 |   0.9001 |
| direct                          |  496 |  0.0659 |       2.2103 |             3.6061 |                  0.7303 |      0.0141 |      0.004  |      0.0027 |   1.0101 |
| set_play                        |  245 |  0.0326 |       4.6776 |             7.7037 |                  0.4378 |      0.4122 |      0.0531 |      0.0023 |   1.5796 |
| create                          | 2278 |  0.3028 |       8.036  |             5.6945 |                  0.625  |      0.0413 |      0.0048 |      0.0023 |   2.9728 |

## Does the direction a receiver faces predict what the possession becomes?

- 6,007 pass receptions by a player moving faster than 2.0 m/s.
- Receiving while facing the opposition goal is worth 8.4 m of progression more than receiving facing your own (+6.5 m vs -1.9 m; Kruskal-Wallis H=2650, p=<1e-16).
- Forward-facing receptions lead to a shot 16.3% of the time against 2.7% for backward-facing ones, and add +0.0035 EPV against -0.0008 (H=88, p=<1e-16).
- The gap holds inside every pressure band, so it is not simply a proxy for being unmarked: high_pressure: +7.1 m; low_pressure: +10.3 m; medium_pressure: +8.6 m; no_pressure: +9.5 m; very_high_pressure: +7.6 m
- Direction of causation is not established. A player already running forward is often already in a transition, and transitions progress the ball for reasons that have nothing to do with body shape.

| orientation   |    n |   progression_m |     epv |   forward_pass_share |   lead_to_shot |   line_breaking |   mean_pressure_id |   retention |
|:--------------|-----:|----------------:|--------:|---------------------:|---------------:|----------------:|-------------------:|------------:|
| forward       | 2749 |          6.4554 |  0.0035 |               0.179  |         0.1626 |          0.3336 |             2.803  |      0.8785 |
| sideways      | 1749 |          1.5776 |  0.0001 |               0.1961 |         0.0617 |          0.0961 |             2.3998 |      0.9262 |
| backward      | 1509 |         -1.9013 | -0.0008 |               0.1093 |         0.0272 |          0.0139 |             2.497  |      0.943  |

## Which off-ball runs get found, and which are worth finding?

- 9,894 off-ball runs across 20 matches, in ten SkillCorner subtypes.
- `cross_receiver` runs carry the highest xThreat (0.104) and are targeted 31% of the time.
- `support` runs are the least likely to be found (28% targeted) despite an xThreat of 0.018.
- Runs that break the defensive line are 1.9x as likely to end in a shot as runs that do not.

| event_subtype         |    n |   targeted_rate |   received_rate |   dangerous_rate |   lead_to_shot |   xthreat |   distance_m |   speed_ms | line_break_rate   |   value_if_found |
|:----------------------|-----:|----------------:|----------------:|-----------------:|---------------:|----------:|-------------:|-----------:|:------------------|-----------------:|
| cross_receiver        |  928 |          0.3093 |          0.1272 |           0.9731 |         0.2759 |    0.1045 |      18.0073 |    19.3414 | <NA>              |           0.0323 |
| behind                |  741 |          0.4629 |          0.2294 |           0.6856 |         0.1984 |    0.0525 |      16.6229 |    20.7763 | 0.2254            |           0.0243 |
| run_ahead_of_the_ball | 2721 |          0.333  |          0.2554 |           0.3186 |         0.1136 |    0.0203 |      13.4498 |    18.881  | <NA>              |           0.0067 |
| underlap              |  228 |          0.3202 |          0.2281 |           0.3114 |         0.1404 |    0.0196 |      16.9275 |    20.2702 | <NA>              |           0.0063 |
| overlap               |  302 |          0.394  |          0.3444 |           0.3013 |         0.1854 |    0.0195 |      22.0495 |    21.0307 | <NA>              |           0.0077 |
| support               | 1623 |          0.2785 |          0.2372 |           0.2773 |         0.1516 |    0.0182 |      13.6767 |    18.4651 | <NA>              |           0.0051 |
| pulling_half_space    |  299 |          0.2843 |          0.2375 |           0.0502 |         0.0535 |    0.0082 |       9.9203 |    17.2561 | <NA>              |           0.0023 |
| pulling_wide          |  642 |          0.4548 |          0.405  |           0.0125 |         0.0358 |    0.0042 |      12.4443 |    17.741  | <NA>              |           0.0019 |
| coming_short          | 1312 |          0.3613 |          0.3377 |           0.0122 |         0.0213 |    0.0027 |       9.9338 |    17.0019 | <NA>              |           0.001  |
| dropping_off          | 1098 |          0.3934 |          0.3825 |           0.0064 |         0.0182 |    0.0012 |      10.1453 |    17.0908 | <NA>              |           0.0005 |

## Does a team's tactical shape explain its players' physical output?

- 40 team-matches, each with mean per-frame shape and mean per-player physical output.
- Strongest association: sprints vs width_m, rho=-0.35 (p=2.86e-02, n=40).
- With 40 team-matches these correlations are indicative only; the sign is more trustworthy than the magnitude.

| physical_metric   | shape_metric          |   spearman_rho |   p_value |   n |
|:------------------|:----------------------|---------------:|----------:|----:|
| sprints           | width_m               |        -0.3463 |    0.0286 |  40 |
| hsr_per_min       | width_m               |        -0.3366 |    0.0337 |  40 |
| sprints           | space_between_lines_m |        -0.2887 |    0.0708 |  40 |
| hsr_per_min       | length_m              |         0.2872 |    0.0723 |  40 |
| hsr_per_min       | space_between_lines_m |        -0.2775 |    0.083  |  40 |
| sprints           | length_m              |         0.2658 |    0.0974 |  40 |
| high_accels       | space_between_lines_m |        -0.2406 |    0.1349 |  40 |
| hsr_per_min       | def_line_x            |         0.2295 |    0.1544 |  40 |
| m_per_min         | space_between_lines_m |        -0.2248 |    0.1632 |  40 |
| sprints           | compactness_m         |        -0.1772 |    0.2739 |  40 |
| high_accels       | width_m               |        -0.1687 |    0.2981 |  40 |
| hsr_per_min       | compactness_m         |        -0.1548 |    0.3403 |  40 |
| sprints           | def_line_x            |         0.1354 |    0.4049 |  40 |
| m_per_min         | width_m               |        -0.1253 |    0.441  |  40 |
| high_accels       | def_line_x            |         0.1235 |    0.4478 |  40 |
| high_accels       | length_m              |         0.0613 |    0.7073 |  40 |
| high_accels       | compactness_m         |        -0.0381 |    0.8155 |  40 |
| m_per_min         | length_m              |         0.0246 |    0.8803 |  40 |
| m_per_min         | def_line_x            |         0.0163 |    0.9204 |  40 |
| m_per_min         | compactness_m         |         0.0118 |    0.9423 |  40 |

## Does this pipeline reproduce SkillCorner's own PSV99?

- 186 players appear both in this project's tracking-derived physical table and in SkillCorner's season physical aggregates.
- SkillCorner's PSV99 is published in km/h. Converted to m/s, it correlates with this project's per-match PSV99 at r=0.58 (p=<1e-16), with this pipeline reading -1.43 m/s (-5.1 km/h) lower on average.
- The gap is expected and is not a bug: SkillCorner's figure is a season-level 99th percentile over their full pipeline and all matches, while this one is computed from broadcast frames only, over 1-3 matches, after deliberately conservative smoothing (single-frame maxima correlate at rho=0.41, which is the noisier comparison).

|   player_id |   fil_psv99_ms |   fil_top_speed_ms |   matches | player_short_name   | position_group   |   psv99 |   psv99_top5 |   count_match |   sc_psv99_ms |   diff_ms |
|------------:|---------------:|-------------------:|----------:|:--------------------|:-----------------|--------:|-------------:|--------------:|--------------:|----------:|
|         211 |         6.2969 |             8.5092 |         1 | A. Taggart          | Center Forward   |   28.83 |        29.98 |            24 |        8.0083 |   -1.7115 |
|         218 |         6.518  |             8.5916 |         1 | A. Traoré           | Full Back        |   30.07 |        30.24 |             6 |        8.3528 |   -1.8348 |
|        2858 |         6.1517 |            11.0314 |         1 | Douglas Costa       | Wide Attacker    |   27.12 |        27.12 |             4 |        7.5333 |   -1.3817 |
|        4322 |         6.6592 |            11.8331 |         6 | H. Sakai            | Full Back        |   28.88 |        30.64 |            21 |        8.0222 |   -1.363  |
|        4792 |         6.2851 |             8.2783 |         1 | Javi López          | Full Back        |   27.6  |        27.94 |             6 |        7.6667 |   -1.3816 |
|        5468 |         6.0538 |             8.5583 |         2 | J. Brillante        | Midfield         |   27.11 |        28.78 |            20 |        7.5306 |   -1.4768 |
|        5521 |         5.9117 |             7.6288 |         1 | Juan Mata           | Midfield         |   25.73 |        25.73 |             3 |        7.1472 |   -1.2355 |
|        5805 |         5.5206 |             7.8894 |         1 | K. Boli             | Central Defender |   27.52 |        28.48 |             8 |        7.6444 |   -2.1238 |
|        6871 |         6.6466 |             9.4532 |         2 | M. Jakoliš          | Wide Attacker    |   29.93 |        31.68 |            20 |        8.3139 |   -1.6673 |
|        7252 |         6.4679 |             7.9549 |         1 | M. Leckie           | Midfield         |   28.83 |        29.3  |             6 |        8.0083 |   -1.5404 |
|        9175 |         5.9517 |             9.5786 |         2 | Roderick Miranda    | Central Defender |   28.42 |        30.1  |            28 |        7.8944 |   -1.9428 |
|        9520 |         5.5415 |             7.9765 |         3 | S. Souprayen        | Central Defender |   27.45 |        28.88 |            20 |        7.625  |   -2.0835 |
|       10549 |         6.2664 |            10.148  |         4 | T. Sainsbury        | Central Defender |   28.45 |        29.68 |            13 |        7.9028 |   -1.6363 |
|       10623 |         5.67   |             8.5208 |         1 | V. Germain          | Center Forward   |   28.26 |        30    |            16 |        7.85   |   -2.18   |
|       11117 |         6.7028 |            11.7546 |         3 | Z. Machach          | Midfield         |   28.37 |        30.5  |            25 |        7.8806 |   -1.1778 |
|       11885 |         7.0632 |             9.7014 |         3 | D. Arzani           | Wide Attacker    |   29.28 |        30.88 |            18 |        8.1333 |   -1.0701 |
|       11891 |         6.5068 |            11.9171 |         3 | A. Behich           | Full Back        |   29.52 |        31.24 |            25 |        8.2    |   -1.6932 |
|       11895 |         5.6643 |             7.5342 |         1 | M. Jurman           | Central Defender |   27.04 |        28.4  |            10 |        7.5111 |   -1.8468 |
|       11897 |         5.6108 |            10.3709 |         1 | J. Risdon           | Full Back        |   28.45 |        29.26 |            15 |        7.9028 |   -2.292  |
|       12185 |         5.9219 |             7.9609 |         1 | L. Amos             | Midfield         |   26.57 |        26.84 |             7 |        7.3806 |   -1.4587 |

## Does controlling more of the pitch translate into progressing the ball?

- 1,992 randomly sampled fully tracked frames across 20 matches, each scored with this project's pitch-control model and matched to how far the ball travelled towards goal over the following 5 s.
- Taken raw, control of the final third correlates *negatively* with subsequent progression (rho=-0.212). That is a ceiling artefact, not a finding: ball position alone correlates with future progression at rho=-0.294, because a ball already near the goal has nowhere left to travel.
- Conditioned on where the ball is, the relationship essentially disappears: own third rho=+0.048 (p=2.56e-01, 10.2 m top quintile vs 10.0 m bottom, n=570); middle third rho=-0.019 (p=5.71e-01, 2.6 m top quintile vs 4.3 m bottom, n=884); final third rho=-0.079 (p=6.90e-02, -4.6 m top quintile vs -3.3 m bottom, n=537)
- This is a null result and is reported as one. Instantaneous spatial control, as modelled here, does not predict how far a team moves the ball over the next five seconds once field position is accounted for. Two readings are available and this data cannot separate them: control may genuinely not be what drives progression at this time scale, or a two-parameter time-to-intercept model on broadcast tracking may be too coarse to detect it. What the analysis does establish is methodological -- any spatial-control metric must be conditioned on field position before it is related to progression, or the ceiling artefact will be mistaken for a finding.

| ball_zone    |   n |   mean_control_att_third |   rho_control_vs_progress |   p_value |   rho_control_vs_progress_share |   progress_top_quintile_m |   progress_bottom_quintile_m |
|:-------------|----:|-------------------------:|--------------------------:|----------:|--------------------------------:|--------------------------:|-----------------------------:|
| own third    | 570 |                   0.0258 |                    0.0476 |    0.2565 |                          0.0649 |                   10.2119 |                      10.0325 |
| middle third | 884 |                   0.1465 |                   -0.0191 |    0.5708 |                          0.0069 |                    2.624  |                       4.271  |
| final third  | 537 |                   0.3917 |                   -0.0785 |    0.069  |                         -0.0781 |                   -4.6367 |                      -3.281  |

## Does 3D body pose agree with movement heading, and by how much?

- 4,953 player-frames where 3D pose and tracking both resolve the same player (match 1925299, source: sample_phase).
- Above 4 m/s, pose-derived torso facing and tracking movement heading agree to a median of 9.0 degrees, and 95% of frames agree within 45 degrees. That is the check that makes the heading proxy in Q5 usable at scale.
- Below 1 m/s they diverge as expected (median 69 degrees): a nearly stationary player has no meaningful movement heading, which is exactly the case where only pose can answer the question.
- Mean per-joint p90 error over these frames is 12.2 cm; at the observed 0.37 m shoulder width that is roughly 18 degrees of orientation uncertainty, which is why only three coarse orientation classes are ever reported.

| speed_band   |    n |   median_abs_diff_deg |   p90_abs_diff_deg |   share_within_45deg |   mean_joint_err_cm |
|:-------------|-----:|----------------------:|-------------------:|---------------------:|--------------------:|
| <1 m/s       |  545 |               68.6646 |           141.6    |               0.2881 |             14.2238 |
| 1-2          | 1115 |               25.0771 |            98.745  |               0.6852 |             11.3901 |
| 2-4          | 2294 |               13.1342 |            87.3353 |               0.8492 |             11.8226 |
| >4           |  999 |                9.0418 |            26.9741 |               0.955  |             12.7544 |
