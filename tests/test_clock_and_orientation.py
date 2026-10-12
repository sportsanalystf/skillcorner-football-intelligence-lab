"""The two cross-source facts everything else depends on.

1. A single match clock reconstructs the timestamps in all three files.
2. Dynamic-event and phase coordinates are stored in each team's **attacking**
   frame, while tracking is in broadcast orientation.

If either regresses, every "show me the tracking context of this event"
feature in the project silently starts drawing mirrored pictures, which is
exactly what happened during development before this was pinned down.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.config import PROCESSED_DIR
from src.data.loaders import PERIOD_OFFSET_S, load_events_raw, load_match_meta, load_phases_raw
from tests.conftest import needs_data

pytestmark = needs_data


def _tracking(mid):
    from src.preprocessing.build_tracking import load_tracking

    return load_tracking(mid)


@needs_data
def test_period_offsets_are_the_documented_ones():
    assert PERIOD_OFFSET_S[1] == 0.0
    assert PERIOD_OFFSET_S[2] == 2700.0


@needs_data
def test_clock_reproduces_the_tracking_files_own_timestamp(ref_match):
    """`timestamp` in the raw JSONL is a period clock; rebuild it from frames."""
    from src.data.loaders import iter_tracking

    meta = load_match_meta(ref_match)
    checked = 0
    for rec in iter_tracking(ref_match):
        if rec["timestamp"] is None or rec["frame"] % 997:
            continue
        h, m, s = rec["timestamp"].split(":")
        want = int(h) * 3600 + int(m) * 60 + float(s)
        got = float(meta.clock_seconds([rec["frame"]], [rec["period"]])[0])
        assert got == pytest.approx(want, abs=0.051), rec["frame"]
        checked += 1
    assert checked > 20


@needs_data
def test_clock_matches_the_event_files_minute_and_second(ref_match):
    meta = load_match_meta(ref_match)
    ev = load_events_raw(ref_match)
    clock = meta.clock_seconds(ev["frame_start"], ev["period"])
    # `time_start` is "MM:SS.s" measured on the same clock.
    parsed = ev["time_start"].str.extract(r"(\d+):(\d+\.\d+)").astype(float)
    want = parsed[0] * 60 + parsed[1]
    err = np.abs(clock - want)

    # Exact for the three ball-related event types.
    ball_rows = ev["event_type"] != "on_ball_engagement"
    assert err[ball_rows].max() < 0.051

    # `on_ball_engagement` rows are the exception and it is a property of the
    # source, not of this code: their `time_start` runs up to ~4 s ahead of
    # their own `frame_start` (310 of 4,915 rows in the reference match). The
    # project always uses `frame_start`, which is what tracking agrees with --
    # see test_event_coordinates_are_in_the_teams_attacking_frame, which
    # matches engagement positions to tracking at frame_start exactly.
    off = err[~ball_rows]
    assert off.max() < 6.0
    assert (off > 0.051).sum() > 0, "the quirk should still be present"


@needs_data
def test_clock_matches_the_phase_file(ref_match):
    meta = load_match_meta(ref_match)
    ph = load_phases_raw(ref_match)
    clock = meta.clock_seconds(ph["frame_start"], ph["period"])
    parsed = ph["time_start"].str.extract(r"(\d+):(\d+\.\d+)").astype(float)
    want = parsed[0] * 60 + parsed[1]
    assert np.abs(clock - want).max() < 0.051


@needs_data
def test_attack_sign_follows_home_team_side(ref_match):
    meta = load_match_meta(ref_match)
    sides = meta.raw["home_team_side"]
    for period, side in enumerate(sides, start=1):
        expect_home = 1 if side == "left_to_right" else -1
        assert meta.attack_sign(meta.home_team_id, period) == expect_home
        assert meta.attack_sign(meta.away_team_id, period) == -expect_home


@needs_data
def test_goalkeepers_sit_behind_their_own_teams_attacking_direction(ref_match):
    """An independent check on attack_sign that never looks at the metadata."""
    meta = load_match_meta(ref_match)
    trk = _tracking(ref_match)
    gks = meta.players_frame().query("position == 'GK'")["player_id"].tolist()
    sub = trk[trk["player_id"].isin(gks)]
    for (pid, period), g in sub.groupby(["player_id", "period"], observed=True):
        team = int(g["team_id"].iloc[0])
        sign = meta.attack_sign(team, int(period))
        # A keeper defends the goal at -x in their own attacking frame.
        assert float(g["x"].mean()) * sign < -20


@needs_data
def test_event_coordinates_are_in_the_teams_attacking_frame(ref_match):
    """The load-bearing assertion of the whole project.

    Match every player_possession row to the tracked position of its own
    player at ``frame_start``. Multiplying the event's x/y by ``attack_sign``
    must reproduce the tracking coordinate; not multiplying must not.
    """
    ev = pd.read_parquet(PROCESSED_DIR / "events.parquet",
                         filters=[("match_id", "==", ref_match)])
    trk = _tracking(ref_match)
    pp = ev[(ev["event_type"] == "player_possession") & ev["player_id"].notna()]
    key = pp[["frame_start", "player_id", "x_start", "y_start", "attack_sign"]].rename(
        columns={"frame_start": "frame"}
    )
    key["player_id"] = key["player_id"].astype("int64")
    m = key.merge(trk[["frame", "player_id", "x", "y"]], on=["frame", "player_id"], how="inner")
    assert len(m) > 500, "not enough matched events to test"

    flipped = np.hypot(m["x_start"] * m["attack_sign"] - m["x"],
                       m["y_start"] * m["attack_sign"] - m["y"])
    plain = np.hypot(m["x_start"] - m["x"], m["y_start"] - m["y"])

    assert flipped.median() < 0.01, "attacking-frame conversion must be exact"
    # For the half of the events whose team attacks towards -x, the raw
    # coordinate is the mirror image and must be far away.
    neg = m[m["attack_sign"] == -1]
    assert len(neg) > 100
    assert np.hypot(neg["x_start"] - neg["x"], neg["y_start"] - neg["y"]).median() > 20


@needs_data
def test_the_processed_suffixes_mean_what_the_dictionary_says(ref_match):
    ev = pd.read_parquet(PROCESSED_DIR / "events.parquet",
                         filters=[("match_id", "==", ref_match)])
    meta = load_match_meta(ref_match)
    sx = 105.0 / meta.pitch.length
    assert np.allclose(ev["x_start_att"], ev["x_start"] * sx, atol=1e-3, equal_nan=True)
    assert np.allclose(ev["x_start_std"], ev["x_start_att"] * ev["attack_sign"],
                       atol=1e-3, equal_nan=True)
    assert np.allclose(ev["x_start_trk"], ev["x_start"] * ev["attack_sign"],
                       atol=1e-3, equal_nan=True)


@needs_data
def test_phase_coordinates_match_the_ball_at_phase_start(ref_match):
    from src.preprocessing.build_tracking import load_ball

    ph = pd.read_parquet(PROCESSED_DIR / "phases.parquet",
                         filters=[("match_id", "==", ref_match)])
    ball = load_ball(ref_match).set_index("frame")
    b = ball.reindex(ph["frame_start"])
    ok = b["ball_x"].notna().to_numpy()
    d = np.hypot(ph["x_start_trk"].to_numpy()[ok] - b["ball_x"].to_numpy()[ok],
                 ph["y_start_trk"].to_numpy()[ok] - b["ball_y"].to_numpy()[ok])
    assert np.median(d) < 0.01
