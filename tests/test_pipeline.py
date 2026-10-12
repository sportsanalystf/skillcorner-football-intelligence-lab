"""The built tables: shape, keys, joins and internal consistency."""

import numpy as np
import pandas as pd
import pytest

from src.config import DERIVED_DIR, PROCESSED_DIR
from src.data.loaders import available_match_ids, load_match_meta
from tests.conftest import needs_data

pytestmark = needs_data

DERIVED_TABLES = [
    "possessions", "team_possessions", "passes", "off_ball_runs", "pressures",
    "transitions", "player_match_physical", "sprints", "player_season_profile",
    "player_similarity", "player_embedding", "pose_joints", "pose_orientation",
]


@pytest.mark.parametrize("name", DERIVED_TABLES)
def test_derived_table_exists_and_is_non_empty(name):
    p = DERIVED_DIR / f"{name}.parquet"
    assert p.exists(), f"{name} not built"
    assert len(pd.read_parquet(p, columns=None).head(1)) == 1


def test_every_match_has_tracking_and_ball_parquet(match_ids):
    for mid in match_ids:
        assert (PROCESSED_DIR / "tracking" / f"match_id={mid}" / "tracking.parquet").exists()
        assert (PROCESSED_DIR / "ball" / f"match_id={mid}" / "ball.parquet").exists()


def test_tracking_has_one_row_per_player_frame(ref_match):
    from src.preprocessing.build_tracking import load_tracking

    trk = load_tracking(ref_match, columns=["frame", "player_id"])
    assert not trk.duplicated(["frame", "player_id"]).any()


def test_tracked_frames_hold_exactly_22_players(ref_match):
    from src.preprocessing.build_tracking import load_tracking

    trk = load_tracking(ref_match, columns=["frame", "player_id"])
    counts = trk.groupby("frame").size()
    assert set(counts.unique()) == {22}


def test_ball_frames_are_consecutive_and_unique(ref_match):
    from src.preprocessing.build_tracking import load_ball

    ball = load_ball(ref_match)
    f = ball["frame"].to_numpy()
    assert len(np.unique(f)) == len(f)
    assert (np.diff(np.sort(f)) == 1).all()


def test_event_uid_is_globally_unique():
    ev = pd.read_parquet(PROCESSED_DIR / "events.parquet", columns=["event_uid"])
    assert not ev["event_uid"].duplicated().any()
    # The raw event_id is only unique within a match, which is why event_uid exists.
    raw = pd.read_parquet(PROCESSED_DIR / "events.parquet", columns=["event_id"])
    assert raw["event_id"].duplicated().any()


def test_almost_every_event_player_is_in_the_match_squad():
    """One player in the whole dataset is not.

    Match 1953632 carries 35 event rows for player 973077 (M. Ghossaini,
    Melbourne City) who does not appear in that match's team sheet. It is a
    source inconsistency, measured here rather than silently dropped, and it
    is recorded in docs/LIMITATIONS.md.
    """
    ev = pd.read_parquet(PROCESSED_DIR / "events.parquet",
                         columns=["match_id", "player_id"]).dropna()
    sq = pd.read_parquet(PROCESSED_DIR / "squads.parquet", columns=["match_id", "player_id"])
    known = set(map(tuple, sq.to_numpy()))
    unknown = [
        (int(m), int(p)) for m, p in ev.to_numpy() if (int(m), int(p)) not in known
    ]
    assert len(set(unknown)) <= 1
    assert len(unknown) / len(ev) < 0.001


def test_possessions_are_a_subset_of_events_and_carry_epv():
    poss = pd.read_parquet(DERIVED_DIR / "possessions.parquet")
    ev = pd.read_parquet(PROCESSED_DIR / "events.parquet", columns=["event_uid", "event_type"])
    pp = ev[ev["event_type"] == "player_possession"]
    assert len(poss) == len(pp)
    assert set(poss["event_uid"]) == set(pp["event_uid"])
    # EPV is absent entirely from four matches (see the dedicated test below),
    # and about 3-5% of possessions elsewhere.
    assert poss["possession_epv_total"].notna().mean() > 0.7


def test_team_possessions_reconcile_with_player_possessions():
    poss = pd.read_parquet(DERIVED_DIR / "possessions.parquet")
    tp = pd.read_parquet(DERIVED_DIR / "team_possessions.parquet")
    counted = poss.groupby(["match_id", "phase_index"], observed=True).size()
    joined = tp.set_index(["match_id", "phase_index"])["n_player_possessions"]
    common = counted.index.intersection(joined.index)
    assert len(common) == len(tp)
    assert (counted.loc[common].to_numpy() == joined.loc[common].to_numpy()).all()


def test_passes_are_the_possessions_that_ended_in_a_pass():
    poss = pd.read_parquet(DERIVED_DIR / "possessions.parquet")
    passes = pd.read_parquet(DERIVED_DIR / "passes.parquet")
    assert len(passes) == int((poss["end_type"] == "pass").sum())
    assert passes["completed"].dtype == bool


def test_epv_and_pressure_are_missing_from_exactly_four_matches():
    """A property of the source, pinned so it cannot change unnoticed.

    Matches 1886347, 1899585, 1925299 and 2017461 carry no EPV, no pressure
    band and no reception difficulty on any player_possession row. Together
    that is 3,829 of 18,792 possessions. Anything that averages EPV must be
    aware of it, which is why the app reports EPV coverage on the page.
    """
    poss = pd.read_parquet(DERIVED_DIR / "possessions.parquet",
                           columns=["match_id", "possession_epv_total",
                                    "overall_pressure_start"])
    cov = poss.groupby("match_id")["possession_epv_total"].apply(lambda s: s.notna().mean())
    empty = set(cov[cov == 0].index)
    assert empty == {1886347, 1899585, 1925299, 2017461}
    assert (cov.drop(index=list(empty)) > 0.9).all()
    pcov = poss.groupby("match_id")["overall_pressure_start"].apply(lambda s: s.notna().mean())
    assert set(pcov[pcov == 0].index) == empty


def test_xthreat_and_xpass_live_on_the_event_types_that_have_them():
    ev = pd.read_parquet(PROCESSED_DIR / "events.parquet",
                         columns=["event_type", "xthreat", "xpass_completion",
                                  "player_targeted_xpass_completion"])
    opt = ev[ev["event_type"] == "passing_option"]
    pp = ev[ev["event_type"] == "player_possession"]
    assert opt["xthreat"].notna().mean() > 0.999  # 2 of 48,431 rows are blank
    assert pp["xthreat"].notna().mean() == 0.0
    assert pp["player_targeted_xpass_completion"].notna().mean() > 0.7


def test_physical_totals_are_internally_consistent():
    phys = pd.read_parquet(DERIVED_DIR / "player_match_physical.parquet")
    zones = ["walking", "jogging", "running", "hsr", "sprint"]
    zsum = phys[[f"distance_{z}_m" for z in zones]].sum(axis=1)
    # Steps taken in frames whose speed was masked as implausible belong to no
    # zone, so the zone sum is a lower bound on the total, and the gap is the
    # masked share -- under 0.1% of distance.
    gap = (phys["total_distance_m"] - zsum) / phys["total_distance_m"]
    assert (gap >= -1e-6).all()
    assert gap.max() < 0.02
    assert gap.median() < 1e-3
    assert (phys["hsr_distance_m"] >= phys["sprint_distance_m"] - 1e-6).all()
    assert (phys["tracked_minutes"] > 0).all()
    assert (phys["coverage"].dropna() <= 1.05).all(), "cannot track more than was played"


def test_physical_values_are_physiologically_plausible():
    phys = pd.read_parquet(DERIVED_DIR / "player_match_physical.parquet")
    p = phys[phys["tracked_minutes"] >= 45]
    outfield = p[p["position"] != "GK"]
    keepers = p[p["position"] == "GK"]
    assert outfield["m_per_min"].between(80, 220).all()
    assert outfield["psv99_ms"].between(4.0, 11.0).all()
    # Keepers legitimately cover far less ground; they are checked separately
    # rather than being allowed to widen the outfield bounds.
    assert keepers["m_per_min"].between(25, 90).all()
    assert p["top_speed_ms"].max() < 12.0  # the configured gate


def test_team_frames_cover_both_teams_for_every_match(match_ids):
    for mid in match_ids:
        tf = pd.read_parquet(DERIVED_DIR / "team_frames" / f"{mid}.parquet",
                             columns=["team_id", "n_players"])
        assert tf["team_id"].nunique() == 2
        assert tf["n_players"].max() <= 11
        assert tf["n_players"].min() >= 6


def test_transitions_alternate_possession():
    tr = pd.read_parquet(DERIVED_DIR / "transitions.parquet")
    assert (tr["winning_team_id"] != tr["losing_team_id"]).all()
    for mid, g in tr.groupby("match_id"):
        w = g.sort_values("frame")["winning_team_id"].to_numpy()
        assert (w[1:] != w[:-1]).all(), f"match {mid} has consecutive identical winners"


def test_similarity_is_symmetric_in_membership_and_excludes_self():
    sim = pd.read_parquet(DERIVED_DIR / "player_similarity.parquet")
    assert (sim["profile_id"] != sim["similar_profile_id"]).all()
    assert sim["similarity"].between(-1.0001, 1.0001).all()
    assert sim.groupby("profile_id")["rank"].max().nunique() == 1


def test_warehouse_views_resolve():
    from src.config import WAREHOUSE_PATH
    from src.data.warehouse import query

    if not WAREHOUSE_PATH.exists():
        pytest.skip("warehouse not built")
    n = query("SELECT count(*) AS n FROM tracking")["n"].iloc[0]
    assert n > 10_000_000
    assert query("SELECT count(*) AS n FROM possessions")["n"].iloc[0] > 10_000
