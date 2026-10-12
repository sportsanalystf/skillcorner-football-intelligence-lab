"""Player style similarity from the season aggregate files.

Scope
-----
The similarity model uses the **season aggregates** (physical, passing,
off-ball runs) rather than the 20 tracking matches, because the aggregates
cover 400+ players across the whole 2024/25 A-League while the tracking
matches give at most a couple of appearances each. A similarity model built on
one or two matches per player would mostly measure fixture, not style.

A property of these files that is easy to get wrong: they are **per-match
averages, not season totals**. ``minutes_full_all`` averages 86 and tops out
at 105; ``total_distance_full_all`` averages 10.5 km. ``count_match`` /
``performance_included_count`` is how many performances went into the average,
and SkillCorner only includes performances of 60+ minutes. ``psv99`` is in
**km/h**, unlike everything this project computes from tracking, which is in
m/s.

The grain is **(player_id, team_id, position_group)**, not player: a player who
appeared in two position groups has two rows, each averaging only the matches
in that role. 268 distinct players produce 406 rows. That grain is kept here,
because "N. Player as a full back" and "N. Player as a centre back" really are
different statistical profiles; ``profile_id`` identifies a row and
``count_match`` says how much evidence sits behind it.

Method
------
1. Select rate-based features only -- ``*_p30tip`` per-30-minutes-in-possession
   counts, percentages, and per-minute physical rates. Raw totals would make
   the model a minutes-played detector.
2. Drop features missing for more than 20% of players; median-impute the rest.
3. Rank-transform each feature to a percentile within the league, then
   standardise. Ranking removes the long right tails that a handful of the
   metrics have and makes the Euclidean geometry behave.
4. PCA to the number of components explaining 90% of variance, for the
   exploration plot.
5. Similarity is **cosine similarity in the standardised feature space**, not
   in PCA space, so no information is dropped from the ranking itself.

What this is not
----------------
It is a description of statistical profile, not a recruitment recommendation.
Two players can have near-identical profiles and be worth very different
amounts; the model has no information about level of opposition, contract,
age curve, or whether a player's numbers are a product of their team's style.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.config import DERIVED_DIR, PROCESSED_DIR

log = logging.getLogger(__name__)

PROFILE_PATH = DERIVED_DIR / "player_season_profile.parquet"
SIMILARITY_PATH = DERIVED_DIR / "player_similarity.parquet"
EMBEDDING_PATH = DERIVED_DIR / "player_embedding.parquet"

ID_COLS = ["player_id", "player_name", "player_short_name", "team_name", "team_id",
           "position_group", "player_birthdate", "minutes", "count_match", "profile_id"]

#: Physical metrics are totals over the season; these become per-90 rates.
PHYSICAL_RATE_BASE = [
    "total_distance_full_all", "running_distance_full_all", "hsr_distance_full_all",
    "sprint_distance_full_all", "hi_distance_full_all",
    "hsr_count_full_all", "sprint_count_full_all", "hi_count_full_all",
    "medaccel_count_full_all", "highaccel_count_full_all",
    "meddecel_count_full_all", "highdecel_count_full_all",
    "explacceltohsr_count_full_all", "explacceltosprint_count_full_all",
]
#: Already rates or peaks; used as delivered. ``psv99`` is km/h.
PHYSICAL_ABSOLUTE = ["psv99", "psv99_top5", "total_metersperminute_full_all",
                     "timetohsr_top3", "timetosprint_top3"]


def build_profile(min_matches: int = 4) -> pd.DataFrame:
    """Join the three aggregate files into one player-season profile table."""
    phys = pd.read_parquet(PROCESSED_DIR / "season_physical.parquet")
    pas = pd.read_parquet(PROCESSED_DIR / "season_passing.parquet")
    obr = pd.read_parquet(PROCESSED_DIR / "season_obr.parquet")

    phys = phys.rename(columns={"minutes_full_all": "minutes"})
    base = phys[[c for c in ID_COLS if c in phys.columns]].copy()

    per90 = phys[PHYSICAL_RATE_BASE].div(phys["minutes"].replace(0, np.nan), axis=0) * 90
    per90.columns = [f"{c.replace('_full_all', '')}_per90" for c in PHYSICAL_RATE_BASE]
    feats = pd.concat([base, per90, phys[PHYSICAL_ABSOLUTE]], axis=1)

    # Possession / out-of-possession split as a style descriptor.
    feats["tip_share"] = phys["minutes_full_tip"] / phys["minutes"].replace(0, np.nan)
    feats["hsr_tip_share"] = phys["hsr_distance_full_tip"] / phys["hsr_distance_full_all"].replace(0, np.nan)

    keep_pass = [c for c in pas.columns if c.endswith(("_p30tip", "_pct_completed", "_avgdistance"))
                 or c.startswith("pass_pct") or c.startswith("pass_avg")]
    keep_obr = [c for c in obr.columns if c.endswith(("_p30tip", "_avgdistance", "_pct"))]

    feats["count_match"] = phys["count_match"].to_numpy()
    feats["psv99_kmh"] = feats.pop("psv99")
    feats["psv99_top5_kmh"] = feats.pop("psv99_top5")

    key = ["player_id", "team_id", "position_group"]
    feats = feats.merge(pas[key + keep_pass], on=key, how="left", validate="one_to_one")
    feats = feats.merge(obr[key + keep_obr], on=key, how="left", validate="one_to_one")

    feats["profile_id"] = (
        feats["player_id"].astype(str) + ":" + feats["position_group"].astype(str)
    )
    feats = feats[feats["count_match"] >= min_matches].reset_index(drop=True)
    return feats


def feature_matrix(profile: pd.DataFrame, max_missing: float = 0.2):
    num = profile.drop(columns=[c for c in ID_COLS if c in profile.columns], errors="ignore")
    num = num.select_dtypes("number")
    num = num.loc[:, num.isna().mean() <= max_missing]
    num = num.loc[:, num.std(numeric_only=True) > 0]
    imputed = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(num), columns=num.columns
    )
    # Rank -> percentile -> standardise.
    ranked = imputed.rank(pct=True)
    scaled = StandardScaler().fit_transform(ranked)
    return pd.DataFrame(scaled, columns=num.columns), ranked


def build(min_matches: int = 4, n_neighbours: int = 12) -> dict[str, int]:
    profile = build_profile(min_matches)
    X, ranked = feature_matrix(profile)

    # Cosine similarity in standardised space.
    M = X.to_numpy()
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    S = (M / norm) @ (M / norm).T
    np.fill_diagonal(S, -np.inf)

    ids = profile["profile_id"].to_numpy()
    pids = profile["player_id"].to_numpy()
    names = profile["player_short_name"].fillna(profile["player_name"]).to_numpy()
    teams = profile["team_name"].to_numpy()
    pos = profile["position_group"].to_numpy()

    rows = []
    k = min(n_neighbours, len(ids) - 1)
    for i in range(len(ids)):
        top = np.argsort(-S[i])[:k]
        for rank, j in enumerate(top, start=1):
            rows.append(
                {
                    "profile_id": ids[i], "player_id": int(pids[i]), "player_name": names[i],
                    "position_group": pos[i], "team_name": teams[i],
                    "rank": rank,
                    "similar_profile_id": ids[j], "similar_player_id": int(pids[j]),
                    "similar_player_name": names[j],
                    "similar_position_group": pos[j], "similar_team_name": teams[j],
                    "similarity": float(S[i, j]),
                }
            )
    sim = pd.DataFrame(rows)

    n_comp = min(len(X.columns), len(X) - 1)
    pca_full = PCA(n_components=n_comp).fit(M)
    cum = np.cumsum(pca_full.explained_variance_ratio_)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    emb2 = PCA(n_components=2).fit_transform(M)
    embedding = profile[[c for c in ID_COLS if c in profile.columns]].copy()
    embedding["pc1"] = emb2[:, 0]
    embedding["pc2"] = emb2[:, 1]
    embedding["n_components_for_90pct"] = k90
    embedding["pc1_var"] = pca_full.explained_variance_ratio_[0]
    embedding["pc2_var"] = pca_full.explained_variance_ratio_[1]

    # Percentile table for the player cards: within league and within position.
    pct = ranked.copy()
    pct.insert(0, "profile_id", profile["profile_id"].to_numpy())
    pct.insert(1, "player_id", profile["player_id"].to_numpy())
    pct.insert(2, "position_group", profile["position_group"].to_numpy())
    within_pos = (
        pct.drop(columns=["profile_id", "player_id"]).groupby("position_group").rank(pct=True)
    )
    within_pos.insert(0, "profile_id", profile["profile_id"].to_numpy())
    within_pos.insert(1, "player_id", profile["player_id"].to_numpy())
    within_pos.insert(2, "position_group", profile["position_group"].to_numpy())

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    profile.to_parquet(PROFILE_PATH, index=False, compression="zstd")
    sim.to_parquet(SIMILARITY_PATH, index=False, compression="zstd")
    embedding.to_parquet(EMBEDDING_PATH, index=False, compression="zstd")
    pct.to_parquet(DERIVED_DIR / "player_percentiles_league.parquet", index=False)
    within_pos.to_parquet(DERIVED_DIR / "player_percentiles_position.parquet", index=False)

    return {
        "players": len(profile),
        "features": len(X.columns),
        "similarity_rows": len(sim),
        "pca_components_for_90pct": k90,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(build())
