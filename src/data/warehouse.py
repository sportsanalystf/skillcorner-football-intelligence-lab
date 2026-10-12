"""DuckDB warehouse over the Parquet tables.

The Parquet files are the storage format; DuckDB is how the application asks
questions of them. Creating views rather than importing keeps a single copy of
the data on disk and means a rebuilt Parquet file is picked up without
re-loading anything.

The one thing this buys that matters in practice: slicing 60,000 frames of
tracking down to the 300 an animation needs, by predicate pushdown, in
milliseconds instead of reading a 44 MB file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import pandas as pd

from src.config import DERIVED_DIR, PROCESSED_DIR, WAREHOUSE_PATH

log = logging.getLogger(__name__)

VIEWS = {
    "matches": PROCESSED_DIR / "matches.parquet",
    "squads": PROCESSED_DIR / "squads.parquet",
    "events": PROCESSED_DIR / "events.parquet",
    "phases": PROCESSED_DIR / "phases.parquet",
    "season_physical": PROCESSED_DIR / "season_physical.parquet",
    "season_passing": PROCESSED_DIR / "season_passing.parquet",
    "season_obr": PROCESSED_DIR / "season_obr.parquet",
    "possessions": DERIVED_DIR / "possessions.parquet",
    "team_possessions": DERIVED_DIR / "team_possessions.parquet",
    "passes": DERIVED_DIR / "passes.parquet",
    "off_ball_runs": DERIVED_DIR / "off_ball_runs.parquet",
    "pressures": DERIVED_DIR / "pressures.parquet",
    "transitions": DERIVED_DIR / "transitions.parquet",
    "player_match_physical": DERIVED_DIR / "player_match_physical.parquet",
    "sprints": DERIVED_DIR / "sprints.parquet",
    "player_season_profile": DERIVED_DIR / "player_season_profile.parquet",
    "player_similarity": DERIVED_DIR / "player_similarity.parquet",
    "player_embedding": DERIVED_DIR / "player_embedding.parquet",
    "player_percentiles_position": DERIVED_DIR / "player_percentiles_position.parquet",
    "player_percentiles_league": DERIVED_DIR / "player_percentiles_league.parquet",
    "pose_joints": DERIVED_DIR / "pose_joints.parquet",
    "pose_orientation": DERIVED_DIR / "pose_orientation.parquet",
}

GLOB_VIEWS = {
    "tracking": PROCESSED_DIR / "tracking" / "*" / "tracking.parquet",
    "ball": PROCESSED_DIR / "ball" / "*" / "ball.parquet",
    "team_frames": DERIVED_DIR / "team_frames" / "*.parquet",
}


def build(path: Path = WAREHOUSE_PATH) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    made = {}
    for name, p in VIEWS.items():
        if not p.exists():
            log.warning("skipping view %s: %s missing", name, p)
            continue
        con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{p}')")
        made[name] = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    for name, g in GLOB_VIEWS.items():
        con.execute(
            f"CREATE OR REPLACE VIEW {name} AS "
            f"SELECT * FROM read_parquet('{g}', union_by_name=true, hive_partitioning=true)"
        )
        made[name] = con.execute(f"SELECT count(*) FROM {name}").fetchone()[0]
    con.close()
    return made


def connect(path: Path = WAREHOUSE_PATH, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(path), read_only=read_only)


def query(sql: str, params: list | None = None, path: Path = WAREHOUSE_PATH) -> pd.DataFrame:
    con = connect(path)
    try:
        return con.execute(sql, params or []).df()
    finally:
        con.close()


def tracking_window(match_id: int, frame_lo: int, frame_hi: int,
                    columns: str = "*") -> pd.DataFrame:
    """Frames in [lo, hi] for one match, straight off Parquet."""
    p = PROCESSED_DIR / "tracking" / f"match_id={match_id}" / "tracking.parquet"
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT {columns} FROM read_parquet('{p}') "
            "WHERE frame BETWEEN ? AND ? ORDER BY frame, player_id",
            [frame_lo, frame_hi],
        ).df()
    finally:
        con.close()


def ball_window(match_id: int, frame_lo: int, frame_hi: int) -> pd.DataFrame:
    p = PROCESSED_DIR / "ball" / f"match_id={match_id}" / "ball.parquet"
    con = duckdb.connect()
    try:
        return con.execute(
            f"SELECT * FROM read_parquet('{p}') WHERE frame BETWEEN ? AND ? ORDER BY frame",
            [frame_lo, frame_hi],
        ).df()
    finally:
        con.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for k, v in build().items():
        print(f"{k:32s} {v:>12,}")
