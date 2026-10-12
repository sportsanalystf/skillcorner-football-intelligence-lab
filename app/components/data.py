"""Cached data access for the Streamlit app.

Every read goes through here so that caching, column pruning and the
Parquet/DuckDB split live in one place rather than being sprinkled through the
pages.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.config import DERIVED_DIR, PROCESSED_DIR, REPORTS_DIR
from src.data import warehouse as wh
from src.data.loaders import load_match_meta

TTL = 3600


@st.cache_data(ttl=TTL, show_spinner=False)
def matches() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "matches.parquet")


@st.cache_data(ttl=TTL, show_spinner=False)
def squads() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "squads.parquet")


@st.cache_data(ttl=TTL, show_spinner=False)
def table(name: str) -> pd.DataFrame:
    for base in (DERIVED_DIR, PROCESSED_DIR):
        p = base / f"{name}.parquet"
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"no table named {name}")


@st.cache_resource(show_spinner=False)
def meta(match_id: int):
    return load_match_meta(int(match_id))


@st.cache_data(ttl=TTL, show_spinner=False)
def match_events(match_id: int) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "events.parquet",
                           filters=[("match_id", "==", int(match_id))])


@st.cache_data(ttl=TTL, show_spinner=False)
def match_phases(match_id: int) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "phases.parquet",
                           filters=[("match_id", "==", int(match_id))])


@st.cache_data(ttl=TTL, show_spinner=False)
def match_possessions(match_id: int) -> pd.DataFrame:
    return pd.read_parquet(DERIVED_DIR / "possessions.parquet",
                           filters=[("match_id", "==", int(match_id))])


@st.cache_data(ttl=TTL, show_spinner=False)
def match_team_possessions(match_id: int) -> pd.DataFrame:
    return pd.read_parquet(DERIVED_DIR / "team_possessions.parquet",
                           filters=[("match_id", "==", int(match_id))])


@st.cache_data(ttl=TTL, show_spinner=False)
def match_ball(match_id: int) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "ball" / f"match_id={int(match_id)}" / "ball.parquet")


@st.cache_data(ttl=TTL, show_spinner=False)
def match_team_frames(match_id: int, stride: int = 5) -> pd.DataFrame:
    df = pd.read_parquet(DERIVED_DIR / "team_frames" / f"{int(match_id)}.parquet")
    return df[df["frame"] % stride == 0] if stride > 1 else df


@st.cache_data(ttl=TTL, show_spinner=False)
def tracking_window(match_id: int, lo: int, hi: int) -> pd.DataFrame:
    return wh.tracking_window(int(match_id), int(lo), int(hi))


@st.cache_data(ttl=TTL, show_spinner=False)
def match_tracking(match_id: int, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    return pd.read_parquet(
        PROCESSED_DIR / "tracking" / f"match_id={int(match_id)}" / "tracking.parquet",
        columns=list(columns) if columns else None,
    )


@st.cache_data(ttl=TTL, show_spinner=False)
def validation() -> pd.DataFrame:
    p = REPORTS_DIR / "validation.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=TTL, show_spinner=False)
def research(key: str) -> pd.DataFrame:
    p = REPORTS_DIR / "research" / f"{key}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


@st.cache_data(ttl=TTL, show_spinner=False)
def research_summary() -> dict:
    import json

    p = REPORTS_DIR / "research" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


@st.cache_data(ttl=TTL, show_spinner=False)
def frame_players(match_id: int, frame: int) -> pd.DataFrame:
    """All players at one frame, joined to squad names."""
    df = wh.tracking_window(int(match_id), int(frame), int(frame))
    sq = squads()
    sq = sq[sq["match_id"] == int(match_id)][["player_id", "display_name", "shirt_number"]]
    return df.merge(sq, on="player_id", how="left")


def match_label_map() -> dict[str, int]:
    m = matches()
    return {
        f"{r.date}  ·  {r.home_team} {r.home_score}-{r.away_score} {r.away_team}": int(r.match_id)
        for r in m.itertuples()
    }


def clock_label(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "--:--"
    s = int(round(seconds))
    return f"{s // 60:02d}:{s % 60:02d}"
