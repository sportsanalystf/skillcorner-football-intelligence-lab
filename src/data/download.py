"""Fetch the SkillCorner Open Data payload this project builds on.

The tracking files are stored with Git-LFS. ``git clone`` alone leaves 133-byte
pointer files behind, and ``git lfs pull`` needs credentials on some networks,
so this module downloads the resolved payload over plain HTTPS from
``media.githubusercontent.com`` instead. That endpoint returns the real bytes
for an LFS-tracked path and needs no authentication.

Usage::

    python -m src.data.download              # metadata, events, phases, tracking
    python -m src.data.download --bodypose   # also the two 600 MB pose archives
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import (
    BODYPOSE_HF_BASE,
    BODYPOSE_MATCH_IDS,
    LFS_MEDIA_BASE,
    MATCHES_INDEX_JSON,
    RAW_AGGREGATES_DIR,
    RAW_BODYPOSE_DIR,
    RAW_DIR,
    RAW_MATCHES_DIR,
    RAW_BASE,
)

log = logging.getLogger(__name__)

AGGREGATE_FILES = (
    "aus1league_physicalaggregates_20242025.csv",
    "aus1league_passingaggregates_20242025.csv",
    "aus1league_obraggregates_20242025.csv",
)

PER_MATCH_SMALL = ("{mid}_match.json", "{mid}_dynamic_events.csv", "{mid}_phases_of_play.csv")
PER_MATCH_LFS = ("{mid}_tracking_extrapolated.jsonl",)

BODYPOSE_SUPPORT = ("MANIFEST.json", "README.md", "sample_1925299_phase406.jsonl.gz")


def _get(url: str, dest: Path, min_bytes: int = 1, attempts: int = 3) -> bool:
    """Download ``url`` to ``dest`` unless a plausible copy is already there."""
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=600) as resp, tmp.open("wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            if tmp.stat().st_size >= min_bytes:
                tmp.replace(dest)
                return True
            log.warning("short read for %s (%d bytes)", url, tmp.stat().st_size)
        except Exception as exc:  # noqa: BLE001 - network errors are varied
            log.warning("attempt %d/%d failed for %s: %s", attempt, attempts, url, exc)
    tmp.unlink(missing_ok=True)
    return False


def download_index() -> list[dict]:
    """Fetch ``data/matches.json`` and return the parsed match index."""
    _get(f"{RAW_BASE}/data/matches.json", MATCHES_INDEX_JSON, min_bytes=100)
    return json.loads(MATCHES_INDEX_JSON.read_text())


def download_aggregates() -> None:
    for name in AGGREGATE_FILES:
        _get(f"{RAW_BASE}/data/aggregates/{name}", RAW_AGGREGATES_DIR / name, min_bytes=1000)


def download_match(match_id: int) -> dict[str, bool]:
    """Fetch every per-match file for one match id."""
    out: dict[str, bool] = {}
    dest_dir = RAW_MATCHES_DIR / str(match_id)
    for tmpl in PER_MATCH_SMALL:
        name = tmpl.format(mid=match_id)
        out[name] = _get(f"{RAW_BASE}/data/matches/{match_id}/{name}", dest_dir / name, 500)
    for tmpl in PER_MATCH_LFS:
        name = tmpl.format(mid=match_id)
        # LFS payload: media.githubusercontent.com resolves the pointer.
        out[name] = _get(
            f"{LFS_MEDIA_BASE}/data/matches/{match_id}/{name}",
            dest_dir / name,
            min_bytes=1_000_000,
        )
    return out


def download_bodypose(match_ids: tuple[int, ...] = BODYPOSE_MATCH_IDS) -> dict[int, bool]:
    """Fetch the full body-pose archives from the Hugging Face Hub.

    Roughly 600 MB per match. Networks that block ``huggingface.co`` cannot
    complete this; the committed sample phase under ``data/raw/bodypose/`` is
    what the pose pipeline falls back to, and everything in the app degrades
    gracefully to it.
    """
    for name in BODYPOSE_SUPPORT:
        _get(f"{RAW_BASE}/data/bodypose/{name}", RAW_BODYPOSE_DIR / name, 100)
    results: dict[int, bool] = {}
    for mid in match_ids:
        results[mid] = _get(
            f"{BODYPOSE_HF_BASE}/raw/{mid}.jsonl.zip",
            RAW_BODYPOSE_DIR / f"{mid}.jsonl.zip",
            min_bytes=100_000_000,
        )
        if not results[mid]:
            log.error(
                "Body pose archive for %s could not be downloaded. "
                "See docs/LIMITATIONS.md.", mid,
            )
    return results


def verify_bodypose(path: Path) -> bool:
    """Compare a downloaded pose archive against the SHA256 in MANIFEST.json."""
    manifest = json.loads((RAW_BODYPOSE_DIR / "MANIFEST.json").read_text())
    entry = manifest["files"][f"raw/{path.name}"]
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest() == entry["sha256"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download SkillCorner Open Data")
    parser.add_argument("--bodypose", action="store_true", help="also fetch the pose archives")
    parser.add_argument("--matches", nargs="*", type=int, help="restrict to these match ids")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    index = download_index()
    download_aggregates()
    ids = args.matches or [m["id"] for m in index]
    log.info("fetching %d matches", len(ids))

    failures: list[int] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_match, mid): mid for mid in ids}
        for fut in as_completed(futures):
            mid = futures[fut]
            res = fut.result()
            if not all(res.values()):
                failures.append(mid)
                log.error("match %s incomplete: %s", mid, res)
            else:
                log.info("match %s ok", mid)

    if args.bodypose:
        download_bodypose()

    if failures:
        log.error("failed matches: %s", failures)
        return 1
    log.info("download complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
