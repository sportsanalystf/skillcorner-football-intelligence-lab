"""One command that turns raw SkillCorner files into everything the app reads.

    python -m src.pipeline                 # everything that is missing
    python -m src.pipeline --force         # rebuild from scratch
    python -m src.pipeline --steps tracking events features

Stages, in dependency order:

    download   raw JSON/CSV/JSONL from GitHub (and pose from Hugging Face)
    events     events, phases, squads, matches, season aggregates -> Parquet
    tracking   tracking JSONL -> Parquet, with kinematics attached
    validate   data-quality report
    features   team shape, physical, possessions, chains, passes, runs,
               pressures, transitions
    models     season player profile, percentiles, similarity, PCA embedding
    pose       body-pose joints and orientation
    warehouse  DuckDB views over every Parquet table
    research   the ten research questions
    docs       regenerate the data dictionary from what was built

Each stage is idempotent and logs how long it took.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from src.config import DERIVED_DIR, PROCESSED_DIR, WAREHOUSE_PATH

log = logging.getLogger("pipeline")

STEPS = ["download", "events", "tracking", "validate", "features", "models",
         "pose", "warehouse", "research", "docs"]


def _need(path, force: bool) -> bool:
    return force or not path.exists()


def step_download(force: bool, bodypose: bool = False) -> dict:
    from src.data.download import main as dl

    argv = ["--bodypose"] if bodypose else []
    return {"exit_code": dl(argv)}


def step_events(force: bool) -> dict:
    from src.preprocessing.build_events import run

    if not _need(PROCESSED_DIR / "events.parquet", force):
        return {"skipped": True}
    return run()


def step_tracking(force: bool) -> dict:
    from src.data.loaders import available_match_ids
    from src.preprocessing.build_tracking import build_match

    out = {}
    for mid in available_match_ids():
        out[mid] = build_match(mid, overwrite=force)
    return {"matches": len(out)}


def step_validate(force: bool) -> dict:
    from src.preprocessing.validate import run

    return {"checks": len(run())}


def step_features(force: bool) -> dict:
    from src.events.possessions import build as build_poss
    from src.features.physical import build as build_phys
    from src.features.team_shape import build as build_shape

    out = {}
    out["team_frames"] = len(build_shape())
    out.update(build_phys())
    out.update(build_poss())
    return out


def step_models(force: bool) -> dict:
    from src.models.similarity import build

    return build()


def step_pose(force: bool) -> dict:
    from src.features.pose import build

    try:
        return build()
    except FileNotFoundError as exc:
        log.warning("pose unavailable: %s", exc)
        return {"skipped": str(exc)}


def step_warehouse(force: bool) -> dict:
    from src.data.warehouse import build

    return {"views": len(build())}


def step_research(force: bool) -> dict:
    from src.research.questions import run_all

    return {"questions": len(run_all())}


def step_docs(force: bool) -> dict:
    import subprocess

    from src.config import PROJECT_ROOT

    r = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "make_data_dictionary.py")],
        capture_output=True, text=True, check=False,
    )
    if r.returncode:
        log.error("data dictionary failed: %s", r.stderr[-800:])
    return {"exit_code": r.returncode}


HANDLERS = {
    "download": step_download, "events": step_events, "tracking": step_tracking,
    "validate": step_validate, "features": step_features, "models": step_models,
    "pose": step_pose, "warehouse": step_warehouse, "research": step_research,
    "docs": step_docs,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="*", choices=STEPS, default=None)
    ap.add_argument("--skip", nargs="*", choices=STEPS, default=[])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--bodypose", action="store_true",
                    help="also download the full 3D pose archives (~1.2 GB, needs huggingface.co)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    steps = [s for s in (args.steps or STEPS) if s not in args.skip]
    log.info("running: %s", " -> ".join(steps))

    failed = []
    for name in steps:
        t0 = time.time()
        try:
            kw = {"bodypose": args.bodypose} if name == "download" else {}
            result = HANDLERS[name](args.force, **kw)
            log.info("%-10s %6.1fs  %s", name, time.time() - t0, result)
        except Exception as exc:  # noqa: BLE001
            failed.append(name)
            log.exception("%-10s FAILED after %.1fs: %s", name, time.time() - t0, exc)
    if failed:
        log.error("failed steps: %s", failed)
        return 1
    log.info("pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
