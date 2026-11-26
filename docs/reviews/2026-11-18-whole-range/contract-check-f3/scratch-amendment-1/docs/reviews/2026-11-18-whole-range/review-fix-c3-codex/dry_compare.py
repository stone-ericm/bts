"""Dry comparison using two archived revisions. Only the local scratch tree is written."""

import argparse
from datetime import date, datetime
import json
from pathlib import Path
import subprocess
import sys

from probe_support import BASE, ET, HEAD, archive, offline, python, scratch


def run_replay(path, today):
    from bts import picks

    with offline():
        try:
            value = picks._replay_season_streak(path, today.year, today.isoformat())
        except Exception as error:
            return {"raised": type(error).__name__, "message": str(error)}
        output = {"replay": value}
        if value is None and hasattr(picks, "_fold_season"):
            state = picks._read_reconcile_streak(path / "streak.json")
            markers = picks._replay_markers(state or {}, today.year)
            _, output["first_blocker"] = picks._fold_season(path, today.year, today.isoformat(), markers)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("picks_dir", type=Path)
    parser.add_argument("--today", type=date.fromisoformat, default=datetime.now(ET).date())
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    picks_dir = args.picks_dir.resolve(strict=True)
    if not picks_dir.is_dir():
        parser.error(f"Not a directory: {picks_dir}")
    if args.worker:
        print(json.dumps(run_replay(picks_dir, args.today)))
        return 0
    print(json.dumps({"picks": str(picks_dir), "season_through": str(args.today)}))
    observations = []
    with scratch("real-comparison") as work:
        for version in (BASE, HEAD):
            source = archive(version, Path(work) / version, "src") / "src"
            try:
                proc = python(source, __file__, picks_dir, "--today", args.today, "--worker", timeout=25)
                observation = json.loads(proc.stdout)
            except subprocess.TimeoutExpired:
                observation = {"timeout": True}
            print(json.dumps({"revision": version} | observation))
            observations.append(observation)
    a, b = observations
    if "replay" not in a or "replay" not in b or a["replay"] != b["replay"]:
        return 2
    return 0 if a["replay"] is not None else 3


if __name__ == "__main__":
    sys.exit(main())
