"""Reproduce round-two review findings without touching the runner's outputs."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
from io import StringIO
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from scripts.audit import bullpen_r52_compare as r52

START = datetime(2026, 11, 19, 22)
END = START + timedelta(hours=2)


def stamp(path, when):
    os.utime(path, (when.timestamp(), when.timestamp()))


def season_path(root, arm, seed, season):
    return root / arm / f"seed_{seed}" / f"backtest_{season}.parquet"


def grid(scratch):
    root = scratch / "out"
    log = root.parent / "progress.log"
    lines = []
    for idx, seed in enumerate(r52.SEEDS):
        for j, arm in enumerate(r52.ARMS):
            start = START + timedelta(hours=3 * (2 * idx + j))
            end = start + (END - START)
            lines.extend((f"{start:%F %T} start arm={arm} seed={seed}",
                          f"{end:%F %T} end   arm={arm} seed={seed} rc=0"))
            for season in r52.SEASONS:
                path = season_path(root, arm, seed, season)
                path.parent.mkdir(parents=True, exist_ok=True)
                dates = [date(season, 4, 1) + timedelta(days=i) for i in range(20)]
                data = pd.DataFrame({"date": [d for d in dates for _ in (1, 2)],
                                     "rank": [1, 2] * len(dates), "actual_hit": 1,
                                     "p_game_hit": [0.9, 0.7] * len(dates),
                                     "batter_id": [1, 2] * len(dates), "n_pas": 4})
                data.to_parquet(path)
                stamp(path, end - timedelta(minutes=1))
            done = path.parent / "DONE"
            done.touch()
            stamp(done, end + timedelta(milliseconds=500))
    log.write_text("\n".join(lines) + "\n")
    return root, log


def cli(root, *extra):
    stdout, stderr = StringIO(), StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        try:
            exit_code = r52.main(["--root", str(root), *map(str, extra)])
        except Exception as exc:
            exit_code = type(exc).__name__
    verdict = next((line for line in stdout.getvalue().splitlines() if line.startswith("R52 verdict:")), "no verdict")
    return exit_code, verdict


def validation(scratch):
    root, log = grid(scratch)
    path = season_path(root, "old", r52.SEEDS[0], r52.SEASONS[0])
    clean = pd.read_parquet(path)
    log_text = log.read_text()
    print("unchanged full synthetic grid:", cli(root))
    for label, invalid in (("string rank", clean.assign(rank=clean["rank"].astype(str))),
                           ("infinite rank", clean.assign(rank=float("inf")))):
        invalid.to_parquet(path)
        stamp(path, END - timedelta(minutes=1))
        print(f"{label}:", cli(root))
    clean.to_parquet(path)
    stamp(path, END - timedelta(minutes=1))
    log.write_bytes(b"\xff")
    print("undecodable progress:", cli(root))
    log.write_text(log_text + "2026-11-32 22:00:00 start arm=old seed=2273360\n")
    print("invalid progress date:", cli(root))


def provenance(scratch):
    root, log = grid(scratch)
    path = season_path(root, "old", r52.SEEDS[0], r52.SEASONS[0])
    clean = pd.read_parquet(path)
    log_text = log.read_text()
    pd.DataFrame(clean.assign(actual_hit=0)).to_parquet(path)
    stamp(path, END + timedelta(seconds=3))
    print("post-DONE replacement (end + 3 seconds):", cli(root))
    clean.to_parquet(path)
    stamp(path, END - timedelta(minutes=1))

    lines = log_text.splitlines()
    later = END - timedelta(minutes=5)
    lines.insert(1, f"{later:%F %T} start arm=old seed={r52.SEEDS[0]}")
    log.write_text("\n".join(lines) + "\n")
    print("overlapping starts with one untagged end:", cli(root))
    log.write_text(log_text + f"{END:%F %T} end   arm=old seed={r52.SEEDS[0]} rc=1\n"
                  f"{END:%F %T} end   arm=old seed={r52.SEEDS[0]} rc=0\n")
    print("stray success after failed end:", cli(root))
    log.write_text(log_text)

    new = season_path(root, "new", r52.SEEDS[0], r52.SEASONS[0])
    stamp_time = datetime.fromtimestamp(new.stat().st_mtime)
    edited = clean.assign(actual_hit=[1, 1] + [0, 1] * 19)
    edited.to_parquet(new)
    stamp(new, stamp_time)
    print("full grid with synthetic drop:", cli(root))
    for file, frame, mtime in ((path, clean, END - timedelta(minutes=1)), (new, edited, stamp_time)):
        frame.iloc[:2].to_parquet(file)
        stamp(file, mtime)
    print("same dates cropped in both arms:", cli(root))


def publication(scratch):
    root, log = grid(scratch)
    seed, season = r52.SEEDS[0], r52.SEASONS[0]
    path = season_path(root, "old", seed, season)
    original = path.read_bytes()
    clean = pd.read_parquet(path)
    clean.assign(actual_hit=clean["actual_hit"].astype(float)).to_parquet(path)
    with patch.object(r52, "SEEDS", (seed,)), patch.object(r52, "SEASONS", (season,)):
        print("binary float outcomes, scorecard experiment:", r52.streak_report(root, 1)[f"old/{seed}/{season}"])
    path.write_bytes(original)
    stamp(path, END - timedelta(minutes=1))
    print("JSON target aliases input:", cli(root, "--json", path))
    print("input after the published verdict:", cli(root))


if __name__ == "__main__":
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    assert priority == 19, f"expected priority 19, got {priority}"
    probes = {"validation": validation, "provenance": provenance, "publication": publication}
    with TemporaryDirectory(dir=HERE) as tmp:
        probes[sys.argv[1]](Path(tmp))
