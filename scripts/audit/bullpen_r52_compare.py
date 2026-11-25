"""Paired comparison of the bullpen-fix backtests against ruling R52 (a deploy condition).

Ruling R52 (docs/reviews/2026-11-18-whole-range/FINDINGS.md, written 2026-11-20 before any
result was read): for each arm, seed and season, P@1 is
``bts.validate.scorecard.compute_precision_at_k(..., k=1)`` on
``<root>/<arm>/seed_<seed>/backtest_<season>.parquet``. Per seed, d = (new minus old) P@1
averaged with equal weight over 2021 to 2025, giving one paired value per seed (R52: 8 seeds;
extended by R59 to 24, the final test). PASS when both hold: (1) the one-sided 95% lower bound
mean(d) - t(0.95, n-1) * sd(d) / sqrt(n) is above -0.5 percentage points; (2) no season's mean
difference over the seeds is below -1.0 percentage points.
The rule is unmet until all 48 runs finish. Streak metrics are reported, not ruled on.

The verdict path uses the fixed grid (the queue's 24 seeds, 2021 to 2025); there is no option to
change it. P@1 is carried as an exact fraction (one validated rank-1 pick per day, binary
outcome) and checked equal to the scorecard's float; both floors are compared exactly.
A run counts only when its DONE marker exists, ``progress.log`` (read strictly: every end pairs
with the one open start of the same run, no two different runs are open at once, a start may
supersede an unfinished start of the SAME run, which is how this serial queue restarts) has its
last start followed by ``end ... rc=0``, every season file was written after that start, before the
logged end's whole second (real files precede it by at least 4.3 s) and no later than the DONE
marker, and the DONE marker is no later than the logged end + 2 s (empty, R52's 8 seeds) or
+ 30 s (a receipt, written after hashing the five files). Each file is read once; the bytes scored are hashed into the result. Every season's day set
must be identical across all 48 runs. Since R59 a run's DONE marker holds the runner's sha256 of its
five files: for every seed except R52's original 8 (whose markers are empty) the bytes scored must
match that receipt. Any unexpected input is "rule unmet", never a verdict.

Exit codes: 0 PASS, 1 FAIL, 2 rule unmet (incomplete, unreadable or invalid inputs, or the
result could not be written).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta
from fractions import Fraction
from pathlib import Path

import pandas as pd

# R52's 8 seeds, then R59's 16 more (Eric 2026-11-22: "3"); the final test runs once on all 24.
LEGACY_SEEDS = (2273360, 260991262, 1746737973, 2048, 3629294338, 1277948386, 3219332220, 2207587974)
SEEDS = LEGACY_SEEDS + (1, 7, 137, 1024, 8192, 65536, 524288, 999983, 2041769, 3167584, 3911096,
                        6426836, 6762547, 6992150, 7777777, 7886470)
SEASONS = (2021, 2022, 2023, 2024, 2025)
ARMS = ("old", "new")
LOWER_BOUND_FLOOR = Fraction(-1, 200)  # the lower bound must be strictly above
SEASON_FLOOR = Fraction(-1, 100)  # a season mean below this fails; equal passes
DEFAULT_ROOT = Path.home() / "bts-runs" / "bullpen-bt" / "out"
RECEIPT_SLACK_S = 30  # a receipt-bearing DONE is written after hashing the five files (real gaps under 1 s)
WINDOW_SLACK_S = 2  # DONE is touched just after the end line, which is stamped to the whole second (real gaps 0.03 to 0.97 s)
_LINE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (start|end) +arm=(\w+) seed=(\d+)(?: rc=(\d+))?\s*$")


class Incomplete(Exception):
    pass


def t_quantile_95(df: int) -> float:
    from scipy.stats import t

    return float(t.ppf(0.95, df))


def run_windows(progress: Path) -> dict[tuple[str, int], tuple[datetime, datetime]]:
    """(arm, seed) -> (start, end) of its last successful attempt, from the runner's own log."""
    try:
        lines = progress.read_bytes().decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise Incomplete(f"cannot read {progress}: {type(exc).__name__}") from exc
    open_run: tuple[tuple[str, int], datetime] | None = None
    done: dict[tuple[str, int], tuple[datetime, datetime]] = {}
    last_when = None
    for n, line in enumerate(lines, 1):
        m = _LINE.match(line)
        if not m:
            continue
        ts, kind, arm, seed, rc = m.groups()
        try:
            key, when = (arm, int(seed)), datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            raise Incomplete(f"progress.log line {n}: unreadable time") from exc
        if arm not in ARMS or (kind == "end" and rc is None):
            raise Incomplete(f"progress.log line {n}: malformed record")
        if last_when is not None and when < last_when:
            raise Incomplete(f"progress.log line {n}: time goes backwards")
        last_when = when
        if kind == "start":
            if open_run is not None and open_run[0] != key:
                raise Incomplete(f"progress.log line {n}: {key} starts while {open_run[0]} is open")
            open_run = (key, when)  # a restart of the same unfinished run supersedes it
            done.pop(key, None)  # a later attempt supersedes an earlier success
        else:
            if open_run is None or open_run[0] != key:
                raise Incomplete(f"progress.log line {n}: end of {key} with no open start")
            if rc == "0":
                done[key] = (open_run[1], when)
            else:
                done.pop(key, None)
            open_run = None
    return done


def _cell(root: Path, arm: str, seed: int, season: int, window, seasons=SEASONS):
    tag = f"{arm} seed {seed} {season}"
    run = root / arm / f"seed_{seed}"
    if window is None:
        raise Incomplete(f"{arm} seed {seed}: no successful attempt in progress.log")
    start, end = window
    try:
        done_mtime = datetime.fromtimestamp((run / "DONE").stat().st_mtime)
        receipt = (run / "DONE").read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise Incomplete(f"{arm} seed {seed}: no readable DONE marker") from exc
    recorded = {}
    if receipt and not receipt.endswith("\n"):
        raise Incomplete(f"{arm} seed {seed}: DONE receipt not the runner's LF-terminated shasum lines")
    for line in receipt.split("\n")[:-1] if receipt else []:
        m = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.]+)", line)  # exactly the runner's shasum line
        parts = m.groups() if m else ()
        if not m or parts[1] in recorded:
            raise Incomplete(f"{arm} seed {seed}: DONE receipt line not '<sha256>  <file>' (or a file listed twice)")
        recorded[parts[1]] = parts[0]
    if recorded and set(recorded) != {f"backtest_{y}.parquet" for y in seasons}:
        raise Incomplete(f"{arm} seed {seed}: DONE receipt does not list exactly the season files compared")
    if not recorded and seed not in LEGACY_SEEDS:
        raise Incomplete(f"{arm} seed {seed}: no receipt in DONE (required for every seed but R52's original 8)")
    slack = RECEIPT_SLACK_S if recorded else WINDOW_SLACK_S
    if not (start <= done_mtime <= end + timedelta(seconds=slack)):
        raise Incomplete(f"{arm} seed {seed}: DONE written {done_mtime:%F %T}, outside its run {start:%F %T} to {end:%F %T}")
    path = run / f"backtest_{season}.parquet"
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        data = path.read_bytes()
        df = pd.read_parquet(io.BytesIO(data))
    except Exception as exc:  # missing, unreadable or not parquet
        raise Incomplete(f"{tag}: cannot read {path.name} ({type(exc).__name__})") from exc
    if recorded and recorded[path.name] != hashlib.sha256(data).hexdigest():
        raise Incomplete(f"{tag}: bytes differ from the runner's receipt in DONE")
    if not (start <= mtime <= done_mtime and mtime < end):
        raise Incomplete(f"{tag}: file written {mtime:%F %T}, outside its run {start:%F %T} to its DONE {done_mtime:%F %T}")
    for col in ("date", "rank", "actual_hit"):
        if col not in df.columns:
            raise Incomplete(f"{tag}: no {col} column")
    if df.empty:
        raise Incomplete(f"{tag}: no rows")
    dates = pd.to_datetime(df["date"], errors="coerce")
    if dates.isna().any() or set(dates.dt.year) != {season}:
        raise Incomplete(f"{tag}: dates missing or outside {season}")
    for col in ("rank", "actual_hit"):
        if not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            raise Incomplete(f"{tag}: {col} is not numeric")
    ranks = df["rank"].astype(float)
    if not ranks.map(math.isfinite).all() or (ranks != ranks.round()).any():
        raise Incomplete(f"{tag}: non-integer ranks")
    keyed = pd.DataFrame({"d": dates.dt.date, "r": ranks.astype(int)})
    for d, rs in keyed.groupby("d")["r"]:
        if sorted(rs) != list(range(1, len(rs) + 1)) or len(rs) > 10:
            raise Incomplete(f"{tag}: ranks on {d} are not 1..n (n <= 10)")
    top = df[ranks == 1]
    hits = pd.to_numeric(top["actual_hit"], errors="coerce")
    if hits.isna().any() or not set(hits.astype(float)) <= {0.0, 1.0}:
        raise Incomplete(f"{tag}: rank-1 outcomes not all 0 or 1")
    exact = Fraction(int(hits.sum()), len(top))
    from bts.validate.scorecard import compute_precision_at_k

    scorecard = compute_precision_at_k(df, k_values=[1]).get(1)
    if scorecard is None or not math.isclose(scorecard, float(exact), rel_tol=0, abs_tol=1e-12):
        raise Incomplete(f"{tag}: scorecard P@1 {scorecard} disagrees with the exact {exact}")
    return exact, frozenset(keyed["d"]), hashlib.sha256(data).hexdigest(), df


def compare(root: Path, seeds=SEEDS, seasons=SEASONS, progress: Path | None = None) -> tuple[dict, dict]:
    """The R52 statistic. ``main`` always uses the fixed grid; the parameters exist for tests."""
    seeds, seasons = tuple(seeds), tuple(seasons)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds) or len(set(seasons)) != len(seasons) or not seasons:
        raise Incomplete("seeds and seasons must be distinct, with at least two seeds")
    windows = run_windows(progress if progress is not None else root.parent / "progress.log")
    cells, frames, season_days = {}, {}, {}
    for seed in seeds:
        for season in seasons:
            got = {}
            for arm in ARMS:
                try:
                    got[arm] = _cell(root, arm, seed, season, windows.get((arm, seed)), seasons)
                except Incomplete:
                    raise
                except Exception as exc:  # anything unexpected is unmet, never a verdict
                    raise Incomplete(f"{arm} seed {seed} {season}: {type(exc).__name__}") from exc
                frames[(arm, seed, season)] = got[arm][3]
                days = season_days.setdefault(season, got[arm][1])
                if got[arm][1] != days:
                    raise Incomplete(f"{arm} seed {seed} {season}: covers {len(got[arm][1])} days where "
                                     f"another run of {season} covers {len(days)} (or other days)")
            cells[(seed, season)] = {"old": got["old"][0], "new": got["new"][0], "days": len(got["old"][1]),
                                     "old_sha256": got["old"][2], "new_sha256": got["new"][2]}
    d = [sum((cells[(s, y)]["new"] - cells[(s, y)]["old"] for y in seasons), Fraction(0)) / len(seasons)
         for s in seeds]
    n = len(d)
    mean_d = sum(d, Fraction(0)) / n
    ss = sum(((x - mean_d) ** 2 for x in d), Fraction(0))
    tq = t_quantile_95(n - 1)
    # lower = mean_d - tq * sqrt(ss / (n (n-1))) > floor  <=>  gap > 0 and gap^2 > tq^2 * ss / (n (n-1))
    gap = mean_d - LOWER_BOUND_FLOOR
    cond1 = gap > 0 and gap * gap > Fraction(tq) ** 2 * ss / (n * (n - 1))
    season_means = {y: sum((cells[(s, y)]["new"] - cells[(s, y)]["old"] for s in seeds), Fraction(0)) / n
                    for y in seasons}
    cond2 = all(m >= SEASON_FLOOR for m in season_means.values())
    sd = math.sqrt(float(ss) / (n - 1))
    return {
        "rule": "R52",
        "seeds": list(seeds), "seasons": list(seasons),
        "cells": [{"seed": s, "season": y, "old": float(c["old"]), "new": float(c["new"]),
                   "diff": float(c["new"] - c["old"]), "days": c["days"],
                   "old_exact": str(c["old"]), "new_exact": str(c["new"]),
                   "old_sha256": c["old_sha256"], "new_sha256": c["new_sha256"]}
                  for (s, y), c in cells.items()],
        "per_seed_d": {str(s): float(x) for s, x in zip(seeds, d)},
        "mean_d": float(mean_d), "sd_d": sd, "t_0.95": tq,
        "lower_bound_95": float(mean_d) - tq * sd / math.sqrt(n),
        "season_mean_diff": {str(y): float(m) for y, m in season_means.items()},
        "condition_1_lower_bound_above_-0.005": cond1,
        "condition_2_no_season_below_-0.010": cond2,
        "verdict": "PASS" if (cond1 and cond2) else "FAIL",
    }, frames


def _metric(v):
    if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
        return f"unavailable ({v!r})"
    return v


def streak_report(frames: dict, mc_trials: int) -> dict:
    """Reported only (R52), per arm and seed, each season separately (no carry-over between seasons).

    mean_max_streak: the scorecard's 180-day bootstrap under the `combined` strategy;
    p_57_exact: the scorecard's binned Markov estimate (approximate saver). Neither decides anything.
    """
    from bts.validate.scorecard import compute_full_scorecard

    out = {}
    for (arm, seed, season), df in sorted(frames.items()):
        key = f"{arm}/{seed}/{season}"
        try:
            sc = compute_full_scorecard(df, mc_trials=mc_trials)
            out[key] = {"p_57_exact": _metric(sc["p_57_exact"]),
                        "mean_max_streak": _metric(sc["streak_metrics"]["mean_max_streak"])}
        except Exception as exc:
            out[key] = {"unavailable": f"{type(exc).__name__}: {exc}"[:200]}
    return out


def _write_atomic(path: Path, text: str) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _positive(s: str) -> int:
    v = int(s)
    if v <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return v


def main(argv: list[str] | None = None, *, _grid=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--streak", action="store_true", help="also report streak metrics (not part of the rule)")
    ap.add_argument("--mc-trials", type=_positive, default=10_000)
    ap.add_argument("--json", type=Path, help="write the full result here (before the verdict is printed)")
    args = ap.parse_args(argv)
    seeds, seasons = _grid if _grid is not None else (SEEDS, SEASONS)  # _grid: tests only
    if args.json:
        evidence = args.root.resolve().parent
        target = args.json.resolve()
        if target == evidence or evidence in target.parents:
            print(f"R52: refusing to write the result inside the evidence tree {evidence}", file=sys.stderr)
            return 2
    try:
        result, frames = compare(args.root, seeds, seasons)
    except Incomplete as exc:
        print(f"R52 rule unmet: {exc}", file=sys.stderr)
        return 2
    if args.streak:
        result["streak_metrics_reported_only"] = streak_report(frames, args.mc_trials)
    if args.json:
        try:
            _write_atomic(args.json, json.dumps(result, indent=2) + "\n")
        except OSError as exc:
            print(f"R52 result not written to {args.json} ({type(exc).__name__}); no verdict", file=sys.stderr)
            return 2
    pp = lambda x: f"{100 * x:+.3f} pt"  # noqa: E731
    for c in result["cells"]:
        print(f"seed {c['seed']:>10} {c['season']}: old {c['old']:.4f} new {c['new']:.4f} "
              f"diff {pp(c['diff'])} ({c['days']} days)")
    for s, v in result["per_seed_d"].items():
        print(f"seed {s:>10} d (season-averaged) {pp(v)}")
    for y, m in result["season_mean_diff"].items():
        print(f"season {y} mean diff over seeds {pp(m)}")
    print(f"mean d {pp(result['mean_d'])}, sd {100 * result['sd_d']:.3f} pt, "
          f"one-sided 95% lower bound {pp(result['lower_bound_95'])}")
    for k, v in result.get("streak_metrics_reported_only", {}).items():
        print(f"reported only, {k}: {v}")
    print(f"condition 1 (lower bound above -0.5 pt): {result['condition_1_lower_bound_above_-0.005']}")
    print(f"condition 2 (no season below -1.0 pt): {result['condition_2_no_season_below_-0.010']}")
    print(f"R52 verdict: {result['verdict']}")
    return 0 if result["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
