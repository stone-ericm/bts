"""Bounded, synthetic R52 probes. No runner output is used."""

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from datetime import date, timedelta
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd
from scripts.audit import bullpen_r52_compare as r52
from bts.validate import scorecard


fixtures = runpy.run_path(str(REPO / "tests/scripts/test_bullpen_r52_compare.py"))
write, log, base = (fixtures[name] for name in ("_write", "_log", "BASE"))
group = sys.argv[1]


def check(label, expected, *extra, grid=None, root=None):
    args = ["--root", str(root or queue / "out"), *map(str, extra)]
    stdout, stderr = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            code = r52.main(args, _grid=grid) if grid is not None else r52.main(args)
        except SystemExit as exc:
            code = exc.code
    out = stdout.getvalue()
    terminal = [line.split()[-1] for line in out.splitlines() if line.startswith("R52 verdict:")]
    assert (code, terminal) == (expected, {0: ["PASS"], 1: ["FAIL"], 2: []}[expected]), (label, code, out, stderr.getvalue())
    print(f"{label} -> {code}", stderr.getvalue().splitlines()[-1] if stderr.getvalue() else terminal[0])
    return out


def changed(path, transform):
    original = path.read_bytes()
    stamp = path.stat()
    transform(pd.read_parquet(io.BytesIO(original))).to_parquet(path, index=False)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    return original, stamp


def restore(path, saved):
    original, stamp = saved
    path.write_bytes(original)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))


def populate(seeds, seasons, old, new):
    for seed in seeds:
        for year in seasons:
            write(queue, "old", seed, year, old(seed, year))
            write(queue, "new", seed, year, new(seed, year))
    log(queue, seeds)


def cell(root, arm, seed, year):
    return root / "out" / arm / f"seed_{seed}" / f"backtest_{year}.parquet"


def full_grid_and_provenance():
    populate(r52.SEEDS, r52.SEASONS, lambda s, y: 138, lambda s, y: 137)
    run = queue / "out" / "old" / f"seed_{r52.SEEDS[0]}"
    path = run / "backtest_2021.parquet"
    marker = run / "DONE"
    report = queue.parent / "result.json"
    out = check("strict -0.5 pt floor on fixed grid", 1, "--json", report)
    payload = json.loads(report.read_text())
    assert len(payload["cells"]) == 40 and set(payload["seeds"]) == set(r52.SEEDS)
    assert payload["t_0.95"] == r52.t_quantile_95(7)
    assert payload["cells"][0]["old_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "above -0.5 pt): False" in out and "below -1.0 pt): True" in out
    for flag, value in (("--seeds", "1,1"), ("--seasons", "2021,2021")):
        check(f"duplicate CLI {flag} rejected", 2, flag, value)
    missing = marker.with_name("marker-held")
    marker.rename(missing)
    check("15 completed runs", 2)
    missing.rename(marker)
    saved = changed(path, lambda df: df.iloc[:2])
    check("one readable partial file", 2)
    restore(path, saved)
    saved = changed(path, lambda df: df)
    late = (base + timedelta(hours=1, seconds=3)).timestamp()
    os.utime(path, (late, late))
    check("replacement at end +3 s", 2)
    restore(path, saved)
    for seed in r52.SEEDS:
        for year in r52.SEASONS:
            for arm in r52.ARMS:
                path = cell(queue, arm, seed, year)
                # Day 138 is precisely the day on which old hit and new missed.
                changed(path, lambda df: df.drop(index=[274, 275]))
    check("same day cut in every run and season flips FAIL", 0)

    populate(r52.SEEDS, r52.SEASONS, lambda s, y: 150, lambda s, y: 148 if y == 2021 else 150)
    out = check("inclusive -1.0 pt floor on fixed grid", 0)
    assert "above -0.5 pt): True" in out and "below -1.0 pt): True" in out
    assert "p_57_exact" not in out and "mean_max_streak" not in out
    print("default invocation omits both streak metrics")

    path = cell(queue, "new", r52.SEEDS[0], 2021)
    saved = changed(path, lambda df: df.assign(actual_hit=0))
    check("same-date older-mtime outcome replacement flips PASS", 1)
    restore(path, saved)

    # A deterministic replacement between the comparator's separate stat/read calls.
    target = cell(queue, "new", r52.SEEDS[0], 2021)
    incoming = target.with_suffix(".incoming")
    pd.read_parquet(target).assign(actual_hit=0).to_parquet(incoming, index=False)
    assert incoming.stat().st_mtime > target.with_name("DONE").stat().st_mtime
    read_bytes = Path.read_bytes

    def swap_on_read(current):
        if current == target:
            incoming.replace(target)
        return read_bytes(current)

    with patch.object(Path, "read_bytes", swap_on_read):
        check("after-DONE replacement between stat and read flips PASS", 1)


def log_and_timestamps():
    seeds, seasons = (1, 2), (2021,)
    grid = seeds, seasons
    populate(seeds, seasons, lambda s, y: 150, lambda s, y: 150)
    record = queue / "progress.log"
    lines = record.read_text().splitlines(keepends=True)
    now = (base + timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")
    before = (base - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")

    def with_log(label, content, status):
        record.write_bytes(content if isinstance(content, bytes) else "".join(content).encode())
        check(label, status, grid=grid)

    with_log("invalid UTF-8 after full history", "".join(lines).encode() + b"\xff", 2)
    with_log("matched impossible date", [*lines, "2026-11-32 00:00:00 start arm=old seed=1\n"], 2)
    with_log("overlapping starts", [lines[0], lines[2].replace(lines[2][:19], lines[0][:19]), *lines[1:]], 2)
    with_log("stray successful end after failure", [lines[0], lines[1].replace("rc=0", "rc=1"), *lines[1:]], 2)
    with_log("backward time", [*lines, f"{before} start arm=old seed=2\n"], 2)
    with_log("same-run serial restart", [f"{before} start arm=old seed=1\n", "operator restart note\n", *lines], 0)
    with_log("later unfinished attempt supersedes a success", [*lines, f"{now} start arm=old seed=1\n"], 2)
    with_log("malformed later start discarded", [*lines, f"{now} start arm=old seed=1 extra\n"], 0)
    with_log("malformed failed end discarded, later stray success counted", [lines[0], lines[1].replace("rc=0", "rc=-1"), *lines[1:]], 0)
    with_log("unexpected seed accepted", [*lines, f"{now} start arm=new seed=999\n"], 0)
    with_log("start with a return code accepted", [lines[0].rstrip() + " rc=7\n", *lines[1:]], 0)
    record.write_text("".join(lines))

    path = cell(queue, "old", 1, 2021)
    marker = path.with_name("DONE")
    saved = path.read_bytes(), path.stat()
    marker_saved = marker.read_bytes(), marker.stat()
    for label, offset, marker_offset, status in (
        ("pre-start stale file", -1, 3600.5, 2),
        ("file at exact start", 0, 3600.5, 0),
        ("DONE before logged end", 60, 120, 0),
        ("file writes after the logged end second", 3601.25, 3601.5, 0),
        ("late DONE", 60, 3602.1, 2),
    ):
        t_file = base.timestamp() + offset
        t_marker = base.timestamp() + marker_offset
        os.utime(path, (t_file, t_file))
        os.utime(marker, (t_marker, t_marker))
        check(label, status, grid=grid)
    restore(path, saved)
    restore(marker, marker_saved)
    marker.unlink()
    marker.mkdir()
    stamp = marker_saved[1]
    os.utime(marker, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    check("directory posing as DONE", 0, grid=grid)


def shape_publication_and_streaks():
    seeds, seasons = (1, 2), (2021, 2022)
    grid = seeds, seasons
    populate(seeds, seasons, lambda s, y: 150, lambda s, y: 150)
    path = cell(queue, "old", 1, 2021)
    cases = (
        ("missing date column", lambda df: df.drop(columns="date"), 2),
        ("mixed year", lambda df: df.assign(date=df.date.where(df.index != 0, "2022-01-01")), 2),
        ("rank strings", lambda df: df.assign(rank=df["rank"].astype(str)), 2),
        ("infinite rank", lambda df: df.assign(rank=df["rank"].where(df.index != 0, float("inf"))), 2),
        ("boolean rank", lambda df: df.assign(rank=df["rank"] == 1), 2),
        ("floating rank integers", lambda df: df.assign(rank=df["rank"].astype(float)), 0),
        ("outcome strings", lambda df: df.assign(actual_hit=df.actual_hit.astype(str)), 2),
        ("boolean outcomes", lambda df: df.assign(actual_hit=df.actual_hit.astype(bool)), 2),
        ("top outcome null", lambda df: df.assign(actual_hit=df.actual_hit.where(df.index != 0)), 2),
        ("top outcome 2", lambda df: df.assign(actual_hit=df.actual_hit.where(df.index != 0, 2)), 2),
        ("rank-2 outcome null", lambda df: df.assign(actual_hit=df.actual_hit.where(df.index != 1)), 0),
        ("rank-2 outcome 7", lambda df: df.assign(actual_hit=df.actual_hit.where(df.index != 1, 7)), 0),
        ("integer timestamps posing as writer dates", lambda df: df.assign(date=pd.to_datetime(df.date).astype("datetime64[ns]").astype("int64")), 0),
        ("equal-count shifted date set", lambda df: df.assign(date=(pd.to_datetime(df.date) + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")), 2),
        ("one lower-rank row removed with all dates retained", lambda df: df.drop(index=1), 0),
    )
    for name, alter, status in cases:
        saved = changed(path, alter)
        check(name, status, grid=grid)
        restore(path, saved)
    saved = path.read_bytes(), path.stat()
    path.write_bytes(saved[0][:100])
    os.utime(path, ns=(saved[1].st_atime_ns, saved[1].st_mtime_ns))
    check("byte-truncated Parquet", 2, grid=grid)
    restore(path, saved)

    repeated = pd.DataFrame({"date": ["2021-01-01", "2021-01-01", "2021-01-02"], "rank": [1, 1, 1], "actual_hit": [0, 1, 1]})
    assert scorecard.compute_precision_at_k(repeated, k_values=[1])[1] == 0.75
    saved = changed(path, lambda df: df.assign(rank=df["rank"].where(df.index != 1, 1)))
    check("double rank-1 row rejected despite scorecard daily weighting", 2, grid=grid)
    restore(path, saved)
    for seeds_arg, seasons_arg in [((1, 1), seasons), (seeds, (2021, 2021))]:
        try:
            r52.compare(queue / "out", seeds_arg, seasons_arg)
        except r52.Incomplete:
            print("duplicate Python seam grid rejected")
        else:
            raise AssertionError("duplicate grid accepted")

    record = queue / "progress.log"
    for dest in (path, record):
        before = dest.read_bytes()
        check("ordinary JSON evidence target refused", 2, "--json", dest, grid=grid)
        assert dest.read_bytes() == before
    check("ordinary publication error suppresses verdict", 2, "--json", queue.parent, grid=grid)
    check("nonpositive trials rejected", 2, "--mc-trials", "0", grid=grid)

    def null_scorecard(df, mc_trials):
        assert df["rank"].min() == 1 and mc_trials == 1
        return {"p_57_exact": None, "streak_metrics": {"mean_max_streak": float("nan")}}

    result_path = queue.parent / "streak.json"
    with patch.object(scorecard, "compute_full_scorecard", null_scorecard):
        out = check("null and nonfinite streaks report unavailable", 0, "--streak", "--mc-trials", "1", "--json", result_path, grid=grid)
    metrics = json.loads(result_path.read_text())["streak_metrics_reported_only"]
    assert all(v == {"p_57_exact": "unavailable (None)", "mean_max_streak": "unavailable (nan)"} for v in metrics.values())
    assert "unavailable (None)" in out and "unavailable (nan)" in out
    assert out.index("p_57_exact") < out.index("R52 verdict:")

    # Resolve(root).parent differs from the lexical parent used to find progress.log.
    alias = queue.parent / "alias"
    alias.mkdir()
    (alias / "out").symlink_to(queue / "out", target_is_directory=True)
    alias_log = alias / "progress.log"
    alias_log.write_bytes(record.read_bytes())
    check("JSON replaces actually used log with a symlinked root", 0, "--json", alias_log, grid=grid, root=alias / "out")
    assert json.loads(alias_log.read_text())["verdict"] == "PASS"
    check("rerun after overwriting its log", 2, grid=grid, root=alias / "out")

    # A real subprocess confirms the exit status rather than just an exception from main().
    loop = queue.parent / "loop"
    loop.symlink_to(loop)
    cli = REPO / "scripts/audit/bullpen_r52_compare.py"
    output = subprocess.run([sys.executable, "-B", str(cli), "--root", str(queue / "out"), "--json", str(loop)], capture_output=True, text=True)
    assert output.returncode == 1 and "Symlink loop" in output.stderr and "R52 verdict:" not in output.stdout, output
    print("symlink-loop JSON target raises at resolve -> process exit 1")


def writer_and_discriminating_tests():
    from bts.simulate import backtest_blend as writer
    from bts.simulate.monte_carlo import load_profiles
    from bts.simulate.quality_bins import _rank_pair_frame

    frames = []
    for i in range(4):
        # One batter appears in both games; equal scores retain the first tie.
        pa = pd.DataFrame({"batter_id": [10, 20, 10, 30], "game_pk": [11, 11, 12, 12],
                           "is_home": [False] * 4, "pitcher_id": [1, 1, 2, 2],
                           "is_hit": [i % 2, 1, 1, 0]})
        pred = writer._estimated_pa_game_predictions(
            pa, blend={"baseline": (None, [], lambda model, rows, cols: [0.3] * len(rows))},
            blend_pa_scores={"baseline": [0.4, 0.4, 0.4, float("nan")]},
            side_channel_names=set(), model_train_window=pa, top_n=10, capture_per_model=False,
        )
        assert pred["rank"].tolist() == [1, 2, 3, 4] and pd.isna(pred.iloc[-1]["p_game_hit"])
        assert pred.loc[pred.batter_id == 10, "game_pk"].tolist() == [11, 12]
        pred["date"] = date(2021, 4, 1) + timedelta(days=i)
        frames.append(pred[writer.PROFILE_COLUMNS + writer.ESTIMATED_PA_EXTRA_COLUMNS])
    profiles = pd.concat(frames, ignore_index=True)
    # Use finite scores for the positive streak case and exercise the real writer/reader.
    profiles["p_game_hit"] = profiles["p_game_hit"].fillna(0.2)
    root = queue / "out"
    run = root / "old" / "seed_1"
    path = writer.save_profiles(profiles, 2021, run)
    (run / "DONE").touch()
    log(queue, (1,))
    windows = r52.run_windows(queue / "progress.log")
    exact, day_set, digest, df = r52._cell(root, "old", 1, 2021, windows[("old", 1)])
    assert float(exact) == 0.5 and len(day_set) == 4 and digest == hashlib.sha256(path.read_bytes()).hexdigest()
    res = r52.streak_report({("old", 1, 2021): df}, 1)["old/1/2021"]
    assert all(isinstance(res[key], float) for key in ("p_57_exact", "mean_max_streak")), res
    without_second = df.drop(index=1)
    assert len(load_profiles(without_second)) == 4 and len(_rank_pair_frame(without_second)) == 3
    print("writer date32, ties, doubleheader, NaN rank, numeric streaks and rank-2 fallback verified")

    source = Path(r52.__file__).read_text()
    equal_mean = 'sum((cells[(s, y)]["new"] - cells[(s, y)]["old"] for y in seasons), Fraction(0)) / len(seasons)'
    weighted_mean = ('sum(((cells[(s, y)]["new"] - cells[(s, y)]["old"]) * cells[(s, y)]["days"] '
                     'for y in seasons), Fraction(0)) / sum(cells[(s, y)]["days"] for y in seasons)')
    mutations = {
        "test_lower_bound_exactly_at_floor_fails": [
            ("cond1 = gap > 0 and gap * gap >", "cond1 = gap >= 0 and gap * gap >="),
        ],
        "test_season_exactly_at_floor_passes": [("m >= SEASON_FLOOR", "m > SEASON_FLOOR")],
        "test_seasons_weigh_equally_whatever_their_length": [(equal_mean, weighted_mean)],
    }
    for test_name, replacements in mutations.items():
        altered = source
        for old, new in replacements:
            assert altered.count(old) == 1
            altered = altered.replace(old, new)
        mutant = ModuleType("r52_synthetic_mutant")
        exec(compile(altered, r52.__file__, "exec"), mutant.__dict__)
        test = fixtures[test_name]
        with TemporaryDirectory(dir=queue) as tmp, patch.dict(test.__globals__, {"r52": mutant}):
            try:
                test(Path(tmp))
            except AssertionError:
                print(f"{test_name} catches the corresponding mutant")
            else:
                raise AssertionError(f"mutant survived {test_name}")


if os.getpriority(os.PRIO_PROCESS, 0) != 19:
    raise SystemExit("probe requires nice 19")
with TemporaryDirectory(dir=HERE, prefix="r52fu-synthetic-") as tmp:
    queue = Path(tmp) / "queue"
    queue.mkdir()
    {"grid": full_grid_and_provenance, "log": log_and_timestamps, "shape": shape_publication_and_streaks,
     "writer": writer_and_discriminating_tests}[group]()
