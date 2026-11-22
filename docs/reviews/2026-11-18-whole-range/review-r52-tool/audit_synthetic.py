"""Synthetic probes and source mutants for the R52 review."""

from contextlib import redirect_stderr, redirect_stdout
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from scripts.audit import bullpen_r52_compare as r52
from bts.simulate.backtest_blend import _estimated_pa_game_predictions, save_profiles
from bts.validate.scorecard import compute_precision_at_k

TEST_PATH = REPO / "tests/scripts/test_bullpen_r52_compare.py"
spec = spec_from_file_location("reviewed_tests", TEST_PATH)
suite = module_from_spec(spec)
spec.loader.exec_module(suite)

MUTANTS = {
    "inclusive-lower": ("lower > LOWER_BOUND_FLOOR", "lower >= LOWER_BOUND_FLOOR"),
    "exclusive-season": ("m >= SEASON_FLOOR", "m > SEASON_FLOOR"),
    "day-weighted-season": (
        'sum(cells[(s, y)]["diff"] for y in seasons) / len(seasons)',
        'sum(cells[(s, y)]["diff"] * cells[(s, y)]["old_days"] for y in seasons) / '
        'sum(cells[(s, y)]["old_days"] for y in seasons)',
    ),
    "first-ranked-row-per-day": (
        "compute_precision_at_k(df, k_values=[1])",
        'compute_precision_at_k(df.sort_values("rank").drop_duplicates("date"), k_values=[1])',
    ),
    "two-sided-quantile": ("t.ppf(0.95, df)", "t.ppf(0.975, df)"),
    "wrong-df": ("tq = t_quantile_95(n - 1)", "tq = t_quantile_95(n)"),
    "population-sd": ("** 2 for x in d) / (n - 1)", "** 2 for x in d) / n"),
    "ignore-season-gate": ("if (cond1 and cond2) else", "if cond1 else"),
    "ignore-done": ('if not (run / "DONE").is_file():', "if False:"),
    "ignore-year": ("if years != [season]:", "if False:"),
}


def run_mutants(root):
    source_path = Path(r52.__file__)
    source = source_path.read_text()
    tests = sorted(name for name in vars(suite) if name.startswith("test_"))
    for label, (before, after) in MUTANTS.items():
        assert source.count(before) == 1, label
        changed = ModuleType(label)
        exec(compile(source.replace(before, after, 1), str(source_path), "exec"), changed.__dict__)
        suite.r52 = changed
        caught = []
        for name in tests:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                try:
                    getattr(suite, name)(root / label / name)
                except (AssertionError, pytest.fail.Exception):
                    caught.append(name)
        print(f"{label}: {', '.join(caught) or 'SURVIVED'}")


def run_cli(root, seeds=None, seasons=None, extra=()):
    args = ["--root", str(root)]
    if seeds is not None:
        args += ["--seeds", ",".join(map(str, seeds))]
    if seasons is not None:
        args += ["--seasons", ",".join(map(str, seasons))]
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            status = r52.main(args + list(extra))
        except Exception as exc:
            status = type(exc).__name__
    verdict = [line for line in out.getvalue().splitlines() if line.startswith("R52 verdict:")]
    return status, verdict, out.getvalue()


def boundary_probes(root):
    for name, old, new, expected in (
        ("strict-lower", 138, lambda y: 137, "FAIL"),
        ("inclusive-season", 150, lambda y: 148 if y == 2021 else 150, "PASS"),
    ):
        for seed in r52.SEEDS:
            for year in r52.SEASONS:
                suite._write(root, "old", seed, year, old)
                suite._write(root, "new", seed, year, new(year))
        result = r52.compare(root, r52.SEEDS, r52.SEASONS)
        print(f"{name}: expected {expected}, got {result['verdict']}; "
              f"lower={result['lower_bound_95']!r}, 2021={result['season_mean_diff']['2021']!r}")


def queue_probes(root):
    suite._grid(root, lambda s, y: 150, lambda s, y: 150)
    seeds, years = suite.SEEDS, suite.SEASONS
    print("same partial queue, canonical then overridden:", run_cli(root)[:2], run_cli(root, seeds, years)[:2])
    print("duplicate seeds and seasons:", run_cli(root, [1, 1], [2021, 2021])[:2])

    run = root / "new/seed_1"
    target = run / "backtest_2021.parquet"
    clean = pd.read_parquet(target)
    original = target.read_bytes()
    (run / "DONE").unlink()
    target.write_bytes(original[:len(original) // 2])
    print("truncated, absent marker:", run_cli(root, seeds, years)[:2])
    (run / "DONE").touch()
    print("truncated, marker present:", run_cli(root, seeds, years)[:2])

    def check_frame(label, frame):
        frame.to_parquet(target)
        print(f"{label}:", run_cli(root, seeds, years)[:2])

    check_frame("missing date column", clean.drop(columns="date"))
    changed_dates = clean.copy()
    changed_dates.loc[0, "date"] = "2022-04-01"
    check_frame("mixed years", changed_dates)
    check_frame("one missing rank 1", clean.drop(index=0))
    check_frame("all rank 1 missing", clean[clean["rank"] > 1])
    valid_partial = clean.iloc[:2]
    check_frame("readable partial, marker present", valid_partial)
    stale = clean.copy()
    stale.to_parquet(target)
    os.utime(target, ns=(1_000_000_000, 1_000_000_000))
    print("stale file, marker present:", run_cli(root, seeds, years)[:2])

    shifted = clean.copy()
    shifted["date"] = (pd.to_datetime(shifted["date"]) + pd.Timedelta(days=5)).dt.strftime("%Y-%m-%d")
    check_frame("shifted date set, equal counts", shifted)
    print("shifted date set warnings:", r52.compare(root, seeds, years)["day_count_mismatches"])
    check_frame("unequal counts", clean.iloc[:-2])
    print("unequal day count warnings:", r52.compare(root, seeds, years)["day_count_mismatches"])

    one_null = clean.copy()
    one_null.loc[0, "actual_hit"] = float("nan")
    check_frame("one null outcome", one_null)
    print("one null outcome P@1 and reported days:", r52.load_p_at_1(root, "new", 1, 2021))
    print("same input with --streak:", run_cli(root, seeds, years, ["--streak", "--mc-trials", "1"])[:2])
    check_frame("restored good input", clean)
    print("verdict printed on JSON path error:", run_cli(root, seeds, years, ["--json", str(root)])[:2])


def profile_probes(root):
    rows = pd.DataFrame({
        "batter_id": [1, 1, 1, 2], "game_pk": [10, 10, 20, 20],
        "pitcher_id": [11, 12, 22, 22], "is_home": [True] * 4,
        "lineup_position": [1] * 4, "is_hit": [0, 1, 0, 1],
    })

    def predict(_model, frame, _cols):
        return [0.3] * len(frame)

    def exported(scores):
        frame = _estimated_pa_game_predictions(
            rows, blend={"baseline": (None, [], predict)}, blend_pa_scores={"baseline": scores},
            side_channel_names=set(), model_train_window=rows, top_n=10, capture_per_model=False,
        )
        frame["date"] = pd.Timestamp("2021-04-01").date()
        path = save_profiles(frame[["date", "rank", "batter_id", "game_pk", "p_game_hit", "actual_hit", "n_pas"]], 2021, root)
        return pd.read_parquet(path)

    print("writer tied scores and same batter in two games:", exported([0.3] * 4).to_dict("records"))
    print("writer all-NaN scores still ranked:", exported([float("nan")] * 4)[["rank", "p_game_hit"]].values.tolist())
    odd = pd.DataFrame({"date": ["2021-04-01", "2021-04-01", "2021-04-02", "2021-04-03", "2021-04-04"],
                        "rank": [1, 1, 1, 1, 2], "actual_hit": [0, 1, 1, float("nan"), 1]})
    print("scorecard: tied rank 1, null outcome, and no rank 1:", compute_precision_at_k(odd, [1]))


def streak_probe(root):
    for arm in r52.ARMS:
        for seed in (1, 2):
            for year in (2021, 2022):
                suite._write(root, arm, seed, year, 200)
    status, verdict, output = run_cli(root, [1, 2], [2021, 2022], ["--streak", "--mc-trials", "1"])
    print("streak flag without JSON:", status, verdict, "streak shown:", "streak" in output.lower())


if __name__ == "__main__":
    mode = sys.argv[1]
    priority = os.getpriority(os.PRIO_PROCESS, 0)
    assert priority == 19, f"expected priority 19, got {priority}"
    if mode == "baseline":
        with TemporaryDirectory(dir=HERE) as temp:
            sys.exit(pytest.main([str(TEST_PATH), "-q", "-p", "no:cacheprovider", "--basetemp", temp]))
    probes = {"mutants": run_mutants, "boundary": boundary_probes, "queue": queue_probes,
              "profiles": profile_probes, "streak": streak_probe}
    with TemporaryDirectory(dir=HERE) as temp:
        probes[mode](Path(temp))
