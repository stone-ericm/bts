"""Wall-clock (2027/2028) and scratch-copy mutation probe for tests/test_wr_a_*.py."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time as _time

ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
NEW_TESTS = [
    "tests/test_wr_a_calendar.py",
    "tests/test_wr_a_pick.py",
    "tests/test_wr_a_schedule.py",
]


def _run_under_wall_clock(wall_iso: str) -> int:
    # Import pandas C extensions BEFORE replacing datetime.date/datetime so Cython's
    # PyDateTime_IMPORT sees the C base types, while bts.* and tests.* imported after
    # bind the shifted wall-clock classes.
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import pytest

    assert "bts" not in sys.modules

    real_dt = _dt.datetime
    real_date = _dt.date
    real_time = _time.time
    target = real_dt.fromisoformat(wall_iso)
    t0 = real_time()

    class WallDate(real_date):
        @classmethod
        def today(cls):
            cur = target + _dt.timedelta(seconds=real_time() - t0)
            return real_date(cur.year, cur.month, cur.day)

    class WallDatetime(real_dt):
        @classmethod
        def now(cls, tz=None):
            cur = target + _dt.timedelta(seconds=real_time() - t0)
            return cur.astimezone(tz) if tz is not None else cur.replace(tzinfo=None)

        @classmethod
        def utcnow(cls):
            cur = (target + _dt.timedelta(seconds=real_time() - t0)).astimezone(
                _dt.timezone.utc
            )
            return cur.replace(tzinfo=None)

    _dt.date = WallDate
    _dt.datetime = WallDatetime
    _time.time = lambda: target.timestamp() + (real_time() - t0)

    return int(pytest.main(["-p", "no:cacheprovider", "-q", *NEW_TESTS]))


MUTANTS = [
    (
        "M1a_sched_dm_exempt",
        "bts/scheduler.py",
        "        refusal = saved_pick_refusal(daily, decision_state)",
        (
            "        fresh_choice = (\n"
            "            selection is not None and selection.decision is not None\n"
            "            and selection.pick_result is not None and selection.pick_result.daily is daily\n"
            "        )\n"
            '        refusal = None if (fresh_choice and mode == "dm") else saved_pick_refusal(daily, decision_state)'
        ),
    ),
    (
        "M1b_sched_public_exempt",
        "bts/scheduler.py",
        "        refusal = saved_pick_refusal(daily, decision_state)",
        (
            "        fresh_choice = (\n"
            "            selection is not None and selection.decision is not None\n"
            "            and selection.pick_result is not None and selection.pick_result.daily is daily\n"
            "        )\n"
            '        refusal = None if (fresh_choice and mode == "public") else saved_pick_refusal(daily, decision_state)'
        ),
    ),
    (
        "M1c_sched_fallback_exempt",
        "bts/scheduler.py",
        "        refusal = saved_pick_refusal(daily, decision_state)",
        (
            "        fresh_choice = (\n"
            "            selection is not None and selection.decision is not None\n"
            "            and selection.pick_result is not None and selection.pick_result.daily is daily\n"
            "        )\n"
            '        refusal = None if (fresh_choice and label == "fallback") else saved_pick_refusal(daily, decision_state)'
        ),
    ),
    (
        "M1d_sched_final_fallback_exempt",
        "bts/scheduler.py",
        "        refusal = saved_pick_refusal(daily, decision_state)",
        (
            "        fresh_choice = (\n"
            "            selection is not None and selection.decision is not None\n"
            "            and selection.pick_result is not None and selection.pick_result.daily is daily\n"
            "        )\n"
            '        refusal = None if (fresh_choice and label == "final fallback") else saved_pick_refusal(daily, decision_state)'
        ),
    ),
    (
        "M1e_cli_run_fresh_exempt",
        "bts/cli.py",
        "    if should_post_now(daily.pick.game_time, daily.bluesky_posted):\n        refusal = saved_pick_refusal(daily, decision_state)",
        "    if should_post_now(daily.pick.game_time, daily.bluesky_posted):\n        refusal = saved_pick_refusal(daily, decision_state) if sel.decision is None else None",
    ),
    (
        "M1f_cli_run_catchup_base_cond",
        "bts/cli.py",
        "                raise SystemExit(1) from exc\n            refusal = saved_pick_refusal(result.daily, decision_state)",
        "                raise SystemExit(1) from exc\n            refusal = saved_pick_refusal(result.daily, decision_state) if sel.decision is None else None",
    ),
    (
        "M1f2_cli_run_catchup_skip_refusal",
        "bts/cli.py",
        "                raise SystemExit(1) from exc\n            refusal = saved_pick_refusal(result.daily, decision_state)",
        "                raise SystemExit(1) from exc\n            refusal = None",
    ),
    (
        "M1g_orchestrate_fresh_exempt",
        "bts/orchestrator.py",
        "    if should_post_now(daily.pick.game_time, daily.bluesky_posted):\n        if required:\n            refusal = saved_pick_refusal(daily, decision_state)",
        "    if should_post_now(daily.pick.game_time, daily.bluesky_posted):\n        if required and sel.decision is None:\n            refusal = saved_pick_refusal(daily, decision_state)",
    ),
    (
        "M1h_orchestrate_catchup_base_cond",
        "bts/orchestrator.py",
        "            if required:\n                refusal = saved_pick_refusal(result.daily, decision_state)",
        "            if required and sel.decision is None:\n                refusal = saved_pick_refusal(result.daily, decision_state)",
    ),
    (
        "M1h2_orchestrate_catchup_skip_refusal",
        "bts/orchestrator.py",
        "            if required:\n                refusal = saved_pick_refusal(result.daily, decision_state)",
        "            if False:\n                refusal = saved_pick_refusal(result.daily, decision_state)",
    ),
    (
        "M2_compare_only_streak",
        "bts/picks.py",
        (
            '        if record["kind"] == live_kind and (\n'
            '            live_kind == "automatic" or (live_kind == "dated" and record["dated"] == live_dated)\n'
            "        ):\n"
            "            return None"
        ),
        '        if record["streak"] == decision_state.streak:\n            return None',
    ),
    (
        "M3a_refuse_auto_to_auto_all",
        "bts/picks.py",
        'live_kind == "automatic" or ',
        "False or ",
    ),
    (
        "M3b_refuse_auto_to_auto_diff_streak",
        "bts/picks.py",
        'live_kind == "automatic"',
        '(live_kind == "automatic" and record["streak"] == decision_state.streak)',
    ),
    (
        "M4_2027_back_to_0926",
        "bts/strategy.py",
        'SEASON_END_DATES = {2026: "2026-09-28", 2027: "2027-09-27"}',
        'SEASON_END_DATES = {2026: "2026-09-28", 2027: "2027-09-26"}',
    ),
    (
        "M5_inclusive_2026",
        "bts/strategy.py",
        "    return max(0, (end - today).days)",
        "    return max(0, (end - today).days + 1)",
    ),
    (
        "M6_nonlist_dates_empty",
        "bts/scheduler.py",
        '    if not isinstance(dates, list):\n        raise ValueError("MLB schedule dates must be a list")',
        "    if not isinstance(dates, list):\n        return []",
    ),
    (
        "M7_skip_filtered_game_validation",
        "bts/scheduler.py",
        (
            "        for game_index, game in enumerate(day_games):\n"
            "            if not isinstance(game, dict):\n"
            '                raise ValueError(f"MLB schedule {location}.games[{game_index}] must be an object")\n'
            "        games.extend(day_games)"
        ),
        "        games.extend(g for g in day_games if isinstance(g, dict) and is_regular_season_game(g))",
    ),
]


def _run_nice_pytest(extra_env: dict[str, str], args: list[str]) -> tuple[int, str]:
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    mpl = REVIEW_DIR / ".mpl"
    tmp = REVIEW_DIR / ".tmp"
    mpl.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl)
    env["TMPDIR"] = str(tmp)
    env.update(extra_env)
    proc = subprocess.run(
        ["/usr/bin/nice", "-n", "19", str(ROOT / ".venv/bin/python"), *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    return proc.returncode, (proc.stdout + "\n" + proc.stderr).strip()


def run_suite() -> dict:
    wall_clocks = [
        "2027-03-24T12:00:00-04:00",
        "2027-10-15T12:00:00-04:00",
        "2028-02-29T12:00:00-05:00",
        "2028-06-15T12:00:00-04:00",
    ]
    wall_results = []
    for wc in wall_clocks:
        rc, out = _run_nice_pytest({}, [str(Path(__file__).resolve()), "--wall-clock", wc])
        assert rc == 0, f"Wall clock {wc} failed:\n{out}"
        last_line = [ln for ln in out.splitlines() if "passed" in ln][-1]
        wall_results.append({"wall_clock": wc, "exit_code": rc, "summary": last_line})

    scratch = REVIEW_DIR / ".tmp" / "mutant_src"
    if scratch.exists():
        shutil.rmtree(scratch)
    shutil.copytree(ROOT / "src", scratch)

    try:
        rc0, out0 = _run_nice_pytest(
            {"PYTHONPATH": f"{scratch}:{ROOT}"},
            ["-m", "pytest", "-p", "no:cacheprovider", "-q", *NEW_TESTS],
        )
        assert rc0 == 0, f"Unmodified scratch failed:\n{out0}"

        mutant_results = []
        for name, rel_file, target, replacement in MUTANTS:
            fpath = scratch / rel_file
            original = fpath.read_text()
            assert original.count(target) == 1, (name, rel_file, original.count(target))
            try:
                fpath.write_text(original.replace(target, replacement, 1))
                rc, out = _run_nice_pytest(
                    {"PYTHONPATH": f"{scratch}:{ROOT}"},
                    ["-m", "pytest", "-p", "no:cacheprovider", "-q", *NEW_TESTS],
                )
                killed_by_new = rc != 0
                summary_line = [ln for ln in out.splitlines() if "passed" in ln or "failed" in ln][-1]
                killed_by_cs5a = None
                if not killed_by_new:
                    rc5, _ = _run_nice_pytest(
                        {"PYTHONPATH": f"{scratch}:{ROOT}"},
                        ["-m", "pytest", "-p", "no:cacheprovider", "-q", "tests/test_saved_pick_paths_cs5a.py"],
                    )
                    killed_by_cs5a = rc5 != 0
                mutant_results.append({
                    "mutant": name,
                    "file": rel_file,
                    "killed_by_new_tests": killed_by_new,
                    "new_tests_summary": summary_line,
                    "killed_by_cs5a_if_survived": killed_by_cs5a,
                })
            finally:
                fpath.write_text(original)
    finally:
        if scratch.exists():
            shutil.rmtree(scratch)

    return {"wall_clocks": wall_results, "mutants": mutant_results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wall-clock", default=None)
    ns = ap.parse_args()
    if ns.wall_clock is not None:
        sys.exit(_run_under_wall_clock(ns.wall_clock))
    res = run_suite()
    (REVIEW_DIR / "probe_mutants_and_clocks.json").write_text(
        json.dumps(res, indent=2)
    )
    print(json.dumps(res, indent=2))
