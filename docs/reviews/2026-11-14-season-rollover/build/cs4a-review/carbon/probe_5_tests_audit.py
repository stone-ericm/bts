"""Adversarial Probe 5: Audit of tests/test_contest_dated_cs4a.py, Fake-Clock Runs & Mutations (Item 8)."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET_XML
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEW_DIR = REPO_ROOT / ".codex-review" / "cs4a-review"
BASE_SRC = REVIEW_DIR / "base_tree" / "src"
BRANCH_SRC = REPO_ROOT / "src"
TEST_FILE = REPO_ROOT / "tests" / "test_contest_dated_cs4a.py"


FAKE_CLOCK_PLUGIN = '''"""Pytest plugin pinning wall-clock datetime.datetime.now and datetime.date.today."""
import datetime as _dt
import os
from zoneinfo import ZoneInfo

import pandas  # import before replacing datetime classes so Cython types initialize cleanly

_TARGET = _dt.datetime.fromisoformat(os.environ["BTS_FAKE_WALL_ISO"])
_REAL_DT = _dt.datetime
_REAL_DATE = _dt.date


class FakeDate(_REAL_DATE):
    @classmethod
    def today(cls):
        d = _TARGET.astimezone(ZoneInfo("America/New_York")).date()
        return cls(d.year, d.month, d.day)


class FakeDateTime(_REAL_DT):
    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return _TARGET.replace(tzinfo=None)
        return _TARGET.astimezone(tz)

    @classmethod
    def utcnow(cls):
        return _TARGET.astimezone(_dt.timezone.utc).replace(tzinfo=None)


def pytest_configure(config):
    _dt.date = FakeDate
    _dt.datetime = FakeDateTime
'''


def run_pytest_xml(pythonpath: str, extra_args: list[str], extra_env: dict[str, str] | None = None) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        xml_path = Path(tmp) / "junit.xml"
        env = dict(
            os.environ,
            PYTHONPATH=pythonpath,
            PYTHONDONTWRITEBYTECODE="1",
            TZ="America/New_York",
        )
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run(
            [
                sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
                f"--junitxml={xml_path}", "-q", *extra_args,
            ],
            env=env, capture_output=True, text=True,
        )
        passed, failed = [], []
        if xml_path.exists():
            root = ET_XML.parse(xml_path).getroot()
            for tc in root.iter("testcase"):
                name = f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}"
                if tc.find("failure") is not None or tc.find("error") is not None:
                    failed.append(name)
                elif tc.find("skipped") is None:
                    passed.append(name)
        return {
            "returncode": proc.returncode,
            "passed_count": len(passed),
            "failed_count": len(failed),
            "passed": passed,
            "summary_line": proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "",
        }


def run_mutations() -> list[dict]:
    mutations = [
        (
            "M1_weak_saver_available_hint",
            "bts/cli.py",
            '"--saver-available/--saver-unavailable" if name == "saver_available"',
            '"--saver" if name == "saver_available"',
        ),
        (
            "M2_isinstance_int_allows_bool_season",
            "bts/contest_state.py",
            "if type(season) is not int or season != effective_date.year:",
            "if not isinstance(season, int) or season != effective_date.year:",
        ),
        (
            "M3_decision_reads_flag_when_dated",
            "bts/contest_state.py",
            "saver = (SaverState(state=contest.saver_state, season=contest.effective_date.year)\n             if dated_in_effect else load_saver_state(picks_dir, season=now_year))",
            "_ = load_saver_state(picks_dir, season=now_year)\n    saver = (SaverState(state=contest.saver_state, season=contest.effective_date.year)\n             if dated_in_effect else _)",
        ),
        (
            "M4_health_omits_early_return_reads_flag",
            "bts/health/contest_state.py",
            "        return alerts\n\n    # The flag belongs",
            "        pass\n\n    # The flag belongs",
        ),
        (
            "M5_best_streak_uses_strict_gt_expiry_at_eod",
            "bts/contest_state.py",
            "    if dated_observation_in_effect(contest, now):\n        return best, \"trusted\"",
            "    if False:\n        return best, \"trusted\"",
        ),
        (
            "M6_decision_consults_local_picks_for_freshness",
            "bts/contest_state.py",
            "if dated_in_effect or contest_state_is_fresh(contest, picks_dir):",
            "if contest_state_is_fresh(contest, picks_dir):",
        ),
        (
            "M7_health_fires_legacy_warn_on_out_of_date_v3",
            "bts/health/contest_state.py",
            "if not dated and (exp_dt is None or exp_dt <= now):",
            "if exp_dt is None or exp_dt <= now:",
        ),
        (
            "M8_noop_rewrites_identical_file",
            "bts/cli.py",
            "observation, state, unchanged = snapshot, snapshot_state, True",
            "observation, state, unchanged = snapshot, snapshot_state, False",
        ),
        (
            "M9_out_of_date_v3_falls_back_when_auto_absent",
            "bts/contest_state.py",
            "manual = None",
            "pass",
        ),
        (
            "M10_double_clock_read_in_decision_loader",
            "bts/contest_state.py",
            "contest = load_contest_streak_state(\n        picks_dir, now=now_dt, require_dated_auto=require_contest_state)",
            "contest = load_contest_streak_state(\n        picks_dir, now=now, require_dated_auto=require_contest_state)",
        ),
    ]

    results = []
    for m_name, rel_file, target, replacement in mutations:
        with tempfile.TemporaryDirectory() as tmp:
            mut_src = Path(tmp) / "src"
            shutil.copytree(BRANCH_SRC, mut_src)
            fpath = mut_src / rel_file
            txt = fpath.read_text(encoding="utf-8")
            assert target in txt, f"Target not found for {m_name}"
            fpath.write_text(txt.replace(target, replacement, 1), encoding="utf-8")
            res = run_pytest_xml(str(mut_src), [str(TEST_FILE)])
            results.append({
                "mutation": m_name,
                "killed": res["failed_count"] > 0,
                "failed_count": res["failed_count"],
                "passed_count": res["passed_count"],
                "summary": res["summary_line"],
            })
    return results


def main() -> None:
    # 1. Base run of tests/test_contest_dated_cs4a.py
    base_run = run_pytest_xml(str(BASE_SRC), [str(TEST_FILE)])

    # 2. Fake wall-clock runs in 2027 and 2028
    plugin_dir = REVIEW_DIR / "probes" / "_clock_plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "fake_wall_clock.py").write_text(FAKE_CLOCK_PLUGIN, encoding="utf-8")

    fake_clock_runs = {}
    for iso_wall in [
        "2027-07-15T12:00:00-04:00",
        "2027-12-31T23:59:59-05:00",
        "2028-02-29T12:00:00-05:00",
        "2028-07-15T12:00:00-04:00",
    ]:
        r = run_pytest_xml(
            f"{BRANCH_SRC}:{plugin_dir}",
            ["-p", "fake_wall_clock", str(TEST_FILE)],
            extra_env={"BTS_FAKE_WALL_ISO": iso_wall},
        )
        fake_clock_runs[iso_wall] = {
            "returncode": r["returncode"],
            "passed_count": r["passed_count"],
            "failed_count": r["failed_count"],
            "summary": r["summary_line"],
        }

    # 3. Mutation runs
    mut_results = run_mutations()

    report = {
        "base_run": {
            "passed_count": base_run["passed_count"],
            "failed_count": base_run["failed_count"],
            "passed_nodeids": base_run["passed"],
            "summary": base_run["summary_line"],
        },
        "fake_clock_runs": fake_clock_runs,
        "mutations": mut_results,
    }
    (REVIEW_DIR / "probe5_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
