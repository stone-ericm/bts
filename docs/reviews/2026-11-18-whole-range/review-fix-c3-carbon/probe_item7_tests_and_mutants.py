"""Item 7 probe: Base RED verification, 2027/2028 wall-clock runs, and scratch-copy mutation sweep.

Runs pytest sequentially ONE invocation at a time under `/usr/bin/nice -n 19` (since sandbox
denies `/Users/eric/bts-runs/yield_run.sh`), mutating only isolated scratch copies inside
`.codex-review/wr-c3-review/` and leaving every tracked file untouched.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET_XML


REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
VENV_PYTHON = str(REPO_ROOT / ".venv/bin/python")
NEW_TEST_FILE = "tests/test_wr_c3_reconcile.py"


MUTANTS = [
    (
        "1_skip_every_unresolved_past_pick",
        "bts/picks.py",
        '        elif day < today:\n            if scoreable_commit_from(decision, daily):\n                return None, (day, "committed pick awaiting a result")\n            if decision is None and daily.delivery_attempted:\n                return None, (day, "delivery attempt without a decision or result")',
        "        elif day < today:\n            continue",
    ),
    (
        "2_skip_committed_pending_pick",
        "bts/picks.py",
        '            if scoreable_commit_from(decision, daily):\n                return None, (day, "committed pick awaiting a result")',
        '            if False and scoreable_commit_from(decision, daily):\n                return None, (day, "committed pick awaiting a result")',
    ),
    (
        "3_ignore_applied_date_marker",
        "bts/picks.py",
        '        elif day in markers:\n            return None, (day, "applied-date marker without a result")',
        '        elif False and day in markers:\n            return None, (day, "applied-date marker without a result")',
    ),
    (
        "4_fold_void_as_miss",
        "bts/picks.py",
        '        elif daily.result in _TERMINAL_RESULTS:\n            if daily.result != "void":',
        "        elif daily.result in _TERMINAL_RESULTS:\n            if True:",
    ),
    (
        "5_drop_season_filter",
        "bts/picks.py",
        '    dates = {path.stem for path in picks_dir.glob(f"{season}-*.json")}\n    dates.update(path.parent.name for path in picks_dir.glob(f"{season}-*/decision.json"))',
        '    dates = {path.stem for path in picks_dir.glob("*.json")}\n    dates.update(path.parent.name for path in picks_dir.glob("*/decision.json"))',
    ),
    (
        "6_never_block_invalid_decision",
        "bts/picks.py",
        '        except _REPLAY_READ_ERRORS:\n            return None, (day, "unreadable or invalid decision record")',
        "        except _REPLAY_READ_ERRORS:\n            decision = None",
    ),
    (
        "7_print_recalculated_for_retained",
        "bts/cli.py",
        '        click.echo(f"Streak retained: {streak}")',
        '        click.echo(f"Streak recalculated: {streak}")',
    ),
    (
        "8a_write_streak_direct_write_text",
        "bts/picks.py",
        '    elif legacy is not None:\n        data["season"] = legacy["season"]\n    atomic_write_text(path, json.dumps(data))',
        '    elif legacy is not None:\n        data["season"] = legacy["season"]\n    Path(path).write_text(json.dumps(data))',
    ),
    (
        "8b_write_streak_nonatomic_unmocked",
        "bts/picks.py",
        'def _write_streak(path, streak, saver, applied, *, season, legacy=None):\n    data = {\n        "streak": streak,\n        "saver_available": saver,\n        "updated": datetime.now(timezone.utc).isoformat(),\n        "applied_dates": applied,\n    }\n    if season is not None:\n        data["season"] = season\n    elif legacy is not None:\n        data["season"] = legacy["season"]\n    atomic_write_text(path, json.dumps(data))',
        '_ORIG_ATOMIC = atomic_write_text\n\n\ndef _write_streak(path, streak, saver, applied, *, season, legacy=None):\n    data = {\n        "streak": streak,\n        "saver_available": saver,\n        "updated": datetime.now(timezone.utc).isoformat(),\n        "applied_dates": applied,\n    }\n    if season is not None:\n        data["season"] = season\n    elif legacy is not None:\n        data["season"] = legacy["season"]\n    if atomic_write_text is _ORIG_ATOMIC:\n        Path(path).write_text(json.dumps(data))\n    else:\n        atomic_write_text(path, json.dumps(data))',
    ),
]


def _run_one_pytest(
    label: str,
    src_dir: Path,
    test_files: list[str],
    xml_path: Path,
    *,
    wall_iso: str | None = None,
) -> dict:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["TZ"] = "America/New_York"
    env["PYTHONPATH"] = f"{src_dir}:{REVIEW_DIR}"
    env["BTS_ASSERT_SRC_PREFIX"] = str(src_dir)
    if wall_iso:
        env["BTS_PINNED_WALL_ISO"] = wall_iso
    else:
        env.pop("BTS_PINNED_WALL_ISO", None)

    cmd = [
        "/usr/bin/nice",
        "-n",
        "19",
        VENV_PYTHON,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "-p",
        "wallclock_plugin",
        "-q",
        f"--junitxml={xml_path}",
        *test_files,
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=60)

    tree = ET_XML.parse(xml_path)
    passed: list[str] = []
    failed: list[dict] = []
    for tc in tree.findall(".//testcase"):
        node = f"{tc.attrib.get('classname', '')}::{tc.attrib.get('name', '')}"
        fail_el = tc.find("failure")
        err_el = tc.find("error")
        if fail_el is not None:
            failed.append({"node": node, "kind": "failure", "message": (fail_el.attrib.get("message") or "").splitlines()[0][:220]})
        elif err_el is not None:
            failed.append({"node": node, "kind": "error", "message": (err_el.attrib.get("message") or "").splitlines()[0][:220]})
        elif tc.find("skipped") is None:
            passed.append(node)

    return {
        "label": label,
        "returncode": proc.returncode,
        "passed_count": len(passed),
        "failed_count": len(failed),
        "passed": passed,
        "failed": failed,
    }


def main() -> int:
    report: dict = {"base_red": {}, "wall_clocks": {}, "mutants_new_file_only": {}, "survivors_against_streak_suite": {}}

    with TemporaryDirectory(dir=REVIEW_DIR, prefix="_item7_") as scratch_str:
        scratch = Path(scratch_str)

        # 1. Base RED run on c2039541
        base_dir = scratch / "base_c2039541"
        base_dir.mkdir()
        with subprocess.Popen(["git", "archive", "c2039541", "src"], cwd=REPO_ROOT, stdout=subprocess.PIPE) as arch:
            assert arch.stdout is not None
            with tarfile.open(fileobj=arch.stdout, mode="r|") as tf:
                tf.extractall(base_dir, filter="data")
            assert arch.wait() == 0

        report["base_red"] = _run_one_pytest(
            "base_c2039541",
            base_dir / "src",
            [NEW_TEST_FILE],
            scratch / "base.xml",
        )

        # 2. Wall clocks in 2027 and 2028 (mid-season + New Year's Day)
        for wc_label, wc_iso in [
            ("wall_2027_midseason", "2027-07-15T14:00:00-04:00"),
            ("wall_2027_nye", "2027-12-31T23:55:00-05:00"),
            ("wall_2028_new_year", "2028-01-01T00:05:00-05:00"),
            ("wall_2028_midseason", "2028-07-15T14:00:00-04:00"),
        ]:
            res = _run_one_pytest(
                wc_label,
                REPO_ROOT / "src",
                [
                    NEW_TEST_FILE,
                    "tests/test_lightgbm_optional.py",
                    "tests/test_streak_replay.py",
                    "tests/test_streak_applied_dates.py",
                    "tests/test_streak_season_cs2.py",
                    "tests/test_streak_season_cs2_review.py",
                    "tests/test_streak_season_cs2b_review.py",
                ],
                scratch / f"{wc_label}.xml",
                wall_iso=wc_iso,
            )
            report["wall_clocks"][wc_label] = {
                "returncode": res["returncode"],
                "passed_count": res["passed_count"],
                "failed_count": res["failed_count"],
                "failed": res["failed"],
            }

        # 3. Mutants against tests/test_wr_c3_reconcile.py
        for m_name, rel_file, target, replacement in MUTANTS:
            m_src = scratch / f"mut_{m_name}" / "src"
            shutil.copytree(REPO_ROOT / "src", m_src)
            fpath = m_src / rel_file
            orig = fpath.read_text()
            cnt = orig.count(target)
            assert cnt == 1, f"Mutant {m_name}: expected count 1 in {rel_file}, got {cnt}"
            fpath.write_text(orig.replace(target, replacement, 1))

            m_res = _run_one_pytest(
                m_name,
                m_src,
                [NEW_TEST_FILE],
                scratch / f"mut_{m_name}.xml",
            )
            killed = m_res["failed_count"] > 0
            report["mutants_new_file_only"][m_name] = {
                "status": "KILLED" if killed else "SURVIVED",
                "failed_count": m_res["failed_count"],
                "passed_count": m_res["passed_count"],
                "killing_tests": [f["node"] for f in m_res["failed"][:8]],
            }
            if not killed:
                # Check if existing streak suite kills the survivor!
                s_res = _run_one_pytest(
                    f"{m_name}_streak_suite",
                    m_src,
                    ["tests/test_streak_season_cs2b_review.py"],
                    scratch / f"mut_{m_name}_cs2b.xml",
                )
                report["survivors_against_streak_suite"][m_name] = {
                    "status": "KILLED" if s_res["failed_count"] > 0 else "SURVIVED",
                    "failed_count": s_res["failed_count"],
                    "killing_tests": [f["node"] for f in s_res["failed"][:5]],
                }

    out_path = REVIEW_DIR / "item7_results.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "base_red": {"passed": report["base_red"]["passed_count"], "failed": report["base_red"]["failed_count"], "passed_nodes": report["base_red"]["passed"]},
        "wall_clocks": report["wall_clocks"],
        "mutants_new_file_only": report["mutants_new_file_only"],
        "survivors_against_streak_suite": report["survivors_against_streak_suite"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
