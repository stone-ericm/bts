"""Item 8 pytest driver (run inside ~/bts-runs/yield_run.sh):
1. Runs new CS4c test files against archived base 1c6cbc77 src and classifies every pass/fail.
2. Mutates isolated scratch copies of branch src across 24 mutants (covering every brief item-8
   mutation plus unmasked validator/clock mutations), runs author tests, and verifies adoptable
   regression tests kill every surviving mutant.
3. Runs all 4 added/touched test files + adopt/test_cs4c_adversarial_review.py under wall clocks
   pinned to 2027-07-15 and 2028-07-15 via clock_plugin.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = Path(__file__).resolve().parent
BASE_SRC = REVIEW_DIR / "scratch" / "base_1c6cbc77" / "src"
ADOPT_TEST = str(REVIEW_DIR / "adopt" / "test_cs4c_adversarial_review.py")
AUTHOR_NEW_TESTS = [
    "tests/test_automatic_reading_cs4c.py",
    "tests/test_stale_delivery_cs4c.py",
]
ALL_TOUCHED_AND_ADOPT = [
    "tests/test_automatic_reading_cs4c.py",
    "tests/test_stale_delivery_cs4c.py",
    "tests/test_pick_chosen_on_cs5a.py",
    "tests/test_saved_pick_paths_cs5a.py",
    ADOPT_TEST,
]


def _verify_src_import(src_dir: Path) -> None:
    env = dict(os.environ, PYTHONPATH=f"{src_dir}:{REPO_ROOT}", TZ="America/New_York", PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.check_output(
        [sys.executable, "-c", "import bts; print(bts.__file__)"],
        cwd=str(REPO_ROOT), env=env, text=True,
    ).strip()
    assert Path(out).resolve().is_relative_to(src_dir.resolve()), (
        f"Expected bts from {src_dir}, got {out}"
    )


def _run_pytest(args: list[str], *, src_dir: Path | None = None, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    py_paths = []
    if src_dir is not None:
        py_paths.append(str(src_dir))
    py_paths.extend([str(REVIEW_DIR), str(REPO_ROOT)])
    env["PYTHONPATH"] = ":".join(py_paths)
    if extra_env:
        env.update(extra_env)
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *args]
    return subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)


def run_base_audit() -> dict:
    _verify_src_import(BASE_SRC)
    proc = _run_pytest([*AUTHOR_NEW_TESTS, "-v", "--tb=no"], src_dir=BASE_SRC)
    passed = []
    failed = []
    for line in proc.stdout.splitlines():
        if " PASSED" in line and "::" in line:
            passed.append(line.split(" PASSED")[0].strip())
        elif " FAILED" in line and "::" in line:
            failed.append(line.split(" FAILED")[0].strip())
    (REVIEW_DIR / "base_pytest_audit.json").write_text(
        json.dumps({"passed": passed, "failed": failed, "summary": proc.stdout.splitlines()[-3:]}, indent=2)
    )
    print(f"=== BASE 1c6cbc77 AUDIT: {len(passed)} passed, {len(failed)} failed ===")
    for p in passed:
        print(f"  BASE_PASS: {p}")
    return {"passed_count": len(passed), "failed_count": len(failed), "passed": passed}


MUTANTS: list[tuple[str, str, str, str]] = [
    (
        "m01_drop_stamp_dm",
        "bts/scheduler.py",
        "        _stamp_reading(now)\n        daily.delivery_attempted = True",
        "        daily.delivery_attempted = True",
    ),
    (
        "m02_drop_stamp_public",
        "bts/scheduler.py",
        "\n    _stamp_reading(now)\n    daily.delivery_attempted = True",
        "\n    daily.delivery_attempted = True",
    ),
    (
        "m03_stamp_before_cutoff_dm",
        "bts/scheduler.py",
        "        now = _now_et()\n        if now >= cutoff:\n            _refuse_late_delivery(daily, config, picks_dir, state, date, label, cutoff, now)\n            return False\n        _stamp_reading(now)",
        "        now = _now_et()\n        _stamp_reading(now)\n        if now >= cutoff:\n            _refuse_late_delivery(daily, config, picks_dir, state, date, label, cutoff, now)\n            return False",
    ),
    (
        "m04_stamp_before_cutoff_public",
        "bts/scheduler.py",
        "\n    now = _now_et()\n    if now >= cutoff:\n        _refuse_late_delivery(daily, config, picks_dir, state, date, label, cutoff, now)\n        return False\n    _stamp_reading(now)",
        "\n    now = _now_et()\n    _stamp_reading(now)\n    if now >= cutoff:\n        _refuse_late_delivery(daily, config, picks_dir, state, date, label, cutoff, now)\n        return False",
    ),
    (
        "m05_warn_before_send",
        "bts/scheduler.py",
        "        _stamp_reading(now)\n        daily.delivery_attempted = True",
        "        _stamp_reading(now)\n        if required:\n            _warn_stale_contest_reading(daily.automatic_reading, config, picks_dir)\n        daily.delivery_attempted = True",
    ),
    (
        "m06_warn_on_failed_send",
        "bts/scheduler.py",
        "        except Exception as e:\n            daily.delivery_attempted = False  # known failure → clear so a later cycle retries\n            save_pick(daily, picks_dir)\n            print(f\"  Pick DM failed: {e}\", file=sys.stderr)\n            return False",
        "        except Exception as e:\n            daily.delivery_attempted = False\n            save_pick(daily, picks_dir)\n            print(f\"  Pick DM failed: {e}\", file=sys.stderr)\n            if required:\n                _warn_stale_contest_reading(daily.automatic_reading, config, picks_dir)\n            return False",
    ),
    (
        "m07_shared_health_status_file",
        "bts/scheduler.py",
        '"contest_stale_reading_dm_delivery_status.json"',
        '"health_dm_delivery_status.json"',
    ),
    (
        "m08_drop_now_et_date",
        "bts/scheduler.py",
        "                now_et_date=warning_day,\n",
        "",
    ),
    (
        "m09_drop_incident_key",
        "bts/scheduler.py",
        ', incident_key="contest_state:stale_reading"',
        "",
    ),
    (
        "m10_stale_hours_13",
        "bts/contest_state.py",
        "AUTOMATIC_READING_STALE_HOURS = 14",
        "AUTOMATIC_READING_STALE_HOURS = 13",
    ),
    (
        "m11_stale_hours_ge",
        "bts/contest_state.py",
        "elif elapsed > timedelta(hours=AUTOMATIC_READING_STALE_HOURS):",
        "elif elapsed >= timedelta(hours=AUTOMATIC_READING_STALE_HOURS):",
    ),
    (
        "m12_future_4min",
        "bts/contest_state.py",
        "if elapsed < timedelta(minutes=-5):",
        "if elapsed < timedelta(minutes=-4):",
    ),
    (
        "m13_future_le_5min",
        "bts/contest_state.py",
        "if elapsed < timedelta(minutes=-5):",
        "if elapsed <= timedelta(minutes=-5):",
    ),
    (
        "m14_val_accept_int_for_float",
        "bts/contest_state.py",
        "type(age) is not float",
        "type(age) not in (float, int)",
    ),
    (
        "m15_val_accept_bool_for_float",
        "bts/contest_state.py",
        "type(age) is not float",
        "type(age) not in (float, bool)",
    ),
    (
        "m16_val_accept_int_for_bool",
        "bts/contest_state.py",
        "type(stale) is not bool",
        "type(stale) not in (bool, int)",
    ),
    (
        "m17_val_accept_float_for_bool",
        "bts/contest_state.py",
        "type(stale) is not bool",
        "type(stale) not in (bool, float)",
    ),
    (
        "m18_val_unbounded_recorded_at",
        "bts/contest_state.py",
        "if recorded_at is not None and _bounded_reading_time(recorded_at) is None:",
        "if recorded_at is not None and not isinstance(recorded_at, str):",
    ),
    (
        "m19_val_no_abs_age",
        "bts/contest_state.py",
        "or abs(age) > 87_660_000",
        "or age > 87_660_000",
    ),
    (
        "m20_non_automatic_kind_produces_record",
        "bts/contest_state.py",
        '    if s.observation_kind != "automatic":\n        return None\n',
        "",
    ),
    (
        "m21_let_warning_exc_escape",
        "bts/scheduler.py",
        "    except Exception as error:\n        _report_stale_reading_warning_failure(error)",
        "    except Exception as error:\n        raise",
    ),
    (
        "m22_stamp_in_private_mode",
        "bts/scheduler.py",
        '    if mode == "private":\n        save_pick(daily, picks_dir)',
        '    if mode == "private":\n        daily.automatic_reading = automatic_reading_record(load_decision_streak_state(picks_dir, require_contest_state=False), now)\n        save_pick(daily, picks_dir)',
    ),
    (
        "m23_stamp_when_not_required",
        "bts/scheduler.py",
        "    def _stamp_reading(attempt_time: datetime) -> None:\n        if required:",
        "    def _stamp_reading(attempt_time: datetime) -> None:\n        if True:",
    ),
    (
        "m24_warn_from_delivered_recovery",
        "bts/scheduler.py",
        '        print(f"  LOCKED ({label}) — pick already delivered.", file=sys.stderr)\n        return True',
        '        print(f"  LOCKED ({label}) — pick already delivered.", file=sys.stderr)\n        _warn_stale_contest_reading(daily.automatic_reading, config, picks_dir)\n        return True',
    ),
]


def run_mutation_suite() -> list[dict]:
    print("=== MUTATION SUITE (24 MUTANTS) ===")
    summary = []
    with tempfile.TemporaryDirectory() as td:
        scratch_src = Path(td) / "src"
        shutil.copytree(REPO_ROOT / "src", scratch_src)
        _verify_src_import(scratch_src)

        for mid, rel_file, target, replacement in MUTANTS:
            fpath = scratch_src / rel_file
            original = fpath.read_text()
            count = original.count(target)
            assert count == 1, f"{mid}: expected 1 match in {rel_file}, got {count}"
            fpath.write_text(original.replace(target, replacement, 1))
            try:
                p_author = _run_pytest([*AUTHOR_NEW_TESTS, "-q", "--tb=no"], src_dir=scratch_src)
                killed_author = p_author.returncode != 0
                killed_adopt = None
                if not killed_author:
                    p_adopt = _run_pytest([ADOPT_TEST, "-q", "--tb=no"], src_dir=scratch_src)
                    killed_adopt = p_adopt.returncode != 0
                    assert killed_adopt, f"Mutant {mid} survived BOTH author and adopt tests!"
                row = {
                    "mutant": mid,
                    "killed_by_author": killed_author,
                    "killed_by_adopt": killed_adopt,
                    "author_tail": p_author.stdout.strip().splitlines()[-1] if p_author.stdout.strip() else "",
                }
                summary.append(row)
                status_str = "KILLED by author" if killed_author else "SURVIVED author -> KILLED by adopt"
                print(f"  {mid}: {status_str} ({row['author_tail']})")
            finally:
                fpath.write_text(original)

    (REVIEW_DIR / "mutation_results.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_clock_suite() -> dict:
    print("=== WALL CLOCK SUITE (2027-07-15 & 2028-07-15) ===")
    out = {}
    for label, iso in [
        ("clock_2027", "2027-07-15T14:00:00-04:00"),
        ("clock_2028", "2028-07-15T14:00:00-04:00"),
    ]:
        proc = _run_pytest(
            ["-p", "clock_plugin", *ALL_TOUCHED_AND_ADOPT, "-q"],
            extra_env={"FAKE_WALL_CLOCK_ISO": iso},
        )
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        print(f"  {label} ({iso}): rc={proc.returncode} | {tail}")
        assert proc.returncode == 0, f"{label} failed:\n{proc.stdout}\n{proc.stderr}"
        out[label] = tail
    return out


def main() -> None:
    base_res = run_base_audit()
    mut_res = run_mutation_suite()
    clk_res = run_clock_suite()
    survived = [m["mutant"] for m in mut_res if not m["killed_by_author"]]
    print(f"=== ITEM 8 SUMMARY: base_passed={base_res['passed_count']}, base_failed={base_res['failed_count']}, "
          f"author_killed={len(mut_res) - len(survived)}/{len(mut_res)}, survived_author={survived}, clocks={clk_res} ===")
    print("ITEM_8_DRIVER_PASSED")


if __name__ == "__main__":
    main()
