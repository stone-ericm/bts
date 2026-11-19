"""Adversarial review Items 4 & 5: edited test check, RED on base, focused 1-process run,
2027/2028 wall-clock runs, and scratch mutation testing.

Runs every pytest invocation sequentially ONE at a time via /usr/bin/nice -n 19,
storing logs under .codex-review/wr-c3v-review/. Never touches tracked files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
BASE_SRC = REVIEW_DIR / "base_88b69a16" / "src"
HEAD_SRC = REPO_ROOT / "src"
WORK_DIR = REVIEW_DIR / "item4_item5_work"

NEW_TESTS = [
    "tests/test_wr_c3v_reconcile.py",
    "tests/test_wr_c3v_corpus.py",
    "tests/test_wr_c3v_delivery.py",
    "tests/test_wr_c3_r2_entered_pick_blockers.py",
]

FOCUSED_EXTRA = [
    "tests/test_streak_replay.py",
    "tests/test_streak_season_cs2.py",
    "tests/test_reconcile_receipt.py",
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
]

ALL_FOCUSED = NEW_TESTS + FOCUSED_EXTRA


PLUGIN_SOURCE = '''"""Pytest helper plugin for verifying src path and pinning wall clocks."""
import datetime as _dt
import os
from pathlib import Path
import time as _time

_REAL_DATETIME = _dt.datetime
_REAL_DATE = _dt.date
_REAL_MONO = _time.monotonic
_START_MONO = _REAL_MONO()
_WALL_ISO = os.environ.get("BTS_REVIEW_WALL_CLOCK_ISO")

if _WALL_ISO:
    _BASE_WALL = _REAL_DATETIME.fromisoformat(_WALL_ISO)

    def _current_wall():
        return _BASE_WALL + _dt.timedelta(seconds=_REAL_MONO() - _START_MONO)

    class _ReviewWallDate(_REAL_DATE):
        @classmethod
        def today(cls):
            w = _current_wall().astimezone()
            return _REAL_DATE(w.year, w.month, w.day)

    class _ReviewWallDatetime(_REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            w = _current_wall()
            if tz is None:
                loc = w.astimezone()
                return _REAL_DATETIME(
                    loc.year, loc.month, loc.day, loc.hour, loc.minute, loc.second, loc.microsecond
                )
            return w.astimezone(tz)

        @classmethod
        def utcnow(cls):
            u = _current_wall().astimezone(_dt.timezone.utc)
            return _REAL_DATETIME(
                u.year, u.month, u.day, u.hour, u.minute, u.second, u.microsecond
            )

    _dt.date = _ReviewWallDate
    _dt.datetime = _ReviewWallDatetime


def pytest_sessionstart(session):
    expected_src = os.environ.get("BTS_REVIEW_EXPECTED_SRC")
    if expected_src:
        import bts.cli
        import bts.picks
        import bts.reconcile_receipt

        exp = Path(expected_src).resolve()
        for mod in (bts.picks, bts.cli, bts.reconcile_receipt):
            actual = Path(mod.__file__).resolve()
            if not actual.is_relative_to(exp):
                raise RuntimeError(f"Expected {mod.__name__} from {exp}, got {actual}")
'''


def _parse_pytest_summary(output: str) -> dict:
    summary_line = ""
    for line in reversed(output.splitlines()):
        if re.search(r"\b(passed|failed|error|errors|skipped)\b", line) and "=" in line:
            summary_line = line.strip()
            break
    counts = {}
    for key in ("passed", "failed", "error", "errors", "skipped", "deselected", "xfailed"):
        m = re.search(rf"(\d+)\s+{key}\b", summary_line)
        if m:
            counts[key] = int(m.group(1))
    failed_ids = []
    for line in output.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            parts = line.split(" - ", 1)
            node = parts[0].split(None, 1)[1].strip()
            reason = parts[1].strip() if len(parts) > 1 else ""
            failed_ids.append({"node": node, "reason": reason})
    return {"summary_line": summary_line, "counts": counts, "failures": failed_ids}


def run_pytest_once(
    log_name: str,
    args: list[str],
    *,
    src_override: Path | None = None,
    wall_clock_iso: str | None = None,
    timeout: int = 240,
) -> dict:
    plugin_dir = WORK_DIR / "plugin"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    (plugin_dir / "_review_plugin.py").write_text(PLUGIN_SOURCE)

    env = dict(os.environ)
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    py_paths = [str(plugin_dir)]
    if src_override is not None:
        py_paths.insert(0, str(src_override.resolve()))
        env["BTS_REVIEW_EXPECTED_SRC"] = str(src_override.resolve())
    else:
        env["BTS_REVIEW_EXPECTED_SRC"] = str(HEAD_SRC.resolve())
    if env.get("PYTHONPATH"):
        py_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(py_paths)

    if wall_clock_iso is not None:
        env["BTS_REVIEW_WALL_CLOCK_ISO"] = wall_clock_iso
    else:
        env.pop("BTS_REVIEW_WALL_CLOCK_ISO", None)

    cmd = [
        "/usr/bin/nice", "-n", "19",
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "-m", "pytest",
        "-p", "no:cacheprovider",
        "-p", "_review_plugin",
        "-ra",
        *args,
    ]
    proc = subprocess.run(
        cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout,
    )
    combined = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
    log_path = REVIEW_DIR / log_name
    log_path.write_text(combined)
    parsed = _parse_pytest_summary(proc.stdout)
    parsed["returncode"] = proc.returncode
    parsed["log"] = str(log_path.relative_to(REPO_ROOT))
    return parsed


def check_item4_edited_test() -> dict:
    old_test_dir = WORK_DIR / "old_cs2"
    old_test_dir.mkdir(parents=True, exist_ok=True)
    old_cs2_bytes = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "show", "88b69a16:tests/test_streak_season_cs2.py"],
        timeout=10,
    )
    old_cs2_file = old_test_dir / "test_streak_season_cs2_base.py"
    old_cs2_file.write_bytes(old_cs2_bytes)

    res = run_pytest_once(
        "item4_unmodified_test_row7_on_head.log",
        [f"{old_cs2_file}::test_row7_reconcile_summary_uses_replay_et_clock"],
    )
    diff_stat = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "diff", "88b69a16", "HEAD", "--stat", "--", "tests"],
        text=True, timeout=10,
    )
    return {
        "diff_stat_tests": diff_stat.strip().splitlines(),
        "unmodified_88b69a16_test_row7_against_head": res,
    }


def _build_mutants() -> list[tuple[str, str, callable]]:
    cli_orig = (HEAD_SRC / "bts" / "cli.py").read_text()
    picks_orig = (HEAD_SRC / "bts" / "picks.py").read_text()

    def _replace_once(src: str, target: str, repl: str) -> str:
        assert src.count(target) == 1, f"Expected exact match once for:\n{target}"
        return src.replace(target, repl, 1)

    detail_orig = '''    streak, saver = 0, True
    for f in sorted(picks_dir.glob(f"{season}-*.json")):
        if not _ISO_DATE_RE.match(f.stem) or f.stem > today_iso:
            continue  # skip non-date files and future previews
        try:
            data = json.loads(f.read_text())
        except Exception:
            return None, (f.stem, "pick file unreadable or not valid JSON")
        r = data.get("result")
        if f.stem == today_iso and r not in ("hit", "miss", "void"):
            if today_iso in load_applied_dates(picks_dir):
                # A scorer applied today's result to streak.json and has not saved the pick's terminal
                # result yet (it died between the two writes, or is between them now). Replaying without
                # today would save the streak WITHOUT that result while keeping the marker that stops
                # the scorer's retry from re-applying it: the result would be lost (review pp1 B1).
                # Fail closed: keep the live-tracked streak until the scorer finishes.
                return None, (f.stem, "today's result is applied to streak.json but not yet saved on the pick")
            continue  # today's pick, not yet resolved: an unplayed preview or a game in progress
        # Today's pick IS replayed once it carries a terminal result: that result was already applied to
        # streak.json when it was written, so leaving it out rolled the streak back (and restored a saver
        # a miss had consumed) on any reconcile run later the same day (incident L02, fixed 2026-11-13).
        if r == "void":
            continue
        if r not in ("hit", "miss"):
            return None, (f.stem, "no final result on the pick")
        if r == "hit":
            daily = load_pick(f.stem, picks_dir)
            if daily is None:
                return None, (f.stem, "hit pick could not be loaded")
            streak, saver = _apply_streak_day(
                streak, saver, True, streak_increment_for_resolved_hit(daily)
            )
        else:
            streak, saver = _apply_streak_day(streak, saver, False, 0)
    return (streak, saver), None'''

    assert picks_orig.count(detail_orig) == 1

    m2_body = '''    streak, saver = 0, True
    last_blocker = None
    for f in sorted(picks_dir.glob(f"{season}-*.json")):
        if not _ISO_DATE_RE.match(f.stem) or f.stem > today_iso:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            last_blocker = (f.stem, "pick file unreadable or not valid JSON")
            continue
        r = data.get("result")
        if f.stem == today_iso and r not in ("hit", "miss", "void"):
            if today_iso in load_applied_dates(picks_dir):
                last_blocker = (f.stem, "today's result is applied to streak.json but not yet saved on the pick")
            continue
        if r == "void":
            continue
        if r not in ("hit", "miss"):
            last_blocker = (f.stem, "no final result on the pick")
            continue
        if r == "hit":
            daily = load_pick(f.stem, picks_dir)
            if daily is None:
                last_blocker = (f.stem, "hit pick could not be loaded")
                continue
            streak, saver = _apply_streak_day(
                streak, saver, True, streak_increment_for_resolved_hit(daily)
            )
        else:
            streak, saver = _apply_streak_day(streak, saver, False, 0)
    if last_blocker is not None:
        return None, last_blocker
    return (streak, saver), None'''

    m5a_body = '''    streak, saver = 0, True
    for f in sorted(picks_dir.glob(f"{season}-*.json")):
        if not _ISO_DATE_RE.match(f.stem) or f.stem > today_iso:
            continue
        try:
            data = json.loads(f.read_text())
        except Exception:
            return None, (f.stem, "pick file unreadable or not valid JSON")
        r = data.get("result")
        if r == "void":
            continue
        if r not in ("hit", "miss"):
            return None, (f.stem, "no final result on the pick")
        if f.stem == today_iso and r not in ("hit", "miss", "void"):
            if today_iso in load_applied_dates(picks_dir):
                return None, (f.stem, "today's result is applied to streak.json but not yet saved on the pick")
            continue
        if r == "hit":
            daily = load_pick(f.stem, picks_dir)
            if daily is None:
                return None, (f.stem, "hit pick could not be loaded")
            streak, saver = _apply_streak_day(
                streak, saver, True, streak_increment_for_resolved_hit(daily)
            )
        else:
            streak, saver = _apply_streak_day(streak, saver, False, 0)
    return (streak, saver), None'''

    m5b_body = '''    streak, saver = 0, True
    for f in sorted(picks_dir.glob(f"{season}-*.json")):
        if not _ISO_DATE_RE.match(f.stem) or f.stem > today_iso:
            continue
        if f.stem == today_iso and today_iso in load_applied_dates(picks_dir):
            return None, (f.stem, "today's result is applied to streak.json but not yet saved on the pick")
        try:
            data = json.loads(f.read_text())
        except Exception:
            return None, (f.stem, "pick file unreadable or not valid JSON")
        r = data.get("result")
        if f.stem == today_iso and r not in ("hit", "miss", "void"):
            continue
        if r == "void":
            continue
        if r not in ("hit", "miss"):
            return None, (f.stem, "no final result on the pick")
        if r == "hit":
            daily = load_pick(f.stem, picks_dir)
            if daily is None:
                return None, (f.stem, "hit pick could not be loaded")
            streak, saver = _apply_streak_day(
                streak, saver, True, streak_increment_for_resolved_hit(daily)
            )
        else:
            streak, saver = _apply_streak_day(streak, saver, False, 0)
    return (streak, saver), None'''

    m5c_body = _replace_once(
        detail_orig,
        '        except Exception:\n            return None, (f.stem, "pick file unreadable or not valid JSON")',
        '        except Exception:\n'
        '            if f.stem == today_iso and today_iso in load_applied_dates(picks_dir):\n'
        '                return None, (f.stem, "today\'s result is applied to streak.json but not yet saved on the pick")\n'
        '            return None, (f.stem, "pick file unreadable or not valid JSON")',
    )

    m5d_body = _replace_once(
        detail_orig,
        '        if r not in ("hit", "miss"):\n            return None, (f.stem, "no final result on the pick")',
        '        try:\n'
        '            _probe_daily = load_pick(f.stem, picks_dir)\n'
        '        except Exception:\n'
        '            _probe_daily = "ERR"\n'
        '        if _probe_daily is None:\n'
        '            return None, (f.stem, "hit pick could not be loaded")\n'
        '        if r not in ("hit", "miss"):\n'
        '            return None, (f.stem, "no final result on the pick")',
    )

    return [
        (
            "M1_drop_stderr",
            "Drop the stderr warning line in bts reconcile when blocked",
            lambda root: (root / "bts" / "cli.py").write_text(
                _replace_once(cli_orig, "    if blocked:\n        click.echo(", "    if False and blocked:\n        click.echo(")
            ),
        ),
        (
            "M2_last_blocking_date",
            "Report the LAST blocking date instead of the FIRST in _replay_season_streak_detail",
            lambda root: (root / "bts" / "picks.py").write_text(_replace_once(picks_orig, detail_orig, m2_body)),
        ),
        (
            "M3a_recalculated_on_blocked_with_corrections",
            "Print 'Streak recalculated' on a blocked replay when corrections are present",
            lambda root: (root / "bts" / "cli.py").write_text(
                _replace_once(
                    cli_orig,
                    '        label = "Streak retained" if blocked else "Streak recalculated"',
                    '        label = "Streak recalculated"',
                )
            ),
        ),
        (
            "M3b_recalculated_on_blocked_no_corrections",
            "Print 'Streak recalculated' on a blocked replay when no corrections are present",
            lambda root: (root / "bts" / "cli.py").write_text(
                _replace_once(
                    cli_orig,
                    '        label = "Streak retained" if blocked else "Streak"',
                    '        label = "Streak recalculated" if blocked else "Streak"',
                )
            ),
        ),
        (
            "M4_skip_unresolved_past_pick",
            "Make _replay_season_streak_detail skip an unresolved past pick instead of blocking",
            lambda root: (root / "bts" / "picks.py").write_text(
                _replace_once(
                    picks_orig,
                    '        if r not in ("hit", "miss"):\n            return None, (f.stem, "no final result on the pick")',
                    '        if r not in ("hit", "miss"):\n            continue',
                )
            ),
        ),
        (
            "M5a_order_pending_before_today",
            "Check r not in ('hit', 'miss') before today's unresolved-pick branch",
            lambda root: (root / "bts" / "picks.py").write_text(_replace_once(picks_orig, detail_orig, m5a_body)),
        ),
        (
            "M5b_order_today_applied_before_json_loads",
            "Check today in load_applied_dates before json.loads and terminal-result check",
            lambda root: (root / "bts" / "picks.py").write_text(_replace_once(picks_orig, detail_orig, m5b_body)),
        ),
        (
            "M5c_order_today_applied_before_corrupt",
            "Prioritize today's applied-marker reason ahead of unreadable/corrupt JSON on today's file",
            lambda root: (root / "bts" / "picks.py").write_text(_replace_once(picks_orig, detail_orig, m5c_body)),
        ),
        (
            "M5d_order_load_pick_none_before_pending",
            "Check load_pick(f.stem, picks_dir) is None ahead of r not in ('hit', 'miss')",
            lambda root: (root / "bts" / "picks.py").write_text(_replace_once(picks_orig, detail_orig, m5d_body)),
        ),
        (
            "M6_not_reached_prints_retained_with_corrections",
            "Use replay['state'] != 'saved' instead of blocked in corrections branch (breaks not_reached + corrections)",
            lambda root: (root / "bts" / "cli.py").write_text(
                _replace_once(
                    cli_orig,
                    '        label = "Streak retained" if blocked else "Streak recalculated"',
                    '        label = "Streak retained" if replay["state"] != "saved" else "Streak recalculated"',
                )
            ),
        ),
    ]


def main() -> int:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    # 1. Item 4 check
    item4 = check_item4_edited_test()

    # 2. Focused 1-process run on HEAD (all 9 requested files)
    focused_head = run_pytest_once("focused_head.log", ALL_FOCUSED)

    # Determine if test_web_saver socket test failed solely due to sandbox loopback PermissionError
    socket_deselect: list[str] = []
    if focused_head["counts"].get("failed") == 1:
        only_fail = focused_head["failures"][0]
        if "test_web_saver.py::" in only_fail["node"] and "PermissionError" in only_fail["reason"]:
            socket_deselect = ["-k", "not socket_wiring and not do_post_saver_transition"]
            # Look up exact node name from failure
            test_fn_name = only_fail["node"].split("::")[-1]
            socket_deselect = ["--deselect", only_fail["node"]]

    # 3. RED run on base (88b69a16) for the 4 new test files
    red_on_base = run_pytest_once("red_on_base.log", NEW_TESTS, src_override=BASE_SRC)

    # 4. Wall clocks in 2027 and 2028 (4 timestamps covering mid-season and New Year boundaries)
    wall_clocks = {}
    for label, iso_val in [
        ("2027_midseason", "2027-07-15T14:00:00-04:00"),
        ("2027_dec31_utc_2028", "2027-12-31T23:30:00-05:00"),
        ("2028_jan01", "2028-01-01T00:30:00-05:00"),
        ("2028_midseason", "2028-07-15T14:00:00-04:00"),
    ]:
        wall_clocks[label] = run_pytest_once(
            f"wallclock_{label}.log",
            [*ALL_FOCUSED, *socket_deselect],
            wall_clock_iso=iso_val,
        )

    # 5. Scratch mutants (run against the 4 new test files first; if a mutant survives the 4 new files,
    #    also run it against the full focused suite to see whether existing tests kill it!)
    mutants_report = {}
    mut_base_dir = WORK_DIR / "mutants"
    for m_name, m_desc, apply_fn in _build_mutants():
        m_src = mut_base_dir / m_name / "src"
        shutil.copytree(HEAD_SRC, m_src, symlinks=True)
        apply_fn(m_src)

        res_new = run_pytest_once(f"mutant_{m_name}_new.log", NEW_TESTS, src_override=m_src)
        survived_new = res_new["returncode"] == 0
        entry = {
            "description": m_desc,
            "survived_new_tests": survived_new,
            "new_tests_counts": res_new["counts"],
            "new_tests_sample_failures": res_new["failures"][:5],
            "new_tests_log": res_new["log"],
        }
        if survived_new:
            res_focused = run_pytest_once(
                f"mutant_{m_name}_focused.log",
                [*FOCUSED_EXTRA, *socket_deselect],
                src_override=m_src,
            )
            entry["survived_focused_suite"] = res_focused["returncode"] == 0
            entry["focused_extra_counts"] = res_focused["counts"]
            entry["focused_extra_failures"] = res_focused["failures"][:5]
            entry["focused_log"] = res_focused["log"]
        else:
            entry["survived_focused_suite"] = False
        mutants_report[m_name] = entry
        shutil.rmtree(m_src.parent)

    shutil.rmtree(WORK_DIR)

    summary = {
        "item4": item4,
        "focused_head": focused_head,
        "red_on_base": red_on_base,
        "wall_clocks": wall_clocks,
        "mutants": mutants_report,
    }
    out_path = REVIEW_DIR / "item4_item5_results.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
