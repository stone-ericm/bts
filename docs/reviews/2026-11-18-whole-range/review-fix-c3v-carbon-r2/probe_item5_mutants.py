"""Adversarial probe for Item 5: RED on base (88b69a16), 1-process focused suite, wall-clock checks, and scratch mutants.

Enforces the >= 06:35 ET start rule. Tries ~/bts-runs/yield_run.sh (with pgrep idle check) first;
if refused by the sandbox, falls back cleanly to sequential /usr/bin/nice -n 19 pytest runs.
"""
from __future__ import annotations

from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import time
from zoneinfo import ZoneInfo


ET = ZoneInfo("America/New_York")
EARLIEST_START_ET = datetime(2026, 11, 19, 6, 35, 0, tzinfo=ET)

REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
HEAD_SRC = REPO_ROOT / "src"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"
WRAPPER = Path.home() / "bts-runs" / "yield_run.sh"
WORK_DIR = REVIEW_DIR / "_item5_scratch"
BASE_REV = "88b69a16"

NEW_TEST_FILES = [
    "tests/test_wr_c3v_reconcile.py",
    "tests/test_wr_c3v_corpus.py",
    "tests/test_wr_c3v_delivery.py",
    "tests/test_wr_c3_r2_entered_pick_blockers.py",
    "tests/test_wr_c3v_review_gaps.py",
]

EXISTING_FOCUSED_FILES = [
    "tests/test_streak_replay.py",
    "tests/test_streak_season_cs2.py",
    "tests/test_reconcile_receipt.py",
    "tests/test_lightgbm_optional.py",
    "tests/test_web_saver.py",
]

ALL_FOCUSED_FILES = NEW_TEST_FILES + EXISTING_FOCUSED_FILES


PLUGIN_CODE = '''"""Pytest helper plugin: verifies imported bts src location and supports optional wall-clock shift."""
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


def _check_wrapper_usable() -> tuple[bool, str]:
    try:
        pg = subprocess.run(["pgrep", "-f", "yield_run.sh"], capture_output=True, text=True, timeout=5)
        if pg.returncode not in (0, 1):
            return False, f"pgrep exited {pg.returncode}: {(pg.stderr or pg.stdout).strip()}"
    except OSError as exc:
        return False, f"pgrep OSError: {exc}"
    try:
        if not os.access(WRAPPER, os.R_OK | os.X_OK):
            return False, f"wrapper not readable/executable in sandbox: {WRAPPER}"
    except OSError as exc:
        return False, f"wrapper access error: {exc}"
    return True, "usable"


def _wait_wrapper_idle(timeout_s: int = 300) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        pg = subprocess.run(["pgrep", "-f", "yield_run.sh"], capture_output=True, text=True, timeout=5)
        if pg.returncode != 0:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Timed out waiting for yield_run.sh to become idle")
        time.sleep(3)


def _parse_pytest_output(text: str) -> dict:
    summary = ""
    for line in reversed(text.splitlines()):
        if re.search(r"\b(passed|failed|error|errors|skipped)\b", line) and "=" in line:
            summary = line.strip()
            break
    counts = {}
    for key in ("passed", "failed", "error", "errors", "skipped", "deselected", "xfailed"):
        m = re.search(rf"(\d+)\s+{key}\b", summary)
        if m:
            counts[key] = int(m.group(1))
    failures = []
    for line in text.splitlines():
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            parts = line.split(" - ", 1)
            node = parts[0].split(None, 1)[1].strip()
            reason = parts[1].strip() if len(parts) > 1 else ""
            failures.append({"node": node, "reason": reason})
    return {"summary_line": summary, "counts": counts, "failures": failures}


class Runner:
    def __init__(self, plugin_dir: Path):
        self.plugin_dir = plugin_dir
        self.use_wrapper, self.wrapper_note = _check_wrapper_usable()

    def run_pytest(
        self,
        log_name: str,
        pytest_args: list[str],
        *,
        src_override: Path | None = None,
        wall_clock_iso: str | None = None,
        timeout: int = 300,
    ) -> dict:
        now_et = datetime.now(ET)
        if now_et < EARLIEST_START_ET:
            raise RuntimeError(f"Refusing to run pytest before 06:35 ET (now={now_et.isoformat()})")

        log_path = REVIEW_DIR / log_name
        expected_src = (src_override or HEAD_SRC).resolve()
        py_paths = [str(self.plugin_dir.resolve())]
        if src_override is not None:
            py_paths.insert(0, str(expected_src))
        if os.environ.get("PYTHONPATH"):
            py_paths.append(os.environ["PYTHONPATH"])
        py_path_str = os.pathsep.join(py_paths)

        env = dict(os.environ)
        env["TZ"] = "America/New_York"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = py_path_str
        env["BTS_REVIEW_EXPECTED_SRC"] = str(expected_src)
        if wall_clock_iso is not None:
            env["BTS_REVIEW_WALL_CLOCK_ISO"] = wall_clock_iso
        else:
            env.pop("BTS_REVIEW_WALL_CLOCK_ISO", None)

        pytest_cmd = [
            str(PYTHON_BIN),
            "-m", "pytest",
            "-p", "no:cacheprovider",
            "-p", "_review_plugin",
            "-ra",
            *pytest_args,
        ]

        if self.use_wrapper:
            _wait_wrapper_idle()
            env_kv = [
                "env",
                "TZ=America/New_York",
                "PYTHONDONTWRITEBYTECODE=1",
                f"PYTHONPATH={py_path_str}",
                f"BTS_REVIEW_EXPECTED_SRC={expected_src}",
            ]
            if wall_clock_iso is not None:
                env_kv.append(f"BTS_REVIEW_WALL_CLOCK_ISO={wall_clock_iso}")
            cmd = [str(WRAPPER), str(log_path), *env_kv, *pytest_cmd]
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout)
            if proc.returncode == 126 or "Operation not permitted" in (proc.stderr or ""):
                self.use_wrapper = False
                self.wrapper_note = f"yield_run.sh refused ({proc.returncode}): {(proc.stderr or '').strip()}"
            else:
                text = log_path.read_text() if log_path.exists() else (proc.stdout + "\n" + proc.stderr)
                parsed = _parse_pytest_output(text)
                parsed["returncode"] = proc.returncode
                parsed["runner"] = "yield_run.sh"
                parsed["log"] = str(log_path.relative_to(REPO_ROOT))
                return parsed

        cmd = ["/usr/bin/nice", "-n", "19", *pytest_cmd]
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout)
        combined = proc.stdout + ("\n--- STDERR ---\n" + proc.stderr if proc.stderr else "")
        log_path.write_text(combined)
        parsed = _parse_pytest_output(proc.stdout)
        parsed["returncode"] = proc.returncode
        parsed["runner"] = "nice_-n_19"
        parsed["log"] = str(log_path.relative_to(REPO_ROOT))
        return parsed


def _replace_once(src: str, target: str, repl: str) -> str:
    count = src.count(target)
    assert count == 1, f"Expected count==1, got {count} for:\n{target}"
    return src.replace(target, repl, 1)


def _build_mutants() -> list[tuple[str, str, object]]:
    cli_orig = (HEAD_SRC / "bts" / "cli.py").read_text()
    picks_orig = (HEAD_SRC / "bts" / "picks.py").read_text()
    rr_orig = (HEAD_SRC / "bts" / "reconcile_receipt.py").read_text()

    cli_escape_block = (
        '        blocked_date = replay["blocked_date"].encode("unicode_escape").decode("ascii")\n'
        '        reason = replay["reason"].encode("unicode_escape").decode("ascii")'
    )
    assert cli_orig.count(cli_escape_block) == 1

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

    def _apply_escape_receipt_too(src_root: Path):
        (src_root / "bts" / "reconcile_receipt.py").write_text(
            _replace_once(
                rr_orig,
                '    def replayed(self, replay, **detail) -> None:\n'
                '        self.replay = ({"state": "unavailable", **detail} if replay is None\n'
                '                       else {"state": "saved", "streak": replay[0], "saver_available": replay[1]})',
                '    def replayed(self, replay, **detail) -> None:\n'
                '        esc = {k: (v.encode("unicode_escape").decode("ascii") if isinstance(v, str) else v) for k, v in detail.items()}\n'
                '        self.replay = ({"state": "unavailable", **esc} if replay is None\n'
                '                       else {"state": "saved", "streak": replay[0], "saver_available": replay[1]})',
            )
        )
        # Avoid double-escaping stderr so ONLY receipt-value assertions can detect this mutant!
        (src_root / "bts" / "cli.py").write_text(
            _replace_once(
                cli_orig,
                cli_escape_block,
                '        blocked_date = replay["blocked_date"]\n        reason = replay["reason"]',
            )
        )

    return [
        (
            "M_remove_escaping",
            "Remove unicode_escape rendering on stderr (revert to raw interpolation)",
            lambda r: (r / "bts" / "cli.py").write_text(
                _replace_once(
                    cli_orig,
                    cli_escape_block,
                    '        blocked_date = replay["blocked_date"]\n        reason = replay["reason"]',
                )
            ),
        ),
        (
            "M_escape_receipt_too",
            "Escape blocked_date and reason in ReconcileReceipt.replayed too (keeping stderr single-escaped)",
            _apply_escape_receipt_too,
        ),
        (
            "M_escape_with_repr",
            "Escape blocked_date and reason with repr() so ordinary dates and reasons gain quotes",
            lambda r: (r / "bts" / "cli.py").write_text(
                _replace_once(
                    cli_orig,
                    cli_escape_block,
                    '        blocked_date = repr(replay["blocked_date"])\n        reason = repr(replay["reason"])',
                )
            ),
        ),
        (
            "M5b_today_applied_before_json_loads",
            "Carbon r1 M5b: check today in load_applied_dates before json.loads and terminal-result check",
            lambda r: (r / "bts" / "picks.py").write_text(_replace_once(picks_orig, detail_orig, m5b_body)),
        ),
        (
            "M5c_today_applied_before_corrupt",
            "Carbon r1 M5c: prioritize today's applied-marker reason ahead of corrupt JSON on today.json",
            lambda r: (r / "bts" / "picks.py").write_text(_replace_once(picks_orig, detail_orig, m5c_body)),
        ),
        (
            "M5d_load_pick_none_before_pending",
            "Carbon r1 M5d: check load_pick(f.stem, picks_dir) is None ahead of r not in ('hit', 'miss')",
            lambda r: (r / "bts" / "picks.py").write_text(_replace_once(picks_orig, detail_orig, m5d_body)),
        ),
        (
            "M6_state_ne_saved_with_corrections",
            "Carbon r1 M6: use replay['state'] != 'saved' instead of blocked in corrections branch",
            lambda r: (r / "bts" / "cli.py").write_text(
                _replace_once(
                    cli_orig,
                    '        label = "Streak retained" if blocked else "Streak recalculated"',
                    '        label = "Streak retained" if replay["state"] != "saved" else "Streak recalculated"',
                )
            ),
        ),
        (
            "X1_escape_date_only",
            "Escape blocked_date on stderr, leave reason unescaped",
            lambda r: (r / "bts" / "cli.py").write_text(
                _replace_once(
                    cli_orig,
                    cli_escape_block,
                    '        blocked_date = replay["blocked_date"].encode("unicode_escape").decode("ascii")\n'
                    '        reason = replay["reason"]',
                )
            ),
        ),
        (
            "X2_escape_lf_only",
            "Only replace '\\n' with '\\\\n' instead of unicode_escape (leaves CR, ESC, LS/PS, non-ASCII unescaped)",
            lambda r: (r / "bts" / "cli.py").write_text(
                _replace_once(
                    cli_orig,
                    cli_escape_block,
                    '        blocked_date = replay["blocked_date"].replace("\\n", r"\\n")\n'
                    '        reason = replay["reason"].replace("\\n", r"\\n")',
                )
            ),
        ),
        (
            "X3_drop_blocked_date_guard",
            "Drop 'and \"blocked_date\" in replay' in cli.py so bare {'state': 'unavailable'} hits KeyError",
            lambda r: (r / "bts" / "cli.py").write_text(
                _replace_once(
                    cli_orig,
                    '    blocked = replay["state"] == "unavailable" and "blocked_date" in replay',
                    '    blocked = replay["state"] == "unavailable"',
                )
            ),
        ),
        (
            "X4_regex_ascii_flag",
            "Add re.ASCII to _ISO_DATE_RE in picks.py (changes replay scan on non-ASCII Unicode digit stems)",
            lambda r: (r / "bts" / "picks.py").write_text(
                _replace_once(
                    picks_orig,
                    '_ISO_DATE_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}$")',
                    '_ISO_DATE_RE = re.compile(r"^\\d{4}-\\d{2}-\\d{2}$", re.ASCII)',
                )
            ),
        ),
    ]


def main() -> int:
    if WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
    WORK_DIR.mkdir(parents=True)

    try:
        plugin_dir = WORK_DIR / "plugin"
        plugin_dir.mkdir()
        (plugin_dir / "_review_plugin.py").write_text(PLUGIN_CODE)

        base_extract = WORK_DIR / "base_88b69a16"
        base_extract.mkdir()
        archive = subprocess.check_output(["git", "-C", str(REPO_ROOT), "archive", BASE_REV, "src"], timeout=15)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
            tf.extractall(base_extract, filter="data")
        base_src = base_extract / "src"

        runner = Runner(plugin_dir)

        # 1. RED run of all 5 new test files against 88b69a16 base
        red_on_base = runner.run_pytest("item5_red_on_base.log", NEW_TEST_FILES, src_override=base_src)

        # 2. One-process focused suite on HEAD
        focused_head = runner.run_pytest("item5_focused_head.log", ALL_FOCUSED_FILES)

        socket_deselect: list[str] = []
        if focused_head["counts"].get("failed") == 1:
            only_fail = focused_head["failures"][0]
            if "test_web_saver.py::test_do_post_saver_transition_socket_wiring" in only_fail["node"]:
                socket_deselect = ["--deselect", only_fail["node"]]
                focused_no_socket = runner.run_pytest(
                    "item5_focused_head_no_socket.log", [*ALL_FOCUSED_FILES, *socket_deselect]
                )
            else:
                focused_no_socket = None
        else:
            focused_no_socket = None

        # 3. Wall-clock checks in 2027 and 2028 on focused suite
        wall_clocks = {}
        for label, iso_str in (
            ("2027_midseason", "2027-07-15T14:00:00-04:00"),
            ("2028_jan01", "2028-01-01T00:30:00-05:00"),
        ):
            wall_clocks[label] = runner.run_pytest(
                f"item5_wallclock_{label}.log",
                [*ALL_FOCUSED_FILES, *socket_deselect],
                wall_clock_iso=iso_str,
            )

        # 4. Scratch mutants
        mutants = {}
        mut_root = WORK_DIR / "mutants"
        for m_name, m_desc, apply_fn in _build_mutants():
            m_src = mut_root / m_name / "src"
            shutil.copytree(HEAD_SRC, m_src, symlinks=True)
            apply_fn(m_src)

            res_new = runner.run_pytest(f"mutant_{m_name}_new.log", NEW_TEST_FILES, src_override=m_src)
            survived_new = res_new["returncode"] == 0
            rec = {
                "description": m_desc,
                "survived_new_tests": survived_new,
                "new_tests_counts": res_new["counts"],
                "new_tests_sample_failures": res_new["failures"][:6],
                "new_tests_log": res_new["log"],
            }
            if survived_new:
                res_extra = runner.run_pytest(
                    f"mutant_{m_name}_existing.log",
                    [*EXISTING_FOCUSED_FILES, *socket_deselect],
                    src_override=m_src,
                )
                rec["survived_focused_suite"] = res_extra["returncode"] == 0
                rec["existing_focused_counts"] = res_extra["counts"]
                rec["existing_focused_failures"] = res_extra["failures"][:6]
            else:
                rec["survived_focused_suite"] = False
            mutants[m_name] = rec
            shutil.rmtree(m_src.parent)

        summary = {
            "runner_used": "yield_run.sh" if runner.use_wrapper else "nice_-n_19",
            "wrapper_note": runner.wrapper_note,
            "red_on_base": red_on_base,
            "focused_head": focused_head,
            "focused_head_no_socket": focused_no_socket,
            "wall_clocks": wall_clocks,
            "mutants": mutants,
        }
        out_path = REVIEW_DIR / "item5_results.json"
        out_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote {out_path}")
        print("runner:", summary["runner_used"], "|", summary["wrapper_note"])
        print("red_on_base:", red_on_base["summary_line"])
        print("focused_head:", focused_head["summary_line"])
        if focused_no_socket:
            print("focused_head_no_socket:", focused_no_socket["summary_line"])
        for wc_k, wc_v in wall_clocks.items():
            print(f"wall_clock[{wc_k}]:", wc_v["summary_line"])
        for m_k, m_v in mutants.items():
            print(f"mutant[{m_k}]: survived_new={m_v['survived_new_tests']} survived_focused={m_v['survived_focused_suite']} counts={m_v['new_tests_counts']}")
    finally:
        if WORK_DIR.exists():
            shutil.rmtree(WORK_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
