"""Prepare isolated source mutations and run targeted tests sequentially."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import monotonic


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRATCH = HERE / "scratch"
PATCHED = HERE / "patched" / "src"
SOURCE_FILE = "bts/scheduler.py"


@dataclass(frozen=True)
class Edit:
    old: str
    new: str
    count: int = 1


@dataclass(frozen=True)
class Mutation:
    selection: str
    edits: tuple[Edit, ...]


MUTATIONS = {
    "no-byte-match": Mutation("fallback_reload and (replacement or whitespace)", (
        Edit(' and raw == saved_bytes:', ':'),
        Edit('or current_bytes != saved_bytes):', 'or current_bytes is None):'),
    )),
    "sleep-reuse": Mutation("fallback_reload and sleep", (
        Edit('if not slept_to_deadline and fresh_decision is not None and saved_bytes is not None:',
             'if fresh_decision is not None and saved_bytes is not None:'),
    )),
    "no-post-marker-cutoff": Mutation("reuse_refuses_when_attempt_write_reaches_cutoff", (
        Edit('if now < cutoff and now.date() == snapshot_day:', 'if now.date() == snapshot_day:'),
    )),
    "no-et-day-check": Mutation("reuse_checks_live_day_at_midnight and (during_read or during_marker)", (
        Edit('if _now_et().date() == snapshot_day:', 'if True:', count=2),
        Edit('if now < cutoff and now.date() == snapshot_day:', 'if now < cutoff:'),
    )),
    "no-scoreable-check": Mutation("fallback_reload and scoreable", (
        Edit('if record is not None and record["scoreable"]:', 'if False:', count=2),
    )),
    "apply-late-results": Mutation("pending_window_replans and partial", (
        Edit('def _reuse_lookups(calls, deadline: float):',
             'def _reuse_lookups(calls, deadline: float, on_late=None):'),
        Edit('        else:\n            completed.put((time.monotonic(), result, None))',
             '        else:\n'
             '            if on_late is not None and time.monotonic() > deadline:\n'
             '                on_late(result)\n'
             '            completed.put((time.monotonic(), result, None))'),
        Edit('        for confirmation in _reuse_lookups(lookups, deadline):\n'
             '            confirmed_sides.update(\n'
             '                (pk, side) for pk, sides in confirmation.items() for side in sides\n'
             '            )',
             '        def merge(confirmation):\n'
             '            confirmed_sides.update(\n'
             '                (pk, side) for pk, sides in confirmation.items() for side in sides\n'
             '            )\n'
             '        for confirmation in _reuse_lookups(lookups, deadline, on_late=merge):\n'
             '            merge(confirmation)'),
    )),
    "unchecked-selection": Mutation("warmup_cached_selection", (
        Edit('selection=_reuse_commit_selection(refresh.selection, daily),',
             'selection=refresh.selection,'),
    )),
    "coalesce-reuse-defer": Mutation("reuse_defer_keeps_an_overdue_confirmed_check_available", (
        Edit('if not reused_check_cascade:\n                            for j',
             'if True:\n                            for j'),
    )),
    "no-final-decision-read": Mutation("pending_window_replans and changed_decision_fifo", (
        Edit('current_record = _reuse_decision_snapshot(date, picks_dir)',
             'current_record = reuse_snapshot'),
    )),
    "separate-status-budget": Mutation("pending_window_replans and (slow_status or partial)", (
        Edit('_reuse_lookups([lambda: _delivery_game_statuses(date)], deadline)',
             '_reuse_lookups([lambda: _delivery_game_statuses(date)], time.monotonic() + 2.0)'),
    )),
    "poll-mismatched-commit": Mutation("fallback_reload and scoreable", (
        Edit('_pick_matches_reuse_commit(daily_for_poll, authoritative_reuse_commit)', 'True'),
    )),
    "no-final-pick-read": Mutation("(pending_window_replans and attempt) or missed_pick_alert_survives", (
        Edit('current, current_bytes = load_pick_bytes(\n'
             '                                date, picks_dir, regular_only=True)',
             'current, current_bytes = daily, saved_bytes'),
    )),
    "game-not-side": Mutation("pending_window_replans and partial", (
        Edit('(pk, side) for pk, sides in confirmation.items() for side in sides',
             '(pk, side) for pk, sides in confirmation.items() if sides for side in ("away", "home")'),
    )),
    "capture-from-readback": Mutation("fallback_reload and audit_replacement", (
        Edit('    with capture_pick_write(pick_result.daily) as written:\n'
             '        save_pick(pick_result.daily, picks_dir)\n'
             '    saved_pick_bytes = written[0] if written else None',
             '    saved_path = save_pick(pick_result.daily, picks_dir)\n'
             '    saved_pick_bytes = saved_path.read_bytes()'),
    )),
}


def prepare():
    source = (PATCHED / SOURCE_FILE).read_text()
    for name, mutation in MUTATIONS.items():
        target = SCRATCH / name / "src"
        shutil.copytree(PATCHED / "bts", target / "bts", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        changed = source
        for edit in mutation.edits:
            actual = changed.count(edit.old)
            assert actual == edit.count, (name, edit.old, edit.count, actual)
            changed = changed.replace(edit.old, edit.new)
        assert changed != source
        compile(changed, SOURCE_FILE, "exec")
        (target / SOURCE_FILE).write_text(changed)
        print(f"Prepared {name}", flush=True)


def run():
    from datetime import datetime, time
    from zoneinfo import ZoneInfo

    assert datetime.now(ZoneInfo("America/New_York")).time() >= time(9, 45)
    for name, mutation in MUTATIONS.items():
        started = monotonic()
        source = SCRATCH / name / "src"
        assert (source / SOURCE_FILE).is_file()
        startup = ("from pathlib import Path; import sys; import bts, pytest; "
                   "assert Path(bts.__file__).resolve().is_relative_to(Path(sys.argv[1])); "
                   "raise SystemExit(pytest.main(sys.argv[2:]))")
        cmd = [sys.executable, "-c", startup, str(source), "-p", "no:cacheprovider", "-q",
               "--tb=short", "tests/test_wr_w3.py", "-k", mutation.selection]
        env = os.environ | {"PYTHONPATH": os.pathsep.join([str(HERE / "offline_python"), str(source)])}
        try:
            result = subprocess.run(cmd, cwd=ROOT, env=env, timeout=45, check=False,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except subprocess.TimeoutExpired as exc:
            output = exc.stdout or b""
            status = "TIMEOUT"
        else:
            output = result.stdout
            status = {0: "SURVIVED", 1: "DETECTED"}.get(result.returncode, f"TEST ERROR {result.returncode}")
        if len(output) > 260_000:
            output = output[:130_000] + b"\n[trimmed]\n" + output[-130_000:]
        (HERE / f"mutation-{name}.log").write_bytes(output)
        summary = output.decode(errors="replace").strip().splitlines()
        print(f"{name:24} {status:12} {monotonic() - started:5.2f}s  "
              f"{summary[-1] if summary else '[no pytest summary]'}", flush=True)


if __name__ == "__main__":
    {"prepare": prepare, "run": run}[sys.argv[1]]()
