"""Sensitivity of the E77/L03 delivery predicate (Codex phase-1 r5 #5), outside the strict sweep.

    python fixture_predicate_mutants.py <build-worktree>

The strict sweep runs the tooling tests only (its baseline must pass every node, and the fixture file's
strict XFAILs are skips in JUnit). This script measures the fixture file itself, one mutant at a time,
restoring every file byte for byte:

* PREDICATE mutants weaken one clause of the fixture's delivery predicate; each must fail at least one
  fixture test;
* SOURCE mutants are production formatter changes that send something other than a pick; each must
  fail the E77 and L03 positive controls and leave no incident node at its declared shape except E77's
  (whose bad path never reaches the formatter);
* REPAIR sketches are throwaway production repairs; each must turn its incident node XPASS(strict)
  with every control of that incident passing, so the required branch stays reachable.
"""
import ast
import os
import subprocess
import sys
from pathlib import Path

FIXTURES = "tests/test_incident_register_2026.py"
SCHEDULER = "src/bts/scheduler.py"
ENV = {**os.environ, "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1"}

PREDICATE = [
    ("P1 affirmative content (not a name match)", "    return re.fullmatch(pattern, text) is not None\n",
     "    return pick.batter_name in text\n"),
    ("P2 send and record on the declared day", "    return t.astimezone(ET).date().isoformat() == day and t < cutoff\n",
     "    return t < cutoff\n"),
    ("P3 strictly before the cutoff", "    return t.astimezone(ET).date().isoformat() == day and t < cutoff\n",
     "    return t.astimezone(ET).date().isoformat() == day and t <= cutoff\n"),
    ("P4 the record's delivered_at bounded", '    return (_on_day_before(dm["at"], day, cutoff) and _on_day_before(delivered, day, cutoff)\n',
     '    return (_on_day_before(dm["at"], day, cutoff) and True\n'),
    ("P5 E77: no other send naming the batter", "    if len(picks) == 1 and not others and _record_matches(",
     "    if len(picks) == 1 and _record_matches("),
    ("P6 L03: another send naming the batter is other", "    if [d for d in _mentions(obs, L03_BATTER) if d not in picks]:\n",
     "    if False:\n"),
    ("P7 the record is the declared day's", "    if not (daily and daily.date == day and daily.notification_sent",
     "    if not (daily and daily.notification_sent"),
]
SOURCE = [
    ("F1 named status formatter", "_format_pick_delivery_text", '    return "No pick today for " + daily.pick.batter_name\n'),
    ("F2 unqualified no-pick formatter", "_format_pick_delivery_text", '    return "No pick today"\n'),
]
def _noon_refetch(text: str) -> str:
    """E77 repair sketch: re-fetch the schedule at noon, after the declared 12:00 move."""
    anchor = "    games = fetch_schedule(date)\n"
    start = text.index("def run_day(")
    body = text[start:]
    assert body.count(anchor) == 1
    return text[:start] + body.replace(anchor, anchor + (
        "    noon = _now_et().replace(hour=12, minute=0, second=0, microsecond=0)\n"
        "    if _now_et() < noon:\n"
        "        time.sleep((noon - _now_et()).total_seconds())\n"
        "        games = fetch_schedule(date)\n"), 1)


REPAIR = [
    ("R1 L03 postponement guard at the delivery chokepoint",
     lambda t: _insert(t, "_deliver_and_lock_pick",
                       "    from bts.picks import get_game_statuses_detailed\n"
                       "    if any(s.get('detailed') == 'Postponed' for s in get_game_statuses_detailed(date).values()):\n"
                       "        return False\n"), "l03", "postponed_game_is_never"),
    ("R2 E77 noon schedule re-fetch", _noon_refetch, "e77_singleton or e77_positive", "e77_singleton_slate"),
]
FOUR = "e77_singleton or e77_positive or l03_postponed or l03_positive"


def _pytest(build: Path, select: str | None = None) -> tuple[int, list[str]]:
    cmd = [str(build / ".venv/bin/python"), "-B", "-m", "pytest", FIXTURES, "-q", "-p", "no:cacheprovider",
           "--tb=short", "-rfEX"] + (["-k", select] if select else [])     # --tb=no hides the XPASS(strict) reason
    p = subprocess.run(cmd, cwd=build, capture_output=True, text=True, env=ENV)
    lines = p.stdout.splitlines()
    # a strict XPASS is printed on its own "[XPASS(strict)] <reason>" line; its node's line says FAILED
    return p.returncode, [l for l in lines if l.startswith(("FAILED", "XPASS", "ERROR")) or "XPASS(strict)" in l] + lines[-1:]


def _insert(text: str, func: str, body: str) -> str:
    node = next(x for x in ast.parse(text).body if isinstance(x, ast.FunctionDef) and x.name == func)
    lines = text.splitlines(True)
    lines.insert(node.body[1].lineno - 1, body)       # after the docstring
    return "".join(lines)


def main(build: Path) -> int:
    fixtures, scheduler = build / FIXTURES, build / SCHEDULER
    fx0, sc0 = fixtures.read_bytes(), scheduler.read_bytes()
    ok = True
    try:
        rc, tail = _pytest(build)
        print(f"BASELINE rc={rc} | {tail[-1]}", flush=True)
        for label, old, new in PREDICATE:
            text = fx0.decode()
            assert text.count(old) == 1, label
            fixtures.write_text(text.replace(old, new))
            rc, out = _pytest(build)
            fixtures.write_bytes(fx0)
            caught = rc != 0 and any(l.startswith("FAILED") for l in out)
            ok &= caught
            print(f"{label}: {'CAUGHT' if caught else 'MISSED'} | {out[-1]}", flush=True)
        for label, func, body in SOURCE:
            scheduler.write_text(_insert(sc0.decode(), func, body))
            rc, out = _pytest(build, FOUR)
            scheduler.write_bytes(sc0)
            failed = {l.split("::", 1)[1].split(" ")[0] for l in out if l.startswith("FAILED")}
            need = {"test_e77_positive_execution_control_correct_schedule_delivers",
                    "test_l03_positive_execution_control_cached_fallback_delivers_a_playable_game",
                    "test_l03_postponed_game_is_never_delivered_by_the_cached_fallback"}
            caught = need <= failed
            ok &= caught
            print(f"{label}: {'CAUGHT' if caught else 'MISSED'} | failed {sorted(failed)} | {out[-1]}", flush=True)
        for label, transform, select, incident in REPAIR:
            scheduler.write_text(transform(sc0.decode()))
            rc, out = _pytest(build, select)
            scheduler.write_bytes(sc0)
            xpass = any("XPASS(strict)" in l for l in out)
            others = [l for l in out if l.startswith("FAILED") and incident not in l]
            reached = xpass and not others
            ok &= reached
            print(f"{label}: {'REACHES THE REQUIRED BRANCH' if reached else 'DOES NOT'} | {out[-1]}", flush=True)
    finally:
        fixtures.write_bytes(fx0)
        scheduler.write_bytes(sc0)
        assert fixtures.read_bytes() == fx0 and scheduler.read_bytes() == sc0
    print("ALL AS REQUIRED" if ok else "NOT ALL AS REQUIRED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve()))
