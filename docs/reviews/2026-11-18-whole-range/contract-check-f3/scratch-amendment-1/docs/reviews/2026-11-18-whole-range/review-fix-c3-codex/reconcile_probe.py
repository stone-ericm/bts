"""Offline CLI, cutoff, locking, receipt, and special-file probes."""

from contextlib import contextmanager, nullcontext
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from probe_support import DAY, ET, REPO, RUN, STREAK, decision, load, offline
from probe_support import plant, python_env, run_cli, scratch, seed_streak


MIDNIGHT = datetime(2027, 4, 20, 0, 1, tzinfo=ET)
CUTOFF = datetime(2027, 4, 20, 8, tzinfo=ET)
BEFORE_CUTOFF = CUTOFF.replace(hour=7, minute=59, second=59)
EARLY = "2027-04-15"


def receipts(root):
    from bts.reconcile_receipt import discover, receipts_root

    directory = receipts_root(root)
    return [record for day in sorted(directory.iterdir()) for record in discover(root, day.name)] if directory.exists() else []


def report(name, root, result, before):
    target = root / STREAK
    record = {
        "case": name, "exit": result.exit_code, "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(), "exception": type(result.exception).__name__ if result.exception else None,
        "before": before, "after": target.read_text() if target.is_file() else None,
        "receipts": receipts(root),
    }
    print(json.dumps(record, sort_keys=True), flush=True)
    return record


@contextmanager
def menu(answer=None, seen=None, after_lookup=None):
    from bts import picks

    def response(*args, **kwargs):
        assert answer is not None, "Unexpected feed lookup"
        if seen is not None:
            seen.append(args)
        if after_lookup is not None:
            after_lookup()
        return answer

    with patch.object(picks, "get_game_statuses_detailed", return_value={}), patch.object(picks, "check_hit", response):
        yield


def eligibility(work):
    def refuse(root, day):
        from bts.contest_state import DecisionStreakState
        from bts.picks import chosen_on_record, load_pick, saved_pick_refusal

        chosen = DecisionStreakState(0, True, True, "model", "fresh", 0, True)
        plant(root, day, chosen_on=chosen_on_record(chosen, required=False))
        live = DecisionStreakState(21, False, False, "contest", "fresh", 0, True, observation_kind="automatic")
        assert "chosen without the live check" in saved_pick_refusal(load_pick(day, root), live)

    def decide(root, day, *, status, scoreable, result=None, missing=False):
        decision(root, day, status=status, scoreable=scoreable)
        if not missing:
            plant(root, day, result)

    def marker(root, day, season):
        plant(root, day)
        seed_streak(root, markers=(day,), season=season)

    def invalid(root, day, path_kind):
        if path_kind == "pick":
            (root / f"{day}.json").write_text("[")
        else:
            decision(root, day, scoreable=True).write_text("{}")
            plant(root, day, "hit")

    items = {
        "03:00 refused preview": (EARLY, refuse, None),
        "private terminal": (EARLY, lambda r, d: decide(r, d, status="private_locked", scoreable=True, result="hit"), None),
        "private pending": (EARLY, lambda r, d: decide(r, d, status="private_locked", scoreable=True), "committed pick awaiting a result"),
        "locked unconfirmed": (EARLY, lambda r, d: decide(r, d, status="locked_unconfirmed", scoreable=True), "committed pick awaiting a result"),
        "bare attempt": (EARLY, lambda r, d: plant(r, d, delivery_attempted=True), "delivery attempt without a decision or result"),
        "delivered pending": (EARLY, lambda r, d: plant(r, d, bluesky_posted=True), "committed pick awaiting a result"),
        "skip with leftover": (EARLY, lambda r, d: decide(r, d, status="skipped", scoreable=False), None),
        "skip without root": (EARLY, lambda r, d: decide(r, d, status="skipped", scoreable=False, missing=True), None),
        "commit without root": (EARLY, lambda r, d: decide(r, d, status="delivered", scoreable=True, missing=True), "committed pick file missing"),
        "past applied tagged": (DAY, lambda r, d: marker(r, d, 2027), "applied-date marker without a result"),
        "past applied tagless": (DAY, lambda r, d: marker(r, d, None), "applied-date marker without a result"),
        "past applied foreign": (DAY, lambda r, d: marker(r, d, 2026), None),
        "today applied tagged": (RUN.date().isoformat(), lambda r, d: marker(r, d, 2027), "applied-date marker without a result"),
        "today applied tagless": (RUN.date().isoformat(), lambda r, d: marker(r, d, None), "applied-date marker without a result"),
        "today applied foreign": (RUN.date().isoformat(), lambda r, d: marker(r, d, 2026), None),
        "today committed unmarked": (RUN.date().isoformat(), lambda r, d: plant(r, d, bluesky_posted=True), None),
        "today attempt unmarked": (RUN.date().isoformat(), lambda r, d: plant(r, d, delivery_attempted=True), None),
        "today terminal": (RUN.date().isoformat(), lambda r, d: plant(r, d, "hit"), None),
        "bad older pick": (EARLY, lambda r, d: invalid(r, d, "pick"), "unreadable or invalid pick file"),
        "bad recent pick": (DAY, lambda r, d: invalid(r, d, "pick"), "unreadable or invalid pick file"),
        "bad decision beside hit": (EARLY, lambda r, d: invalid(r, d, "decision"), "unreadable or invalid decision record"),
    }
    for index, (name, (day, fixture, reason)) in enumerate(items.items()):
        root = work / str(index) / "picks"
        state = seed_streak(root)
        fixture(root, day)
        before = state.read_text()
        with menu():
            result = run_cli(root, MIDNIGHT)
        record = report(name, root, result, before)
        assert result.exit_code == 0
        if reason is None:
            assert not result.stderr and record["receipts"][0]["replay"]["state"] == "saved"
        else:
            assert result.stderr.strip() == f"Season replay unavailable for {day} ({reason})"
            assert "Streak retained:" in result.stdout and "recalculated" not in result.output
            assert record["before"] == record["after"]
            assert record["receipts"][0]["replay"]["blocked_date"] == day
            assert record["receipts"][0]["replay"]["reason"] == reason


def corrections(work):
    from bts import picks

    for before_result, after_result in (("hit", "miss"), ("miss", "hit")):
        for boundary in ("02:00", "07:59:59", "08:00", "during fetch", "waiting for lock"):
            root = work / f"{before_result}-{boundary}" / "picks"
            path = plant(root, DAY, before_result, bluesky_posted=True)
            plant(root, "2027-04-17", "hit")
            streak = seed_streak(root)
            before = streak.read_text()
            now = {"02:00": RUN, "08:00": CUTOFF}.get(boundary, BEFORE_CUTOFF)
            seen = []
            lock = picks.scoring_lock

            def advance():
                nonlocal now
                now = CUTOFF

            @contextmanager
            def late_lock(picks_dir):
                with lock(picks_dir):
                    advance()
                    yield

            with menu(after_result, seen, advance if boundary == "during fetch" else None), (
                patch.object(picks, "scoring_lock", late_lock) if boundary == "waiting for lock" else nullcontext()
            ):
                result = run_cli(root, lambda: now)
            record = report(boundary + " " + before_result + " to " + after_result, root, result, before)
            changed = boundary in ("02:00", "07:59:59")
            assert result.exit_code == 0 and not result.stderr
            assert bool(seen) == (boundary != "08:00")
            assert load(path)["result"] == (after_result if changed else before_result)
            assert ("Streak recalculated:" in result.stdout) == changed
            assert ("CORRECTIONS FOUND" in result.stdout) == changed
            assert record["receipts"][0]["replay"]["state"] == "saved"
            expected = 2 if load(path)["result"] == "hit" else 0
            assert load(streak)["streak"] == expected

    root = work / "retained-after-correction" / "picks"
    path = plant(root, DAY, "hit", bluesky_posted=True)
    plant(root, EARLY, bluesky_posted=True)
    target = seed_streak(root)
    before = target.read_text()
    with menu("miss"):
        result = run_cli(root)
    rec = report("applied correction with blocked season replay", root, result, before)
    assert result.exit_code == 0 and "hit -> miss" in result.stdout and "Streak retained: 21" in result.stdout
    assert "recalculated" not in result.output and load(path)["result"] == "miss" and rec["after"] == before


def failures(work):
    from bts import util

    for case, raw in (("bad JSON", "{broken"), ("array", "[]"), ("bad streak field", '{"season": 2027, "streak": "seven"}')):
        root = work / case / "picks"
        plant(root, EARLY, bluesky_posted=True)
        target = root / STREAK
        target.write_text(raw)
        with menu():
            result = run_cli(root)
        result_info = report(case + " blocked", root, result, raw)
        assert result.exit_code == 1 and "Traceback" not in result.output and target.read_text() == raw
        assert result_info["receipts"][0]["replay"]["retained_streak"] is None
        plant(root, EARLY, "hit")
        old_inode = target.stat().st_ino
        with target.open() as old, menu():
            result = run_cli(root)
            assert old.read() == raw
        report(case + " rebuilt", root, result, raw)
        assert result.exit_code == 0 and load(target)["streak"] == 1 and target.stat().st_ino != old_inode

    root = work / "interrupt" / "picks"
    target = seed_streak(root)
    before = target.read_text()
    path = plant(root, DAY, "miss", bluesky_posted=True)
    replace = util.os.replace

    def interrupt(src, dst):
        if Path(dst) == target:
            raise KeyboardInterrupt()
        return replace(src, dst)

    with menu("hit"), patch.object(util.os, "replace", interrupt):
        result = run_cli(root)
    report("interrupt before streak rename", root, result, before)
    assert result.exit_code == 1 and load(path)["result"] == "hit" and target.read_text() == before
    assert not list(root.glob("*.tmp"))
    with menu("hit"):
        result = run_cli(root)
    report("retry after interrupt", root, result, before)
    assert result.exit_code == 0 and load(target)["streak"] == 1


def special_worker(root, which, kind):
    from bts.reconcile_receipt import receipts_root

    answer = None
    if which == "picks_root":
        path = root
        if kind == "directory":
            return  # a directory is the expected type here
    else:
        root.mkdir()
        if which in ("old_pick", "recent_pick"):
            d = EARLY if which == "old_pick" else DAY
            path = root / f"{d}.json"
        elif which in ("old_decision", "recent_decision", "decision_parent"):
            d = DAY if which == "recent_decision" else EARLY
            plant(root, d, "hit")
            answer = "miss" if d == DAY else None
            path = root / d
            if which != "decision_parent":
                path.mkdir()
                path /= "decision.json"
            elif kind == "directory":
                return
        elif which == "streak":
            path = root / STREAK
        elif which == "scoring_lock":
            path = root / ".scoring.lock"
        elif which == "lineup_evolution":
            plant(root, DAY, "hit")
            path = next(root.glob("*lineup_evolution*.jsonl"))
            path.unlink()
            answer = "miss"
        else:
            path = receipts_root(root)
            if kind == "directory":
                path.mkdir()
                path = path / RUN.date().isoformat()
                # A receipt directory is legitimate, so use the type inversion at publication below.
                kind = "fifo"

    if kind == "fifo":
        os.mkfifo(path)
    else:
        path.mkdir()
    with menu(answer):
        result = run_cli(root)
    print(json.dumps({"path": str(path), "exit": result.exit_code, "stderr": result.stderr,
                      "stdout": result.stdout, "exception": repr(result.exception),
                      "fifo_survived": path.is_fifo()}))


def special(work):
    src = REPO / "src"
    for which in ("old_pick", "recent_pick", "old_decision", "recent_decision", "decision_parent", "streak",
                  "picks_root", "scoring_lock", "lineup_evolution", "receipt_directory"):
        for kind in ("fifo", "directory"):
            directory = work / f"{which}-{kind}"
            directory.mkdir(parents=True)
            try:
                proc = subprocess.run([sys.executable, __file__, "special-worker", str(directory / "picks"), which, kind],
                                      text=True, capture_output=True, env=python_env(src), timeout=3, check=True)
                info = json.loads(proc.stdout) if proc.stdout else {"skipped": "directory expected"}
            except subprocess.TimeoutExpired:
                info = {"timeout_seconds": 3}
            print(json.dumps({"case": which, "type": kind} | info), flush=True)


def race_worker(root, ready):
    seen = []
    with menu("hit", seen, lambda: ready.touch()):
        result = run_cli(root)
    print(json.dumps({"exit": result.exit_code, "stdout": result.stdout, "stderr": result.stderr, "seen": seen}))


def concurrent(work):
    from bts.picks import scoring_lock

    root = work / "picks"
    target = seed_streak(root)
    before = target.read_text()
    plant(root, DAY, "miss", bluesky_posted=True)
    with scoring_lock(root):
        readiness = [work / f"ready-{index}" for index in range(2)]
        processes = [subprocess.Popen([sys.executable, __file__, "race-worker", str(root), str(ready)],
                                      text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      env=python_env(REPO / "src")) for ready in readiness]
        try:
            deadline = time.monotonic() + 5
            while not all(path.exists() for path in readiness) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert all(path.exists() for path in readiness) and not any(proc.poll() is not None for proc in processes)
            assert target.read_text() == before
        except BaseException:
            for proc in processes:
                proc.terminate()
                proc.communicate(timeout=3)
            raise
    results = []
    for proc in processes:
        try:
            out, err = proc.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            raise
        assert proc.returncode == 0, err
        results.append(json.loads(out))
    print(json.dumps({"case": "two simultaneous reconciles", "results": results, "before": before,
                      "after": target.read_text(), "receipts": receipts(root)}), flush=True)
    assert sum("CORRECTIONS FOUND" in result["stdout"] for result in results) == 1
    assert all(result["exit"] == 0 for result in results) and load(target)["streak"] == 1
    assert len(receipts(root)) == 2


def main():
    mode = sys.argv[1]
    with offline():
        if mode == "special-worker":
            special_worker(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
        elif mode == "race-worker":
            race_worker(Path(sys.argv[2]), Path(sys.argv[3]))
        else:
            with scratch(mode) as directory:
                globals()[mode](Path(directory))


if __name__ == "__main__":
    main()
