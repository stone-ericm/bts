import os
import sys
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import bts

DOCS = Path.cwd() / "docs/reviews/2026-11-18-whole-range"
sys.path.insert(0, str(DOCS / "review-fix-k2-codex"))
from probe_journal_pipe import PipeJournal
from probe_startup import DAY, Startup, game, scheduler

from bts import contest_state, dm, picks
from bts.daily_decision import load_decision

PREFIX = b"SEASON BOUNDARY BEHIND SCHEDULE"
TOMORROW = "2027-09-28"


def daily_at(pitch):
    return picks.DailyPick(
        date=DAY, run_time=f"{DAY}T10:05:00-04:00",
        pick=picks.Pick(batter_name="Fixture Batter", batter_id=1, team="NYM",
                        lineup_position=1, pitcher_name="Fixture Pitcher", pitcher_id=2,
                        p_game_hit=.8, flags=[], projected_lineup=True, game_pk=301,
                        game_time=f"{DAY}T{pitch}-04:00"),
        double_down=None, runner_up=None,
    )


def replay(root, budget, at, pitch, delay, enabled):
    start = Startup(root, mode="dm")
    pick_dir = root / "picks"
    daily = daily_at(pitch)
    checks, fallbacks, deliveries, sends, lines, workers = [], [], [], [], [], []
    worker_entered = Event()
    original_write = os.write
    original_boundary = scheduler._season_boundary_check
    original_deliver = scheduler._deliver_and_lock_pick
    original_next = scheduler._next_day_wakeup
    prefix_at_hook = []

    def advance():
        start.now += timedelta(seconds=delay)

    journal = PipeJournal(budget, advance)

    class JournalWorker(Thread):
        def start(self):
            workers.append(self)
            super().start()

    def write(fd, data):
        if fd == 2 and data.startswith(PREFIX):
            lines.append(data)
            worker_entered.set()
            journal.write(data.decode())
            return len(data)
        return original_write(fd, data)

    def boundary(*args):
        prefix_at_hook.append(journal.getvalue())
        if enabled:
            original_boundary(*args)

    def lookahead(*args):
        if enabled:
            assert worker_entered.wait(1)
        return original_next(*args)

    def check(**kwargs):
        checks.append((start.now.isoformat(), sorted(kwargs["unavailable_game_pks"])))
        if len(checks) == 1:
            picks.save_pick(daily, pick_dir)
        return {"new_lineups": 0, "skipped": True, "pick_result": None, "should_post": False}

    def refresh(*args, **kwargs):
        fallbacks.append((start.now.isoformat(), sorted(kwargs["unavailable_game_pks"])))
        return SimpleNamespace(daily=daily, selection=None)

    def deliver(candidate, config, picks_dir, state, date, label, **kwargs):
        deliveries.append((start.now.isoformat(), label, picks.submission_cutoff_et(candidate).isoformat()))
        return original_deliver(candidate, config, picks_dir, state, date, label, **kwargs)

    def send(recipient, body):
        sends.append((start.now.isoformat(), recipient, body))
        return "fixture-id"

    mocks = [
        patch.object(os, "write", write),
        patch.object(scheduler, "Thread", JournalWorker),
        patch.object(scheduler, "_season_boundary_check", boundary),
        patch.object(scheduler, "_next_day_wakeup", lookahead),
        patch.object(scheduler, "run_single_check", check),
        patch.object(scheduler, "_refresh_pick_at_fallback_decision", refresh),
        patch.object(scheduler, "_void_committed_games", return_value={}),
        patch.object(scheduler, "_deliver_and_lock_pick", deliver),
        patch.object(scheduler, "_trigger_live_forward_capture_on_lock", return_value=None),
        patch.object(scheduler, "run_result_polling", return_value="final"),
        patch.object(contest_state, "load_decision_streak_state", return_value=SimpleNamespace(streak=57)),
        patch.object(dm, "send_dm", send),
        patch.object(scheduler.time, "monotonic", lambda: start.now.timestamp()),
    ]
    try:
        start.run([game(start=pitch)], at=at, journal=journal, extra=mocks)
        for worker in workers:
            worker.join(1)
            assert not worker.is_alive()
        assert not start.bodies
        assert len(lines) == len(workers) == int(enabled)
        assert len(prefix_at_hook) == 1
        stored = picks.load_pick(DAY, pick_dir)
        decision = load_decision(DAY, pick_dir)
        assert decision["scoreable"] and decision["delivery_status"] == "delivered"
        assert stored.notification_id == "fixture-id"
        print(f"free={budget} at={at} line={len(lines[0]) if lines else 0}"
              f" checks={checks} fallback={fallbacks} delivery={deliveries}"
              f" first_block={journal.blocked_write!r}", flush=True)
        return checks, fallbacks, deliveries, sends, stored.delivered_at, prefix_at_hook[0]
    finally:
        journal.finish()


def main():
    print("source:", bts.__file__, flush=True)
    with TemporaryDirectory() as temp:
        root = Path(temp)
        for at, pitch, delay, budgets in (
            ("18:59:59", "19:05:00", 2, (510, 650, 666, 672, 678, 684, 690, 696, 704)),
            ("10:05:00", "13:15:00", 720, (350, 690)),
        ):
            for budget in budgets:
                without, with_line = [
                    replay(root / f"{at}-{budget}-{enabled}", budget, at, pitch, delay, enabled)
                    for enabled in (False, True)
                ]
                assert with_line == without, (budget, at, without, with_line)
                checks, fallbacks, deliveries, sends, _, _ = with_line
                first = at if at == "18:59:59" else "10:15:00"
                assert checks[0] == (f"{DAY}T{first}-04:00", [])
                assert len(checks) == (1 if at == "18:59:59" else 3)
                assert len(fallbacks) == len(deliveries) == len(sends) == 1
                assert fallbacks[0][1] == [] and deliveries[0][1] == "final fallback"
                assert fallbacks[0][0] == deliveries[0][0] == sends[0][0] < deliveries[0][2]


if __name__ == "__main__":
    main()
