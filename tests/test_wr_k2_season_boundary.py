from datetime import date, datetime, timedelta
import os
from pathlib import Path
import runpy
import sys
from threading import Event, Thread, current_thread
from types import SimpleNamespace

import pytest

from bts import picks, scheduler, strategy
from bts.health import alert
from bts.health import tail_policy
from scripts.audit.season_walk import walk


DAY = "2027-09-27"
CHECKS = Path("docs/reviews/2026-11-18-whole-range")
CASES = [(57, 57, "double", "longest_streak"),
         (120, 120, "double", "longest_streak"),
         (55, 55, "double", "reach57"),
         (56, 56, "single", "reach57"),
         (20, 20, "double", "emax_season_best"),
         (20, 25, "skip", "emax_season_best")]


@pytest.mark.parametrize("day", [DAY, "2028-09-29"])
@pytest.mark.parametrize("saver", [False, True])
@pytest.mark.parametrize("streak,best,action,objective", CASES)
def test_selection_and_explicit_context(day, saver, streak, best, action, objective, tmp_path):
    frame = walk._predictions(date.fromisoformat(day), .8)
    statuses = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
    context = strategy.DecisionContext(.8, .79, True, day, True, strategy._load_mdp(),
                                       best, "trusted")
    bare = strategy.resolve_policy_decision(context, streak, saver)
    assert (bare.action, bare.objective, bare.days_effective) == ("skip", "reach57", 0)
    assert strategy.effective_pick_bar(streak, day, saver) is None
    selected = strategy.select_pick(frame, day, tmp_path, streak, saver,
                                    game_statuses_detailed=statuses,
                                    best_streak=best, best_status="trusted")
    decision = selected.decision
    assert (decision.action, decision.objective, decision.days_effective) == (action, objective, 1)
    context.min_days = 1
    assert strategy.resolve_policy_decision(context, streak, saver) == decision
    if action == "skip":
        assert "max reachable 22" in decision.reason
    else:
        assert selected.pick_result.daily.policy_decision["days_effective"] == 1


@pytest.mark.parametrize("streak,best,action,objective", CASES)
def test_walk_makeup(streak, best, action, objective):
    row = walk.walk_one(date.fromisoformat(DAY), walk.State("dated", streak, best, "used", .8))
    assert (row["action"], row["objective"], row["days_left"], row["days_effective"]) == (
        action, objective, 0, 1)
    assert walk.check([row]) == []


def test_walk_empty_after_finale():
    row = walk.walk_one(date(2027, 10, 1), walk.State("dated", 57, 57, "used", .8),
                        empty_slate=True)
    assert row["days_left"] == 0
    assert row["action"] is row["days_effective"] is None
    assert walk.check([row]) == []


def no_pick_check(**kwargs):
    return {"new_lineups": 0, "skipped": True, "pick_result": None, "should_post": False}


def game(day=DAY, **changes):
    return {"gamePk": 123, "gameType": "R", "officialDate": day,
            "gameDate": f"{day}T19:05:00-04:00",
            "teams": {"away": {"team": {"name": "Away"}},
                      "home": {"team": {"name": "Home"}}},
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"}} | changes


@pytest.fixture
def startup(monkeypatch, tmp_path):
    import io
    import json

    bodies = []
    workers = []
    config = {"orchestrator": {"picks_dir": str(tmp_path / "picks"),
                               "heartbeat_path": str(tmp_path / "heartbeat")},
              "scheduler": {"pick_delivery": "private"},
              "health_checks": {"enabled": False},
              "bluesky": {"dm_recipient": "test"}}

    def send(recipient, body):
        bodies.append(body)
        return "message-id"

    class RecordWorker(Thread):
        def start(self):
            workers.append(self)
            super().start()

    monkeypatch.setattr(scheduler, "Thread", RecordWorker)
    monkeypatch.setattr(alert, "send_dm", send)
    monkeypatch.setattr(scheduler, "run_single_check", no_pick_check)
    monkeypatch.setattr(scheduler, "_idle_until_next_wakeup", lambda *a: None)
    monkeypatch.setattr(scheduler, "_next_day_wakeup", lambda day, config:
                        datetime.fromisoformat(day).replace(tzinfo=scheduler.ET)
                        + timedelta(days=1, hours=10))
    monkeypatch.setattr(scheduler, "_maybe_alert_missed_pick", lambda *a: None)
    monkeypatch.setattr(scheduler, "notify_ready", lambda: None)
    monkeypatch.setattr(scheduler, "notify_watchdog", lambda: None)

    def start(games, *, day=DAY, at=None, dry=False, journal_delay=lambda: 0,
              before_join=lambda: None):
        now = datetime.fromisoformat(at or f"{day}T10:00").replace(tzinfo=scheduler.ET)
        monkeypatch.setattr(scheduler, "_now_et", lambda: now + timedelta(seconds=journal_delay()))
        def advance(seconds):
            nonlocal now
            now += timedelta(seconds=seconds)
        monkeypatch.setattr(scheduler, "_watchdog_ping_sleep", advance)
        monkeypatch.setattr(scheduler, "retry_urlopen", lambda *a, **kw: io.BytesIO(
            json.dumps({"dates": [{"date": day, "games": games}]}).encode()))
        scheduler.run_day(day, config, dry_run=dry)
        before_join()
        for worker in workers:
            worker.join(1)
            assert not worker.is_alive()
        workers.clear()

    return start, bodies, config


@pytest.mark.parametrize("mode", ["private", "dm", "public"])
def test_qualifying_start_sends_no_dm_and_writes_no_receipt(startup, tmp_path, mode):
    start, bodies, config = startup
    config["scheduler"]["pick_delivery"] = mode

    start([game()])

    receipt = tmp_path / "health_state" / "season_boundary" / f"{DAY}.json"
    assert (bodies, receipt.exists()) == ([], False)


def test_configured_boundary_journal_counts_each_start(startup, capfd):
    start, _, _ = startup
    start([game()])
    start([game(), game(gamePk=124, status={"detailedState": "Final"}),
           game(gamePk=125, status={"detailedState": "Postponed"}),
           game(gamePk=126, status={"detailedState": "Canceled"})])
    start([game()])

    expected = (
        "SEASON BOUNDARY BEHIND SCHEDULE — 2027-09-27 ET: 1 regular-season game(s) "
        "with that official date and a reported status other than postponed or canceled. "
        "The configured exclusive boundary is 2027-09-27, giving "
        "a raw count of zero. A fresh pick selection with an available predicted candidate "
        "uses the last-day policy. Ask the BTS lead session or Eric to check MLB's last "
        "regular-season game date, including pending makeup games, and to update the "
        "stored boundary for a later run."
    )
    lines = [line for line in capfd.readouterr().err.splitlines()
             if line.startswith("SEASON BOUNDARY")]
    assert lines == [expected, expected.replace("1 regular-season", "2 regular-season"), expected]


@pytest.mark.parametrize("at,pitch,first_check,stall", [
    ("10:05:00", "13:15:00", "10:15:00", 720),
    ("10:10:00", "13:15:00", "10:15:00", 720),
    ("18:59:59", "19:05:00", "18:59:59", 2),
], ids=["thin-slate-first-check", "thin-slate-start-near-check", "restart-before-entry-cutoff"])
def test_stalled_boundary_journal_preserves_first_check(
        startup, monkeypatch, at, pitch, first_check, stall):
    start, _, _ = startup
    scheduler_thread = current_thread()
    blocked = Event()
    release = Event()
    workers = []
    checks = []
    elapsed = 0
    original_write = os.write
    original_stderr = sys.stderr
    def stall_warning():
        nonlocal elapsed
        if current_thread() is scheduler_thread:
            elapsed += stall
        else:
            blocked.set()
            assert release.wait(2)

    class Journal:
        def write(self, value):
            if (current_thread() is scheduler_thread
                    and value.startswith("SEASON BOUNDARY BEHIND SCHEDULE")):
                stall_warning()
            return original_stderr.write(value)

        def flush(self):
            original_stderr.flush()

    def write(fd, data):
        if fd == 2 and data.startswith(b"SEASON BOUNDARY BEHIND SCHEDULE"):
            stall_warning()
            return len(data)
        return original_write(fd, data)

    class RecordWorker(Thread):
        def start(self):
            workers.append(self)
            super().start()

    def check(**kwargs):
        checks.append((scheduler._now_et().strftime("%H:%M:%S"), kwargs["unavailable_game_pks"]))
        return no_pick_check(**kwargs)

    def release_writer():
        assert checks[0] == (first_check, set())
        assert blocked.wait(1)
        release.set()

    monkeypatch.setattr(scheduler, "run_single_check", check)
    monkeypatch.setattr(scheduler, "Thread", RecordWorker)
    monkeypatch.setattr(sys, "stderr", Journal())
    monkeypatch.setattr(os, "write", write)
    try:
        start([game(gameDate=f"{DAY}T{pitch}-04:00")], at=f"{DAY}T{at}",
              journal_delay=lambda: elapsed, before_join=release_writer)
    finally:
        release.set()
        for worker in workers:
            Thread.join(worker, 2)
            assert not worker.is_alive()


def test_boundary_journal_follows_last_lineup_check_and_fallback(startup, monkeypatch, capfd):
    start, _, _ = startup
    daily = SimpleNamespace(pick=SimpleNamespace(game_time=f"{DAY}T19:05:00-04:00"),
                            double_down=None)
    monkeypatch.setattr(picks, "load_pick", lambda *a: daily)
    monkeypatch.setattr(picks, "pick_was_delivered", lambda *a: False)
    monkeypatch.setattr(scheduler, "_refresh_pick_at_fallback_decision",
                        lambda *a, **kw: SimpleNamespace(daily=daily, selection=None))
    delivered = []
    monkeypatch.setattr(scheduler, "_deliver_and_lock_pick",
                        lambda *a, **kw: delivered.append(a[-1]))

    # Run the queued writer to completion at its enqueue point so thread scheduling
    # cannot hide a warning queued too early.
    workers = []
    original_queue = scheduler._queue_season_boundary_line

    class JournalWorker(Thread):
        def start(self):
            workers.append(self)
            super().start()

    def queue(line):
        original_queue(line)
        workers[-1].join(1)
        assert not workers[-1].is_alive()

    monkeypatch.setattr(scheduler, "Thread", JournalWorker)
    monkeypatch.setattr(scheduler, "_queue_season_boundary_line", queue)
    start([game()])

    assert delivered == ["final fallback"]
    journal = capfd.readouterr().err
    assert (0 <= journal.rfind("Running lineup check...")
            < journal.index("FALLBACK —")
            < journal.index("SEASON BOUNDARY BEHIND SCHEDULE")
            < journal.index("Tomorrow's wake-up:"))


def test_shared_journal_with_690_bytes_free_preserves_restart_cutoff(startup, monkeypatch):
    start, _, _ = startup
    elapsed = 0
    checks = []
    lines = []
    wrote = Event()
    original_queue = scheduler._queue_season_boundary_line
    original_write = os.write
    reader, writer = os.pipe()

    def charge(data):
        nonlocal elapsed
        try:
            written = original_write(writer, data)
        except BlockingIOError:
            # Simulate two seconds of backpressure before the reader resumes.
            elapsed += 2
            os.read(reader, capacity)
            written = original_write(writer, data)
        assert written == len(data)
        return written

    class SharedJournal:
        def write(self, value):
            charge(value.encode())
            return len(value)

        def flush(self):
            pass

    def write(fd, data):
        if fd == 2 and data.startswith(b"SEASON BOUNDARY BEHIND SCHEDULE"):
            charge(data)
            lines.append(data)
            wrote.set()
            return len(data)
        return original_write(fd, data)

    def queue(line):
        original_queue(line)
        assert wrote.wait(1)

    def check(**kwargs):
        checks.append((scheduler._now_et().strftime("%H:%M:%S"), kwargs["unavailable_game_pks"]))
        return no_pick_check(**kwargs)

    monkeypatch.setattr(sys, "stderr", SharedJournal())
    monkeypatch.setattr(os, "write", write)
    monkeypatch.setattr(scheduler, "_queue_season_boundary_line", queue)
    monkeypatch.setattr(scheduler, "run_single_check", check)
    try:
        os.set_blocking(writer, False)
        capacity = 0
        for size in (1024, 1):
            while True:
                try:
                    capacity += original_write(writer, b"x" * size)
                except BlockingIOError:
                    break
        assert len(os.read(reader, 690)) == 690
        start([game()], at=f"{DAY}T18:59:59", journal_delay=lambda: elapsed)
    finally:
        os.close(writer)
        os.close(reader)

    assert len(lines) == 1
    assert checks[0] == ("18:59:59", set())


def test_positive_day_grid(capsys):
    import json

    probe = runpy.run_path(str(CHECKS / "contract-check-k2b/probe_policy.py"))
    probe["grid"]()
    assert json.loads(capsys.readouterr().out) == json.loads(
        (CHECKS / "contract-check-k2/base-grid.json").read_text())


def test_health_retains_raw_boundary():
    from dataclasses import asdict
    import gzip
    import json

    with gzip.open(CHECKS / "contract-check-k2b/base-policy.out.gz", "rt") as stored:
        expected = json.load(stored)["health"]
    actual = []
    for day, *_ in expected:
        today = date.fromisoformat(day)
        actual.append([day, strategy._days_remaining(day),
                       [asdict(a) for a in tail_policy.check(
                           today=today, base_path=Path("data/models/mdp_policy.npz"),
                           tail_path=Path("data/models/mdp_tail_policy.npz"))],
                       [asdict(a) for a in tail_policy.season_end_alerts(today=today)]])
    assert actual == expected


def test_fallback_warning_and_unclassifiable_rows(startup, capfd):
    start, _, _ = startup
    day = "2028-09-29"
    start([game(day), game(day, gamePk=124, officialDate="2028-09-30"),
           game(day, gamePk=125, status=None), game(day, gamePk=126, gameType=None),
           game(day, gamePk=127, status={"detailedState": " "})], day=day)
    lines = [line for line in capfd.readouterr().err.splitlines()
             if line.startswith("SEASON BOUNDARY")]
    assert lines == [
        "SEASON BOUNDARY BEHIND SCHEDULE — 2028-09-29 ET: 1 regular-season game(s) "
        "with that official date and a reported status other than postponed or canceled. "
        "The fallback exclusive boundary for the unconfigured year 2028 is 2028-09-28, giving "
        "a raw count of zero. A fresh pick selection with an available predicted candidate "
        "uses the last-day policy. Ask the BTS lead session or Eric to check MLB's last "
        "regular-season game date, including pending makeup games, and to update the "
        "stored boundary for a later run."
    ]


def test_real_morning_refusal_unchanged_at_boundary(tmp_path):
    import json

    probe = runpy.run_path(str(CHECKS / "contract-check-k2b/probe_scheduler.py"))
    config = probe["fixture"](tmp_path, mode="dm", required=True)
    result = probe["start"](config, DAY, [probe["game"](DAY)], at=f"{DAY}T15:54:55")
    base_output = (CHECKS / "contract-check-k2b/base-scheduler.out").read_text()
    base = json.loads(base_output)
    assert result["bodies"] == base["S5_original_11s"]["bodies"]


@pytest.mark.parametrize("streak,best,objective", [
    (57, 57, "longest_streak"), (20, 25, "emax_season_best"),
])
def test_saved_decision_keeps_schema(tmp_path, streak, best, objective):
    from bts.daily_decision import DECISION_SCHEMA
    from bts.picks import load_pick

    probe = runpy.run_path(str(CHECKS / "contract-check-k2b/probe_scheduler.py"))
    config = probe["fixture"](tmp_path)
    picks = Path(config["orchestrator"]["picks_dir"])
    check = probe["selection"](DAY, picks, streak, best)
    result = probe["start"](config, DAY, [probe["game"](DAY, i) for i in range(3)],
                            at=f"{DAY}T18:30", check_result=check)
    decision = result["decision_json"]
    assert decision["schema_version"] == DECISION_SCHEMA
    assert decision["objective"] == objective
    assert "days_effective" not in decision
    assert check["selection"].decision.days_effective == 1
    if streak == 57:
        assert load_pick(DAY, picks).policy_decision["days_effective"] == 1




@pytest.mark.parametrize("day,games,dry", [
    ("2027-09-26", [game("2027-09-26")], False),
    (DAY, [], False),
    (DAY, [game(gameType="F"), game(gameType="S")], False),
    (DAY, [game()], True),
    (DAY, [game(status={"detailedState": "Postponed"})], False),
    (DAY, [game(officialDate="2027-09-26")], False),
    ("2027-12-31", [], False),
    ("2028-01-01", [], False),
])
def test_startup_without_warning(startup, capfd, day, games, dry):
    start, bodies, _ = startup
    start(games, day=day, dry=dry)
    assert not bodies
    assert "SEASON BOUNDARY BEHIND SCHEDULE" not in capfd.readouterr().err
