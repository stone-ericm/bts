from datetime import date, datetime, timedelta
import os
from pathlib import Path
import runpy
import sys
from threading import Event, Thread, current_thread

import pytest

from bts import scheduler, strategy
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


class FirstCheck(BaseException):
    pass


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

    def check(**kwargs):
        raise FirstCheck

    class RecordWorker(Thread):
        def start(self):
            workers.append(self)
            super().start()

    monkeypatch.setattr(scheduler, "Thread", RecordWorker)
    monkeypatch.setattr(alert, "send_dm", send)
    monkeypatch.setattr(scheduler, "run_single_check", check)
    monkeypatch.setattr(scheduler, "_idle_until_next_wakeup", lambda *a: None)
    monkeypatch.setattr(scheduler, "notify_ready", lambda: None)
    monkeypatch.setattr(scheduler, "notify_watchdog", lambda: None)

    def start(games, *, day=DAY, at=None, dry=False, journal_delay=lambda: 0):
        now = datetime.fromisoformat(at or f"{day}T10:00").replace(tzinfo=scheduler.ET)
        monkeypatch.setattr(scheduler, "_now_et", lambda: now + timedelta(seconds=journal_delay()))
        def advance(seconds):
            nonlocal now
            now += timedelta(seconds=seconds)
        monkeypatch.setattr(scheduler, "_watchdog_ping_sleep", advance)
        monkeypatch.setattr(scheduler, "retry_urlopen", lambda *a, **kw: io.BytesIO(
            json.dumps({"dates": [{"date": day, "games": games}]}).encode()))
        try:
            scheduler.run_day(day, config, dry_run=dry)
        except FirstCheck:
            pass
        for worker in workers:
            worker.join(1)
            assert not worker.is_alive()
        workers.clear()

    return start, bodies, config


def test_warning_restart_and_count_change(startup, capfd):
    start, bodies, _ = startup
    start([game()])
    start([game(), game(gamePk=124)])
    start([game()])
    assert len(bodies) == 1
    assert "At scheduler startup MLB listed at least one regular-season game" in bodies[0]
    assert "The configured exclusive boundary is 2027-09-27" in bodies[0]
    assert len(bodies[0]) < 1000
    assert capfd.readouterr().err.count("SEASON BOUNDARY BEHIND SCHEDULE") == 3


def test_warning_dm_describes_reported_status(startup):
    start, bodies, _ = startup
    start([game(status={"abstractGameCode": "F", "detailedState": "Final"})])
    assert ("At scheduler startup MLB listed at least one regular-season game with that "
            "official date and a reported status other than postponed or canceled.") in bodies[0]


def test_warning_journal_describes_reported_status(startup, capfd):
    start, _, _ = startup
    start([game()])
    assert ("1 regular-season game(s) with that official date and a reported status "
            "other than postponed or canceled.") in capfd.readouterr().err


def test_warning_journal_describes_past_check(startup, monkeypatch, capfd):
    _, _, config = startup
    now = datetime.fromisoformat(f"{DAY}T16:05").replace(tzinfo=scheduler.ET)
    monkeypatch.setattr(scheduler, "_now_et", lambda: now)
    workers = []

    class RecordWorker(Thread):
        def start(self):
            workers.append(self)
            super().start()

    monkeypatch.setattr(scheduler, "Thread", RecordWorker)
    scheduler._season_boundary_check(
        DAY, [game()], [{"time_et": now}], config, Path(config["orchestrator"]["picks_dir"]))
    for worker in workers:
        worker.join(1)
        assert not worker.is_alive()
    assert "The first planned lineup check is due now or has already passed." in capfd.readouterr().err


@pytest.mark.parametrize("at,pitch,first_check,stall", [
    ("10:05:00", "13:15:00", "10:15:00", 720),
    ("18:59:59", "19:05:00", "18:59:59", 2),
], ids=["thin-slate-first-check", "restart-before-entry-cutoff"])
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

    class ShortJoin(Thread):
        def start(self):
            workers.append(self)
            super().start()

        def join(self, timeout=None):
            nonlocal elapsed
            if timeout:
                assert 59 < timeout <= 60
                assert blocked.wait(1)
                elapsed += timeout

    def check(**kwargs):
        checks.append((scheduler._now_et().strftime("%H:%M:%S"), kwargs["unavailable_game_pks"]))
        raise FirstCheck

    monkeypatch.setattr(scheduler, "run_single_check", check)
    monkeypatch.setattr(scheduler, "Thread", ShortJoin)
    monkeypatch.setattr(sys, "stderr", Journal())
    monkeypatch.setattr(os, "write", write)
    try:
        start([game(gameDate=f"{DAY}T{pitch}-04:00")], at=f"{DAY}T{at}",
              journal_delay=lambda: elapsed)
        assert [(clock, set(unavailable)) for clock, unavailable in checks] == [(first_check, set())]
        assert blocked.wait(1)
    finally:
        release.set()
        for worker in workers:
            Thread.join(worker, 2)
            assert not worker.is_alive()


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
    start, bodies, _ = startup
    day = "2028-09-29"
    start([game(day), game(day, gamePk=124, officialDate="2028-09-30"),
           game(day, gamePk=125, status=None), game(day, gamePk=126, gameType=None),
           game(day, gamePk=127, status={"detailedState": " "})], day=day)
    assert len(bodies) == 1
    assert "The fallback exclusive boundary for the unconfigured year 2028 is 2028-09-28" in bodies[0]
    assert "1 regular-season game(s) with that official date" in capfd.readouterr().err


@pytest.mark.parametrize("outcome", [
    None, " ", OSError("send failed"),
    pytest.param(KeyboardInterrupt(), marks=pytest.mark.filterwarnings(
        "ignore::pytest.PytestUnhandledThreadExceptionWarning")),
])
def test_warning_unconfirmed_send_retried(startup, monkeypatch, outcome):
    start, bodies, _ = startup

    def send(recipient, body):
        bodies.append(body)
        if len(bodies) > 1:
            return "message-id"
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(alert, "send_dm", send)
    for _ in range(3):
        start([game()])
    assert len(bodies) == 2


@pytest.mark.parametrize("kind", ["bad JSON", "schema-less", "symlink", "directory", "fifo"])
def test_warning_invalid_receipt(startup, tmp_path, kind):
    import os

    start, bodies, _ = startup
    receipt = tmp_path / "health_state" / "season_boundary" / f"{DAY}.json"
    receipt.parent.mkdir(parents=True)
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text("{}")
    if kind == "symlink":
        receipt.symlink_to(unrelated)
    elif kind == "directory":
        receipt.mkdir()
    elif kind == "fifo":
        os.mkfifo(receipt)
    else:
        receipt.write_text("{" if kind == "bad JSON" else '{"status": "sent"}')
    start([game()])
    start([game()])
    assert len(bodies) == (1 if kind in {"bad JSON", "schema-less"} else 2)
    assert unrelated.read_text() == "{}"


def test_warning_receipts_keyed_to_slate(startup, capfd):
    start, bodies, _ = startup
    next_day = "2027-09-28"
    start([game()])
    start([game(next_day)], day=next_day)
    start([game()])
    start([game()], at=f"{next_day}T00:01")
    assert len(bodies) == 2
    assert "The first planned lineup check is due now or has already passed." in capfd.readouterr().err


def test_morning_runs_first_with_a_fresh_clock(startup, monkeypatch):
    start, bodies, config = startup
    config["scheduler"].update(pick_delivery="dm", contest_state_required=True)

    def morning(config, picks_dir, day, now, first_check):
        bodies.append("morning")
        monkeypatch.setattr(scheduler, "_now_et", lambda: now + timedelta(seconds=11))

    monkeypatch.setattr(scheduler, "_morning_state_check", morning)
    start([game()], at=f"{DAY}T15:54:55")
    assert bodies == ["morning"]


@pytest.mark.parametrize("lag,boundary_sent", [(0, True), (11, False)])
def test_real_morning_refusal_precedes_boundary(tmp_path, lag, boundary_sent):
    import json

    probe = runpy.run_path(str(CHECKS / "contract-check-k2b/probe_scheduler.py"))
    config = probe["fixture"](tmp_path, mode="dm", required=True)
    result = probe["start"](config, DAY, [probe["game"](DAY)], at=f"{DAY}T15:54:55",
                            morning_lag=lag)
    base_output = (CHECKS / "contract-check-k2b/base-scheduler.out").read_text()
    base = json.loads(base_output)
    assert result["bodies"][0] == base["S5_original_11s"]["bodies"][0]
    assert any("[season_boundary]" in body for body in result["bodies"]) is boundary_sent


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


def test_boundary_transport_can_outlive_wait(startup, monkeypatch, capfd):
    start, _, _ = startup

    sending = Event()
    release = Event()
    deadline_written = Event()
    workers = []
    original_write = os.write

    def write(fd, data):
        size = original_write(fd, data)
        if fd == 2 and b"still sending at its deadline" in data:
            deadline_written.set()
        return size

    def send(recipient, body):
        sending.set()
        assert release.wait(2)
        return "message-id"

    class ThreadWithShortWait(Thread):
        def start(self):
            workers.append(self)
            super().start()

        def join(self, timeout=None):
            assert 59 < timeout <= 60
            assert sending.wait(1)

    monkeypatch.setattr(alert, "send_dm", send)
    monkeypatch.setattr(os, "write", write)
    monkeypatch.setattr(scheduler, "Thread", ThreadWithShortWait, raising=False)
    try:
        start([game()], at=f"{DAY}T15:55:00")
        assert deadline_written.wait(1)
        journal = capfd.readouterr().err
        assert "still sending at its deadline; continuing." in journal, journal
    finally:
        release.set()
        for worker in workers:
            Thread.join(worker, 2)


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
def test_startup_without_warning(startup, capsys, day, games, dry):
    start, bodies, _ = startup
    start(games, day=day, dry=dry)
    assert not bodies
    assert "SEASON BOUNDARY BEHIND SCHEDULE" not in capsys.readouterr().err
