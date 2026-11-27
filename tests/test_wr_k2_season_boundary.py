from datetime import date, datetime, timedelta
import os
from pathlib import Path
import runpy
import sys

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
def test_qualifying_start_writes_no_boundary_line(startup, capfd, mode):
    """Eric 2026-11-27 (K2 option 2): the stale-boundary journal line is dropped; only the floor stays."""
    start, bodies, config = startup
    config["scheduler"]["pick_delivery"] = mode
    start([game()])
    err = capfd.readouterr().err
    assert "SEASON BOUNDARY" not in err and "Season boundary" not in err
    assert bodies == []
    assert not hasattr(scheduler, "_season_boundary_check")


@pytest.mark.parametrize("mode", ["private", "dm", "public"])
def test_qualifying_start_sends_no_dm_and_writes_no_receipt(startup, tmp_path, mode):
    start, bodies, config = startup
    config["scheduler"]["pick_delivery"] = mode

    start([game()])

    receipt = tmp_path / "health_state" / "season_boundary" / f"{DAY}.json"
    assert (bodies, receipt.exists()) == ([], False)


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
