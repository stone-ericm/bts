"""Real selection and outward delivery paths using offline transports."""

import json
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timedelta
from unittest.mock import Mock

import pandas as pd
import pytest
from click.testing import CliRunner

from bts import cli, contest_state, dm, orchestrator, picks, posting, scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import (
    DAY, NOW, daily_pick, record, set_automatic, set_dated,
)


@pytest.fixture
def world(tmp_path, monkeypatch):
    from bts.model import predict

    now = [NOW]

    class DecisionClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now[0].astimezone(tz) if tz else now[0].replace(tzinfo=None)

    picks_dir = tmp_path / "picks"
    statuses = {pk: {"abstract": "P", "detailed": "Scheduled", "code": "S"}
                for pk in (101, 102)}
    predictions = pd.DataFrame([
        {"batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
         "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
         "p_hit_pa": 0.24, "p_game_hit": 0.82, "flags": "", "projected_lineup": False,
         "game_pk": 101, "game_time": f"{DAY}T23:10:00Z"},
        {"batter_id": 665742, "batter_name": "Juan Soto", "team": "NYM", "lineup": 2,
         "pitcher_id": 605483, "pitcher_name": "Blake Snell", "opp": "LAD",
         "p_hit_pa": 0.23, "p_game_hit": 0.81, "flags": "", "projected_lineup": False,
         "game_pk": 102, "game_time": f"{DAY}T23:40:00Z"},
    ])
    config = {
        "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(tmp_path / "heartbeat"),
                         "models_dir": str(tmp_path / "models")},
        "bluesky": {"dm_recipient": "offline.test"},
        "scheduler": {"contest_state_required": True, "pick_delivery": "dm",
                      "live_forward_capture_on_lock": False},
        "health_checks": {"enabled": False},
        "tiers": [],
    }
    monkeypatch.setattr(contest_state, "datetime", DecisionClock)
    monkeypatch.setattr(scheduler, "_now_et", lambda: now[0])
    monkeypatch.setattr(cli, "_now_et", lambda: now[0])
    monkeypatch.setattr(alert, "datetime", DecisionClock)
    monkeypatch.setattr(predict, "run_pipeline", Mock(return_value=predictions))
    monkeypatch.setattr(orchestrator, "run_cascade", Mock(return_value=(predictions, "offline")))
    monkeypatch.setattr(picks, "get_game_statuses_detailed", lambda *a: statuses)
    monkeypatch.setattr(strategy, "get_game_statuses_detailed", lambda *a: statuses)
    monkeypatch.setattr(picks, "get_game_statuses", lambda *a: {pk: "P" for pk in statuses})
    monkeypatch.setattr(scheduler, "fetch_schedule", Mock(return_value=[]))
    monkeypatch.setattr(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext())
    monkeypatch.setattr(scheduler, "retry_urlopen",
                        Mock(side_effect=AssertionError("unexpected MLB transport")))
    monkeypatch.setattr(dm, "send_dm", Mock(return_value="pick-dm"))
    monkeypatch.setattr(alert, "send_dm", Mock(return_value="health-dm"))
    monkeypatch.setattr(posting, "post_to_bluesky", Mock(return_value="at://post"))
    monkeypatch.setattr(posting, "should_post_now", Mock(return_value=True))
    return config, picks_dir, now, statuses


def scheduler_state():
    return scheduler.SchedulerState(
        date=DAY, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None,
        next_wakeup=None,
    )


def invoke(command, config, tmp_path):
    if command == "orchestrate":
        # Exercise the command's real gate and core with the same configuration.
        import tomllib

        config_path = tmp_path / "config.toml"
        lines = ["tiers = []"]
        for key, values in config.items():
            if key != "tiers":
                lines += [f"[{key}]"] + [f"{field} = {json.dumps(value)}"
                                         for field, value in values.items()]
        config_path.write_text("\n".join(lines))
        assert tomllib.loads(config_path.read_text()) == config
        args = ["--config", str(config_path)]
    else:
        args = ["--picks-dir", config["orchestrator"]["picks_dir"],
                "--models-dir", config["orchestrator"]["models_dir"]]
    return CliRunner().invoke(cli.cli, [command, "--date", DAY, *args], catch_exceptions=False)


def captured_fetch(directory, monkeypatch, streak, *, reactivate=False):
    import datetime as calendar
    from types import SimpleNamespace

    from bts import contest_fetch
    from bts.leaderboard import auth

    runner = CliRunner()
    with monkeypatch.context() as fetch:
        fetch.setattr(calendar, "datetime", contest_state.datetime)
        fetch.setattr(auth, "load_session_cookies", lambda: {})
        fetch.setattr(auth, "extract_uid", lambda _: "u")
        fetch.setattr(auth, "fetch_login_session", lambda **_: SimpleNamespace(
            user_id=1, username="offline", xsid="x"))
        fetch.setattr(contest_fetch, "fetch_profile", lambda *_: {
            "activeStreak": streak, "seasonBestStreak": max(9, streak),
            "predictions": [{"roundId": 1, "result": "hit"}],
        })
        fetch.setattr(cli, "_fetch_rounds", lambda: {1: calendar.date(2027, 7, 14)})
        if reactivate:
            for command, args, refusal in (
                ("revoke", ["--reason", "offline restore"], "revoked"),
                ("enable", [], "activation_id"),
            ):
                result = runner.invoke(cli.cli, [
                    "season-contract", command, "--season", "2027", "--picks-dir", str(directory), *args,
                ], catch_exceptions=False)
                assert result.exit_code == 0, result.output
                with pytest.raises(contest_state.ContestStateError, match=refusal):
                    contest_state.load_decision_streak_state(directory, require_contest_state=True)
        result = runner.invoke(cli.cli, [
            "fetch-contest-streak", "--picks-dir", str(directory),
        ], catch_exceptions=False)
        assert result.exit_code == 0, result.output
    assert contest_state.load_decision_streak_state(directory, require_contest_state=True).streak == streak


def test_uncovered_preview_double_is_refused_then_replaced(world, tmp_path, capsys):
    config, directory, _, _ = world
    set_automatic(directory, streak=0, covered=False)
    preview = invoke("preview", config, tmp_path)
    assert preview.exit_code == 0
    cached = picks.load_pick(DAY, directory)
    assert cached.double_down is not None
    set_dated(directory, streak=30, best=30)
    frame, tier = orchestrator.run_cascade.return_value
    replacement = frame.assign(p_game_hit=[0.99, 0.81])
    orchestrator.run_cascade.side_effect = [RuntimeError("cascade failed"),
                                           (replacement, tier)]
    refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, cached, 0.03)
    assert refresh.daily is cached and refresh.selection is None
    path = directory / f"{DAY}.json"
    raw = path.read_bytes()
    state = scheduler_state()
    assert scheduler._deliver_and_lock_pick(
        refresh.daily, config, directory, state, DAY, "fallback", selection=refresh.selection,
    ) is False
    assert path.read_bytes() == raw and not state.pick_locked
    assert load_decision(DAY, directory) is None
    assert not dm.send_dm.called and not posting.post_to_bluesky.called
    assert alert.send_dm.call_count == 1
    assert "chosen without the live check" in alert.send_dm.call_args.args[1]
    assert "CONTEST STATE ERROR — pick delivery blocked" in capsys.readouterr().err
    refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, cached, 0.03)
    assert refresh.daily is refresh.selection.pick_result.daily
    assert refresh.selection.decision is not None and refresh.daily.double_down is None
    assert scheduler._deliver_and_lock_pick(
        refresh.daily, config, directory, state, DAY, "fallback", selection=refresh.selection,
    ) is True
    daily = picks.load_pick(DAY, directory)
    assert daily.chosen_on["dated"]["active_streak"] == 30
    assert daily.delivery_attempted and daily.notification_sent and state.pick_locked
    assert load_decision(DAY, directory)["delivery_status"] == "delivered"
    assert dm.send_dm.call_count == 1 and "Streak: 30" in dm.send_dm.call_args.args[1]
    assert not posting.post_to_bluesky.called and alert.send_dm.call_count == 1


@pytest.mark.parametrize("mode", ["dm", "public"])
@pytest.mark.parametrize("selection_kind", ["fresh", "matching", "reused", "reloaded", "skip", "none"])
def test_scheduler_uses_same_cycle_object_and_policy_decision(
    world, monkeypatch, capsys, mode, selection_kind,
):
    config, directory, _, _ = world
    config["scheduler"]["pick_delivery"] = mode
    set_dated(directory)
    _, selection, _ = orchestrator.run_and_pick(config, DAY)
    daily = selection.pick_result.daily
    path = picks.save_pick(daily, directory)
    # Whole-range fix set A contract A2 compares fresh picks against the final observation.
    if selection_kind != "matching":
        set_dated(directory, best=10)
    if selection_kind == "reloaded":
        daily = picks.load_pick(DAY, directory)
    elif selection_kind == "reused":
        selection.decision = None
    elif selection_kind == "skip":
        selection.pick_result = None
    elif selection_kind == "none":
        selection = None
    before = path.read_bytes()
    state = scheduler_state()
    clock = Mock(return_value=NOW)
    monkeypatch.setattr(scheduler, "_now_et", clock)
    original_formatter = scheduler._format_pick_delivery_text
    formatter = Mock(wraps=original_formatter)
    monkeypatch.setattr(scheduler, "_format_pick_delivery_text", formatter)
    result = scheduler._deliver_and_lock_pick(
        daily, config, directory, state, DAY, "lineup", selection=selection)
    fresh = selection_kind == "matching"
    assert result is fresh
    assert state.pick_locked is fresh and formatter.called is fresh
    persisted = picks.load_pick(DAY, directory)
    assert persisted.delivery_attempted is fresh
    assert persisted.notification_sent is (fresh and mode == "dm")
    assert persisted.bluesky_posted is (fresh and mode == "public")
    assert dm.send_dm.called is (fresh and mode == "dm")
    assert posting.post_to_bluesky.called is (fresh and mode == "public")
    assert alert.send_dm.called is not fresh
    if fresh:
        assert load_decision(DAY, directory)["delivery_status"] == "delivered"
        assert clock.call_count >= 2
    else:
        assert path.read_bytes() == before and load_decision(DAY, directory) is None
        assert clock.call_count == 1
        assert "pick delivery blocked" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["delivered", "attempted", "private", "optional",
                                  "late", "void", "loader_error"])
def test_early_guards_and_optional_delivery_skip_matcher(world, monkeypatch, mode):
    config, directory, now, _ = world
    if mode != "loader_error":
        set_dated(directory)
    daily = daily_pick()
    if mode == "delivered":
        daily.notification_sent = True
        daily.notification_channel = "bluesky_dm"
        daily.notification_id = "prior"
        daily.delivered_at = NOW.isoformat()
    elif mode == "attempted":
        daily.delivery_attempted = True
    elif mode == "optional":
        config["scheduler"]["contest_state_required"] = False
    elif mode == "private":
        config["scheduler"]["pick_delivery"] = "private"
    elif mode == "late":
        now[0] += timedelta(hours=1)
    elif mode == "void":
        monkeypatch.setattr(scheduler, "_void_committed_games", lambda *a: {101: "Cancelled"})
    picks.save_pick(daily, directory)
    judge = Mock(side_effect=AssertionError("matcher must not be called"))
    monkeypatch.setattr(picks, "saved_pick_refusal", judge, raising=False)
    state = scheduler_state()
    assert scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, mode) is (
        mode in {"delivered", "private", "optional"})
    assert not judge.called
    assert dm.send_dm.called is (mode == "optional")
    committed = {"delivered": "delivered", "attempted": "locked_unconfirmed",
                 "private": "private_locked", "optional": "delivered"}
    decision = load_decision(DAY, directory)
    assert (decision["delivery_status"] if decision else None) == committed.get(mode)
    assert state.pick_locked is (mode in committed)


@pytest.mark.parametrize("command", ["run", "orchestrate"])
@pytest.mark.parametrize("path", ["locked", "warmup", "strict-failure"])
@pytest.mark.parametrize("outcome", ["match", "mismatch", "missing", "closed", "read-error"])
def test_legacy_public_reuse(world, tmp_path, monkeypatch, command, path, outcome):
    if path != "warmup" and outcome == "closed":
        pytest.skip("catch-up always bypasses the posting window")
    config, directory, _, statuses = world
    set_dated(directory)
    daily = daily_pick()
    daily.chosen_on = (None if outcome == "missing" else record(required=outcome == "match"))
    # Inject the on-disk record even on the base, whose writer omits the field.
    pick_path = picks.save_pick(daily, directory)
    body = asdict(daily)
    body.pop("chosen_on", None)
    if daily.chosen_on is not None:
        body["chosen_on"] = daily.chosen_on
    pick_path.write_text(json.dumps(body))
    if path == "strict-failure":
        failing_status = Mock(side_effect=RuntimeError("offline detailed failure"))
        monkeypatch.setattr(picks, "get_game_statuses_detailed", failing_status)
        monkeypatch.setattr(strategy, "get_game_statuses_detailed", failing_status)
    else:
        status = ({"abstract": "P", "detailed": "Warmup", "code": "PW"} if path == "warmup"
                  else {"abstract": "L", "detailed": "In Progress", "code": "I"})
        statuses.update(dict.fromkeys(statuses, status))
    posting.should_post_now.return_value = outcome != "closed"
    judge = getattr(picks, "saved_pick_refusal", None)
    judge_spy = Mock(wraps=judge)
    if judge is not None:
        monkeypatch.setattr(picks, "saved_pick_refusal", judge_spy)
    if outcome == "read-error":
        reader = contest_state.load_decision_streak_state
        monkeypatch.setattr(contest_state, "load_decision_streak_state",
                            Mock(side_effect=[reader(directory, now=NOW, require_contest_state=True),
                                              contest_state.ContestStateError("second read failed")]))
    if command == "orchestrate" and outcome == "read-error":
        with pytest.raises(contest_state.ContestStateError, match="second read failed"):
            invoke(command, config, tmp_path)
    else:
        result = invoke(command, config, tmp_path)
        assert result.exit_code == (outcome in {"mismatch", "missing", "read-error"})
        if outcome in {"mismatch", "missing"}:
            assert f"bts {command}: no pick: " in result.stderr
        if outcome == "read-error":
            assert "second read failed" in result.stderr
    posted = outcome == "match"
    assert posting.post_to_bluesky.called is posted
    persisted = picks.load_pick(DAY, directory)
    assert persisted.bluesky_posted is posted
    assert persisted.bluesky_uri == ("at://post" if posted else None)
    assert persisted.chosen_on == daily.chosen_on
    assert posting.should_post_now.called is (path == "warmup" and outcome != "read-error")
    if judge is not None:
        assert judge_spy.called is (outcome not in {"closed", "read-error"})
    saves = 1 + int(path == "warmup") + int(posted)
    assert len((directory / f"lineup_evolution_{DAY}.jsonl").read_text().splitlines()) == saves


@pytest.mark.parametrize("command", ["run", "preview", "orchestrate", "core"])
@pytest.mark.parametrize("kind", ["automatic", "dated"])
def test_fresh_choices_stamp_the_selection_read(world, tmp_path, monkeypatch, command, kind):
    config, directory, _, _ = world
    setter = set_dated if kind == "dated" else set_automatic
    setter(directory, streak=7)
    if command == "run":
        # A real, different second snapshot cannot change the first-read stamp.
        reader = contest_state.load_decision_streak_state
        invoked = []

        def read(*args, **kwargs):
            invoked.append(True)
            if len(invoked) == 2:
                setter(directory, streak=8)
            return reader(*args, **kwargs)

        monkeypatch.setattr(contest_state, "load_decision_streak_state", read)
    if command == "core":
        _, selection, _ = orchestrator.run_and_pick(config, DAY)
        assert selection.decision is not None
        derived = asdict(selection.pick_result.daily).get("chosen_on")
    else:
        result = invoke(command, config, tmp_path)
        # Whole-range fix set A contract A2 refuses a changed dated observation.
        refused = command == "run" and kind == "dated"
        assert result.exit_code == int(refused)
        derived = asdict(picks.load_pick(DAY, directory)).get("chosen_on")
    assert derived == record(kind, required=command != "preview")
    if command == "run":
        assert len(invoked) == 2
        if refused:
            posting.post_to_bluesky.assert_not_called()
            assert "bts run: no pick:" in result.stderr
        else:
            assert "Streak: 8" in posting.post_to_bluesky.call_args.args[0]


@pytest.mark.parametrize("required_via", ["scheduler", "health_checks", None])
@pytest.mark.parametrize("minutes", [9, 0, -3])
def test_missed_pick_alert_copy(world, required_via, minutes):
    config, _, _, _ = world
    config["scheduler"]["contest_state_required"] = required_via == "scheduler"
    config["health_checks"]["contest_state_expected"] = required_via == "health_checks"
    daily = daily_pick()
    scheduler._alert_missed_pick(config, daily, minutes)
    message = alert.send_dm.call_args.args[1]
    assert "NO PICK DELIVERED" in message
    if required_via is None:
        assert daily.pick.batter_name in message and "Deliver manually" in message
        assert f"~{minutes} min to first pitch" in message
    else:
        assert daily.pick.batter_name not in message and "Deliver manually" not in message
        assert "bts season-contract status" in message
        assert ("~9 min to first pitch" if minutes > 0 else "~0 min to first pitch" if minutes == 0 else "first pitch has passed") in message


@pytest.mark.parametrize("change", ["best", "streak", "saver", "rollover", "covered",
                                   "flag", "reactivated", "override"])
def test_failed_refresh_reuses_only_matching_saved_pick(world, monkeypatch, change):
    from tests.test_pick_chosen_on_cs5a import account_file

    config, directory, now, _ = world
    automatic = change in {"covered", "flag", "reactivated", "override"}
    (set_automatic if automatic else set_dated)(directory)
    _, selection, _ = orchestrator.run_and_pick(config, DAY)
    daily = selection.pick_result.daily
    if change in {"rollover", "override"}:
        daily.pick.game_time = "2027-07-16T06:30:00Z"
        if daily.double_down:
            daily.double_down.game_time = "2027-07-16T06:45:00Z"
    path = picks.save_pick(daily, directory)
    raw = path.read_bytes()
    if change == "best":
        set_dated(directory, best=10)
    elif change == "streak":
        set_dated(directory, streak=8)
    elif change == "saver":
        set_dated(directory, saver="active")
    elif change == "rollover":
        now[0] = datetime.fromisoformat("2027-07-16T00:08:00-04:00")
        set_dated(directory, now=now[0])
    elif change == "covered":
        captured_fetch(directory, monkeypatch, 56)
    elif change == "flag":
        account_file(directory, "saver_state.json", {"season": 2027, "state": "active"})
    elif change == "reactivated":
        captured_fetch(directory, monkeypatch, 9, reactivate=True)
    elif change == "override":
        set_dated(directory)

    frame, tier = orchestrator.run_cascade.return_value
    orchestrator.run_cascade.return_value = (None, None)
    cached = picks.load_pick(DAY, directory)
    refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, cached, 0.03)
    assert refresh.daily is cached and refresh.selection is None
    state = scheduler_state()
    matches = change in {"covered", "flag", "reactivated"}
    assert scheduler._deliver_and_lock_pick(
        refresh.daily, config, directory, state, DAY, "fallback", selection=refresh.selection,
    ) is matches
    assert dm.send_dm.called is matches and state.pick_locked is matches
    assert alert.send_dm.called is not matches
    if matches:
        assert load_decision(DAY, directory)["delivery_status"] == "delivered"
        assert picks.load_pick(DAY, directory).notification_sent
    else:
        assert load_decision(DAY, directory) is None and path.read_bytes() == raw
    if change == "override":
        now[0] = datetime.fromisoformat("2027-07-16T00:08:00-04:00")
        assert scheduler._deliver_and_lock_pick(
            refresh.daily, config, directory, state, DAY, "fallback") is True
        assert dm.send_dm.call_count == 1 and alert.send_dm.call_count == 1
    elif not matches:
        if change == "rollover":
            frame = frame.assign(game_time=["2027-07-16T06:30:00Z", "2027-07-16T06:45:00Z"])
        orchestrator.run_cascade.return_value = frame, tier
        retry = scheduler._refresh_pick_at_fallback_decision(config, DAY, cached, 0.03)
        assert retry.selection.decision is not None and retry.daily is not cached
        assert scheduler._deliver_and_lock_pick(
            retry.daily, config, directory, state, DAY, "fallback", selection=retry.selection)
        assert dm.send_dm.call_count == 1 and alert.send_dm.call_count == 1
        assert load_decision(DAY, directory)["delivery_status"] == "delivered"
        assert picks.load_pick(DAY, directory).notification_sent and state.pick_locked


def test_refusal_survives_non_object_health_status(world, capsys):
    config, directory, _, _ = world
    set_dated(directory)
    status = directory.parent / "health_state" / "health_dm_delivery_status.json"
    status.parent.mkdir()
    status.write_text("[]")
    daily = daily_pick()
    path = picks.save_pick(daily, directory)
    raw = path.read_bytes()
    state = scheduler_state()
    assert scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, "fallback") is False
    assert "Contest-state alert failed" not in capsys.readouterr().err
    assert json.loads(status.read_text())["status"] == "sent"
    assert path.read_bytes() == raw
    assert not state.pick_locked and load_decision(DAY, directory) is None
    assert not dm.send_dm.called
    alert.send_dm.assert_called_once()


@pytest.mark.parametrize("recorded", [True, False])
def test_preview_resaves_live_reuse_without_changing_record(world, tmp_path, recorded):
    config, directory, _, statuses = world
    set_dated(directory)
    daily = daily_pick()
    pick_path = picks.save_pick(daily, directory)
    expected = record() if recorded else None
    body = asdict(daily)
    body.pop("chosen_on", None)
    if recorded:
        body["chosen_on"] = expected
    pick_path.write_text(json.dumps(body))
    statuses.update(dict.fromkeys(statuses, {"abstract": "P", "detailed": "Warmup", "code": "PW"}))
    assert invoke("preview", config, tmp_path).exit_code == 0
    assert len((directory / f"lineup_evolution_{DAY}.jsonl").read_text().splitlines()) == 2
    assert getattr(picks.load_pick(DAY, directory), "chosen_on", None) == expected
    assert not dm.send_dm.called and not posting.post_to_bluesky.called


def prepare_run_day(monkeypatch):
    games = [{
        "gamePk": pk, "gameDate": f"{DAY}T{game_time}:00Z",
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
        "teams": {"away": {"team": {"name": f"Away {pk}"}}, "home": {"team": {"name": f"Home {pk}"}}},
    } for pk, game_time in ((101, "23:10"), (102, "23:40"))]
    scheduler.fetch_schedule.side_effect = lambda date: games if date == DAY else []
    monkeypatch.setattr(scheduler, "check_confirmed_lineups", lambda *a: {})
    monkeypatch.setattr(scheduler, "_idle_until_next_wakeup", lambda *a: None)
    monkeypatch.setattr(scheduler, "_watchdog_ping_sleep", lambda *a: None)
    monkeypatch.setattr(scheduler, "run_result_polling", Mock(return_value="final"))


def test_run_day_refusals_dedup_across_restart_and_missed_pick_has_no_candidate(
    world, monkeypatch, tmp_path, capsys,
):
    config, directory, now, _ = world
    set_automatic(directory, streak=0, covered=False)
    assert invoke("preview", config, tmp_path).exit_code == 0
    pick_file = directory / f"{DAY}.json"
    raw = pick_file.read_bytes()
    set_dated(directory, streak=30, best=30)
    config["scheduler"]["entry_intent"] = "enter"
    frame, tier = orchestrator.run_cascade.return_value
    healthy_prediction = (frame.assign(p_game_hit=[0.99, 0.81]), tier)
    orchestrator.run_cascade.return_value = (None, None)
    prepare_run_day(monkeypatch)
    # At the fallback deadline, already past the first lineup check. Stub sleeps
    # without advancing the operation clock so a second invocation models restart.
    for _ in range(2):
        scheduler.run_day(date=DAY, config=config)
        assert pick_file.read_bytes() == raw
        assert load_decision(DAY, directory) is None
        assert not scheduler.load_state(DAY, directory).pick_locked
    assert not scheduler.run_result_polling.called and not dm.send_dm.called
    assert "pick delivery blocked" in capsys.readouterr().err
    messages = [call.args[1] for call in alert.send_dm.call_args_list]
    assert len(messages) == 2
    assert sum("[contest_state]" in msg for msg in messages) == 1
    missed, = [msg for msg in messages if "[missed_pick]" in msg]
    assert "NO PICK DELIVERED" in missed and "Steven Kwan" not in missed and "Juan Soto" not in missed
    assert "bts season-contract status" in missed
    entry_verify = Mock(side_effect=AssertionError("uncommitted pick reached entry verifier"))
    monkeypatch.setattr("bts.leaderboard.auth.load_session_cookies", entry_verify)
    entry = CliRunner().invoke(cli.cli, [
        "check-pick-entered", "--picks-dir", str(directory),
        "--now-et", f"{DAY}T18:40:00", "--expected-username", "offline",
    ], catch_exceptions=False)
    assert entry.exit_code == 0 and "not committed/locked" in entry.stdout
    assert not entry_verify.called
    cfg = tmp_path / "intent.toml"
    cfg.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
    outcome = CliRunner().invoke(cli.cli, [
        "day-outcome", "--date", DAY, "--config", str(cfg),
        "--picks-dir", str(directory), "--expected-username", "offline",
    ], catch_exceptions=False)
    assert outcome.exit_code == 1 and json.loads(outcome.stdout)["state"] == "no_decision"
    orchestrator.run_cascade.return_value = healthy_prediction
    scheduler.run_day(date=DAY, config=config)
    assert scheduler.load_state(DAY, directory).pick_locked
    assert load_decision(DAY, directory)["delivery_status"] == "delivered"
    assert picks.load_pick(DAY, directory).notification_sent
    assert dm.send_dm.call_count == 1


def test_direct_optional_orchestration_preserves_unguarded_reuse(world, tmp_path, monkeypatch):
    config, directory, _, statuses = world
    config["scheduler"]["contest_state_required"] = False
    set_automatic(directory, covered=False)
    picks.save_pick(daily_pick(), directory)
    statuses.update(dict.fromkeys(statuses, {"abstract": "P", "detailed": "Warmup", "code": "PW"}))
    judge = Mock(side_effect=AssertionError("optional orchestrator reached matcher"))
    monkeypatch.setattr(picks, "saved_pick_refusal", judge, raising=False)
    assert orchestrator.orchestrate(tmp_path / "unused.toml", DAY, config=config)
    assert posting.post_to_bluesky.called and not judge.called
    assert picks.load_pick(DAY, directory).bluesky_posted


@pytest.mark.parametrize("mode", ["dm", "public"])
def test_marker_archive_and_grading_saves_preserve_chosen_on(world, monkeypatch, mode):
    config, directory, _, _ = world
    config["scheduler"]["pick_delivery"] = mode
    set_dated(directory)
    daily = daily_pick()
    daily.chosen_on = record()
    picks.attach_provenance(daily)
    assert daily.chosen_on == record()
    path = picks.save_pick(daily, directory)
    path.write_text(json.dumps(dict(asdict(daily), chosen_on=record())))
    state = scheduler_state()
    daily = picks.load_pick(DAY, directory)
    sent = dm.send_dm if mode == "dm" else posting.post_to_bluesky

    def capture_attempt(*args):
        attempted = picks.load_pick(DAY, directory)
        assert attempted.delivery_attempted and not picks.pick_was_delivered(attempted)
        assert attempted.chosen_on == record()
        return "captured"

    sent.side_effect = capture_attempt
    assert scheduler._deliver_and_lock_pick(daily, config, directory, state, DAY, "fallback")
    assert picks.load_pick(DAY, directory).chosen_on == record()
    assert state.pick_locked
    monkeypatch.setattr(picks, "resolve_daily_slot_results", lambda *_: {"primary": "hit"})
    result = CliRunner().invoke(cli.cli, [
        "check-results", "--date", DAY, "--picks-dir", str(directory),
    ], catch_exceptions=False)
    assert result.exit_code == 0
    graded = picks.load_pick(DAY, directory)
    assert graded.result == "hit" and graded.chosen_on == record()
    archive = scheduler._defer_pick_at_fallback(directory, DAY, graded, reason="fixture")
    archived = json.loads(archive.read_text())
    assert list(archived)[-3:] == ["chosen_on", "automatic_reading", "deferred_fallback"]
    assert archived["chosen_on"] == record()


@pytest.mark.parametrize("recorded", [True, False])
@pytest.mark.parametrize("reuse", ["locked", "strict", "coarse", "unavailable"])
def test_real_selection_reuse_preserves_record_or_absence(world, monkeypatch, reuse, recorded):
    config, directory, _, statuses = world
    set_dated(directory)
    daily = daily_pick()
    initial = asdict(daily)
    initial.pop("chosen_on", None)
    if recorded:
        initial["chosen_on"] = record()
    path = directory / f"{DAY}.json"
    path.write_text(json.dumps(initial))
    if reuse == "locked":
        statuses.update(dict.fromkeys(statuses, {"abstract": "L", "detailed": "In Progress", "code": "I"}))
    elif reuse == "unavailable":
        statuses.update(dict.fromkeys(statuses, {"abstract": "P", "detailed": "Warmup", "code": "PW"}))
    elif reuse == "strict":
        monkeypatch.setattr(picks, "get_game_statuses_detailed", Mock(side_effect=OSError("offline")))
        monkeypatch.setattr(strategy, "get_game_statuses_detailed", Mock(side_effect=OSError("offline")))
    else:
        # Only the selection-input lookup fails; the classifier's next detailed
        # read succeeds before select_pick's separate coarse lookup fails.
        monkeypatch.setattr(picks, "get_game_statuses_detailed",
                            Mock(side_effect=[OSError("offline"), statuses, statuses]))
        monkeypatch.setattr(strategy, "get_game_statuses", Mock(side_effect=OSError("offline")))
    if reuse != "strict":
        cached = picks.load_pick(DAY, directory)
        refreshed = scheduler._refresh_pick_at_fallback_decision(config, DAY, cached, 0.03)
        selection = refreshed.selection
        assert refreshed.daily is selection.pick_result.daily and refreshed.daily is not cached
        if reuse == "coarse":
            strategy.get_game_statuses.assert_called_once_with(DAY)
    else:
        _, selection, _ = orchestrator.run_and_pick(config, DAY)
    assert selection.decision is None and selection.pick_result.locked is (reuse != "unavailable")
    assert getattr(selection.pick_result.daily, "chosen_on", None) == (record() if recorded else None)
    if reuse != "strict":
        state = scheduler_state()
        assert scheduler._deliver_and_lock_pick(
            refreshed.daily, config, directory, state, DAY, "fallback", selection=selection,
        ) is recorded
        assert dm.send_dm.called is recorded and state.pick_locked is recorded
        assert alert.send_dm.called is not recorded


@pytest.mark.parametrize("action", ["single", "double", "heuristic", "skip"])
@pytest.mark.parametrize("kind", ["dated", "automatic"])
def test_real_policy_outputs_stamp_only_choices(world, monkeypatch, action, kind):
    config, directory, _, _ = world
    setter = set_dated if kind == "dated" else set_automatic
    streak = 30 if action in {"single", "skip"} else 0
    setter(directory, streak=streak, **({"best": streak} if kind == "dated" else {}))
    monkeypatch.setattr(strategy, "_mdp_cache", {} if action == "heuristic" else None)
    if action in {"single", "skip"}:
        predictions, tier = orchestrator.run_cascade.return_value
        probabilities = [0.99, 0.81] if action == "single" else [0.10, 0.09]
        orchestrator.run_cascade.return_value = (predictions.assign(p_game_hit=probabilities), tier)
    _, selection, _ = orchestrator.run_and_pick(config, DAY)
    assert selection.decision is not None
    if action == "skip":
        assert selection.action == "skip" and selection.pick_result is None
        assert not (directory / f"{DAY}.json").exists()
        return
    if action == "heuristic":
        assert selection.source == "heuristic"
    else:
        assert selection.source == "mdp" and selection.action == action
        assert (selection.pick_result.daily.double_down is not None) is (action == "double")
    stamp = asdict(selection.pick_result.daily)["chosen_on"]
    assert stamp["kind"] == kind and stamp["required"] is True and stamp["streak"] == streak


def test_overrun_reloads_the_same_cycle_choice_and_checks_it_as_saved(world, monkeypatch, capsys):
    config, directory, _, _ = world
    set_dated(directory)
    prepare_run_day(monkeypatch)
    frame, tier = orchestrator.run_cascade.return_value
    frame.loc[1, "flags"] = "PROJECTED"
    orchestrator.run_cascade.side_effect = [(frame, tier), (None, None)]
    check = scheduler.run_single_check

    def correct_after_check(*args, **kwargs):
        result = check(*args, **kwargs)
        if result["pick_result"] is not None:
            assert not result["should_post"] and result["selection"].decision is not None
            assert result["lock_decision"] is not None
            assert result["saved_pick_bytes"] == (directory / f"{DAY}.json").read_bytes()
            set_dated(directory, best=10)
        return result

    monkeypatch.setattr(scheduler, "run_single_check", correct_after_check)
    scheduler.run_day(date=DAY, config=config)
    err = capsys.readouterr().err
    assert "reusing this check's fresh cascade decision" in err
    assert "pick delivery blocked" in err and not dm.send_dm.called
    assert load_decision(DAY, directory) is None
    assert not scheduler.load_state(DAY, directory).pick_locked


def test_automatic_gap_then_real_fetch_and_restart_uses_cached_pick(world, monkeypatch, capsys):
    config, directory, _, _ = world
    set_automatic(directory, streak=5)
    frame, _ = orchestrator.run_cascade.return_value
    frame.loc[1, "flags"] = "PROJECTED"
    monkeypatch.setattr(scheduler, "check_confirmed_lineups", lambda *a: {})
    check = scheduler.run_single_check(DAY, [101, 102], set(), config, 0.03)
    assert not check["should_post"] and "gap=1.0%" in capsys.readouterr().err
    assert picks.load_pick(DAY, directory).pick.batter_id == 680757
    assert not dm.send_dm.called
    captured_fetch(directory, monkeypatch, 6)
    orchestrator.run_cascade.return_value = (None, None)
    prepare_run_day(monkeypatch)
    scheduler.run_day(date=DAY, config=config)
    assert load_decision(DAY, directory)["delivery_status"] == "delivered"
    assert dm.send_dm.call_count == 1 and "Streak: 6" in dm.send_dm.call_args.args[1]
    assert not alert.send_dm.called and not posting.post_to_bluesky.called


def test_run_day_genuine_mdp_skip_still_writes_a_skip(world, monkeypatch):
    config, directory, _, _ = world
    set_dated(directory, streak=56, best=56)
    prepare_run_day(monkeypatch)
    frame, tier = orchestrator.run_cascade.return_value
    orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=0.10), tier)
    scheduler.run_day(date=DAY, config=config)
    decision = load_decision(DAY, directory)
    assert decision["action"] == "skip" and decision["source"] == "mdp"
    assert not decision["scoreable"]
    assert not (directory / f"{DAY}.json").exists() and not alert.send_dm.called
