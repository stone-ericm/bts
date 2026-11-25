"""Offline scheduler startup, live checks and health dispatch."""

import json
import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from bts import cli, contest_state, daily_decision, dm, orchestrator, picks, scheduler, strategy
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, set_automatic, set_dated
from tests.test_saved_pick_paths_cs5a import captured_fetch, prepare_run_day, scheduler_state, world
from tests.test_stale_delivery_cs4c import CRITICAL_FILE, OLD, WARNING_FILE, WARNING_KEY, envelope, seed_automatic


START = NOW.replace(hour=10, minute=0)
FIRST = NOW.replace(hour=16, minute=10)
MORNING_FILE = "contest_morning_check_dm_delivery_status.json"
MORNING_KEY = "CRITICAL:contest_state:morning"
FAILURE = "  Morning state check failed: "
SKIP = "  Morning state check skipped: "
BANNER = "⚠️ Previous health alert DM FAILED to send (1 CRITICAL) — it may have been missed."


def refusal(day=DAY, first="16:10"):
    return (
        "BTS health CRITICAL alert(s):\n- [contest_state] "
        f"Morning check for {day}: no usable contest state was found at startup. "
        "Check MLB and run bts season-contract status with the scheduler's picks directory. "
        "If state still needs correction, follow today's dated-observation instructions while "
        f"a lineup check remains. First planned lineup check at {first} ET."
    )


def warning(detail):
    return (
        "BTS health WARN attention:\n- [contest_state] "
        f"Morning check for {DAY}: the automatic contest reading was stale at startup ({detail}). This "
        "reading is eligible if it remains in use. Check MLB. If MLB differs from status (B8a), use "
        f"bts set-contest-streak --date {DAY} --streak N --best-streak B --saver S --picks-dir DIR on "
        f"{DAY} ET only. DIR is the absolute scheduler picks path. This leaves the saver flag "
        f"unchanged. On {DAY} ET away from midnight, if MLB shows this season's saver used, check bts "
        "saver-state --picks-dir DIR. For active or not_earned, run bts saver-state --use --picks-dir "
        "DIR. For uninitialized, see B8a. Check the latest fetch status."
    )


@pytest.fixture
def morning(world, monkeypatch):
    config, directory, now, _ = world
    now[0] = START
    prepare_run_day(monkeypatch)

    def advance(seconds):
        now[0] += timedelta(seconds=seconds)

    monkeypatch.setattr(scheduler, "_watchdog_ping_sleep", advance)
    monkeypatch.setattr(daily_decision, "datetime", contest_state.datetime)
    monkeypatch.setattr(picks, "datetime", contest_state.datetime)
    monkeypatch.setattr(strategy, "datetime", contest_state.datetime)
    monkeypatch.setattr(cli, "datetime", contest_state.datetime)
    monkeypatch.setattr(scheduler, "datetime", contest_state.datetime)
    monkeypatch.setattr(scheduler, "time", Mock(monotonic=lambda: 0))
    # Leave the policy cache as it was found (reviews, rounds 1 and 2): a test that
    # loads the shipped policy must not decide what a later test's first use sees.
    monkeypatch.setattr(strategy, "_mdp_cache", strategy._mdp_cache)
    frame, tier = orchestrator.run_cascade.return_value
    predictions = frame.assign(p_game_hit=[0.99, 0.81]), tier
    orchestrator.run_cascade.return_value = (None, None)
    return config, directory, now, predictions


def read_status(directory, name):
    return json.loads((directory.parent / "health_state" / name).read_bytes())


def test_refusal_precedes_real_live_refusal(morning, monkeypatch, capsys):
    config, directory, now, predictions = morning
    events = []
    load = contest_state.load_decision_streak_state

    def read(*args, **kwargs):
        events.append(("read", now[0], kwargs))
        return load(*args, **kwargs)

    def send(_, body):
        events.append(("dm", now[0], body))

    def predict(*args, **kwargs):
        events.append(("prediction", now[0]))
        return predictions

    monkeypatch.setattr(contest_state, "load_decision_streak_state", read)
    alert.send_dm.side_effect = send
    orchestrator.run_cascade.side_effect = predict
    with pytest.raises(contest_state.ContestStateError) as error:
        load(directory, require_contest_state=True, now=START)
    scheduler.run_day(DAY, config)
    assert events[:3] == [
        ("read", START, {"require_contest_state": True, "now": START}),
        ("dm", START, refusal()),
        ("prediction", FIRST),
    ]
    assert sum(event[0] == "read" and event[1] == START for event in events) == 1
    assert alert.send_dm.call_count == 2
    assert read_status(directory, MORNING_FILE)["sent_sources"] == [MORNING_KEY]
    assert read_status(directory, CRITICAL_FILE)["sent_sources"] == ["CRITICAL:contest_state"]
    assert f"  MORNING STATE CHECK — no pick can be made yet: {ascii(str(error.value))[:800]}" in (
        capsys.readouterr().err.splitlines()
    )
    dm.send_dm.assert_not_called()


@pytest.mark.parametrize("kind", ["dated", "automatic"])
def test_accepted_state_logs_without_health_writes(morning, capsys, kind):
    config, directory, _, predictions = morning
    (set_dated if kind == "dated" else set_automatic)(directory, now=START, streak=30,
                                                    **({"best": 30} if kind == "dated" else {}))
    orchestrator.run_cascade.return_value = predictions
    scheduler.run_day(DAY, config)
    assert f"  Morning state check: {kind} contest state accepted (streak 30)." in (
        capsys.readouterr().err.splitlines()
    )
    assert not (directory.parent / "health_state").exists()
    alert.send_dm.assert_not_called()
    dm.send_dm.assert_called_once()


@pytest.mark.parametrize("scenario", ["sent", "failed", "ages_into_stale"])
def test_stale_warning_uses_each_snapshot(morning, capsys, caplog, scenario):
    config, directory, now, predictions = morning
    age = 12 if scenario == "ages_into_stale" else 20
    recorded = (START - timedelta(hours=age)).isoformat()
    seed_automatic(directory, recorded, now=START, streak=30)
    events = []

    def health_send(_, body):
        events.append((now[0], body))
        if scenario == "failed" and now[0] == START:
            raise RuntimeError("offline morning WARN")

    def predict(*args, **kwargs):
        events.append((now[0], "prediction"))
        if scenario == "failed" and now[0] == FIRST:
            assert read_status(directory, WARNING_FILE)["status"] == "failed"
        return predictions

    alert.send_dm.side_effect = health_send
    dm.send_dm.side_effect = lambda *_: events.append((now[0], "pick")) or "sent"
    orchestrator.run_cascade.side_effect = predict
    scheduler.run_day(DAY, config)
    delivered_age = age + (FIRST - START).total_seconds() / 3600
    startup_detail = f"last accepted fetch {recorded}, about {age:.2f} hours ago"
    delivery_detail = f"last accepted fetch {recorded}, about {delivered_age:.2f} hours ago"
    expected = [(FIRST, "prediction"), (FIRST, "pick")]
    if scenario != "ages_into_stale":
        expected.insert(0, (START, warning(startup_detail)))
    if scenario != "sent":
        expected.append((FIRST, envelope(detail=delivery_detail)))
    assert events[:len(expected)] == expected
    assert alert.send_dm.call_count == (2 if scenario == "failed" else 1)
    status = read_status(directory, WARNING_FILE)
    assert status["sent_sources"] == [WARNING_KEY]
    assert status["updated_at"] == (START if scenario == "sent" else FIRST).astimezone(
        scheduler.UTC
    ).isoformat()
    saved = picks.load_pick(DAY, directory)
    assert saved.notification_sent and saved.automatic_reading == dict(
        OLD, recorded_at=recorded, age_hours=round(delivered_age, 2),
    )
    err = capsys.readouterr().err
    assert f"  STALE CONTEST READING — {envelope(detail=delivery_detail).split('- [contest_state] ')[1]}" in err
    if scenario != "ages_into_stale":
        assert f"  MORNING STATE CHECK — stale contest reading: {startup_detail}" in err
    if scenario == "failed":
        assert "offline morning WARN" in caplog.text


@pytest.mark.parametrize(("lead", "expected"), [
    (-60, "due"), (0, "due"), (599, "due"), (600, "run"), (None, "empty"), ("mixed", "due"),
])
def test_margin_uses_earliest_planned_target(morning, monkeypatch, capsys, lead, expected):
    config, _, now, _ = morning
    targets = [] if lead is None else [now[0] + timedelta(seconds=lead)] if lead != "mixed" else [
        now[0] + timedelta(hours=1), now[0] - timedelta(minutes=1),
    ]
    monkeypatch.setattr(scheduler, "plan_run_times", lambda *a, **k: [
        {"time_et": target, "game_pks": [101]} for target in targets
    ])
    helper = Mock(wraps=getattr(scheduler, "_morning_state_check", None))
    monkeypatch.setattr(scheduler, "_morning_state_check", helper, raising=False)
    scheduler.run_day(DAY, config)
    err = capsys.readouterr().err
    if expected == "run":
        helper.assert_called_once_with(config, morning[1], DAY, START, targets[0])
        alert.send_dm.assert_called_once_with("offline.test", refusal(first="10:10"))
    else:
        helper.assert_not_called()
        text = ("no lineup check is planned." if expected == "empty"
                else "first lineup check is due within 10 minutes.")
        assert err.splitlines().count(SKIP + text) == 1
        alert.send_dm.assert_not_called()
    assert orchestrator.run_cascade.called is bool(targets)


@pytest.mark.parametrize("gate", ["private", "optional", "prior_lock", "dry_run", "off_day", "other_date"])
def test_ineligible_start_has_no_effects(morning, monkeypatch, capsys, gate):
    config, directory, now, _ = morning
    if gate == "private":
        config["scheduler"]["pick_delivery"] = "private"
    elif gate == "optional":
        config["scheduler"]["contest_state_required"] = False
    elif gate == "prior_lock":
        state = scheduler_state()
        state.pick_locked = True
        scheduler.save_state(state, directory)
    elif gate == "off_day":
        scheduler.fetch_schedule.side_effect = None
    elif gate == "other_date":
        now[0] -= timedelta(days=1)
    helper = Mock(wraps=getattr(scheduler, "_morning_state_check", None))
    monkeypatch.setattr(scheduler, "_morning_state_check", helper, raising=False)
    read = Mock(wraps=contest_state.load_decision_streak_state)
    monkeypatch.setattr(contest_state, "load_decision_streak_state", read)
    scheduler.run_day(DAY, config, dry_run=gate == "dry_run")
    if gate != "other_date":
        helper.assert_not_called()
    read.assert_not_called()
    alert.send_dm.assert_not_called()
    assert "Morning state check" not in capsys.readouterr().err
    assert not (directory.parent / "health_state").exists()


@pytest.mark.parametrize("mode", ["dm", "public"])
@pytest.mark.parametrize("recipient", [True, False])
@pytest.mark.parametrize("flag", ["scheduler", "health"])
def test_either_required_flag_and_outward_mode(morning, capsys, mode, recipient, flag):
    config, directory, _, _ = morning
    config["scheduler"].update(pick_delivery=mode, contest_state_required=flag == "scheduler")
    config["health_checks"]["contest_state_expected"] = flag == "health"
    if not recipient:
        config["bluesky"].clear()
    scheduler.run_day(DAY, config)
    assert "  MORNING STATE CHECK — no pick can be made yet:" in capsys.readouterr().err
    if recipient:
        alert.send_dm.assert_called_once_with("offline.test", refusal())
    else:
        alert.send_dm.assert_not_called()
        assert not (directory.parent / "health_state").exists()


@pytest.mark.parametrize("record", ["scoreable", "delivered", "attempted"])
@pytest.mark.parametrize("unlocked", [False, True])
def test_existing_commit_markers_suppress_startup_read(morning, monkeypatch, capsys, record, unlocked):
    config, directory, _, _ = morning
    if unlocked:
        scheduler.save_state(scheduler_state(), directory)
    if record == "scoreable":
        path = daily_decision.decision_path(DAY, directory)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"schema_version": daily_decision.ACCEPTED_SCHEMAS[0],
                                    "date": DAY, "action": "single", "scoreable": True}))
        assert not (directory / f"{DAY}.json").exists()
    else:
        daily = daily_pick()
        daily.delivery_attempted = record == "attempted"
        daily.notification_sent = record == "delivered"
        daily.notification_id = "confirmed" if daily.notification_sent else None
        picks.save_pick(daily, directory)
    read = Mock(wraps=contest_state.load_decision_streak_state)
    monkeypatch.setattr(contest_state, "load_decision_streak_state", read)
    actual = scheduler.run_single_check

    def check(*args, **kwargs):
        read.assert_not_called()
        return actual(*args, **kwargs)

    monkeypatch.setattr(scheduler, "run_single_check", check)
    scheduler.run_day(DAY, config)
    assert "Morning state check" not in capsys.readouterr().err
    assert not (directory.parent / "health_state" / MORNING_FILE).exists()


def test_schedule_default_date_captured_before_midnight_load(morning, monkeypatch, tmp_path):
    config, directory, now, _ = morning
    config["scheduler"]["entry_intent"] = "enter"
    now[0] = START.replace(hour=23, minute=59, second=59)
    config_path = tmp_path / "scheduler.toml"
    config_path.touch()

    def load(_):
        now[0] += timedelta(seconds=2)
        return config

    monkeypatch.setattr(orchestrator, "load_config", load)
    monkeypatch.setattr(scheduler, "plan_run_times", lambda *a, **k: [
        {"time_et": now[0] + timedelta(minutes=20), "game_pks": [101]},
    ])
    result = CliRunner().invoke(cli.cli, ["schedule", "--config", str(config_path)], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert scheduler.fetch_schedule.call_args_list[0].args == (DAY,)
    assert scheduler.load_state(DAY, directory).runs_completed
    assert "Morning state check" not in result.stderr
    alert.send_dm.assert_not_called()


@pytest.mark.parametrize("fail_first", [False, True])
def test_restarts_dedup_retry_and_roll_date(morning, capsys, fail_first):
    config, directory, now, _ = morning
    if fail_first:
        alert.send_dm.side_effect = [RuntimeError("offline CRITICAL"), "sent", "sent"]
    starts = 3 if fail_first else 2
    for _ in range(starts):
        now[0] = START
        scheduler.run_day(DAY, config)
    assert capsys.readouterr().err.count("  MORNING STATE CHECK — no pick can be made yet:") == starts
    tomorrow = (START + timedelta(days=1)).date().isoformat()
    fetch = scheduler.fetch_schedule.side_effect
    games = [{**game, "gameDate": game["gameDate"].replace(DAY, tomorrow)} for game in fetch(DAY)]
    scheduler.fetch_schedule.side_effect = lambda date: games if date == tomorrow else []
    now[0] = START + timedelta(days=1)
    scheduler.run_day(tomorrow, config)
    expected = [refusal()]
    if fail_first:
        expected.append(BANNER + "\n\n" + refusal())
    assert [call.args[1] for call in alert.send_dm.call_args_list] == expected + [refusal(tomorrow)]
    assert read_status(directory, MORNING_FILE)["sent_day"] == tomorrow


@pytest.mark.parametrize("repair", ["dated", "fetch"])
def test_first_check_reads_repair_after_startup(morning, monkeypatch, repair):
    config, directory, now, predictions = morning
    if repair == "fetch":
        set_automatic(directory, now=START)
        (directory / "account_state" / "contest_streak.json").unlink()
    sleep = scheduler._watchdog_ping_sleep

    def advance(seconds):
        if now[0] == START:
            alert.send_dm.assert_called_once_with("offline.test", refusal())
            if repair == "dated":
                set_dated(directory, now=START, streak=30, best=30)
            else:
                captured_fetch(directory, monkeypatch, 30)
        sleep(seconds)

    monkeypatch.setattr(scheduler, "_watchdog_ping_sleep", advance)
    orchestrator.run_cascade.return_value = predictions
    scheduler.run_day(DAY, config)
    assert "Streak: 30" in dm.send_dm.call_args.args[1]
    assert picks.load_pick(DAY, directory).chosen_on["kind"] == ("dated" if repair == "dated" else "automatic")
    assert alert.send_dm.call_count == 1


def run_without_escape(config):
    try:
        scheduler.run_day(DAY, config)
    except (KeyboardInterrupt, SystemExit) as error:
        pytest.fail(f"morning fault escaped run_day: {error!r}")


@pytest.mark.parametrize("site", ["read", "dispatcher", "transport", "gate"])
@pytest.mark.parametrize("exception", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_startup_fault_reaches_real_first_check(morning, monkeypatch, capsys, caplog, site, exception):
    config, directory, now, _ = morning
    error = exception("line one\n" + "é" * 220)
    events = []

    def fault(*args, **kwargs):
        events.append(("fault", now[0]))
        raise error

    if site == "read":
        monkeypatch.setattr(contest_state, "load_decision_streak_state", fault)
    elif site == "dispatcher":
        monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", fault)
    elif site == "transport":
        alert.send_dm.side_effect = fault
    else:
        required = scheduler._contest_state_required

        def startup_gate(config):
            if now[0] == START:
                fault()
            return required(config)

        monkeypatch.setattr(scheduler, "_contest_state_required", startup_gate)

    def predict(*args, **kwargs):
        events.append(("prediction", now[0]))
        return None, None

    orchestrator.run_cascade.side_effect = predict
    if exception is not RuntimeError:
        # Round-1 fix (Codex B1): an interrupt is not the morning work's to swallow.
        # It leaves run_day, as it would while the base waits for its first check.
        # (At the reviewed commit these six cases asserted containment.)
        escaped = None
        try:
            scheduler.run_day(DAY, config)
        except BaseException as caught:
            escaped = caught
        assert escaped is error and events == [("fault", START)]
        return
    run_without_escape(config)
    assert events[:2] == [("fault", START), ("prediction", FIRST)]
    err = capsys.readouterr().err
    if site == "transport" and exception is RuntimeError:
        assert "health alert DM delivery failed" in caplog.text
        assert read_status(directory, MORNING_FILE)["status"] == "failed"
        assert FAILURE not in err
    else:
        assert FAILURE + ascii(str(error))[:200] in err.splitlines()
        assert not (directory.parent / "health_state" / MORNING_FILE).exists()


@pytest.mark.parametrize("permanent", [False, True])
def test_advisory_stderr_failure_still_dispatches(morning, monkeypatch, capsys, permanent):
    config, _, now, _ = morning
    error = OSError("offline stderr")
    events = []

    def print_to_stderr(*args, **kwargs):
        line = args[0]
        if line.startswith("  MORNING STATE CHECK") or (permanent and line.startswith(FAILURE)):
            events.append(("stderr failure", now[0]))
            raise error
        print(*args, **kwargs)

    def predict(*args, **kwargs):
        events.append(("prediction", now[0]))
        return None, None

    monkeypatch.setattr(scheduler, "print", print_to_stderr, raising=False)
    alert.send_dm.side_effect = lambda *_: events.append(("dm", now[0]))
    orchestrator.run_cascade.side_effect = predict
    run_without_escape(config)
    assert events[:(4 if permanent else 3)] == (
        [("stderr failure", START)] * (2 if permanent else 1)
        + [("dm", START), ("prediction", FIRST)]
    )
    if not permanent:
        assert FAILURE + ascii(str(error)) in capsys.readouterr().err.splitlines()


@pytest.mark.parametrize("outcome", ["sent", "failed", "no_recipient"])
def test_morning_does_not_touch_shared_status(morning, outcome):
    config, directory, _, _ = morning
    shared = directory.parent / "health_state" / CRITICAL_FILE
    shared.parent.mkdir()
    seed = b'{ "identifiable": "pre-existing shared bytes" }\n'
    shared.write_bytes(seed)
    if outcome == "no_recipient":
        config["bluesky"].clear()
    elif outcome == "failed":
        alert.send_dm.side_effect = RuntimeError("offline CRITICAL")

    def predict(*args, **kwargs):
        assert shared.read_bytes() == seed
        if outcome == "no_recipient":
            assert not shared.with_name(MORNING_FILE).exists()
        else:
            assert read_status(directory, MORNING_FILE)["status"] == outcome
        return None, None

    orchestrator.run_cascade.side_effect = predict
    scheduler.run_day(DAY, config)
    orchestrator.run_cascade.assert_called()
    assert shared.read_bytes() == seed


@pytest.mark.parametrize("kind", ["decision", "pick"])
def test_unreadable_startup_record_is_absent(morning, monkeypatch, kind):
    config, _, now, _ = morning
    module, name = (daily_decision, "load_decision") if kind == "decision" else (picks, "load_pick")
    load = getattr(module, name)
    failed = []

    def read(*args, **kwargs):
        if now[0] == START:
            failed.append(now[0])
            raise UnicodeError("broken record encoding")
        return load(*args, **kwargs)

    monkeypatch.setattr(module, name, read)
    scheduler.run_day(DAY, config)
    assert failed == [START]
    alert.send_dm.assert_called_once_with("offline.test", refusal())


@pytest.mark.parametrize("day_type", ["normal", "refused", "skip", "off", "stale"])
def test_non_morning_artifacts(morning, monkeypatch, capsys, day_type):
    config, directory, _, (frame, tier) = morning
    if day_type in {"normal", "skip"}:
        set_dated(directory, now=START, streak=30, best=30)
    elif day_type == "stale":
        seed_automatic(directory, (START - timedelta(hours=20)).isoformat(), now=START, streak=30)
    elif day_type == "off":
        scheduler.fetch_schedule.side_effect = None
    if day_type == "skip":
        frame = frame.assign(p_game_hit=0.1)
    if day_type != "off":
        orchestrator.run_cascade.return_value = frame, tier

    # Pin metadata contributed by non-scheduler code. Heartbeat writes are
    # included in the snapshot, with their own UTC clock pinned.
    from bts import heartbeat, slate

    monkeypatch.setattr(heartbeat, "datetime", contest_state.datetime)
    monkeypatch.setattr(slate, "datetime", contest_state.datetime)
    monkeypatch.setattr(picks, "_git_head_sha", lambda *_: "a" * 40)
    scheduler.run_day(DAY, config)
    final = daily_decision.load_decision(DAY, directory)
    assert (final and final["action"]) == {
        "normal": "single", "refused": None, "skip": "skip", "off": None, "stale": "single",
    }[day_type]
    output = os.environ.get("CS6B1_SNAPSHOT_DIR")
    if output:
        root = directory.parent
        snapshot = {
            "files": {str(path.relative_to(root)): path.read_bytes().hex()
                      for path in sorted(root.rglob("*")) if path.is_file()},
            "health_dms": [call.args for call in alert.send_dm.call_args_list],
            "pick_dms": [call.args for call in dm.send_dm.call_args_list],
            "stderr": capsys.readouterr().err.replace(str(root), "<fixture>"),
        }
        (Path(output) / f"{day_type}.json").write_text(json.dumps(snapshot, indent=2))
