"""Scheduler delivery and stale warnings with real persistence and offline transports."""

import json
from datetime import datetime, timedelta
from unittest.mock import Mock

import pytest

from bts import contest_state, dm, orchestrator, picks, posting, scheduler, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from tests.test_automatic_reading_cs4c import OLD, RECORDED
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, set_automatic, set_dated
from tests.test_saved_pick_paths_cs5a import invoke, prepare_run_day, scheduler_state, world


WARNING_FILE = "contest_stale_reading_dm_delivery_status.json"
CRITICAL_FILE = "health_dm_delivery_status.json"
WARNING_KEY = "WARN_ATTENTION:contest_state:stale_reading"
PREFIX = "  STALE CONTEST READING — "


def message(day=DAY, detail=f"last accepted fetch {RECORDED}, about 20.00 hours ago"):
    return (
        f"The pick was delivered using an automatic contest reading flagged stale ({detail}). Check "
        f"MLB. If MLB differs from status (B8a), use bts set-contest-streak --date {day} --streak N "
        "--best-streak B --saver S --picks-dir DIR. DIR is the absolute scheduler picks path. The sent "
        "pick and saver flag stay as they were. Write only if the pick's slate and today's ET date "
        f"both equal {day}. For a later date, follow B8b(1). On {day} ET away from midnight, if MLB "
        "shows this season's saver used, check bts saver-state --picks-dir DIR. For active or "
        "not_earned, run bts saver-state --use --picks-dir DIR. For uninitialized, see B8a. Check the "
        "latest fetch status."
    )


def envelope(**kwargs):
    return "BTS health WARN attention:\n- [contest_state] " + message(**kwargs)


def seed_automatic(directory, recorded=RECORDED, now=NOW, streak=7):
    set_automatic(directory, now=now, streak=streak)
    path = directory / "account_state" / "contest_streak.json"
    path.write_text(json.dumps(dict(json.loads(path.read_text()), recorded_at=recorded)))


def stamp(daily, directory, stamp_saved_pick, now=NOW):
    return stamp_saved_pick(daily, contest_state.load_decision_streak_state(
        directory, now=now, require_contest_state=True,
    ))


def fresh(daily, world):
    _, directory, now, _ = world
    daily.chosen_on = picks.chosen_on_record(contest_state.load_decision_streak_state(
        directory, now=now[0], require_contest_state=True,
    ), required=True)
    policy = strategy.PolicyDecision(
        "single", "single", "heuristic", "reach57", 7, 80, None, "missing",
        None, None, None, None, None,
    )
    return strategy.SelectionResult(
        strategy.PickResult(daily), "single", "heuristic", None, None, None,
        streak=7, decision=policy,
    )


def send(daily, world, selection=None, state=None):
    config, directory, _, _ = world
    if state is None:
        state = scheduler_state()
        state.date = daily.date
    return scheduler._deliver_and_lock_pick(
        daily, config, directory, state, daily.date, "fixture", selection=selection,
    )


def pick_transport(mode):
    return dm.send_dm if mode == "dm" else posting.post_to_bluesky


def set_reading_on_disk(daily, directory, value=OLD):
    path = picks.save_pick(daily, directory)
    path.write_text(json.dumps(dict(json.loads(path.read_text()), automatic_reading=value), indent=2))
    return path


@pytest.mark.parametrize("mode", ["dm", "public"])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize(("age", "why"), [(12, None), (20, "old")])
def test_attempt_snapshot_and_confirmed_warning(world, stamp_saved_pick, capsys, mode, saved, age, why):
    config, directory, _, _ = world
    config["scheduler"]["pick_delivery"] = mode
    recorded = (NOW - timedelta(hours=age)).isoformat()
    seed_automatic(directory, recorded)
    daily = daily_pick()
    if saved:
        # Same accepted reading was fresh at the earlier choice. The delivery
        # must replace the stored value even when no selection is available.
        chosen = NOW - timedelta(hours=8)
        stamp(daily, directory, stamp_saved_pick, chosen)
        set_reading_on_disk(daily, directory, dict(
            OLD, recorded_at=recorded, age_hours=float(age - 8), stale=False, reason=None,
        ))
        daily = picks.load_pick(DAY, directory)
    expected = dict(OLD, recorded_at=recorded, age_hours=float(age), stale=why is not None, reason=why)
    attempts = []

    def capture(*args):
        attempted = picks.load_pick(DAY, directory)
        assert attempted.delivery_attempted and not picks.pick_was_delivered(attempted)
        assert not alert.send_dm.called
        attempts.append(getattr(attempted, "automatic_reading", None))
        return "sent"

    def capture_warning(recipient, body):
        assert picks.load_pick(DAY, directory).delivered_at == NOW.isoformat()
        assert scheduler.load_state(DAY, directory).pick_locked
        assert load_decision(DAY, directory)["delivery_status"] == "delivered"
        return "captured warning"

    pick_transport(mode).side_effect = capture
    alert.send_dm.side_effect = capture_warning
    assert send(daily, world, None if saved else fresh(daily, world))
    assert attempts == [expected]
    final = picks.load_pick(DAY, directory)
    assert final.automatic_reading == expected and final.delivery_attempted
    assert final.delivered_at == NOW.isoformat() and picks.pick_was_delivered(final)
    assert scheduler.load_state(DAY, directory).pick_locked
    assert load_decision(DAY, directory)["delivery_status"] == "delivered"
    if saved:
        assert final.chosen_on == daily.chosen_on
    err = capsys.readouterr().err
    if why:
        alert.send_dm.assert_called_once_with("offline.test", envelope())
        assert err.splitlines().count(PREFIX + message()) == 1
    else:
        alert.send_dm.assert_not_called()
        assert PREFIX not in err
    assert send(final, world)
    assert pick_transport(mode).call_count == 1
    assert picks.load_pick(DAY, directory).automatic_reading == expected
    assert PREFIX not in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["dm", "public"])
def test_final_cutoff_is_a_no_compute_control(world, monkeypatch, capsys, mode):
    config, directory, _, _ = world
    config["scheduler"]["pick_delivery"] = mode
    seed_automatic(directory)
    daily = daily_pick()
    path = picks.save_pick(daily, directory)
    cutoff = picks.submission_cutoff_et(daily)
    times = iter((NOW, cutoff))
    monkeypatch.setattr(scheduler, "_now_et", lambda: next(times, cutoff))
    spy = Mock(side_effect=AssertionError("cutoff path computed a record"))
    monkeypatch.setattr(contest_state, "automatic_reading_record", spy, raising=False)
    assert not send(daily, world, fresh(daily, world))
    spy.assert_not_called()
    pick_transport(mode).assert_not_called()
    assert "LATE DELIVERY REFUSED" in alert.send_dm.call_args.args[1]
    assert not path.exists() and load_decision(DAY, directory) is None
    archive, = (directory / DAY).glob("refused_delivery*.json")
    assert json.loads(archive.read_bytes())["refused_delivery"]["reason"] == "past_submission_cutoff"
    assert PREFIX not in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["dm", "public"])
def test_missing_recipient_amendment(world, monkeypatch, capsys, mode):
    config, directory, _, _ = world
    config["scheduler"]["pick_delivery"] = mode
    del config["bluesky"]["dm_recipient"]
    seed_automatic(directory)
    dispatch = Mock()
    monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", dispatch)
    daily = daily_pick()
    path = picks.save_pick(daily, directory)
    raw = path.read_bytes()
    if mode == "dm":
        compute = Mock()
        monkeypatch.setattr(contest_state, "automatic_reading_record", compute, raising=False)
        assert not send(daily, world, fresh(daily, world))
        compute.assert_not_called()
        dm.send_dm.assert_not_called()
        assert path.read_bytes() == raw
        assert PREFIX not in capsys.readouterr().err
    else:
        assert send(daily, world, fresh(daily, world))
        posting.post_to_bluesky.assert_called_once()
        assert getattr(picks.load_pick(DAY, directory), "automatic_reading", None) == OLD
        assert PREFIX + message() in capsys.readouterr().err
    dispatch.assert_not_called()
    assert not (directory.parent / "health_state" / WARNING_FILE).exists()


@pytest.mark.parametrize("mode", ["dm", "public"])
def test_dated_attempt_clears_seeded_record(world, stamp_saved_pick, capsys, mode):
    config, directory, _, _ = world
    config["scheduler"]["pick_delivery"] = mode
    set_dated(directory)
    path = set_reading_on_disk(stamp(daily_pick(), directory, stamp_saved_pick), directory)
    snapshots = []

    def capture(*args):
        snapshots.append(json.loads(path.read_bytes()))
        return "sent"

    pick_transport(mode).side_effect = capture
    assert send(picks.load_pick(DAY, directory), world)
    assert snapshots[0]["automatic_reading"] is None
    assert json.loads(path.read_bytes())["automatic_reading"] is None
    assert PREFIX not in capsys.readouterr().err
    alert.send_dm.assert_not_called()


@pytest.mark.parametrize("mode", ["dm", "public"])
def test_pick_send_failure_and_fresh_retry(world, stamp_saved_pick, capsys, mode):
    config, directory, _, _ = world
    config["scheduler"]["pick_delivery"] = mode
    seed_automatic(directory)
    picks.save_pick(stamp(daily_pick(), directory, stamp_saved_pick), directory)
    pick_transport(mode).side_effect = [RuntimeError("pick send failed"), "retry"]
    assert not send(picks.load_pick(DAY, directory), world)
    failed = picks.load_pick(DAY, directory)
    assert not failed.delivery_attempted and not picks.pick_was_delivered(failed)
    assert getattr(failed, "automatic_reading", None) == OLD
    assert load_decision(DAY, directory) is None
    alert.send_dm.assert_not_called()
    assert PREFIX not in capsys.readouterr().err
    seed_automatic(directory, NOW.isoformat())
    assert send(failed, world)
    assert picks.load_pick(DAY, directory).automatic_reading == dict(
        OLD, recorded_at=NOW.isoformat(), age_hours=0.0, reason=None, stale=False,
    )
    alert.send_dm.assert_not_called()


class ProcessExit(BaseException):
    pass


@pytest.mark.parametrize("mode", ["dm", "public"])
@pytest.mark.parametrize("window", ["before_attempt", "after_attempt", "after_delivery"])
def test_restart_preserves_existing_recovery(world, monkeypatch, stamp_saved_pick, capsys, mode, window):
    config, directory, _, _ = world
    config["scheduler"]["pick_delivery"] = mode
    seed_automatic(directory)
    prior = dict(OLD, age_hours=12.0, reason=None, stale=False)
    path = set_reading_on_disk(stamp(daily_pick(), directory, stamp_saved_pick), directory, prior)
    raw = path.read_bytes()
    with monkeypatch.context() as patch:
        if window == "before_attempt":
            patch.setattr(picks, "save_pick", Mock(side_effect=ProcessExit))
        elif window == "after_attempt":
            pick_transport(mode).side_effect = ProcessExit
        else:
            real_save = picks.save_pick

            def crash_after_delivered_save(pick, picks_dir):
                result = real_save(pick, picks_dir)
                if picks.pick_was_delivered(pick):
                    raise ProcessExit
                return result

            patch.setattr(picks, "save_pick", crash_after_delivered_save)
        with pytest.raises(ProcessExit):
            send(picks.load_pick(DAY, directory), world)
    pick_transport(mode).side_effect = None
    alert.send_dm.assert_not_called()
    loaded = picks.load_pick(DAY, directory)
    if window == "before_attempt":
        assert path.read_bytes() == raw
        assert not loaded.delivery_attempted
        assert send(loaded, world)
        assert pick_transport(mode).call_count == 1
        alert.send_dm.assert_called_once_with("offline.test", envelope())
    else:
        assert loaded.delivery_attempted
        assert getattr(loaded, "automatic_reading", None) == OLD
        assert send(loaded, world) is (window == "after_delivery")
        assert pick_transport(mode).call_count == 1
        if window == "after_attempt":
            assert "DELIVERY OUTCOME UNKNOWN" in alert.send_dm.call_args.args[1]
        else:
            alert.send_dm.assert_not_called()
        assert PREFIX not in capsys.readouterr().err
    assert scheduler.load_state(DAY, directory).pick_locked
    assert load_decision(DAY, directory)["delivery_status"] == (
        "locked_unconfirmed" if window == "after_attempt" else "delivered"
    )


@pytest.mark.parametrize("mode", ["dm", "public"])
@pytest.mark.parametrize("crossing", ["before_send", "during_send"])
def test_warning_date_is_emission_date(world, stamp_saved_pick, capsys, mode, crossing):
    config, directory, now, _ = world
    config["scheduler"]["pick_delivery"] = mode
    before = datetime.fromisoformat("2027-07-15T23:59:55-04:00")
    after = before + timedelta(seconds=60)
    now[0] = before if crossing == "during_send" else after
    recorded = (now[0] - timedelta(hours=20)).isoformat()
    seed_automatic(directory, recorded)
    daily = daily_pick()
    daily.pick.game_time = "2027-07-16T05:00:00Z"
    stamp(daily, directory, stamp_saved_pick, before)

    def cross_midnight(*args):
        now[0] = after
        return "midnight"

    pick_transport(mode).side_effect = cross_midnight
    assert send(daily, world)
    actual = picks.load_pick(DAY, directory)
    assert actual.delivered_at == after.isoformat()
    assert getattr(actual, "automatic_reading", None) == dict(OLD, recorded_at=recorded)
    detail = f"last accepted fetch {recorded}, about 20.00 hours ago"
    alert.send_dm.assert_called_once_with("offline.test", envelope(day="2027-07-16", detail=detail))
    assert PREFIX + message(day="2027-07-16", detail=detail) in capsys.readouterr().err
    status = json.loads((directory.parent / "health_state" / WARNING_FILE).read_bytes())
    assert status["sent_day"] == "2027-07-16"


@pytest.mark.parametrize(("recorded", "detail"), [
    ("not an ISO time", "it has no usable recorded_at"),
    ("2027-07-15T23:00:00Z", "its recorded_at 2027-07-15T23:00:00Z is in the future"),
])
def test_other_stale_warning_copy(world, capsys, recorded, detail):
    _, directory, _, _ = world
    seed_automatic(directory, recorded)
    daily = daily_pick()
    assert send(daily, world, fresh(daily, world))
    alert.send_dm.assert_called_once_with("offline.test", envelope(detail=detail))
    assert PREFIX + message(detail=detail) in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["calculation", "dispatcher", "stderr", "transport", "emission_clock"])
def test_warning_failures_cannot_affect_pick(world, monkeypatch, capsys, failure):
    _, directory, _, _ = world
    seed_automatic(directory)
    error = RuntimeError("line 1\n" + "é" * 220)
    if failure == "calculation":
        monkeypatch.setattr(contest_state, "automatic_reading_record", Mock(side_effect=error), raising=False)
    elif failure == "dispatcher":
        monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", Mock(side_effect=error))
    elif failure == "stderr":
        def printed(*args, **kwargs):
            if str(args[0]).startswith((PREFIX, "  Stale-reading warning failed:")):
                raise error
            print(*args, **kwargs)
        monkeypatch.setattr(scheduler, "print", printed, raising=False)
    elif failure == "emission_clock":
        clock = Mock(return_value=NOW)
        monkeypatch.setattr(scheduler, "_now_et", clock)

        def after_capture(*args):
            clock.side_effect = error

        monkeypatch.setattr(scheduler, "_trigger_live_forward_capture_on_lock", after_capture)
    else:
        alert.send_dm.side_effect = error
    daily = daily_pick()
    assert send(daily, world, fresh(daily, world))
    final = picks.load_pick(DAY, directory)
    assert final.delivery_attempted and picks.pick_was_delivered(final)
    assert scheduler.load_state(DAY, directory).pick_locked
    assert load_decision(DAY, directory)["delivery_status"] == "delivered"
    assert getattr(final, "automatic_reading", None) == (None if failure == "calculation" else OLD)
    err = capsys.readouterr().err
    if failure in {"calculation", "dispatcher", "emission_clock"}:
        assert f"  Stale-reading warning failed: {ascii(str(error))[:200]}\n" in err
        alert.send_dm.assert_not_called()
    elif failure == "stderr":
        alert.send_dm.assert_called_once_with("offline.test", envelope())
    else:
        assert "Stale-reading warning failed" not in err
        status = json.loads((directory.parent / "health_state" / WARNING_FILE).read_bytes())
        assert status["status"] == "failed"


@pytest.mark.parametrize("command", ["run", "orchestrate", "preview"])
def test_legacy_reuse_preserves_stamp_without_warning(world, monkeypatch, tmp_path, stamp_saved_pick, command):
    config, directory, _, statuses = world
    set_dated(directory)
    daily = stamp(daily_pick(), directory, stamp_saved_pick)
    path = set_reading_on_disk(daily, directory)
    details = {"abstract": "P", "detailed": "Warmup", "code": "PW"}
    statuses.update(dict.fromkeys(statuses, details))
    compute = Mock(side_effect=AssertionError("legacy command computed an automatic record"))
    monkeypatch.setattr(contest_state, "automatic_reading_record", compute, raising=False)
    result = invoke(command, config, tmp_path)
    assert result.exit_code == 0, result.output
    assert len((directory / f"lineup_evolution_{DAY}.jsonl").read_text().splitlines()) >= 2
    assert json.loads(path.read_bytes()).get("automatic_reading") == OLD
    assert picks.load_pick(DAY, directory).bluesky_posted is (command != "preview")
    assert PREFIX not in result.stderr
    alert.send_dm.assert_not_called()
    compute.assert_not_called()


def test_run_day_chooses_delivers_and_warns_on_covered_old_file(world, monkeypatch, capsys):
    config, directory, _, _ = world
    config["scheduler"]["entry_intent"] = "enter"
    seed_automatic(directory, streak=30)
    frame, tier = orchestrator.run_cascade.return_value
    orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[0.99, 0.81]), tier)
    prepare_run_day(monkeypatch)
    scheduler.run_day(date=DAY, config=config)
    assert not posting.post_to_bluesky.called
    dm.send_dm.assert_called_once()
    daily = picks.load_pick(DAY, directory)
    assert daily.pick.batter_id == 680757 and daily.notification_sent
    assert scheduler.load_state(DAY, directory).pick_locked
    assert load_decision(DAY, directory)["delivery_status"] == "delivered"
    assert getattr(daily, "automatic_reading", None) == OLD
    alert.send_dm.assert_called_once_with("offline.test", envelope())
    assert PREFIX + message() in capsys.readouterr().err


def independently_deliver(world, *, missing_recipient=False):
    config, directory, now, _ = world
    config["scheduler"]["pick_delivery"] = "public"
    if missing_recipient:
        config["bluesky"].pop("dm_recipient")
    daily = daily_pick()
    daily.date = now[0].date().isoformat()
    daily.pick.game_time = (now[0] + timedelta(hours=2)).isoformat()
    recorded = (now[0] - timedelta(hours=20)).isoformat()
    seed_automatic(directory, recorded, now[0])
    assert send(daily, world, fresh(daily, world))
    return f"last accepted fetch {recorded}, about 20.00 hours ago"


def test_warning_dedup_with_two_independent_same_day_deliveries(world, capsys):
    _, directory, now, _ = world
    detail = independently_deliver(world)
    first = envelope(detail=detail)
    config, _, _, statuses = world
    independently_deliver((config, directory.with_name("other-picks"), now, statuses))
    assert posting.post_to_bluesky.call_count == 2
    alert.send_dm.assert_called_once_with("offline.test", first)
    assert capsys.readouterr().err.count(PREFIX) == 2
    status = directory.parent / "health_state" / WARNING_FILE
    assert json.loads(status.read_bytes())["sent_sources"] == [WARNING_KEY]
    now[0] += timedelta(days=1)
    detail = independently_deliver(world)
    assert posting.post_to_bluesky.call_count == 3 and alert.send_dm.call_count == 2
    assert alert.send_dm.call_args.args == ("offline.test", envelope(day="2027-07-16", detail=detail))
    body = json.loads(status.read_bytes())
    assert body["sent_day"] == "2027-07-16" and body["sent_sources"] == [WARNING_KEY]


@pytest.mark.parametrize("prior", ["sent", "failed"])
@pytest.mark.parametrize("outcome", ["sent", "failed", "no_recipient"])
def test_warning_status_is_independent_of_critical_status(world, prior, outcome):
    _, directory, now, _ = world
    health = directory.parent / "health_state"
    critical = health / CRITICAL_FILE
    warning = health / WARNING_FILE
    args = ([alert.Alert("CRITICAL", "contest_state", "critical reading failure")], "offline.test")
    kwargs = dict(status_path=critical, now_et_date=now[0].date())
    if prior == "failed":
        alert.send_dm.side_effect = RuntimeError("critical transport failed")
    alert.dispatch_dm_for_health_alerts(*args, **kwargs)
    before = critical.read_bytes()
    assert json.loads(before)["status"] == prior
    alert.send_dm.reset_mock()
    alert.send_dm.side_effect = RuntimeError("WARN transport failed") if outcome == "failed" else None
    detail = independently_deliver(world, missing_recipient=outcome == "no_recipient")
    assert critical.read_bytes() == before
    if outcome == "no_recipient":
        alert.send_dm.assert_not_called()
        assert not warning.exists()
    else:
        alert.send_dm.assert_called_once_with("offline.test", envelope(detail=detail))
        assert json.loads(warning.read_bytes())["status"] == outcome
    alert.send_dm.reset_mock()
    alert.send_dm.side_effect = None
    assert alert.dispatch_dm_for_health_alerts(*args, **kwargs) is (prior == "failed")
    if prior == "failed":
        assert "Previous health alert DM FAILED" in alert.send_dm.call_args.args[1]
    else:
        alert.send_dm.assert_not_called()
        assert critical.read_bytes() == before


def test_failed_warning_retries_without_resurface_banner(world):
    alert.send_dm.side_effect = [RuntimeError("WARN transport failed"), "warn-sent"]
    detail = independently_deliver(world)
    independently_deliver(world)
    assert posting.post_to_bluesky.call_count == alert.send_dm.call_count == 2
    assert [call.args for call in alert.send_dm.call_args_list] == [
        ("offline.test", envelope(detail=detail)),
        ("offline.test", envelope(detail=detail)),
    ]


def test_grading_reconcile_and_archive_keep_valid_stamp(world, monkeypatch, stamp_saved_pick):
    _, directory, _, _ = world
    seed_automatic(directory)
    daily = stamp(daily_pick(), directory, stamp_saved_pick)
    assert send(daily, world)
    scheduler.save_nonterminal_result(DAY, directory, "unresolved")
    assert getattr(picks.load_pick(DAY, directory), "automatic_reading", None) == OLD
    monkeypatch.setattr(scheduler, "poll_game_result", lambda *_: "final")
    monkeypatch.setattr(picks, "resolve_daily_slot_results", lambda *_: {"primary": "hit"})
    assert scheduler.run_result_polling(101, DAY, directory) == "final"
    assert picks.load_pick(DAY, directory).result == "hit"
    assert picks.load_pick(DAY, directory).automatic_reading == OLD
    monkeypatch.setattr(picks, "resolve_daily_slot_results", lambda *_: {"primary": "miss"})
    assert len(picks.reconcile_results(
        directory, lookback_days=2, clock=lambda: NOW + timedelta(hours=8),
    )) == 1
    graded = picks.load_pick(DAY, directory)
    assert graded.result == "miss" and graded.automatic_reading == OLD
    archive = scheduler._defer_pick_at_fallback(directory, DAY, graded, reason="fixture")
    archived = json.loads(archive.read_bytes())
    assert list(archived)[-3:] == ["chosen_on", "automatic_reading", "deferred_fallback"]
    assert archived["automatic_reading"] == OLD


@pytest.mark.parametrize("mode", ["dm", "public"])
@pytest.mark.parametrize("guard", [
    "already_delivered", "prior_attempt", "private", "optional", "void", "late",
    "loader_error", "chosen_on_refusal",
])
def test_existing_guards_never_calculate_the_reading(world, monkeypatch, mode, guard):
    config, directory, now, statuses = world
    config["scheduler"]["pick_delivery"] = "private" if guard == "private" else mode
    config["scheduler"]["contest_state_required"] = guard != "optional"
    if guard != "loader_error":
        seed_automatic(directory)
    daily = daily_pick()
    if guard == "already_delivered":
        daily.notification_sent = True
        daily.notification_id = "prior-dm"
        daily.delivered_at = NOW.isoformat()
    elif guard == "prior_attempt":
        daily.delivery_attempted = True
    elif guard == "void":
        statuses[101] = {"abstract": "P", "detailed": "Postponed", "code": "PO"}
    elif guard == "late":
        now[0] = picks.submission_cutoff_et(daily)
    path = set_reading_on_disk(daily, directory)
    compute = Mock(side_effect=AssertionError("unexpected automatic reading calculation"))
    monkeypatch.setattr(contest_state, "automatic_reading_record", compute, raising=False)
    daily = picks.load_pick(DAY, directory)
    assert send(daily, world) is (guard in {"already_delivered", "private", "optional"})
    compute.assert_not_called()
    if guard == "optional":
        pick_transport(mode).assert_called_once()
    else:
        pick_transport(mode).assert_not_called()
    if path.exists() and hasattr(daily, "automatic_reading"):
        assert json.loads(path.read_bytes())["automatic_reading"] == OLD
    assert not (directory.parent / "health_state" / WARNING_FILE).exists()
