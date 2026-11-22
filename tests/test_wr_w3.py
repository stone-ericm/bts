"""Exercise in-loop fallback using the real scheduled check and delivery."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import signal
from threading import Event, current_thread, main_thread
from time import monotonic

import pytest

from bts import contest_state, daily_decision, picks, scheduler


DAY = "2027-03-24"


class DecisionReadDeadline(BaseException):
    pass


@pytest.fixture
def offline(tmp_path):
    source = (Path(__file__).resolve().parents[1] /
              "docs/reviews/2026-11-18-whole-range/slice-3-codex/offline.py")
    spec = spec_from_file_location("wr_w3_offline", source)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    module.HERE = tmp_path
    return module


def test_real_overrun_reuses_the_same_candidate_before_cutoff(offline):
    with offline.offline(offline.instant(DAY, 10)) as (world, _):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        starts = []

        def forecast():
            starts.append(world.now)
            if world.now < offline.instant(DAY, 18, 25):
                return None
            if world.now == offline.instant(DAY, 18, 25):
                world.advance(35 * 60)
                return world.frame((0.99,), (True,))
            world.advance(6 * 60)
            return None

        world.predictions = forecast
        scheduler.run_day(DAY, world.config)

        assert starts == [offline.instant(DAY, h, m) for h, m in
                          [(16, 10), (17, 10), (18, 25)]]
        state = scheduler.load_state(DAY, world.picks)
        refresh, = state.fallback_refreshes
        assert refresh["reused_check_cascade"] and refresh["duration_sec"] == 0
        (_, sent_at, text), = [event for event in world.events if event[0] == "dm"]
        assert sent_at < offline.instant(DAY, 19, 5)
        commit = daily_decision.load_decision(DAY, world.picks)
        assert commit["scoreable"] and commit["primary"]["batter_name"] in text


def test_missed_pick_alert_survives_abandoned_reuse(offline, capsys):
    at = lambda hour, minute: offline.instant(DAY, hour, minute)
    with offline.offline(at(18, 25)) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        path = world.picks / f"{DAY}.json"
        starts = []
        check_reload = picks.load_pick_bytes
        changed = False

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(35 * 60)
                return world.frame((0.99,), (True,))
            return None

        def change_before_extra_reload(*args, **kwargs):
            nonlocal changed
            if kwargs.get("regular_only") and not changed:
                changed = True
                path.write_bytes(path.read_bytes() + b"\n")
                world.dated(0, best=1)
            return check_reload(*args, **kwargs)

        world.predictions = forecast
        mp.setattr(picks, "load_pick_bytes", change_before_extra_reload)
        scheduler.run_day(DAY, world.config)

        assert changed and starts == [at(18, 25), at(19, 0)]
        assert not world.sends("dm")
        assert daily_decision.load_decision(DAY, world.picks) is None
        assert not picks.load_pick(DAY, world.picks).delivery_attempted
        alerts = world.sends("health-dm")
        assert sum("[contest_state]" in text for text in alerts) == 1
        assert sum("[missed_pick]" in text for text in alerts) == 1
        assert "reusing this check's fresh cascade decision" not in capsys.readouterr().err


@pytest.mark.parametrize("case", [
    "replacement", "audit_replacement", "whitespace", "sleep", "equality",
    "late_read", "best", "saver", "void", "absence", "dangling",
    "json", "shape", "invalid_time", "unreadable_pick", "directory", "fifo",
    "scoreable", "matching_scoreable", "wrong_date", "invalid_decision",
    "unreadable_decision", "decision_fifo", "decision_dangling", "decision_empty",
    "initial_attempt",
])
def test_fallback_reload(offline, case):
    with offline.offline(offline.instant(DAY, 18, 25)) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        path = world.picks / f"{DAY}.json"
        decision_path = daily_decision.decision_path(DAY, world.picks)
        starts = []
        polled = []
        shadow = []
        if case == "matching_scoreable":
            world.config["scheduler"]["shadow_model"] = True
            mp.setattr(scheduler, "_trigger_shadow_prediction_on_lock",
                       lambda config, date, batter: shadow.append((date, batter)))
        mp.setattr(scheduler, "run_result_polling", lambda pk, *a, **kw: polled.append(pk))
        check = scheduler.run_single_check
        append = picks.append_lineup_evolution
        read = contest_state.load_decision_streak_state
        checked = False
        original_record = None
        bad_pick_bytes = None
        real_open = os.open
        real_path_open = Path.open

        def replacement(daily):
            other = deepcopy(daily)
            other.pick = replace(daily.pick, batter_id=999, batter_name="Replacement Batter")
            picks.save_pick(other, world.picks)

        def append_and_replace(daily, directory):
            append(daily, directory)
            if daily.pick.batter_id != 999 and not checked:
                replacement(daily)

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                minutes = {"sleep": 0, "equality": 10}.get(case, 35)
                world.advance(minutes * 60)
                return world.frame((0.99,), (True,))
            return None

        def mutate_after_check(*args, **kwargs):
            nonlocal checked, original_record, bad_pick_bytes
            result = check(*args, **kwargs)
            checked = True
            if case == "replacement":
                replacement(result["pick_result"].daily)
            elif case == "whitespace":
                path.write_bytes(path.read_bytes() + b"\n")
            elif case in {"best", "saver"}:
                world.dated(0, best=1 if case == "best" else 0,
                            saver="not_earned" if case == "saver" else "used")
            elif case == "void":
                world.schedule[DAY][0]["status"].update(
                    detailedState="Postponed", statusCode="PO")
            elif case in {"absence", "dangling", "directory", "fifo"}:
                path.unlink()
                if case == "dangling":
                    path.symlink_to(world.picks / "missing.json")
                elif case == "directory":
                    path.mkdir()
                elif case == "fifo":
                    os.mkfifo(path)
            elif case in {"json", "shape", "invalid_time"}:
                if case == "invalid_time":
                    content = json.loads(path.read_bytes())
                    content["pick"]["game_time"] = "bad time"
                    path.write_text(json.dumps(content))
                else:
                    path.write_text("{" if case == "json" else "[]")
                bad_pick_bytes = path.read_bytes()
            elif case in {"scoreable", "matching_scoreable", "wrong_date"}:
                primary = {"batter_name": "Previous Batter", "batter_id": 999}
                if case == "matching_scoreable":
                    primary = scheduler._row_from_daily(result["pick_result"].daily.pick)
                daily_decision.write_decision(
                    DAY, world.picks, action="single", source="mdp", scoreable=True,
                    delivery_status="private_locked",
                    primary=primary,
                )
                if case == "wrong_date":
                    record = json.loads(decision_path.read_bytes())
                    decision_path.write_text(json.dumps(dict(record, date="2027-03-23")))
                original_record = decision_path.read_bytes()
            elif case in {"invalid_decision", "unreadable_decision", "decision_empty"}:
                decision_path.write_text("" if case == "decision_empty" else "{")
            elif case == "decision_fifo":
                os.mkfifo(decision_path)
            elif case == "decision_dangling":
                decision_path.symlink_to(decision_path.with_name("missing.json"))
            elif case == "initial_attempt":
                current = picks.load_pick(DAY, world.picks)
                current.delivery_attempted = True
                picks.save_pick(current, world.picks)
            return result

        def read_at_cutoff(*args, **kwargs):
            state = read(*args, **kwargs)
            if checked and world.now < offline.instant(DAY, 19, 5):
                world.advance(5 * 60)
            return state

        def expired(*_):
            if case == "decision_fifo":
                raise DecisionReadDeadline("decision FIFO open blocked")
            raise TimeoutError("FIFO open exceeded test deadline")

        def deny_decision_open(name, flags, *args, **kwargs):
            if name == decision_path and flags & os.O_NONBLOCK:
                raise PermissionError("fixture denies decision read")
            return real_open(name, flags, *args, **kwargs)

        def deny_pick_open(self, *args, **kwargs):
            if checked and self == path:
                raise PermissionError("fixture denies pick read")
            return real_path_open(self, *args, **kwargs)

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", mutate_after_check)
        if case == "audit_replacement":
            mp.setattr(picks, "append_lineup_evolution", append_and_replace)
        if case == "late_read":
            mp.setattr(contest_state, "load_decision_streak_state", read_at_cutoff)
        if case == "unreadable_decision":
            mp.setattr(os, "open", deny_decision_open)
        if case == "unreadable_pick":
            mp.setattr(Path, "open", deny_pick_open)
        errors = {"json": json.JSONDecodeError, "shape": TypeError,
                  "invalid_time": ValueError, "unreadable_pick": PermissionError,
                  "directory": IsADirectoryError, "fifo": TimeoutError}
        if case in {"fifo", "decision_fifo"}:
            prior = signal.signal(signal.SIGALRM, expired)
            signal.alarm(3)
        try:
            if case in errors:
                with pytest.raises(errors[case]):
                    scheduler.run_day(DAY, world.config)
            else:
                scheduler.run_day(DAY, world.config)
        finally:
            if case in {"fifo", "decision_fifo"}:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prior)

        sends = world.sends("dm")
        alerts = world.sends("health-dm")
        if case in errors or case in {"absence", "dangling"}:
            assert not sends
            assert os.path.lexists(path) == (case != "absence")
            if bad_pick_bytes is not None:
                assert path.read_bytes() == bad_pick_bytes
            return
        if case in {"scoreable", "matching_scoreable"}:
            assert not sends and decision_path.read_bytes() == original_record
            assert not picks.pick_was_delivered(picks.load_pick(DAY, world.picks))
            assert polled == ([101] if case == "matching_scoreable" else [])
            if case == "matching_scoreable":
                assert shadow == [(DAY, "Synthetic Batter 101")]
            return
        if case in {"late_read", "best", "saver", "void"}:
            code = {"late_read": "late_delivery", "void": "postponed_pick"}.get(
                case, "contest_state")
            assert not sends and any(f"[{code}]" in text for text in alerts)
            return
        if case == "initial_attempt":
            assert not sends and len(starts) == 2
            assert any("OUTCOME UNKNOWN" in text for text in alerts)
            assert daily_decision.load_decision(DAY, world.picks)["delivery_status"] == "locked_unconfirmed"
            return
        sent, = sends
        state = scheduler.load_state(DAY, world.picks)
        refresh, = state.fallback_refreshes
        assert refresh["reused_check_cascade"] == (case == "equality")
        assert len(starts) == (1 if case == "equality" else 2)
        record = daily_decision.load_decision(DAY, world.picks)
        assert record["primary"]["batter_name"] in sent
        if case in {"replacement", "audit_replacement"}:
            assert record["primary"]["batter_id"] == 999


@pytest.mark.parametrize("action", ["deliver", "defer"])
def test_reused_postponed_pick_takes_void_refusal_at_cutoff(offline, action):
    at = lambda hour, minute: offline.instant(DAY, hour, minute)
    with offline.offline(at(18, 25)) as (world, mp):
        if action == "deliver":
            world.schedule[DAY] = world.schedule[DAY][:1]
            world.dated(0)
        else:
            world.dated(30)
            world.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
            mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                {"time_et": at(18, 25), "game_pks": [101]},
                {"time_et": at(18, 50), "game_pks": [102]},
            ])
            mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
                pk: {"away", "home"} for pk in pks if pk == 101})
        check = scheduler.run_single_check
        statuses = scheduler._delivery_game_statuses
        save_state = scheduler.save_state
        starts = []
        void_read = False
        crossed_during_save = False

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(2430 if action == "deliver" else 2396)
                n = len(world.schedule[DAY])
                return world.frame((0.99, 0.40)[:n], (True, True)[:n])
            return None

        def postpone(*args, **kwargs):
            result = check(*args, **kwargs)
            if len(starts) == 1:
                world.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
            return result

        def read_statuses(date):
            nonlocal void_read
            result = statuses(date)
            if current_thread() is not main_thread():
                void_read = result[101]["detailed"] == "Postponed"
            return result

        def slow_state_save(state, directory):
            nonlocal crossed_during_save
            saved = save_state(state, directory)
            if void_read and not crossed_during_save:
                assert world.now < at(19, 5)
                world.advance(5)
                crossed_during_save = True
            return saved

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", postpone)
        mp.setattr(scheduler, "_delivery_game_statuses", read_statuses)
        mp.setattr(scheduler, "save_state", slow_state_save)
        scheduler.run_day(DAY, world.config)

        state = scheduler.load_state(DAY, world.picks)
        row = state.fallback_refreshes[0]
        assert row["reused_check_cascade"] and row["action"] == action
        assert crossed_during_save == (action == "defer")
        refusal, = state.delivery_refusals
        assert refusal["reason"] == "game_not_playable"
        assert not world.sends("dm")
        alert, = world.sends("health-dm")
        assert "[postponed_pick]" in alert and "Synthetic Batter 101" not in alert
        assert daily_decision.load_decision(DAY, world.picks) is None


@pytest.mark.parametrize("mode", ["dm", "public"])
@pytest.mark.parametrize("clear_error", [False, True])
def test_reuse_refuses_when_attempt_write_reaches_cutoff(offline, mode, clear_error):
    with offline.offline(offline.instant(DAY, 18, 25)) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.config["scheduler"]["pick_delivery"] = mode
        world.dated(0)
        starts = []
        writes = []
        clear_failed = False
        write = picks.atomic_write_text

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(2399)
                return world.frame((0.99,), (True,))
            world.advance(2)
            return None

        def write_and_advance(path, text):
            nonlocal clear_failed
            if path == world.picks / f"{DAY}.json":
                attempted = json.loads(text)["delivery_attempted"]
                if clear_error and writes == [False, True] and not attempted and not clear_failed:
                    clear_failed = True
                    raise OSError("fixture transient marker-clear failure")
                writes.append(attempted)
                if attempted:
                    world.advance(1.25)
            write(path, text)

        world.predictions = forecast
        mp.setattr(picks, "atomic_write_text", write_and_advance)
        scheduler.run_day(DAY, world.config)

        assert not world.sends("dm") and not world.sends("post")
        assert clear_failed is clear_error
        assert len(starts) == 1 and writes == [False, True, False]
        assert sum("[late_delivery]" in text for text in world.sends("health-dm")) == 1
        archived, = (world.picks / DAY).glob("refused_delivery*.json")
        assert json.loads(archived.read_bytes())["delivery_attempted"] is False
        assert daily_decision.load_decision(DAY, world.picks) is None


@pytest.mark.parametrize("day", ["2027-06-18", "2026-12-31"])
@pytest.mark.parametrize("crossing", ["before_read", "during_read", "during_marker", "clear_error"])
def test_reuse_checks_live_day_at_midnight(offline, day, crossing):
    from bts import strategy

    start = offline.instant(day, 23, 20)
    with offline.offline(start) as (world, mp):
        world.schedule[day] = [offline.game(101, start + timedelta(minutes=70))]
        world.config["scheduler"].update(
            fallback_deadline_min=40, fallback_deadline_min_morning=40)
        world.dated(0)
        starts = []
        reads = []
        check = scheduler.run_single_check
        read = contest_state.load_decision_streak_state
        write = picks.atomic_write_text
        checked = False
        marker_written = False
        clear_failed = False

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(2399)
                return world.frame((0.99,), (True,))
            world.advance(2)
            return None

        def crossing_check(*args, **kwargs):
            nonlocal checked
            result = check(*args, **kwargs)
            checked = True
            if crossing == "before_read":
                world.advance(2)
            return result

        def crossing_read(*args, **kwargs):
            if checked:
                reads.append(world.now.date().isoformat())
            result = read(*args, **kwargs)
            if checked and crossing == "during_read" and len(reads) == 1:
                world.advance(2)
            return result

        def crossing_write(path, text):
            nonlocal marker_written, clear_failed
            if crossing in {"during_marker", "clear_error"} and path == world.picks / f"{day}.json":
                if json.loads(text)["delivery_attempted"]:
                    marker_written = True
                    world.advance(2)
                elif crossing == "clear_error" and marker_written and not clear_failed:
                    clear_failed = True
                    raise OSError("fixture transient marker-clear failure")
            write(path, text)

        world.predictions = forecast
        mp.setattr(strategy, "_load_mdp", lambda: None)
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": start, "game_pks": [101]}])
        mp.setattr(scheduler, "run_single_check", crossing_check)
        mp.setattr(contest_state, "load_decision_streak_state", crossing_read)
        mp.setattr(picks, "atomic_write_text", crossing_write)
        scheduler.run_day(day, world.config)

        assert not world.sends("dm")
        assert any("[contest_state]" in text for text in world.sends("health-dm"))
        assert daily_decision.load_decision(day, world.picks) is None
        first_refresh = scheduler.load_state(day, world.picks).fallback_refreshes[0]
        assert first_refresh["reused_check_cascade"]
        assert reads[-1] != day
        assert (day in reads) is (crossing != "before_read")
        assert not picks.load_pick(day, world.picks).delivery_attempted
        if crossing == "clear_error":
            assert clear_failed
            alerts = [when for kind, when, body in world.events
                      if kind == "health-dm" and "ET date changed after the final contest-state read" in body]
            assert len(alerts) == 1


@pytest.mark.parametrize("case", [
    "confirm", "attempt_deliver", "attempt_defer", "void", "reload_cutoff", "slow_status", "partial",
    "changed_decision_fifo", "poll_keyboard_interrupt", "poll_system_exit",
    "attempt_scoreable", "attempt_invalid", "shared_budget",
])
def test_pending_window_replans_without_losing_later_checks(offline, case):
    at = lambda hour, minute: offline.instant(DAY, hour, minute)
    with offline.offline(at(18, 25)) as (world, mp):
        world.dated(0)
        world.schedule[DAY][1]["gameDate"] = at(19, 20).isoformat()
        later_runs = [(18, 25, 101), (18, 50, 102)]
        if case == "partial":
            world.schedule[DAY].append(offline.game(103, at(19, 50)))
            later_runs.append((19, 5, 103))
        mp.setattr(scheduler, "plan_run_times", lambda *args, **kwargs: [
            {"time_et": at(hour, minute), "game_pks": [pk]} for hour, minute, pk in later_runs])
        mp.setattr(scheduler, "run_result_polling", lambda *_a, **_kw: "fixture")
        shadow = []
        if case.startswith("attempt"):
            world.config["scheduler"]["shadow_model"] = True
            mp.setattr(scheduler, "_trigger_shadow_prediction_on_lock",
                       lambda config, date, batter: shadow.append((date, batter)))
        starts = []
        replans = []
        workers = []
        live_confirmations = None
        measured = []
        published_record = None
        real_plan = scheduler.plan_fallback_action
        real_read = picks.load_pick_bytes
        real_check = scheduler.run_single_check
        real_statuses = scheduler._delivery_game_statuses
        real_replan = scheduler._replan_after_reuse_poll
        decision_path = daily_decision.decision_path(DAY, world.picks)
        first_returned = False
        poll_changed_pick = False
        cutoff_at_reload = False
        body_started = Event()
        release_body = Event()
        body_finished = Event()

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(22 * 60)
                second = 0.40 if case == "void" else 0.70
                return world.frame((0.99, second, 0.60), (True, True, True))
            if world.now < at(18, 50):
                return world.frame((0.99, 0.70, 0.60), (True, True, True))
            frame = world.frame((0.70, 0.99, 0.60), (True, False, True))
            return frame.iloc[[1, 0] + ([2] if case == "partial" else [])]

        def check(*args, **kwargs):
            nonlocal first_returned
            if first_returned and case == "changed_decision_fifo":
                assert decision_path.is_fifo()
                decision_path.unlink()
            result = real_check(*args, **kwargs)
            first_returned = True
            return result

        def confirmations(game_pks):
            nonlocal poll_changed_pick, published_record
            if first_returned and game_pks == [102] and not poll_changed_pick:
                if case.startswith("attempt"):
                    current = picks.load_pick(DAY, world.picks)
                    current.delivery_attempted = True
                    picks.save_pick(current, world.picks)
                    poll_changed_pick = True
                    if case == "attempt_scoreable":
                        daily_decision.write_decision(
                            DAY, world.picks, action="double" if current.double_down else "single",
                            source="mdp", primary=scheduler._row_from_daily(current.pick),
                            double_down=scheduler._row_from_daily(current.double_down),
                            delivery_status="locked_unconfirmed", scoreable=True,
                        )
                        published_record = decision_path.read_bytes()
                    if case == "attempt_invalid":
                        decision_path.write_text(json.dumps(dict(
                            schema_version=daily_decision.DECISION_SCHEMA, date=DAY,
                            action=[], scoreable=False)))
                if case == "shared_budget":
                    assert not Event().wait(1.2)
                if case in {"confirm", "attempt_deliver"}:
                    return {102: {"away", "home"}}
                if case == "partial":
                    return {102: {"away"}}
                if case == "changed_decision_fifo":
                    assert not decision_path.exists()
                    os.mkfifo(decision_path)
                if case == "poll_keyboard_interrupt":
                    raise KeyboardInterrupt()
                if case == "poll_system_exit":
                    raise SystemExit()
            if first_returned and case == "partial" and game_pks == [103]:
                slow_body()
                return {103: {"away", "home"}}
            return {pk: {"away", "home"} for pk in game_pks if pk == 101}

        def plan(**kwargs):
            nonlocal live_confirmations
            live_confirmations = kwargs["confirmed_sides"]
            replans.append(set(kwargs["confirmed_sides"]))
            return real_plan(**kwargs)

        def measure_replan(*args, **kwargs):
            started = monotonic()
            result = real_replan(*args, **kwargs)
            measured.append(monotonic() - started)
            return result

        def slow_body():
            workers.append(current_thread())
            body_started.set()
            release_body.wait(timeout=8)
            body_finished.set()

        def statuses(date):
            if case in {"slow_status", "shared_budget"} and world.now < at(18, 50):
                slow_body()
                return {}
            return real_statuses(date)

        def reload_at_cutoff(*args, **kwargs):
            nonlocal cutoff_at_reload
            result = real_read(*args, **kwargs)
            if first_returned and not cutoff_at_reload and not kwargs.get("regular_only"):
                cutoff_at_reload = True
                world.advance(18 * 60)
            return result

        if case == "void":
            def postpone_after_check(*args, **kwargs):
                result = check(*args, **kwargs)
                world.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
                return result

            mp.setattr(scheduler, "run_single_check", postpone_after_check)
        else:
            mp.setattr(scheduler, "run_single_check", check)
        world.predictions = forecast
        mp.setattr(scheduler, "check_confirmed_lineups", confirmations)
        mp.setattr(scheduler, "plan_fallback_action", plan)
        mp.setattr(scheduler, "_replan_after_reuse_poll", measure_replan)
        mp.setattr(scheduler, "_delivery_game_statuses", statuses)
        if case == "reload_cutoff":
            mp.setattr(picks, "load_pick_bytes", reload_at_cutoff)
        if case.startswith("poll_"):
            error = KeyboardInterrupt if case.endswith("interrupt") else SystemExit
            with pytest.raises(error):
                scheduler.run_day(DAY, world.config)
            assert not world.sends("dm") and not picks.load_pick(DAY, world.picks).delivery_attempted
            assert daily_decision.load_decision(DAY, world.picks) is None
            return
        if case == "changed_decision_fifo":
            def expired(*_):
                raise DecisionReadDeadline("decision FIFO installed during poll blocked")

            prior = signal.signal(signal.SIGALRM, expired)
            signal.alarm(3)
        try:
            scheduler.run_day(DAY, world.config)
            if case in {"slow_status", "partial", "shared_budget"}:
                assert body_started.is_set() and not body_finished.is_set()
        finally:
            if case == "changed_decision_fifo":
                signal.alarm(0)
                signal.signal(signal.SIGALRM, prior)
            release_body.set()
            if body_started.is_set():
                assert body_finished.wait(timeout=1)
            for worker in workers:
                worker.join(timeout=1)
                assert not worker.is_alive()

        record = daily_decision.load_decision(DAY, world.picks)
        state = scheduler.load_state(DAY, world.picks)
        if case.startswith("attempt"):
            assert poll_changed_pick and len(starts) == 1 and not world.sends("dm")
            assert shadow == [(DAY, "Synthetic Batter 101")]
            assert record["delivery_status"] == "locked_unconfirmed"
            assert sum("OUTCOME UNKNOWN" in text for text in world.sends("health-dm")) == 1
            if case == "attempt_scoreable":
                assert decision_path.read_bytes() == published_record
                scheduler._alert_delivery_outcome_unknown(world.config, world.picks, DAY)
                assert sum("OUTCOME UNKNOWN" in text for text in world.sends("health-dm")) == 1
        elif case == "confirm":
            assert len(starts) == 1 and len(world.sends("dm")) == 1
            assert state.fallback_refreshes[0]["reused_check_cascade"]
            assert state.fallback_refreshes[0]["action"] == "deliver"
        else:
            assert len(starts) == 2
            assert starts[1] == (at(19, 5) if case == "reload_cutoff" else at(18, 50))
            assert record["primary"]["game_pk"] == 102
            if case == "changed_decision_fifo":
                assert all(row["action"] != "defer" for row in state.fallback_refreshes or [])
                assert not list((world.picks / DAY).glob("deferred_fallback*.json"))
                return
            assert state.fallback_refreshes[0]["reused_check_cascade"]
            if case == "partial":
                assert any((102, "away") in sides and (102, "home") not in sides
                           and (103, "home") not in sides for sides in replans)
                assert (103, "home") not in live_confirmations
            if case == "shared_budget":
                elapsed, = measured
                assert 1.9 <= elapsed < 2.6
            if case in {"void", "reload_cutoff"}:
                code = "postponed_pick" if case == "void" else "late_delivery"
                assert sum(f"[{code}]" in text for text in world.sends("health-dm")) == 1


@pytest.mark.parametrize("correction", [False, True])
def test_reuse_replans_gap_after_status_read(offline, correction):
    at = lambda hour, minute, second=0: offline.instant(DAY, hour, minute, second)
    with offline.offline(at(18, 25)) as (world, mp):
        world.dated(30)
        starts = []
        plans = []
        real_plan = scheduler.plan_fallback_action
        real_status = scheduler._delivery_game_statuses

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(17 * 60 + 59)
                return world.frame((0.99, 0.98), (False, True))
            return None

        def statuses(date):
            if current_thread() is not main_thread():
                assert world.now == at(18, 42, 59)
                world.advance(2)
                if correction:
                    world.dated(30, best=31)
            return real_status(date)

        def plan(**kwargs):
            result = real_plan(**kwargs)
            plans.append((kwargs["now"], result.reason))
            return result

        world.predictions = forecast
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at(18, 25), "game_pks": [101]},
            {"time_et": at(18, 42), "game_pks": [102]},
        ])
        mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
            pk: {"away", "home"} for pk in pks if pk == 101})
        mp.setattr(scheduler, "_delivery_game_statuses", statuses)
        mp.setattr(scheduler, "plan_fallback_action", plan)
        scheduler.run_day(DAY, world.config)

        assert plans[0] == (at(18, 42, 59), "gap_contender_window_feasible")
        assert plans[-1] == (at(18, 43, 1), "gap_no_feasible_window")
        assert not list((world.picks / DAY).glob("deferred_fallback*.json"))
        refresh = scheduler.load_state(DAY, world.picks).fallback_refreshes[0]
        assert refresh["action"] == "deliver" and refresh["reused_check_cascade"]
        record = daily_decision.load_decision(DAY, world.picks)
        if correction:
            assert not world.sends("dm") and record is None
            alerts = world.sends("health-dm")
            assert any("[contest_state]" in text for text in alerts)
            assert any("[missed_pick]" in text for text in alerts)
        else:
            assert len(world.sends("dm")) == 1 and record["primary"]["game_pk"] == 101


@pytest.mark.parametrize("mode", ["dm", "public"])
def test_reuse_confirms_stale_automatic_reading(offline, mode):
    with offline.offline(offline.instant(DAY, 18, 25)) as (world, _):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.config["scheduler"]["pick_delivery"] = mode
        world.automatic(0, age_hours=15)

        def forecast():
            if world.now == offline.instant(DAY, 18, 25):
                world.advance(35 * 60)
                return world.frame((0.99,), (True,))
            return None

        world.predictions = forecast
        scheduler.run_day(DAY, world.config)
        assert len(world.sends("dm" if mode == "dm" else "post")) == 1
        state = scheduler.load_state(DAY, world.picks)
        assert state.fallback_refreshes[0]["reused_check_cascade"]
        pick = picks.load_pick(DAY, world.picks)
        assert pick.automatic_reading["stale"]
        assert sum("stale" in text.lower() for text in world.sends("health-dm")) == 1


@pytest.mark.parametrize("failed", [False, True])
def test_reuse_warns_about_stale_reading_only_after_confirmed_send(offline, failed):
    from bts import dm

    with offline.offline(offline.instant(DAY, 18, 25)) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.automatic(0, age_hours=15)
        write = picks.atomic_write_text
        send = dm.send_dm
        attempted_writes = []
        calls = 0

        def forecast():
            if world.now == offline.instant(DAY, 18, 25):
                world.advance(35 * 60)
                return world.frame((0.99,), (True,))
            return None

        def write_attempt(path, text):
            write(path, text)
            if path == world.picks / f"{DAY}.json":
                attempted_writes.append(json.loads(text)["delivery_attempted"])

        def send_outcome(recipient, text):
            nonlocal calls
            calls += 1
            assert not world.sends("health-dm")
            if not failed:
                return None
            if calls == 1:
                raise RuntimeError("fixture transport failure")
            return send(recipient, text)

        world.predictions = forecast
        mp.setattr(picks, "atomic_write_text", write_attempt)
        mp.setattr(dm, "send_dm", send_outcome)
        scheduler.run_day(DAY, world.config)
        assert scheduler.load_state(DAY, world.picks).fallback_refreshes[0]["reused_check_cascade"]
        assert calls == (2 if failed else 1)
        assert attempted_writes == ([False, True, False, True, True] if failed else [False, True, True])
        assert sum("stale" in text.lower() for text in world.sends("health-dm")) == failed


def test_warmup_cached_selection_commits_saved_pick_rows(offline):
    with offline.offline(offline.instant(DAY, 18, 25)) as (world, _):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        world.predictions = lambda: world.frame((0.99,), (True,))
        checked = scheduler.run_single_check(DAY, [101], set(), world.config, 0.03)
        cached = checked["pick_result"].daily
        world.schedule[DAY][0]["status"].update(detailedState="Warmup", statusCode="PW")

        def forecast():
            world.advance(35 * 60)
            return world.frame((0.99,), (True,))

        world.predictions = forecast
        scheduler.run_day(DAY, world.config)
        state = scheduler.load_state(DAY, world.picks)
        assert state.fallback_refreshes[0]["reused_check_cascade"]
        record = daily_decision.load_decision(DAY, world.picks)
        assert record["primary"] == scheduler._row_from_daily(cached.pick)


def test_reuse_keeps_both_slots_when_only_one_leg_is_void(offline):
    with offline.offline(offline.instant(DAY, 18, 25)) as (world, mp):
        world.dated(0)
        check = scheduler.run_single_check

        def forecast():
            if world.now == offline.instant(DAY, 18, 25):
                world.advance(35 * 60)
                return world.frame((0.99, 0.98), (True, True))
            return None

        def postpone(*args, **kwargs):
            result = check(*args, **kwargs)
            assert result["pick_result"].daily.double_down.game_pk == 102
            world.schedule[DAY][0]["status"].update(detailedState="Postponed", statusCode="PO")
            return result

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", postpone)
        scheduler.run_day(DAY, world.config)
        sent, = world.sends("dm")
        assert "Synthetic Batter 101" in sent and "Synthetic Batter 102" in sent
        record = daily_decision.load_decision(DAY, world.picks)
        assert record["action"] == "double"
        assert [record[slot]["game_pk"] for slot in ("primary", "double_down")] == [101, 102]
        assert scheduler.load_state(DAY, world.picks).fallback_refreshes[0]["reused_check_cascade"]


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("stage", ["before_marker", "after_marker", "after_confirmation"])
def test_restart_uses_durable_delivery_state(offline, error, stage):
    from bts import dm

    at = offline.instant(DAY, 18, 25)
    with offline.offline(at) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        starts = []
        format_text = scheduler._format_pick_delivery_text
        send = dm.send_dm
        commit = scheduler._commit_decision_for_pick
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at, "game_pks": [101]}])

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(35 * 60)
                return world.frame((0.99,), (True,))
            return None

        def interrupt(*args, **kwargs):
            raise error()

        world.predictions = forecast
        if stage == "before_marker":
            mp.setattr(scheduler, "_format_pick_delivery_text", interrupt)
        elif stage == "after_marker":
            mp.setattr(dm, "send_dm", interrupt)
        else:
            mp.setattr(scheduler, "_commit_decision_for_pick", interrupt)
        with pytest.raises(error):
            scheduler.run_day(DAY, world.config)

        before = picks.load_pick(DAY, world.picks)
        assert before.delivery_attempted == (stage != "before_marker")
        assert picks.pick_was_delivered(before) == (stage == "after_confirmation")
        assert len(world.sends("dm")) == (stage == "after_confirmation")
        assert daily_decision.load_decision(DAY, world.picks) is None
        history = scheduler.load_state(DAY, world.picks).fallback_refreshes
        assert len(history) == 1 and history[0]["reused_check_cascade"]

        mp.setattr(scheduler, "_format_pick_delivery_text", format_text)
        mp.setattr(dm, "send_dm", send)
        mp.setattr(scheduler, "_commit_decision_for_pick", commit)
        scheduler.run_day(DAY, world.config)
        record = daily_decision.load_decision(DAY, world.picks)
        assert record["delivery_status"] == (
            "locked_unconfirmed" if stage == "after_marker" else "delivered")
        assert len(world.sends("dm")) == (stage != "after_marker")
        assert any("OUTCOME UNKNOWN" in text for text in world.sends("health-dm")) == (
            stage == "after_marker")
        history = scheduler.load_state(DAY, world.picks).fallback_refreshes
        assert sum(row["reused_check_cascade"] for row in history) == 1


@pytest.mark.parametrize("case", ["no_predictions", "contest_error", "locked", "write_error"])
def test_early_check_returns_keep_their_shape(offline, case):
    at = offline.instant(DAY, 18, 25)
    with offline.offline(at) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        world.predictions = lambda: world.frame((0.99,), (True,))

        def check():
            return scheduler.run_single_check(DAY, [101], set(), world.config, 0.03)

        if case == "locked":
            daily = check()["pick_result"].daily
            daily.delivery_attempted = True
            picks.save_pick(daily, world.picks)
        if case == "no_predictions":
            world.predictions = lambda: None
        if case == "contest_error":
            def contest_error():
                raise contest_state.ContestStateError("fixture invalid observation")

            world.predictions = contest_error
        if case == "write_error":
            def write_error(*_):
                raise OSError("fixture write failed")

            mp.setattr(picks, "atomic_write_text", write_error)
            with pytest.raises(OSError, match="fixture write failed"):
                check()
            return
        result = check()
        expected = {"skipped", "new_lineups", "should_post", "pick_result",
                    "pick_name", "pick_p", "selection"}
        if case == "no_predictions":
            expected.add("skip_summary")
        assert result.keys() == expected
        assert not result["should_post"]
        assert (result["pick_result"] is not None) == (case == "locked")
        if case == "locked":
            assert result["pick_result"].locked


def test_reuse_defer_keeps_an_overdue_confirmed_check_available(offline):
    at = lambda hour, minute: offline.instant(DAY, hour, minute)
    with offline.offline(at(18, 25)) as (world, mp):
        world.dated(0)
        world.schedule[DAY].append(offline.game(103, at(21, 10)))
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at(hour, minute), "game_pks": [pk]}
            for hour, minute, pk in [(18, 25, 101), (18, 45, 102), (19, 25, 103)]])
        mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
            pk: {"away", "home"} for pk in pks if pk in {101, 102}})
        check = scheduler.run_single_check
        checks = []
        scheduled = False

        def observe_check(*args, **kwargs):
            nonlocal scheduled
            checks.append(world.now)
            scheduled = True
            try:
                return check(*args, **kwargs)
            finally:
                scheduled = False

        def forecast():
            if len(checks) == 1 and scheduled:
                world.advance(35 * 60)
            if len(checks) == 2 and scheduled:
                return world.frame((0.40, 0.99, 0.40), (True, False, True)).iloc[[1, 0, 2]]
            return world.frame((0.99, 0.40, 0.40), (True, True, True))

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", observe_check)
        scheduler.run_day(DAY, world.config)
        assert checks == [at(18, 25), at(19, 0)]
        state = scheduler.load_state(DAY, world.picks)
        assert state.fallback_refreshes[0]["action"] == "defer"
        assert state.fallback_refreshes[0]["reused_check_cascade"]
        assert len(state.fallback_refreshes) == 1
        assert not any(run.get("reason") == "coalesced_after_fallback"
                       for run in state.runs_completed)
        assert daily_decision.load_decision(DAY, world.picks)["primary"]["game_pk"] == 102


@pytest.mark.parametrize("cached", [False, True])
def test_standing_mdp_skip_survives_cache_and_yields_to_genuine_pick(offline, cached):
    from bts import strategy

    first = offline.instant(DAY, 17, 40)
    second = offline.instant(DAY, 18, 25)
    with offline.offline(first) as (world, mp):
        world.schedule[DAY] = world.schedule[DAY][:1]
        world.dated(0)
        world.predictions = lambda: world.frame((0.99,), (True,))
        check = scheduler.run_single_check
        assert check(DAY, [101], set(), world.config, 0.03)["pick_result"] is not None
        resolve = strategy.resolve_policy_decision
        selection_actions = []

        def genuine_skip(*args, **kwargs):
            decision = resolve(*args, **kwargs)
            return replace(decision, source="mdp") if decision.action == "skip" else decision

        def observe_check(*args, **kwargs):
            result = check(*args, **kwargs)
            selection_actions.append(result["selection"].action)
            if len(selection_actions) == 1:
                assert "lock_decision" not in result and "saved_pick_bytes" not in result
                if cached:
                    world.schedule[DAY][0]["status"].update(detailedState="Warmup", statusCode="PW")
            return result

        def forecast():
            if world.now == first:
                return world.frame((0.10,), (True,))
            if world.now == second:
                world.advance(35 * 60)
                return world.frame((0.99,), (True,))
            return None

        mp.setattr(strategy, "resolve_policy_decision", genuine_skip)
        mp.setattr(scheduler, "run_single_check", observe_check)
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at, "game_pks": [101]} for at in (first, second)])
        world.predictions = forecast
        scheduler.run_day(DAY, world.config)
        assert selection_actions == ["skip", None if cached else "single"]
        assert sum(text.startswith("Today's pick") for text in world.sends("dm")) == (not cached)
        record = daily_decision.load_decision(DAY, world.picks)
        assert record["action"] == ("skip" if cached else "single")
