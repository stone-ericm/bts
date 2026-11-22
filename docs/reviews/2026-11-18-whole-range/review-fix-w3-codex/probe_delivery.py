"""Run W3 delivery guards, transport outcomes, and durable interruption recovery."""

from datetime import timedelta
import json
import sys

from bts import contest_state, daily_decision, dm, picks, scheduler, strategy

from review_world import DAY, at, events, game, health, offline, pending_window


IS_BASE = "base/src" in scheduler.__file__
EXPECT_CLEAR_ERROR_ALERT = IS_BASE or "scratch/repair/src" in scheduler.__file__


def one_game(world, mp, date, start):
    world.schedule[date] = world.schedule[date][:1]
    mp.setattr(scheduler, "plan_run_times", lambda *_args, **_kwargs: [
        {"time_et": start, "game_pks": [101]},
    ])


def cutoff(mode, crossing):
    with offline(at(18, 25)) as (world, mp):
        one_game(world, mp, DAY, world.now)
        world.dated(0)
        world.config["scheduler"]["pick_delivery"] = mode
        forecasts = []
        tried_markers = []
        refused = []
        read = contest_state.load_decision_streak_state
        write = picks.atomic_write_text
        original = scheduler._refuse_late_delivery

        def forecast():
            forecasts.append(world.now)
            if len(forecasts) == 1:
                world.advance(2399)
                return world.frame((0.99,), (True,))
            if crossing == "attempt":
                world.advance(2)
            return None

        def after_read(*args, **kwargs):
            reading = read(*args, **kwargs)
            if crossing == "read" and world.now == at(19, 4, 59):
                world.advance(1)
            return reading

        def after_write(file, data):
            write(file, data)
            if file == world.picks / f"{DAY}.json":
                marker = json.loads(data)["delivery_attempted"]
                tried_markers.append(marker)
                if crossing == "attempt" and marker:
                    world.advance(1.25)

        def refusal(daily, config, directory, state, date, label, deadline, now):
            refused.append((deadline, now, daily.delivery_attempted))
            return original(daily, config, directory, state, date, label, deadline, now)

        world.predictions = forecast
        mp.setattr(contest_state, "load_decision_streak_state", after_read)
        mp.setattr(picks, "atomic_write_text", after_write)
        mp.setattr(scheduler, "_refuse_late_delivery", refusal)
        scheduler.run_day(DAY, world.config)
        print("CUTOFF", json.dumps({"base": IS_BASE, "mode": mode, "crossing": crossing,
                                    "forecasts": [t.isoformat() for t in forecasts],
                                    "markers": tried_markers,
                                    "refusal": [(d.isoformat(), n.isoformat(), flag) for d, n, flag in refused],
                                    "health": health(world)}), flush=True)
        assert not world.sends("dm") and not world.sends("post")
        assert not (world.picks / f"{DAY}.json").exists()
        assert daily_decision.load_decision(DAY, world.picks) is None
        assert len(refused) == 1 and refused[0][0] == at(19, 5) and not refused[0][2]
        assert sum("[late_delivery]" in msg for msg in health(world)) == 1
        assert tried_markers == ([False, True, False] if not IS_BASE and crossing == "attempt" else [False])


def cutoff_clear_error(mode):
    with offline(at(18, 25)) as (world, mp):
        world.dated(30)
        world.config["scheduler"]["pick_delivery"] = mode
        mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
            {"time_et": at(18, 25), "game_pks": [101]},
            {"time_et": at(19, 15), "game_pks": [102]},
        ])
        real_write = picks.atomic_write_text
        forecasts = []
        marker_crossed = False
        clear_failed = False

        def forecast():
            forecasts.append(world.now)
            if len(forecasts) == 1:
                world.advance(2399)
                return world.frame((0.99, 0.30), (True, True))
            if IS_BASE and len(forecasts) == 2:
                world.advance(2)
            return None

        def write(file, text):
            nonlocal marker_crossed, clear_failed
            if file == world.picks / f"{DAY}.json":
                marker = json.loads(text)["delivery_attempted"]
                if marker_crossed and not marker and not clear_failed:
                    clear_failed = True
                    raise OSError("fixture EIO on first known-unsent marker clear")
                real_write(file, text)
                if marker and not marker_crossed:
                    marker_crossed = True
                    world.advance(1.25)
            else:
                real_write(file, text)

        world.predictions = forecast
        mp.setattr(picks, "atomic_write_text", write)
        scheduler.run_day(DAY, world.config)
        kinds = sorted(file.name.rsplit("_", 1)[0] for file in (world.picks / DAY).glob("*.json"))
        print("CLEAR_ERROR", json.dumps({"base": IS_BASE, "mode": mode, "marker_crossed": marker_crossed,
                                         "clear_failed": clear_failed, "artifacts": kinds,
                                         "health": health(world), "pick_dm": events(world, "dm"),
                                         "post": events(world, "post")}), flush=True)
        assert not world.sends("dm") and not world.sends("post")
        assert daily_decision.load_decision(DAY, world.picks) is None
        assert not (world.picks / f"{DAY}.json").exists()
        assert marker_crossed is clear_failed is not IS_BASE
        assert any("[late_delivery]" in msg for msg in health(world)) is EXPECT_CLEAR_ERROR_ALERT
        assert any("stale_pick" in name for name in kinds) is not EXPECT_CLEAR_ERROR_ALERT


def calendar(day, crossing):
    start = at(23, 20, day=day)
    with offline(start) as (world, mp):
        one_game(world, mp, day, start)
        world.schedule[day][0] = game(101, start + timedelta(minutes=70))
        world.config["scheduler"].update(fallback_deadline_min=40, fallback_deadline_min_morning=40)
        mp.setattr(strategy, "_load_mdp", lambda: None)
        if crossing in {"automatic", "double-roll"}:
            world.automatic(0, age_hours=15)
        else:
            world.dated(0)
        original_check = scheduler.run_single_check
        original_write = picks.atomic_write_text
        original_read = contest_state.load_decision_streak_state
        dial = False
        rolled = False
        frames = 0
        readings = []
        markers = []

        def forecast():
            nonlocal frames
            frames += 1
            if frames == 1:
                world.advance(2399)
                return world.frame((0.99,), (True,))
            return None

        def check(*args, **kwargs):
            nonlocal dial
            result = original_check(*args, **kwargs)
            assert picks.submission_cutoff_et(result["pick_result"].daily).isoformat() == (
                start + timedelta(minutes=65)).isoformat()
            dial = True
            if crossing == "before-read":
                world.advance(2)
            return result

        def read(*args, **kwargs):
            nonlocal rolled
            if dial:
                readings.append((world.now.isoformat(), kwargs.get("now") is not None))
            result = original_read(*args, **kwargs)
            if dial and crossing in {"during-read", "automatic", "double-roll"} and not rolled:
                world.advance(2)
                rolled = True
            if dial and crossing == "double-roll" and len(readings) == 2:
                world.now += timedelta(days=1)
            return result

        def write(file, data):
            nonlocal rolled
            original_write(file, data)
            if file == world.picks / f"{day}.json":
                marker = json.loads(data)["delivery_attempted"]
                markers.append(marker)
                if crossing == "marker" and marker and not rolled:
                    rolled = True
                    world.advance(2)

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", check)
        mp.setattr(contest_state, "load_decision_streak_state", read)
        mp.setattr(picks, "atomic_write_text", write)
        scheduler.run_day(day, world.config)
        outcome = events(world, "dm")
        record = daily_decision.load_decision(day, world.picks)
        print("CALENDAR", json.dumps({"date": day, "case": crossing, "readings": readings,
                                      "markers": markers, "send": outcome, "health": health(world)}), flush=True)
        if crossing == "automatic":
            assert len(outcome) == 1 and (start + timedelta(days=1)).date().isoformat() in outcome[0][0]
            assert readings[0][0][:10] != readings[1][0][:10] and record["scoreable"]
        else:
            assert not outcome and record is None
            assert sum("[contest_state]" in msg for msg in health(world)) >= 1
            if crossing == "double-roll":
                assert any("ET date changed before contest state could be validated" in msg
                           for msg in health(world))
                assert markers == [False]
            else:
                assert not picks.load_pick(day, world.picks).delivery_attempted
            if crossing == "marker":
                assert markers == [False, True, False]
                assert any("ET date changed after the final contest-state read" in msg
                           for msg in health(world))


def interruption(error_type, stage):
    start = at(18, 25)
    with offline(start) as (world, mp):
        world.dated(30)
        if stage.endswith("poll"):
            pending_window(world, mp, scheduler)
        else:
            one_game(world, mp, DAY, start)
        starts = []
        interrupted = False
        sent = dm.send_dm
        real_confirm = scheduler.check_confirmed_lineups
        status = scheduler._delivery_game_statuses
        format_text = scheduler._format_pick_delivery_text
        commit = scheduler._commit_decision_for_pick

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(22 * 60 if stage.endswith("poll") else 35 * 60)
                return world.frame((0.99, 0.70), (True, True))
            return None

        def raise_once(*_args, **_kwargs):
            nonlocal interrupted
            interrupted = True
            raise error_type("fixture interruption")

        def confirm(game_pks):
            if stage == "confirmation-poll" and game_pks == [102] and starts and not interrupted:
                return raise_once()
            return {pk: {"away", "home"} for pk in game_pks if pk == 101}

        def statuses(date):
            if stage == "status-poll" and starts and not interrupted:
                return raise_once()
            return status(date)

        world.predictions = forecast
        if stage == "pre-marker":
            mp.setattr(scheduler, "_format_pick_delivery_text", raise_once)
        if stage == "post-marker":
            mp.setattr(dm, "send_dm", raise_once)
        if stage == "post-confirmation":
            mp.setattr(scheduler, "_commit_decision_for_pick", raise_once)
        if stage.endswith("poll"):
            mp.setattr(scheduler, "check_confirmed_lineups", confirm)
            mp.setattr(scheduler, "_delivery_game_statuses", statuses)
        try:
            scheduler.run_day(DAY, world.config)
        except error_type:
            assert interrupted
        else:
            raise AssertionError("scheduler swallowed the interruption")
        before = picks.load_pick(DAY, world.picks)
        before_bytes = (world.picks / f"{DAY}.json").read_bytes()
        prior = scheduler.load_state(DAY, world.picks).fallback_refreshes or []
        assert before.delivery_attempted == (stage in {"post-marker", "post-confirmation"})
        assert picks.pick_was_delivered(before) == (stage == "post-confirmation")
        assert daily_decision.load_decision(DAY, world.picks) is None
        mp.setattr(dm, "send_dm", sent)
        mp.setattr(scheduler, "_format_pick_delivery_text", format_text)
        mp.setattr(scheduler, "_commit_decision_for_pick", commit)
        mp.setattr(scheduler, "check_confirmed_lineups", real_confirm)
        mp.setattr(scheduler, "_delivery_game_statuses", status)
        assert (world.picks / f"{DAY}.json").read_bytes() == before_bytes
        scheduler.run_day(DAY, world.config)
        after = scheduler.load_state(DAY, world.picks).fallback_refreshes or []
        record = daily_decision.load_decision(DAY, world.picks)
        print("INTERRUPT", json.dumps({"stage": stage, "exception": error_type.__name__,
                                       "first_history": prior, "after_history": after,
                                       "send": events(world, "dm"), "health": health(world),
                                       "record": record and record["delivery_status"]}), flush=True)
        assert sum(r["reused_check_cascade"] for r in after) == sum(r["reused_check_cascade"] for r in prior)
        assert record["delivery_status"] == ("locked_unconfirmed" if stage == "post-marker" else "delivered")
        assert len(world.sends("dm")) == (0 if stage == "post-marker" else 1)
        assert any("OUTCOME UNKNOWN" in msg for msg in health(world)) == (stage == "post-marker")


def outcomes(case):
    with offline(at(18, 25)) as (world, mp):
        one_game(world, mp, DAY, world.now)
        starts = []
        marks = []
        attempted = []
        real_check = scheduler.run_single_check
        real_write = picks.atomic_write_text
        real_dm = dm.send_dm
        world.config["scheduler"]["pick_delivery"] = "public" if case == "stale-public" else "dm"
        if case in {"best", "saver"}:
            world.dated(0)
        else:
            world.automatic(0, age_hours=15)

        def forecast():
            starts.append(world.now)
            if len(starts) == 1:
                world.advance(35 * 60)
                return world.frame((0.99,), (True,))
            return None

        def check(*args, **kwargs):
            result = real_check(*args, **kwargs)
            if case == "best":
                world.dated(0, best=1)
            if case == "saver":
                world.dated(0, saver="not_earned")
            return result

        def write(file, text):
            real_write(file, text)
            if file == world.picks / f"{DAY}.json":
                marks.append(json.loads(text)["delivery_attempted"])

        def send(*args):
            attempted.append(world.now)
            if case == "no-message-id":
                return None
            if case == "failure" and len(attempted) == 1:
                assert not health(world)
                raise RuntimeError("caught send failure in fixture")
            return real_dm(*args)

        world.predictions = forecast
        mp.setattr(scheduler, "run_single_check", check)
        mp.setattr(picks, "atomic_write_text", write)
        mp.setattr(dm, "send_dm", send)
        scheduler.run_day(DAY, world.config)
        record = daily_decision.load_decision(DAY, world.picks)
        daily = picks.load_pick(DAY, world.picks)
        facts = {"base": IS_BASE, "case": case, "starts": [start.isoformat() for start in starts],
                 "dm": events(world, "dm"), "post": events(world, "post"), "attempts": len(attempted),
                 "marks": marks, "status": record and record["delivery_status"],
                 "health": health(world), "delivered": picks.pick_was_delivered(daily)}
        print("OUTCOME", json.dumps(facts), flush=True)
        if case in {"best", "saver"}:
            assert not facts["delivered"] and record is None and not any(marks)
            assert sorted(line.split("]")[0] for line in health(world)) == ["- [contest_state", "- [missed_pick"]
        elif case == "no-message-id":
            assert not facts["delivered"] and facts["attempts"] == 1 and marks == [False, True, True]
            assert not health(world)
        else:
            assert facts["delivered"] and record["scoreable"] and len(facts["dm"] + facts["post"]) == 1
            assert sum("stale" in line.lower() for line in health(world)) == 1
            if case == "failure":
                assert facts["attempts"] == 2 and marks == [False, True, False, True, True]


if __name__ == "__main__":
    if sys.argv[1] == "cutoff":
        for delivery_mode in ["dm", "public"]:
            for crossing in ["attempt", "read"]:
                cutoff(delivery_mode, crossing)
    elif sys.argv[1] == "clear-error":
        for delivery_mode in ["dm", "public"]:
            cutoff_clear_error(delivery_mode)
    elif sys.argv[1] == "calendar":
        for date in ["2027-06-18", "2026-12-31"]:
            for crossing in ["before-read", "during-read", "marker"]:
                calendar(date, crossing)
        calendar("2027-06-18", "automatic")
        calendar("2027-06-18", "double-roll")
    elif sys.argv[1] == "interrupt":
        for error in [KeyboardInterrupt, SystemExit]:
            for stage in ["pre-marker", "post-marker", "post-confirmation", "confirmation-poll", "status-poll"]:
                interruption(error, stage)
    else:
        for case in ["best", "saver", "stale-dm", "stale-public", "failure", "no-message-id"]:
            outcomes(case)
