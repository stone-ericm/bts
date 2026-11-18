"""Independent timestamp, pick-wire and scheduler replay probes."""

import argparse
import hashlib
import io
import itertools
import json
import math
import random
import tempfile
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from bts import contest_state, dm, orchestrator, picks, posting, scheduler
from bts.daily_decision import load_decision
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world
from tests.test_stale_delivery_cs4c import OLD, WARNING_FILE, envelope, fresh, seed_automatic


SCRATCH = Path(__file__).resolve().parent
UTC = timezone.utc
ET = ZoneInfo("America/New_York")
SCHEMA = "bts_pick_automatic_reading_v1"
KEYS = ("schema", "recorded_at", "age_hours", "stale", "reason")


def exact_string(text):
    return isinstance(text, str) and len(text) <= 128 and all(32 <= ord(c) <= 126 for c in text)


def utc(text):
    if not exact_string(text):
        return None
    try:
        parsed = datetime.fromisoformat(text)
        return parsed.astimezone(UTC) if parsed.utcoffset() is not None else None
    except (ValueError, OverflowError):
        return None


def expected(kind, text, now):
    if kind != "automatic":
        return None
    parsed = utc(text)
    if parsed is None:
        age, reason = None, "unreadable"
    else:
        diff = now.astimezone(UTC) - parsed
        ticks = (diff.days * 86_400 + diff.seconds) * 10**6 + diff.microseconds
        age = round(ticks / 3_600_000_000, 2)
        reason = "old" if ticks > 50_400_000_000 else "future" if ticks < -300_000_000 else None
    return dict(zip(KEYS, (SCHEMA, text if exact_string(text) else None, age, reason is not None, reason)))


def expected_valid(value):
    if type(value) is not dict or set(value) != set(KEYS):
        return None
    schema, text, age, stale, reason = (value[key] for key in KEYS)
    if schema != SCHEMA or (text is not None and not exact_string(text)):
        return None
    if type(stale) is not bool or stale != (reason is not None):
        return None
    parsed = utc(text)
    if reason == "unreadable":
        return {key: value[key] for key in KEYS} if parsed is None and age is None else None
    limits = [(None, -0.08, 14.0), ("old", 14.0, 87_660_000.0),
              ("future", -87_660_000.0, -0.08)]
    if parsed is not None and type(age) is float and math.isfinite(age) and round(age, 2) == age:
        for label, lower, upper in limits:
            if reason == label and lower <= age <= upper:
                return {key: value[key] for key in KEYS}
    return None


def pure_cases():
    malformed = [
        "", "bad", "2027-07-15T18:35", "2027-07-15\n18:35Z", "2027-07-15\ud80018:35Z",
        "2027-07-15\r18:35Z", "2027-07-15\x0018:35Z", "2027-07-15\x1f18:35Z",
        "2027-07-15\x7f18:35Z", "2027-07-15é18:35Z", "a" * 128, "a" * 129,
        "0001-01-01T00:00:00+00:01", "9999-12-31T23:59:59.999999-00:01",
        None, 0, True, {}, [], 0.0,
    ]
    for kind, text in itertools.product(("automatic", "dated", "legacy_manual", "model", "other"),
                                       malformed + ["2027-07-14T22:35:00-04:00"]):
        yield kind, text, NOW
    for limit, nearby in itertools.product((timedelta(hours=14), timedelta(minutes=-5)), range(-1, 2)):
        then = NOW.astimezone(UTC) - limit - timedelta(microseconds=nearby)
        yield "automatic", then.isoformat().removesuffix("+00:00") + "Z", NOW
    for now in (datetime(2027, 3, 14, 3, 30, tzinfo=ET),
                datetime(2027, 11, 7, 1, 30, tzinfo=ET, fold=1),
                datetime.min.replace(tzinfo=UTC), datetime.max.replace(tzinfo=UTC)):
        for text in ("2027-03-13T13:30:00-05:00", "2027-11-07T01:30:00-04:00",
                     datetime.min.isoformat() + "Z", datetime.max.isoformat() + "Z"):
            yield "automatic", text, now
    rng = random.Random(466)
    begin = datetime.min.replace(tzinfo=UTC)
    for _ in range(5000):
        then = begin + timedelta(days=rng.randrange(1, 3_652_058), seconds=rng.randrange(86_400),
                                 microseconds=rng.randrange(10**6))
        now = begin + timedelta(days=rng.randrange(1, 3_652_058), seconds=rng.randrange(86_400),
                                microseconds=rng.randrange(10**6))
        offset = timezone(timedelta(minutes=rng.randint(-1439, 1439)))
        yield "automatic", then.astimezone(offset).isoformat(), now


def pure_and_wire(root):
    state = contest_state.DecisionStreakState(7, False, True, "contest", "fresh", 0, False)
    all_inputs = list(pure_cases())
    truths = [expected(*case) for case in all_inputs]
    no_io_or_clock = Mock(side_effect=AssertionError("pure helper used I/O or clock"))

    class Clock(datetime):
        now = today = utcnow = no_io_or_clock

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(contest_state, "datetime", Clock)
        patch.setattr(Path, "stat", no_io_or_clock)
        patch.setattr(Path, "open", no_io_or_clock)
        patch.setattr("builtins.open", no_io_or_clock)
        for (kind, text, now), truth in zip(all_inputs, truths, strict=True):
            supplied = replace(state, observation_kind=kind, contest_recorded_at=text)
            result = contest_state.automatic_reading_record(supplied, now)
            assert result == truth, (text, now, result, truth)
            if result is not None:
                assert tuple(result) == KEYS
                assert type(result["stale"]) is bool
                assert result["age_hours"] is None or type(result["age_hours"]) is float
                assert contest_state.valid_automatic_reading(result) == result, result
    no_io_or_clock.assert_not_called()
    print(f"{len(all_inputs)} pure inputs and generated producer/validator round trips; no I/O or clock")

    supplied = replace(state, observation_kind="automatic", contest_recorded_at=OLD["recorded_at"])
    print("Outside the now precondition: naive gives",
          contest_state.automatic_reading_record(supplied, NOW.replace(tzinfo=None))["reason"])
    try:
        contest_state.automatic_reading_record(
            supplied, datetime.min.replace(tzinfo=timezone(timedelta(hours=1))),
        )
    except OverflowError:
        print("Outside the now precondition: nonrepresentable UTC instant raises OverflowError")
    else:
        raise AssertionError("expected out-of-contract overflow")

    cases = [None, False, 5, "str", [], {}, dict(OLD, extra=1),
             *(dict((k, v) for k, v in OLD.items() if k != key) for key in KEYS),
             *(dict(OLD, schema=value) for value in (None, [], 1, True, "wrong"))]
    for text, age, stale, reason in itertools.product(
        (None, "", "bad", OLD["recorded_at"], "2027-07-15T18:35", [], "x" * 129, "\ud800",
         "0001-01-01T00:00:00+00:01", "9999-12-31T23:59:59-02:00"),
        (None, True, False, 0, 1, 20, [], -87_660_000.01, -87_660_000.0, -0.09, -0.08,
         -0.07, -0.0, 0.0, 0.001, 13.99, 14.0, 14.01, 87_660_000.0, 87_660_000.01,
         math.inf, -math.inf, math.nan),
        (False, True, None, 0, 1, 1.0),
        (None, "old", "future", "unreadable", "", [], {}),
    ):
        cases.append(dict(zip(KEYS, (SCHEMA, text, age, stale, reason))))
    body = asdict(daily_pick())
    normal = root / f"{DAY}.json"
    shadow = root / f"{DAY}.shadow.json"
    for case in cases:
        want = expected_valid(case)
        held = dict(reversed(list(case.items()))) if isinstance(case, dict) else case
        raw = json.dumps(dict(body, automatic_reading=held)).encode()
        normal.write_bytes(raw)
        shadow.write_bytes(raw)
        from_bytes, exact = picks.load_pick_bytes(DAY, root)
        assert exact == raw
        for loaded in (picks.load_pick(DAY, root), from_bytes, picks.load_shadow_pick(DAY, root)):
            assert loaded.automatic_reading == want, case
            if want is not None:
                assert tuple(loaded.automatic_reading) == KEYS
    print(f"{len(cases)} decoded validator shapes through each of the three pick loaders")

    for prior_fields in (("chosen_on", "automatic_reading"), ("automatic_reading",)):
        before = json.dumps({k: v for k, v in body.items() if k not in prior_fields}, indent=2).encode()
        normal.write_bytes(before)
        loaded = picks.load_pick(DAY, root)
        tail = {key: None for key in prior_fields}
        assert picks.save_pick(loaded, root).read_bytes() == before[:-2] + b",\n" + (
            json.dumps(tail, indent=2).encode()[2:]
        )
    shadow_saved = picks.save_shadow_pick(replace(daily_pick(), automatic_reading=OLD), root)
    assert list(json.loads(shadow_saved.read_bytes()))[-2:] == ["chosen_on", "automatic_reading"]
    assert picks.load_shadow_pick(DAY, root).automatic_reading == OLD
    print("Canonical pre-5a and 5a exact bytes, normal and shadow tail order")

    from bts.experiment.artifacts import load_production_pick_snapshot
    from bts.state.export import _pick_to_historical
    snapshots, history = [], []
    for data in (dict(body, automatic_reading=None), dict(body, automatic_reading=OLD)):
        raw = json.dumps(data).encode()
        normal.write_bytes(raw)
        snapshots.append(load_production_pick_snapshot(normal, expected_date=DAY))
        assert snapshots[-1]["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert snapshots[-1]["production_pick_json"] == data
        history.append(_pick_to_historical(data))
    assert snapshots[0]["slots"] == snapshots[1]["slots"]
    assert snapshots[0]["source_sha256"] != snapshots[1]["source_sha256"]
    assert history[0] == history[1]
    print("Experiment raw snapshot and hash record the field, normalized slots and historical export match")


def replay_day(root, case):
    with pytest.MonkeyPatch.context() as patch:
        config, directory, clock, _ = world.__wrapped__(root, patch)
        config["scheduler"]["entry_intent"] = "enter"
        prepare_run_day(patch)
        patch.setattr(scheduler, "notify_ready", Mock())
        patch.setattr(scheduler, "notify_watchdog", Mock())
        frame, tier = orchestrator.run_cascade.return_value
        prediction = (frame.assign(p_game_hit=[0.99, 0.81]), tier)
        orchestrator.run_cascade.return_value = prediction
        sleeps = []
        if case == "dated":
            set_dated(directory, streak=30, best=30)
        else:
            if case == "turns-stale":
                clock[0] -= timedelta(minutes=10)
                recorded = clock[0] - timedelta(hours=13, minutes=55)
                prediction = (prediction[0].assign(flags=["PROJECTED", "PROJECTED"],
                                                   projected_lineup=[True, True]), tier)
                orchestrator.run_cascade.side_effect = [prediction, (None, None)]

                def sleep(seconds):
                    sleeps.append(seconds)
                    assert 0 <= seconds < 3600
                    clock[0] += timedelta(seconds=seconds)

                patch.setattr(scheduler, "_watchdog_ping_sleep", sleep)
            else:
                recorded = clock[0] - timedelta(hours={"old": 20, "fresh": 10}[case])
            seed_automatic(directory, recorded.isoformat(), clock[0], streak=30)

        delivered = Mock(wraps=scheduler._deliver_and_lock_pick)
        patch.setattr(scheduler, "_deliver_and_lock_pick", delivered)
        scheduler.run_day(date=DAY, config=config)
        loaded = picks.load_pick(DAY, directory)
        assert loaded.pick.batter_id == 680757 and picks.pick_was_delivered(loaded)
        assert scheduler.load_state(DAY, directory).pick_locked
        assert load_decision(DAY, directory)["delivery_status"] == "delivered"
        want = {"dated": None, "fresh": None, "old": "old", "turns-stale": "old"}[case]
        assert (loaded.automatic_reading or {}).get("reason") == want
        assert (loaded.automatic_reading is None) is (case == "dated")
        assert dm.send_dm.call_count == 1 and alert.send_dm.call_count == int(want is not None)
        assert not posting.post_to_bluesky.called
        if case == "turns-stale":
            assert sleeps and delivered.call_args.args[5] == "fallback"
            assert delivered.call_args.kwargs["selection"] is None
            assert loaded.automatic_reading["age_hours"] == 14.08
        scheduler.run_day(date=DAY, config=config)
        assert dm.send_dm.call_count == 1 and alert.send_dm.call_count == int(want is not None)
        assert picks.load_pick(DAY, directory).automatic_reading == loaded.automatic_reading
        print(f"{case}: {loaded.automatic_reading}, fallback sleeps={sleeps}; restart sent nothing")


def snapshot_race(root, mode):
    with pytest.MonkeyPatch.context() as patch:
        config, directory, clock, _ = world.__wrapped__(root, patch)
        config["scheduler"]["pick_delivery"] = mode
        start = NOW - timedelta(minutes=1)
        recorded = (NOW - timedelta(hours=14, seconds=1)).isoformat()
        seed_automatic(directory, recorded)
        reads = []
        real_read = contest_state.load_decision_streak_state
        real_compute = contest_state.automatic_reading_record

        def read(*args, **kwargs):
            value = real_read(*args, **kwargs)
            reads.append(value)
            seed_automatic(directory, NOW.isoformat(), streak=12)
            return value

        compute = Mock(wraps=real_compute)
        patch.setattr(contest_state, "load_decision_streak_state", read)
        patch.setattr(contest_state, "automatic_reading_record", compute)
        times = iter((start, NOW))
        patch.setattr(scheduler, "_now_et", lambda: next(times, clock[0]))
        daily = daily_pick()
        assert scheduler._deliver_and_lock_pick(
            daily, config, directory, scheduler_state(), DAY, "snapshot", selection=fresh(daily),
        )
        assert len(reads) == 1 and compute.call_count == 1
        assert compute.call_args.args[0] is reads[0] and compute.call_args.args[1] is NOW
        stamp = picks.load_pick(DAY, directory).automatic_reading
        assert stamp["reason"] == "old" and stamp["age_hours"] == 14.0
        assert stamp["recorded_at"] == recorded
        print(f"{mode}: exactly one read, source replaced on disk, final sample yields {stamp}")


def notification_edges(root):
    for mode in ("dm", "public"):
        for value in (None, ""):
            with pytest.MonkeyPatch.context() as patch:
                config, directory, _, _ = world.__wrapped__(root / f"{mode}-{value!r}", patch)
                config["scheduler"]["pick_delivery"] = mode
                seed_automatic(directory)
                calls = []
                adapter = dm if mode == "dm" else posting
                if mode == "dm":
                    patch.setattr(dm, "send_dm", ORIGINAL_DM)
                    patch.setattr(dm, "get_dm_password", lambda: "offline")
                    responses = [{"accessJwt": "test"}, {"did": "did:test"},
                                 {"convo": {"id": "test"}}, {"id": value}]
                else:
                    patch.setattr(posting, "post_to_bluesky", ORIGINAL_POST)
                    patch.setattr(posting, "get_bluesky_password", lambda: "offline")
                    responses = [{"accessJwt": "test", "did": "did:test"}, {"uri": value}]
                replies = iter(responses)

                def capture(request, **kwargs):
                    calls.append((request.full_url, json.loads(request.data) if request.data else None))
                    return io.BytesIO(json.dumps(next(replies)).encode())

                patch.setattr(adapter, "retry_urlopen", capture)
                daily = daily_pick()
                assert scheduler._deliver_and_lock_pick(
                    daily, config, directory, scheduler_state(), DAY, "null id", selection=fresh(daily),
                )
                saved = picks.load_pick(DAY, directory)
                assert len(calls) == len(responses)
                assert alert.send_dm.call_args.args == ("offline.test", envelope())
                print(f"{mode}: actual adapter parsed {value!r}, warning sent, "
                      f"delivered predicate={picks.pick_was_delivered(saved)}")

    with pytest.MonkeyPatch.context() as patch:
        first = root / "dm-first"
        config, directory, clock, statuses = world.__wrapped__(first, patch)
        from tests.test_stale_delivery_cs4c import independently_deliver
        seed_automatic(directory)
        daily = daily_pick()
        assert scheduler._deliver_and_lock_pick(
            daily, config, directory, scheduler_state(), DAY, "dedup first", selection=fresh(daily),
        )
        original = alert.dispatch_dm_for_health_alerts
        counted_dispatch = Mock(wraps=original)
        patch.setattr(alert, "dispatch_dm_for_health_alerts", counted_dispatch)

        class OtherDay(datetime):
            @classmethod
            def now(cls, tz=None):
                return (NOW + timedelta(days=100)).astimezone(tz)

        patch.setattr(alert, "datetime", OtherDay)
        independently_deliver((config, directory.with_name("public-second"), clock, statuses))
        assert alert.send_dm.call_count == 1
        assert counted_dispatch.call_args.kwargs["now_et_date"] == NOW.date()
        clock[0] += timedelta(days=1)
        independently_deliver((config, directory.with_name("public-third"), clock, statuses))
        assert dm.send_dm.call_count == 1 and posting.post_to_bluesky.call_count == 2
        assert alert.send_dm.call_count == 2
        status = directory.parent / "health_state" / WARNING_FILE
        assert json.loads(status.read_bytes())["sent_day"] == clock[0].date().isoformat()
        print("Independent DM and public picks same ET date plus next day; dispatcher clock skewed +100 days")


ORIGINAL_DM = dm.send_dm
ORIGINAL_POST = posting.post_to_bluesky


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("probe", choices=["wire", "daily", "notifications"])
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(dir=SCRATCH, prefix=f"{args.probe}-") as folder:
        root = Path(folder)
        if args.probe == "wire":
            pure_and_wire(root)
        elif args.probe == "daily":
            for case in ("old", "fresh", "dated", "turns-stale"):
                replay_day(root / case, case)
            for mode in ("dm", "public"):
                snapshot_race(root / mode, mode)
        else:
            notification_edges(root)


if __name__ == "__main__":
    main()
