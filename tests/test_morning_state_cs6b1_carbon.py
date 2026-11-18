"""Adopted adversarial regression tests for CS6b-1 (the morning state check).

Rules satisfied:
- No import of bts.model.predict or lightgbm at module level.
- No sys.addaudithook or process-wide state left installed.
- Any injected BaseException that escapes fails the test cleanly via pytest.fail,
  never terminating the pytest session.
"""
from __future__ import annotations

import datetime as dt
import json
from datetime import timedelta
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from bts import contest_state, daily_decision, dm, orchestrator, picks, scheduler, strategy
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, NOW, set_automatic, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world
from tests.test_stale_delivery_cs4c import WARNING_FILE, seed_automatic

ET = ZoneInfo("America/New_York")
START = NOW.replace(hour=10, minute=0)
FIRST = NOW.replace(hour=16, minute=10)
MORNING_FILE = "contest_morning_check_dm_delivery_status.json"


@pytest.fixture
def morning_env(world, monkeypatch):
    config, directory, now, _ = world
    now[0] = START
    prepare_run_day(monkeypatch)
    monkeypatch.setattr(
        scheduler,
        "_watchdog_ping_sleep",
        lambda secs, *a, **k: now.__setitem__(0, now[0] + timedelta(seconds=max(0.0, float(secs)))),
    )
    monkeypatch.setattr(daily_decision, "datetime", contest_state.datetime)
    monkeypatch.setattr(picks, "datetime", contest_state.datetime)
    monkeypatch.setattr(strategy, "datetime", contest_state.datetime)
    monkeypatch.setattr(scheduler, "datetime", contest_state.datetime)
    monkeypatch.setattr(scheduler, "time", Mock(monotonic=lambda: 0.0))
    orchestrator.run_cascade.return_value = (None, None)
    return config, directory, now


def _run_day_containing_base_exceptions(config: dict) -> None:
    try:
        scheduler.run_day(DAY, config)
    except BaseException as escaped:
        if isinstance(escaped, Exception):
            raise
        pytest.fail(f"non-Exception BaseException escaped run_day: {escaped!r}")


@pytest.mark.parametrize("record_kind", ["decision", "pick"])
def test_b1_deeply_nested_unreadable_decision_or_pick_counts_as_absent(
    morning_env, monkeypatch, capsys, record_kind,
):
    """A 12,000-deep JSON file in decision.json or <date>.json raises RecursionError on decode.
    Per the helper contract, an unreadable decision or pick file must count as 'nothing found',
    not raise out of _morning_state_check or suppress the morning ContestStateError DM."""
    config, directory, now = morning_env
    nested = b"[" * 12000 + b"]" * 12000

    # 1. Direct helper call: must not raise RecursionError, and must send the morning CRITICAL DM.
    direct_dir = directory.parent / "direct_sub" / "picks"
    direct_dir.mkdir(parents=True, exist_ok=True)
    direct_target = (
        daily_decision.decision_path(DAY, direct_dir)
        if record_kind == "decision"
        else direct_dir / f"{DAY}.json"
    )
    direct_target.parent.mkdir(parents=True, exist_ok=True)
    direct_target.write_bytes(nested)
    direct_cfg = {
        **config,
        "orchestrator": {**config["orchestrator"], "picks_dir": str(direct_dir)},
    }
    scheduler._morning_state_check(direct_cfg, direct_dir, DAY, START, FIRST)
    assert alert.send_dm.call_count == 1
    assert "Morning check for 2027-07-15: no usable contest state was found at startup." in (
        alert.send_dm.call_args.args[1]
    )

    # 2. Through real run_day: must log Row 2 and send the morning DM before the first check.
    alert.send_dm.reset_mock()
    capsys.readouterr()
    runday_target = (
        daily_decision.decision_path(DAY, directory)
        if record_kind == "decision"
        else directory / f"{DAY}.json"
    )
    runday_target.parent.mkdir(parents=True, exist_ok=True)
    runday_target.write_bytes(nested)

    reached = []

    def stop_at_first_check(**kwargs):
        reached.append(now[0])
        # Remove the seeded corrupt file once the first check is reached so untouched
        # post-loop readers in run_day:3283 / _maybe_alert_missed_pick:244 do not trip on it.
        runday_target.unlink(missing_ok=True)
        return {"new_lineups": 0, "skipped": True, "pick_result": None, "should_post": False}

    monkeypatch.setattr(scheduler, "run_single_check", stop_at_first_check)
    _run_day_containing_base_exceptions(config)
    err = capsys.readouterr().err
    assert reached[0] == FIRST
    assert alert.send_dm.call_count == 1
    assert "  MORNING STATE CHECK — no pick can be made yet:" in err
    assert "  Morning state check failed:" not in err


@pytest.mark.parametrize("lane", ["critical", "warn"])
def test_morning_dispatches_pass_sampled_now_et_date_not_wall_clock(
    morning_env, monkeypatch, lane,
):
    """Kills mutants that omit now_et_date=now.date() in Row 2 or Row 3 when alert.datetime
    is on a different calendar date than the sampled _now_et()."""
    config, directory, _ = morning_env
    if lane == "warn":
        seed_automatic(directory, (START - timedelta(hours=20)).isoformat(), now=START, streak=30)

    class DifferentWallDate(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            other = dt.datetime(2028, 1, 9, 12, 0, 0, tzinfo=dt.timezone.utc)
            return other.astimezone(tz) if tz is not None else other.replace(tzinfo=None)

    monkeypatch.setattr(alert, "datetime", DifferentWallDate)
    _run_day_containing_base_exceptions(config)
    status_name = MORNING_FILE if lane == "critical" else WARNING_FILE
    payload = json.loads((directory.parent / "health_state" / status_name).read_text())
    assert payload["sent_day"] == DAY
    assert alert.send_dm.call_count == 1


@pytest.mark.parametrize("permanent", [False, True])
@pytest.mark.parametrize("exc_cls", [KeyboardInterrupt, SystemExit])
def test_advisory_stderr_interrupt_propagates(
    morning_env, monkeypatch, capsys, permanent, exc_cls,
):
    """Round-1 fix (Codex B1): an interrupt raised at an advisory stderr write is not
    the morning work's to swallow; it leaves run_day before any DM or check.
    (At the reviewed commit this test asserted containment, to kill the mutants that
    narrowed BaseException to Exception; the narrowing is now the required behaviour.)"""
    config, _, now = morning_env
    events = []
    real_print = print

    def faulty_print(*args, **kwargs):
        line = str(args[0]) if args else ""
        if line.startswith("  MORNING STATE CHECK") or (
            permanent and line.startswith("  Morning state check failed:")
        ):
            events.append(("stderr_interrupt", now[0]))
            raise exc_cls("stderr signal")
        return real_print(*args, **kwargs)

    monkeypatch.setattr(scheduler, "print", faulty_print, raising=False)
    alert.send_dm.side_effect = lambda *_: events.append(("dm", now[0])) or "h1"
    orchestrator.run_cascade.side_effect = lambda *a, **k: (events.append(("check", now[0])) or (None, None))
    escaped = None
    try:
        scheduler.run_day(DAY, config)
    except BaseException as caught:  # must fail the test, not end the session
        escaped = caught
    assert isinstance(escaped, exc_cls)
    assert events == [("stderr_interrupt", START)]


@pytest.mark.parametrize("state_case", ["refusal", "dated", "fresh_auto", "stale_auto"])
def test_morning_helper_single_state_read_and_zero_scheduler_state_writes(
    morning_env, monkeypatch, state_case,
):
    """Kills mutants that call load_decision_streak_state twice (including on Rows 3 & 4)
    or write scheduler_state.json inside _morning_state_check."""
    config, directory, now = morning_env
    if state_case == "dated":
        set_dated(directory, now=START, streak=19, best=25)
    elif state_case == "fresh_auto":
        set_automatic(directory, now=START, streak=19)
    elif state_case == "stale_auto":
        seed_automatic(directory, (START - timedelta(hours=20)).isoformat(), now=START, streak=19)

    reads = []
    real_load = contest_state.load_decision_streak_state

    def spy_load(*args, **kwargs):
        reads.append(now[0])
        return real_load(*args, **kwargs)

    save_calls = []
    real_save_state = scheduler.save_state

    def spy_save_state(st, pdir):
        save_calls.append(now[0])
        return real_save_state(st, pdir)

    monkeypatch.setattr(contest_state, "load_decision_streak_state", spy_load)
    monkeypatch.setattr(scheduler, "save_state", spy_save_state)

    observed_at_first_check = []

    def on_first_cascade(*a, **k):
        if not observed_at_first_check:
            observed_at_first_check.append((list(reads), list(save_calls)))
        return None, None

    orchestrator.run_cascade.side_effect = on_first_cascade
    _run_day_containing_base_exceptions(config)

    startup_reads, startup_saves = observed_at_first_check[0]
    assert startup_reads == [START]
    # Exactly one save_state at START: run_day's own initial save before the morning block.
    assert startup_saves == [START]


def test_row3_future_and_unreadable_details_and_row2_truncation(
    morning_env, monkeypatch, capsys,
):
    config, directory, _ = morning_env
    early_target = START.replace(hour=9, minute=5)

    # Row 3 future
    fut = (START + timedelta(minutes=10)).isoformat()
    seed_automatic(directory, fut, now=START, streak=10)
    scheduler._morning_state_check(config, directory, DAY, START, early_target)
    assert f"its recorded_at {fut} is in the future" in alert.send_dm.call_args.args[1]
    assert f"  MORNING STATE CHECK — stale contest reading: its recorded_at {fut} is in the future" in (
        capsys.readouterr().err.splitlines()
    )

    # Row 3 unreadable (next day to avoid same-day dedup)
    next_start = START + timedelta(days=1)
    next_day = next_start.date().isoformat()
    seed_automatic(directory, "2027-07-16T10:00:00", now=next_start, streak=10)
    scheduler._morning_state_check(config, directory, next_day, next_start, early_target + timedelta(days=1))
    assert "it has no usable recorded_at" in alert.send_dm.call_args.args[1]
    assert "  MORNING STATE CHECK — stale contest reading: it has no usable recorded_at" in (
        capsys.readouterr().err.splitlines()
    )

    # Row 2 leading-zero HH:MM and >800-char multiline non-ASCII journal truncation
    long_err = "PATH=/secret/dir\nsecond line — café " + ("Z" * 1000)
    monkeypatch.setattr(
        contest_state,
        "load_decision_streak_state",
        Mock(side_effect=contest_state.ContestStateError(long_err)),
    )
    capsys.readouterr()
    scheduler._morning_state_check(config, directory, DAY, START, early_target)
    dm_body = alert.send_dm.call_args.args[1]
    assert "PATH=/secret/dir" not in dm_body
    assert dm_body.endswith("First planned lineup check at 09:05 ET.")
    err_lines = capsys.readouterr().err.splitlines()
    assert err_lines == [f"  MORNING STATE CHECK — no pick can be made yet: {ascii(long_err)[:800]}"]


@pytest.mark.parametrize("kind", ["generator_exit", "base_group", "unprintable_base", "unprintable"])
def test_exotic_base_exceptions_propagate_and_unprintable_errors_are_isolated(
    morning_env, monkeypatch, kind,
):
    """Round-1 fix (Codex B1): BaseExceptions that are not Exceptions leave run_day;
    an ordinary exception is contained even when it cannot be printed.
    (At the reviewed commit all of these were contained.)"""
    config, _, now = morning_env

    class BrokenStrBase(BaseException):
        def __str__(self) -> str:
            raise RuntimeError("str failed")

    class BrokenStrError(Exception):
        def __str__(self) -> str:
            raise RuntimeError("str failed")

    exc = {
        "generator_exit": GeneratorExit("gen"),
        "base_group": BaseExceptionGroup("grp", [KeyboardInterrupt("inner")]),
        "unprintable_base": BrokenStrBase(),
        "unprintable": BrokenStrError(),
    }[kind]
    injected = []

    def boom(*a, **k):
        if not injected:
            injected.append(now[0])
            raise exc
        return None, None

    monkeypatch.setattr(contest_state, "load_decision_streak_state", boom)
    reached = []
    orchestrator.run_cascade.side_effect = lambda *a, **k: (reached.append(now[0]) or (None, None))
    escaped = None
    try:
        scheduler.run_day(DAY, config)
    except BaseException as caught:  # must fail the test, not end the session
        escaped = caught
    assert injected == [START]
    if kind == "unprintable":
        assert escaped is None and reached[0] == FIRST
    else:
        assert escaped is exc and reached == []
