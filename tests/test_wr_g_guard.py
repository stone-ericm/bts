"""Change set G against the shipped artifacts and the unmodified base."""
import csv
from collections import Counter
from dataclasses import fields, replace
from datetime import date, timedelta
from functools import cache
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from bts import daily_decision, picks, scheduler, strategy
from bts.simulate import mdp
from scripts.audit.season_walk import walk

CONTRACT = Path("docs/reviews/2026-11-18-whole-range/contract-g")
END = date(2027, 9, 27)
GUARD = dict(rule="monotone_skip_v1", bin=4, from_bin=3, table_action="skip")


@cache
def base_module(name, revision="f83e0d0b"):
    path = f"{revision}:src/bts/{name}.py"
    source = subprocess.run(["git", "show", path], check=True, capture_output=True, timeout=5).stdout
    module = ModuleType(f"wr_g_{revision}_{name.replace('/', '_')}")
    sys.modules[module.__name__] = module
    exec(compile(source, path, "exec"), vars(module))
    return module


@pytest.fixture(scope="module")
def artifact():
    return strategy._load_mdp()


@pytest.fixture(scope="module")
def affected():
    with (CONTRACT / "affected-states.csv").open() as stream:
        return {
            tuple(int(row[key]) for key in ("streak", "days_left", "saver", "bin")):
            (row["guarded_action"], dict(GUARD, bin=int(row["bin"]), from_bin=int(row["from_bin"])))
            for row in csv.DictReader(stream)
        }


@pytest.fixture(scope="module")
def probabilities(artifact):
    edges = [0.0, *artifact["boundaries"], 1.0]
    return [
        (lo, (lo + hi) / 2, np.nextafter(hi, lo) if hi < 1.0 else 0.999)
        for lo, hi in zip(edges, edges[1:])
    ]


def test_regenerate_csv(tmp_path):
    output = tmp_path / "affected.csv"
    subprocess.run([sys.executable, str(CONTRACT / "enumerate_states.py"), ".", str(output)],
                   check=True, capture_output=True, timeout=10)
    assert output.read_bytes() == (CONTRACT / "affected-states.csv").read_bytes()


def test_raw_and_guarded_lookup_grid(artifact, affected, probabilities):
    raw = base_module("simulate/mdp").lookup_action
    args = artifact["policy_table"], artifact["boundaries"]
    calls, changes = 0, 0
    for s, d, saver, q in product(range(58), range(181), (False, True), range(5)):
        hit = affected.get((s, d, saver, q))
        for p in probabilities[q]:
            expected = raw(*args, s, d, saver, p)
            assert mdp.lookup_action(*args, s, d, saver, p) == expected
            actual = mdp.lookup_action_guarded(*args, s, d, saver, p)
            assert actual == (hit or (expected, None))
            calls += 1
            changes += actual[0] != expected
    assert (calls // 3, calls, changes) == (104980, 314940, 4533)
    for s, d in ((58, 180), (10, -1)):
        assert mdp.lookup_action_guarded(None, None, s, d, False, 0.99) == ("skip", None)


def context(p, day, artifact, streak):
    return strategy.DecisionContext(p, 0.95, True, day, True, artifact, streak, "trusted")


def test_affected_rows_and_independent_clamps(artifact, affected, probabilities):
    assert len(affected) == 1511
    counts = Counter()
    for (s, d, saver, q), (action, guard) in affected.items():
        p = probabilities[q][1]
        assert mdp.lookup_action_guarded(
            artifact["policy_table"], artifact["boundaries"], s, d, saver, p
        ) == (action, guard)
        days = [END - timedelta(days=d)]
        if d == 180:
            days.extend(date(2027, 3, n) for n in range(24, 31))
        for day in days:
            ctx = context(p, day.isoformat(), artifact, s)
            actual = strategy.resolve_policy_decision(ctx, s, saver)
            assert (actual.action, actual.policy_action, actual.objective, actual.source,
                    actual.pick_bar, actual.guard) == (action, action, "reach57", "mdp", None, guard)
            assert type(actual.guard["bin"]) is type(actual.guard["from_bin"]) is int
            counts["decisions"] += 1
            if action == "double":
                for clamped in (replace(ctx, allow_double=False),
                                replace(ctx, has_diff_game=False, second_p=None)):
                    actual = strategy.resolve_policy_decision(clamped, s, saver)
                    assert (actual.action, actual.policy_action, actual.guard) == ("single", "double", guard)
                counts["doubles"] += 1
    assert counts == {"decisions": 1567, "doubles": 184}


@pytest.mark.parametrize("start,stop", [(0, 46), (46, 93), (93, 140), (140, 188)])
def test_resolver_and_direct_bar_grid(start, stop, artifact, affected, probabilities, monkeypatch):
    base = base_module("strategy")
    for module in (strategy, base):
        monkeypatch.setattr(module, "_days_remaining", cache(module._days_remaining))
    field_names = [field.name for field in fields(base.PolicyDecision)]
    seen, days = set(), set()
    calls, changed = 0, 0
    for offset in range(start, stop):
        day = (date(2027, 3, 24) + timedelta(days=offset)).isoformat()
        for s, saver in product(range(58), (False, True)):
            bar = strategy.effective_pick_bar(s, day, saver, artifact, s, "trusted")
            old_bar = base.effective_pick_bar(s, day, saver, artifact, s, "trusted")
            assert bar == old_bar
            for q, group in enumerate(probabilities):
                for p in group:
                    ctx = context(p, day, artifact, s)
                    old = base.resolve_policy_decision(ctx, s, saver)
                    result = strategy.resolve_policy_decision(ctx, s, saver)
                    days.add(result.days_effective)
                    key = s, old.days_effective, saver, q
                    hit = affected.get(key) if old.objective == "reach57" else None
                    shared = {name: getattr(old, name) for name in field_names}
                    if hit:
                        action, guard = hit
                        assert old.action == old.policy_action == "skip" and old.pick_bar is not None
                        shared.update(action=action, policy_action=action, pick_bar=None)
                        assert result.guard == guard
                        seen.add(key)
                        changed += 1
                    else:
                        assert getattr(result, "guard", None) is None
                    assert {name: getattr(result, name) for name in field_names} == shared
                    if old.objective == "reach57":
                        if bar is None:
                            assert result.action == "skip"
                        elif p >= bar:
                            assert result.action != "skip"
                        if p == bar:
                            assert old.action != "skip" and result.guard is None
                    calls += 1
    effective_days = {min(180, 187 - offset) for offset in range(start, stop)}
    assert days == effective_days
    assert seen == {key for key in affected if key[1] in effective_days}
    assert calls == (stop - start) * 58 * 2 * 5 * 3
    repeats = 56 if start == 0 else 0
    assert changed == 3 * (len(seen) + repeats)


def test_lookup_edge_shapes():
    table = np.zeros((58, 181, 2, 4), dtype=np.int8)
    table[10, 100, 0] = [0, 2, 0, 0]
    lookup = mdp.lookup_action_guarded
    assert lookup(table, [0.8, 0.8], 10, 100, False, 0.8) == (
        "double", dict(GUARD, bin=2, from_bin=1))
    assert lookup(table, [], 10, 100, False, 0.99) == ("skip", None)
    with pytest.raises(TypeError):
        lookup(table, None, 10, 100, False, 0.99)
    assert lookup(table[..., :2], [0.8, 0.8], 10, 100, False, 0.79) == ("skip", None)
    with pytest.raises(IndexError):
        lookup(table[..., :2], [0.8, 0.8], 10, 100, False, 0.8)
    table[10, 100, 0] = [2, 1, 0, 0]
    for q, p in ((2, 0.85), (3, 0.95), (2, 0.85)):
        assert lookup(table, [0.7, 0.8, 0.9], 10, 105, False, p, 100) == (
            "single", dict(GUARD, bin=q, from_bin=1))
    assert lookup(table, [0.7], 10, 100, False, 0.95) == ("single", None)


@pytest.mark.parametrize("s,d,p,saver,allow,second,expected", [
    (50, 180, 0.83, False, True, True, ""),
    (50, 180, 0.96, False, True, True, "single/single"),
    (1, 29, 0.72, False, True, True, "double/double"),
    (1, 29, 0.72, False, False, True, "double/single"),
    (1, 29, 0.72, False, True, False, "double/single"),
])
def test_guard_journal(artifact, probabilities, capsys, s, d, p, saver, allow, second, expected):
    if expected.startswith("double"):
        p = probabilities[1][1]
    ctx = replace(context(p, (END - timedelta(days=d)).isoformat(), artifact, s),
                  allow_double=allow, has_diff_game=second)
    dec = strategy.resolve_policy_decision(ctx, s, saver)
    scheduler._log_policy_decision(SimpleNamespace(decision=dec, saver_available=saver))
    lines = capsys.readouterr().err.splitlines()
    if not expected:
        assert lines == []
        return
    policy_action, action = expected.split("/")
    q, j = (1, 0) if policy_action == "double" else (4, 3)
    assert lines == [
        f"  POLICY GUARD — At streak {s} with {d} policy days left and saver input unavailable, "
        f"the table skips bin {q}. Bin {j} supplies {policy_action}. Selected action is {action}."
    ]


def test_decision_round_trip_and_shadow(tmp_path):
    day = "2027-03-24"
    primary = {"batter_id": 100, "game_pk": 1, "p_game_hit": 0.96}
    for action, guard, expected in (("single", GUARD, GUARD), ("single", None, None),
                                     ("skip", GUARD, None)):
        record = daily_decision.write_decision(
            day, tmp_path, action=action, source="mdp", primary=primary, policy_guard=guard,
            delivery_status="delivered" if action == "single" else "not_applicable",
            scoreable=action == "single")
        assert record["policy_guard"] == expected
        for revision in ("f83e0d0b", "619abda7"):
            reader = base_module("daily_decision", revision)
            read = reader.load_decision(day, tmp_path)
            assert read["policy_guard"] == expected
            assert read["primary"]["batter_id"] == 100
            assert read["scoreable"] == (action == "single")
        assert daily_decision.is_reach57_mdp_skip(record) == (action == "skip")
    path = daily_decision.decision_path(day, tmp_path)
    legacy = dict(record)
    del legacy["policy_guard"]
    path.write_text(json.dumps(legacy))
    assert daily_decision.load_decision(day, tmp_path).get("policy_guard") is None
    assert path.read_text() == json.dumps(legacy)


DAY = "2027-03-24"
NOW = walk.REAL_DATETIME(2027, 3, 24, 16, tzinfo=walk.ET)


class Interrupt(BaseException):
    pass


def state(**extra):
    values = dict(date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
                  confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                  pick_locked_at=None, result_status=None, next_wakeup=None)
    return scheduler.SchedulerState(**(values | extra))


def select(root, p=0.96, day=DAY, streak=50):
    predictions = walk._predictions(date.fromisoformat(day), p)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in predictions.game_pk}
    return strategy.select_pick(predictions, day, root, streak=streak, saver_available=False,
                                game_statuses_detailed=statuses, require_detailed_statuses=True)


def guarded_pick(root):
    selection = select(root)
    assert selection.decision.action == "single"
    assert selection.decision.guard == GUARD
    return selection.pick_result.daily, selection


def decision(root, status):
    record = daily_decision.load_decision(DAY, root)
    assert record["delivery_status"] == status
    assert record["scoreable"]
    assert record["policy_guard"] == GUARD
    return record


def classify(root, daily, **kwargs):
    scheduler._write_classification_decision(
        root, DAY, action="single", primary=scheduler._row_from_daily(daily.pick),
        double_down=None, state=state(), daily=daily, **kwargs)


def test_pick_readers_and_fresh_recovered_commits(tmp_path):
    with walk.offline(NOW):
        daily, selection = guarded_pick(tmp_path)
        path = picks.save_pick(daily, tmp_path)
        original = path.read_bytes()
        for revision in ("f83e0d0b", "619abda7"):
            loaded = base_module("picks", revision).load_pick(DAY, tmp_path)
            assert loaded.policy_decision["guard"] == GUARD
            assert loaded.pick.batter_id == daily.pick.batter_id
            assert loaded.double_down is None
        legacy = json.loads(original)
        del legacy["policy_decision"]["guard"]
        path.write_text(json.dumps(legacy))
        assert picks.load_pick(DAY, tmp_path).policy_decision.get("guard") is None
        assert path.read_text() == json.dumps(legacy)
        path.write_bytes(original)
        scheduler._commit_decision_for_pick(
            tmp_path, DAY, daily, selection=selection, delivery_status="private_locked", state=state())
        fresh = decision(tmp_path, "private_locked")
        daily_decision.decision_path(DAY, tmp_path).unlink()
        classify(tmp_path, picks.load_pick(DAY, tmp_path), delivered=True)
        recovered = decision(tmp_path, "delivered")
        for key in ("policy_guard", "action", "primary"):
            assert recovered[key] == fresh[key]
        assert recovered["double_down"] is None
        assert (fresh["streak"], fresh["saver_available"]) == (50, False)
        assert (recovered["streak"], recovered["saver_available"]) == (None, None)


def test_interrupt_before_save_and_atomic_replace(tmp_path, monkeypatch):
    with walk.offline(NOW):
        daily, _ = guarded_pick(tmp_path)
        path = picks.save_pick(daily, tmp_path)
        before = path.read_bytes()

        def interrupt(*args, **kwargs):
            raise Interrupt

        monkeypatch.setattr("bts.util.os.replace", interrupt)
        with pytest.raises(Interrupt):
            picks.save_pick(daily, tmp_path)
        assert path.read_bytes() == before
        assert daily_decision.load_decision(DAY, tmp_path) is None
        empty = tmp_path / "empty"
        selected, _ = guarded_pick(empty)
        assert selected.policy_decision["guard"] == GUARD
        assert picks.load_pick(DAY, empty) is None
        with pytest.raises(Interrupt):
            picks.save_pick(selected, empty)
        assert picks.load_pick(DAY, empty) is None
        assert daily_decision.load_decision(DAY, empty) is None


@pytest.fixture
def local_delivery(monkeypatch):
    for name in ("_trigger_live_forward_capture_on_lock", "_alert_delivery_outcome_unknown"):
        monkeypatch.setattr(scheduler, name, lambda *args, **kwargs: None)
    monkeypatch.setattr(scheduler, "_void_committed_games", lambda *args: {})
    monkeypatch.setattr(scheduler, "_now_et", lambda: NOW)


@pytest.mark.parametrize("recovery", ["direct", "classify", "empty_dm", "cached", "private_interrupt"])
def test_delivery_and_restart(tmp_path, monkeypatch, local_delivery, recovery):
    from contextlib import nullcontext
    from unittest.mock import Mock
    from bts import dm

    config = {"orchestrator": {"picks_dir": str(tmp_path)},
              "bluesky": {"dm_recipient": "probe"}, "scheduler": {"pick_delivery": "dm"}}
    with walk.offline(NOW):
        daily, _ = guarded_pick(tmp_path)
        picks.save_pick(daily, tmp_path)

        def deliver(daily):
            return scheduler._deliver_and_lock_pick(daily, config, tmp_path, state(), DAY, "probe")

        if recovery == "cached":
            config["scheduler"]["pick_delivery"] = "private"
            monkeypatch.setattr(scheduler, "run_and_pick", lambda *args, **kwargs: (None, None, None))
            monkeypatch.setattr(scheduler, "heartbeat_watchdog", lambda *args, **kwargs: nullcontext())
            refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, daily, 0.03)
            assert refresh.selection is None
            assert deliver(refresh.daily)
            decision(tmp_path, "private_locked")
            return
        if recovery == "private_interrupt":
            config["scheduler"]["pick_delivery"] = "private"
            monkeypatch.setattr(scheduler, "_trigger_live_forward_capture_on_lock",
                                Mock(side_effect=Interrupt))
            with pytest.raises(Interrupt):
                deliver(daily)
            loaded = picks.load_pick(DAY, tmp_path)
            classify(tmp_path, loaded, delivered=False)
            assert loaded.policy_decision["guard"] == GUARD
            assert daily_decision.load_decision(DAY, tmp_path) is None
            return
        if recovery == "empty_dm":
            monkeypatch.setattr(dm, "send_dm", Mock(return_value=""))
            assert deliver(daily)
            assert not picks.pick_was_delivered(picks.load_pick(DAY, tmp_path))
            decision(tmp_path, "delivered")
            return
        monkeypatch.setattr(dm, "send_dm", Mock(side_effect=RuntimeError("probe failed send")))
        assert not deliver(daily)
        daily = picks.load_pick(DAY, tmp_path)
        assert not daily.delivery_attempted
        assert daily_decision.load_decision(DAY, tmp_path) is None
        monkeypatch.setattr(dm, "send_dm", Mock(side_effect=Interrupt))
        with pytest.raises(Interrupt):
            deliver(daily)
        daily = picks.load_pick(DAY, tmp_path)
        assert daily.delivery_attempted
        assert daily_decision.load_decision(DAY, tmp_path) is None
        transport = Mock()
        monkeypatch.setattr(dm, "send_dm", transport)
        if recovery == "direct":
            assert not deliver(daily)
        else:
            classify(tmp_path, daily, delivered=False, attempted=True)
        transport.assert_not_called()
        decision(tmp_path, "locked_unconfirmed")


def test_restart_after_decision_before_flag_save(tmp_path, monkeypatch):
    from unittest.mock import Mock

    with walk.offline(NOW):
        daily, selection = guarded_pick(tmp_path)
        picks.save_pick(daily, tmp_path)
        scheduler.save_state(state(), tmp_path)
        with monkeypatch.context() as crash:
            crash.setattr(scheduler, "save_state", Mock(side_effect=Interrupt))
            with pytest.raises(Interrupt):
                scheduler._commit_decision_for_pick(
                    tmp_path, DAY, daily, selection=selection,
                    delivery_status="delivered", state=state())
        path = daily_decision.decision_path(DAY, tmp_path)
        before = path.read_bytes()
        restored = scheduler.load_state(DAY, tmp_path)
        assert not restored.committed_pick_written
        scheduler._write_endofday_skip(
            tmp_path, DAY, state(final_skip_candidate={"source": "mdp"}))
        assert path.read_bytes() == before
        classify(tmp_path, picks.load_pick(DAY, tmp_path), delivered=True)
        assert path.read_bytes() == before
        scheduler._commit_decision_for_pick(
            tmp_path, DAY, daily, selection=None,
            delivery_status="locked_unconfirmed", state=state())
        assert path.read_bytes() == before
        decision(tmp_path, "delivered")


def test_unaffected_controls(artifact, probabilities):
    base = base_module("strategy")
    for injected, day, s, p in (
        (artifact, "2027-09-17", 0, 0.96),
        (dict(artifact, tail=None, tail_error="corrupt"), "2027-09-17", 0, 0.96),
        (None, "2027-09-17", 0, 0.96),
        (None, DAY, 1, 0.96),
        (None, DAY, 1, 0.72),
        (artifact, (END - timedelta(days=29)).isoformat(), 1, probabilities[0][1]),
    ):
        for allow, second in product((True, False), repeat=2):
            ctx = replace(context(p, day, injected, s), allow_double=allow, has_diff_game=second)
            current = strategy.resolve_policy_decision(ctx, s, False)
            old = base.resolve_policy_decision(ctx, s, False)
            assert all(getattr(current, f.name) == getattr(old, f.name) for f in fields(old))
            assert current.guard is None


def test_corrupt_directory_and_fifo(tmp_path, monkeypatch):
    import os
    import signal

    corrupt, directory, fifo = (tmp_path / name for name in ("corrupt", "directory", "fifo"))
    corrupt.write_bytes(b"invalid NPZ")
    directory.mkdir()
    os.mkfifo(fifo)
    for path, error in ((corrupt, "ValueError"), (directory, "IsADirectoryError")):
        monkeypatch.setattr(mdp, "DEFAULT_POLICY_PATH", path)
        monkeypatch.setattr(strategy, "_mdp_cache", None)
        artifact = strategy._load_mdp()
        assert error in artifact["base_error"]
        assert artifact["tail"] is None and "unverifiable" in artifact["tail_error"]
        resolved = strategy.resolve_policy_decision(
            context(0.96, "2027-09-17", artifact, 0), 0, False)
        assert resolved.action == "single" and resolved.degraded_reason
        assert resolved.guard is None

    def deadline(*_):
        raise Interrupt("loader blocked at unwritten FIFO")

    monkeypatch.setattr(mdp, "DEFAULT_POLICY_PATH", fifo)
    monkeypatch.setattr(strategy, "_mdp_cache", None)
    original = signal.signal(signal.SIGALRM, deadline)
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.15)
        with pytest.raises(Interrupt, match="unwritten FIFO"):
            strategy._load_mdp()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)


def test_year_boundary(artifact, tmp_path):
    import pandas as pd

    for day, raw, effective in (("2026-12-31", 0, 0), ("2027-01-01", 269, 180)):
        assert strategy._days_remaining(day) == raw
        assert strategy.resolve_policy_decision(
            context(0.96, day, artifact, 50), 50, False).days_effective == effective
        now = walk.REAL_DATETIME.fromisoformat(day + "T12:00:00").replace(tzinfo=walk.ET)
        with walk.offline(now):
            no_games = strategy.select_pick(pd.DataFrame(), day, tmp_path, streak=50,
                                            saver_available=False)
            assert no_games.pick_result is None
            if day == "2027-01-01":
                assert select(tmp_path, day=day).action == "single"
    assert walk.boundary_checks()["switch_2027_at_2028_01_01_0001"].startswith("refused")
