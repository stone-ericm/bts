"""Compare unaffected decisions, invalid artifacts and standing conditions to base."""
from contextlib import ExitStack
from dataclasses import asdict
from datetime import date, timedelta
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from bts import daily_decision, orchestrator, picks, strategy
from bts.simulate import mdp, tail_policy
from commit_lifecycle import DAY, data, setup
from scripts.audit.season_walk import walk

HERE = Path(__file__).resolve().parent
END = date(2027, 9, 27)
ARTIFACT = strategy._load_mdp()
GUARDED = hasattr(mdp, "lookup_action_guarded")


class Interrupt(BaseException):
    pass


def decision(s, d, p, artifact, best=56, saver=False, allow=True, second=True):
    day = (END - timedelta(days=d)).isoformat()
    ctx = strategy.DecisionContext(p, .92 if second else None, second, day, allow, artifact, best, "trusted")
    return strategy.resolve_policy_decision(ctx, s, saver)


def shared(value):
    data = asdict(value)
    assert data.pop("guard", None) is None
    return data


def unaffected(root):
    cases = []
    for s, d in [(44, 6), (45, 6), (46, 6), (57, 5), (58, 5), (56, 0)]:
        cases.append(shared(decision(s, d, .81, ARTIFACT)))
    assert [case["objective"] for case in cases[:3]] == ["emax_season_best", "reach57", "reach57"]
    assert [case["action"] for case in cases[-3:]] == ["skip"] * 3
    for p in (.6, .96):
        cases.append(shared(decision(50, 180, p, None)))
    assert cases[-2]["action"] == "skip" and cases[-1]["source"] == "heuristic"
    for allow in (False, True):
        for second in (False, True):
            cases.append(shared(decision(1, 29, .79, ARTIFACT, allow=allow, second=second)))
    assert cases[-1]["action"] == "double"

    table = ARTIFACT["policy_table"].copy()
    table[44, 6, 0] = [0, 0, 0, 1, 0]
    tail_hole = ARTIFACT | {"policy_table": table}
    assert shared(decision(44, 6, .96, tail_hole)) == shared(decision(44, 6, .96, ARTIFACT))
    assert decision(44, 6, .96, tail_hole).action == "skip"
    for kind in ("missing", "corrupt", "directory"):
        path = root / kind
        if kind == "corrupt":
            path.write_bytes(b"invalid npz")
        if kind == "directory":
            path.mkdir()
        with patch.object(mdp, "DEFAULT_POLICY_PATH", path), patch.object(strategy, "_mdp_cache", None):
            broken = strategy._load_mdp()
        assert broken["tail"] is None and "pairing unverifiable" in broken["tail_error"]
        assert broken["base_error"].startswith({"missing": "FileNotFoundError",
                                               "corrupt": "ValueError",
                                               "directory": "IsADirectoryError"}[kind])
        for best, expected in [(44, "single"), (56, "skip")]:
            actual = shared(decision(44, 6, .96, broken, best))
            assert actual["action"] == expected and actual["degraded_reason"]
            # Absolute fixture paths differ across interpreters.
            actual["degraded_reason"] = actual["degraded_reason"].replace(str(root), "<fixture>")
            cases.append(actual)
    fifo = root / "fifo"
    os.mkfifo(fifo)

    def timed_out(*_):
        raise Interrupt("unwritten FIFO")

    original = signal.signal(signal.SIGALRM, timed_out)
    try:
        with patch.object(mdp, "DEFAULT_POLICY_PATH", fifo), patch.object(strategy, "_mdp_cache", None):
            signal.setitimer(signal.ITIMER_REAL, .15)
            try:
                strategy._load_mdp()
            except Interrupt:
                print("Unwritten base FIFO blocked for 150 ms")
            else:
                raise AssertionError("Unwritten FIFO returned unexpectedly")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)
    return cases


def odd_tables(root):
    def inspect(lookup, table, edges, p, saver=False, streak=10, days=100, season=180):
        try:
            return lookup(table, edges, streak, days, saver, p, season)
        except (ValueError, IndexError, TypeError) as exc:
            return type(exc).__name__

    table = np.zeros((58, 181, 2, 4), dtype=np.int8)
    table[10, 100, 0] = [0, 2, 0, 0]
    cases = [
        ((table, [.8, .8], .8), "skip", ("double", 2, 1)),
        ((table[..., :2], [.8, .8], .79), "skip", None),
        ((table[..., :2], [.8, .8], .8), "IndexError", None),
        ((table, [], .9), "skip", None),
        ((table, None, .9), "TypeError", None),
        ((table, [.8, .8], .8, False, 10, 250, 100), "skip", ("double", 2, 1)),
        ((None, None, .8, 2, 58), "skip", None),
        ((None, None, .8, 2, 10, -1), "skip", None),
    ]
    table = table.copy()
    table[10, 100, 0] = [1, 3, 0, 0]
    path = root / "malformed-donor.npz"
    np.savez(path, policy_table=table, boundaries=[.7, .8], season_length=180)
    with patch.object(mdp, "DEFAULT_POLICY_PATH", path), patch.object(strategy, "_mdp_cache", None):
        loaded = strategy._load_mdp()
    assert loaded["base_error"] is None
    cases += [
        ((loaded["policy_table"], loaded["boundaries"], .8), "skip", ("single", 2, 0)),
        ((loaded["policy_table"], loaded["boundaries"], .7), "IndexError", None),
    ]
    for args, expected, intended_change in cases:
        assert inspect(mdp.lookup_action, *args) == expected
        if GUARDED:
            actual = inspect(mdp.lookup_action_guarded, *args)
            if intended_change:
                action, q, j = intended_change
                assert actual == (action, {"rule": "monotone_skip_v1", "table_action": "skip",
                                           "bin": q, "from_bin": j})
            else:
                assert actual == (expected if expected.endswith("Error") else (expected, None))
    for saver in (0, 1, -1, "0", "1", np.bool_(True), 2, "false"):
        args = ARTIFACT["policy_table"], ARTIFACT["boundaries"], .96
        raw = inspect(mdp.lookup_action, *args, saver, 50, 180)
        if saver == 2 or saver == "false":
            assert raw == ("IndexError" if saver == 2 else "ValueError")
        else:
            assert raw == inspect(mdp.lookup_action, *args, bool(int(saver)), 50, 180)
        if GUARDED:
            assert inspect(mdp.lookup_action_guarded, *args, saver, 50, 180) == (
                raw if raw.endswith("Error") else inspect(mdp.lookup_action_guarded, *args, bool(int(saver)), 50, 180))
    print("Tied, narrow, missing boundaries, out-of-range saver and queried-cell errors checked")


def clocks_and_interrupt(root):
    for at, expected in [(walk.REAL_DATETIME(2026, 12, 31, 23, 59, tzinfo=walk.ET), (0, 0)),
                         (walk.REAL_DATETIME(2027, 1, 1, 0, 1, tzinfo=walk.ET), (269, 180))]:
        with walk.offline(at):
            day = at.date().isoformat()
            ctx = strategy.DecisionContext(.96, None, False, day, True, ARTIFACT, 50, "trusted")
            dec = strategy.resolve_policy_decision(ctx, 50, False)
            assert (strategy._days_remaining(day), dec.days_effective) == expected
            assert strategy.select_pick(pd.DataFrame(), day, root, streak=50).pick_result is None
            assert dec.action == ("single" if GUARDED and at.year == 2027 else "skip")
            if dec.action == "single":
                assert dec.guard["bin"] == 4 and dec.guard["from_bin"] == 3
            print(at.isoformat(), expected, "synthetic decision", dec.action, "empty slate has no pick")
    result = walk.boundary_checks()
    assert not result["problems"] and result["switch_2027_at_2028_01_01_0001"].startswith("refused")
    assert result["switch_2027_at_2027_12_31_2359"].startswith("accepted")
    with ExitStack() as stack:
        config, _ = setup(stack, root, data(.96))
        original = strategy.resolve_policy_decision

        def interrupt_after_lookup(*args):
            assert original(*args).action == ("single" if GUARDED else "skip")
            raise Interrupt("after lookup")

        with patch.object(strategy, "resolve_policy_decision", interrupt_after_lookup):
            try:
                orchestrator.run_and_pick(config, DAY)
            except Interrupt:
                assert picks.load_pick(DAY, root) is None
                assert daily_decision.load_decision(DAY, root) is None
            else:
                raise AssertionError("Expected interruption")
    return result


if __name__ == "__main__":
    side = sys.argv[1]
    with TemporaryDirectory() as temp:
        root = Path(temp)
        observations = unaffected(root)
        odd_tables(root)
        clocks = clocks_and_interrupt(root)
    (HERE / f"controls-{side}.json").write_text(json.dumps([observations, clocks], sort_keys=True))
    print(side, "controls completed")
