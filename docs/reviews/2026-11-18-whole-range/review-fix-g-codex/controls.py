"""Policy boundaries, unusual artifacts, and interruption before a durable save."""
from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import signal
import sys

import numpy as np

from bts import daily_decision, orchestrator, picks, strategy
from bts.simulate import mdp, tail_policy
from commit_paths import DAY, data, setup
from scripts.audit.season_walk import walk

side = sys.argv[1]
END = date(2027, 9, 27)
ARTIFACT = strategy._load_mdp()
GUARDED = hasattr(mdp, 'lookup_action_guarded')


def resolve(s, d, p=.96, saver=False, artifact=ARTIFACT, best=56):
    day = (END - timedelta(days=d)).isoformat()
    return strategy.resolve_policy_decision(strategy.DecisionContext(p, .95, True, day, True,
                                            artifact, best, 'trusted'), s, saver)


def snap(result):
    return (result.objective, result.policy_action, result.action, result.pick_bar,
            bool(result.degraded_reason), getattr(result, 'guard', None))


def get(action, *args):
    try:
        return action(*args)
    except (IndexError, TypeError, ValueError) as exc:
        return type(exc).__name__


def variants():
    cases = ((44, 6), (45, 6), (46, 6), (56, 1), (57, 1), (58, 1), (56, 0))
    print('regime/edge controls:', [snap(resolve(s, d, p=.81)) for s, d in cases])
    assert [resolve(s, d).objective for s, d in cases[:3]] == ['emax_season_best', 'reach57', 'reach57']
    assert [resolve(s, d).action for s, d in cases[-3:]] == ['skip'] * 3
    assert resolve(50, 180, .60, artifact=None).action == 'skip'
    assert resolve(50, 180, .96, artifact=None).source == 'heuristic'
    table = ARTIFACT['policy_table'].copy()
    table[44, 6, 0] = [0, 0, 0, 1, 0]
    forced_tail_hole = ARTIFACT | {'policy_table': table}
    tail_control = resolve(44, 6, artifact=forced_tail_hole)
    assert snap(tail_control) == snap(resolve(44, 6)) and tail_control.action == 'skip'
    print('synthetic tail hole:', snap(tail_control))
    for value in (0, 1, -1, '0', '1', np.bool_(True), 2, 'false'):
        args = (ARTIFACT['policy_table'], ARTIFACT['boundaries'], 50, 180, value, .96)
        print('non-bool', repr(value), get(mdp.lookup_action, *args),
              get(mdp.lookup_action_guarded, *args) if GUARDED else None)


def artifact_shapes(root):
    lookup = mdp.lookup_action_guarded if GUARDED else mdp.lookup_action
    shaped = np.zeros((58, 181, 2, 4), dtype=np.int8)
    shaped[10, 100, 0, :] = [0, 2, 0, 0]
    results = [get(lookup, t, edges, 10, 100, False, p)
               for t, edges, p in [(shaped, [0.8, 0.8], .8), (shaped[..., :2], [0.8, 0.8], .79),
                                  (shaped[..., :2], [0.8, 0.8], .8), (shaped, [], .9),
                                  (shaped, None, .9)]]
    assert results[2] == 'IndexError' and results[4] == 'TypeError'
    assert (results[0][0] if GUARDED else results[0]) == ('double' if GUARDED else 'skip')
    print('tied, narrow below, narrow edge, empty, None:', results)
    for name in ('missing', 'corrupt', 'directory'):
        path = root / name
        if name == 'corrupt':
            path.write_bytes(b'invalid NPZ')
        elif name == 'directory':
            path.mkdir()
        with patch.object(mdp, 'DEFAULT_POLICY_PATH', path), patch.object(strategy, '_mdp_cache', None):
            result = strategy._load_mdp()
        assert result['base_error'] and 'unverifiable' in result['tail_error'] and result['tail'] is None
        forced = [resolve(44, 6, artifact=result, best=value) for value in (44, 56)]
        assert [r.action for r in forced] == ['single', 'skip'] and all(r.degraded_reason for r in forced)
        assert all(getattr(r, 'guard', None) is None for r in forced)
        print(name, result['base_error'].split(':')[0], [snap(r) for r in forced])
    with patch.object(mdp, 'DEFAULT_POLICY_PATH', root/'missing'), patch.object(strategy, '_mdp_cache', None), \
         patch.object(tail_policy, 'DEFAULT_TAIL_POLICY_PATH', root/'missing-tail'):
        assert strategy._load_mdp() is None
    malformed = root / 'malformed.npz'
    shaped[10, 100, 0, :] = [1, 3, 0, 0]
    np.savez(malformed, policy_table=shaped, boundaries=[.7, .8], season_length=180)
    with patch.object(mdp, 'DEFAULT_POLICY_PATH', malformed), patch.object(strategy, '_mdp_cache', None):
        inherited = strategy._load_mdp()
    assert inherited['base_error'] is None
    outcome = get(resolve, 10, 100, .8, False, inherited, 10)
    assert (outcome if GUARDED else outcome.action) == ('IndexError' if GUARDED else 'skip')
    print('loaded row [single,invalid,skip]:', outcome)
    fifo = root / 'fifo'
    os.mkfifo(fifo)
    def alarm(*_):
        raise InterruptedError('loader open timed out')
    original = signal.signal(signal.SIGALRM, alarm)
    try:
        with patch.object(mdp, 'DEFAULT_POLICY_PATH', fifo), patch.object(strategy, '_mdp_cache', None):
            # Raise BaseException so _load_mdp cannot treat the deadline as an artifact error.
            def stop(*_):
                raise KeyboardInterrupt('unwritten FIFO')
            signal.signal(signal.SIGALRM, stop)
            signal.setitimer(signal.ITIMER_REAL, .15)
            try:
                strategy._load_mdp()
            except KeyboardInterrupt:
                print('unwritten FIFO blocked until 150 ms deadline')
            else:
                raise AssertionError('FIFO unexpectedly returned')
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)


def midnight_and_interrupt(root):
    import pandas as pd

    for at in (walk.REAL_DATETIME(2026, 12, 31, 23, 59, tzinfo=walk.ET),
               walk.REAL_DATETIME(2027, 1, 1, 0, 1, tzinfo=walk.ET)):
        with walk.offline(at):
            day = at.date().isoformat()
            raw = strategy._days_remaining(day)
            ctx = strategy.DecisionContext(.96, .95, True, day, True, ARTIFACT, 50, 'trusted')
            dec = strategy.resolve_policy_decision(ctx, 50, False)
            assert (raw, dec.days_effective) == ((0, 0) if at.year == 2026 else (269, 180))
            assert strategy.select_pick(pd.DataFrame(), day, root, streak=50).pick_result is None
            frame = walk._predictions(at.date(), .96)
            statuses = {pk: {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame.game_pk}
            if at.year == 2027:
                sel = strategy.select_pick(frame, day, root, streak=50, saver_available=False,
                                           game_statuses_detailed=statuses, require_detailed_statuses=True)
                assert sel.action == ('single' if GUARDED else 'skip')
            print('ET boundary:', at, raw, dec.days_effective, dec.action)
    boundary = walk.boundary_checks()
    assert not boundary['problems'] and boundary['switch_2027_at_2028_01_01_0001'].startswith('refused')
    if GUARDED:
        with ExitStack() as stack:
            cfg, _ = setup(stack, root, data(.96))
            original = strategy.resolve_policy_decision
            def cut(*args):
                selected = original(*args)
                assert selected.guard is not None
                raise KeyboardInterrupt('after lookup')
            with patch.object(strategy, 'resolve_policy_decision', cut):
                try:
                    orchestrator.run_and_pick(cfg, DAY)
                except KeyboardInterrupt:
                    assert picks.load_pick(DAY, root) is None and daily_decision.load_decision(DAY, root) is None
                else:
                    raise AssertionError('unobserved interruption')
            print('interrupted after the real lookup: no pick or decision')


with TemporaryDirectory() as tmp:
    root = Path(tmp)
    variants()
    artifact_shapes(root)
    midnight_and_interrupt(root)
