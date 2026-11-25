import argparse
import json
import os
import signal
from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from bts import cli, contest_state as cs, daily_decision, picks, scheduler, strategy
from bts.health import contest_state as health
from bts.saver_state import load_saver_state
from probe_live import at, cfg_for, fetch, inputs, model_and_switch, select, sched_state, state, wrong_case
from probe_validators import call, invoke
from scripts.audit.season_walk import walk as w

DAY = date(2027, 9, 26)
HIGH = hasattr(cs, 'MAX_SEASON_STREAK')


def rollover():
    day = date(2027, 12, 31)
    with TemporaryDirectory() as tmp, w.offline(at(day, 23, 59)):
        base = Path(tmp)
        for kind in ('automatic', 'dated'):
            directory = base / kind / 'picks'
            w._set_clock(at(day, 23, 59))
            model_and_switch(directory, day, 57)
            w._set_clock(at(day, 23, 59))
            if kind == 'automatic':
                print('year fetch', fetch(directory, day, 58, 58, recipient=False))
            else:
                print('year write', invoke(['set-contest-streak', '--picks-dir', directory, '--date', day,
                                            '--streak', 58, '--best-streak', 58, '--saver', 'used']))
            print('year before', kind, call(lambda: (state(directory).streak, state(directory).observation_kind)))
            w._set_clock(at(day + timedelta(days=1), 0, 0))
            after = call(lambda: state(directory))
            print('year after', kind, after)
            assert 'switch is other_season' in after


def interruption():
    with TemporaryDirectory() as tmp, w.offline(at(DAY)):
        directory = Path(tmp) / 'picks'
        model_and_switch(directory, DAY, 57)
        value = 58 if HIGH else 56
        path = directory / 'account_state/contest_streak.json'
        ledger = directory / 'account_state/contest_ledger.jsonl'
        status = directory.parent / 'health_state/contest_streak_fetch_status.json'
        old = tuple(p.read_bytes() for p in (path, ledger, status))
        real_replace = os.replace
        for stage in ('before observation replace', 'before status replace'):
            target = str(path if stage == 'before observation replace' else status)
            def interrupt(src, dst, *args, **kwargs):
                if str(dst) == target:
                    raise KeyboardInterrupt('synthetic interrupt')
                return real_replace(src, dst, *args, **kwargs)
            with patch.object(os, 'replace', interrupt):
                result = fetch(directory, DAY, value, value, recipient=False)
            same = tuple(p.read_bytes() == prev for p, prev in zip((path, ledger, status), old))
            print('atomic', stage, result, 'unchanged observation, ledger, status', same,
                  'live', state(directory).streak, 'leftover temp', list(directory.parent.rglob('*.tmp')))
            assert same == ((True, True, True) if stage == 'before observation replace' else (False, True, True))


def malformed():
    def time_limit(_signum, _frame):
        raise TimeoutError('bounded probe interrupted a blocking file read')

    with TemporaryDirectory() as tmp, w.offline(at(DAY)):
        root = Path(tmp)
        for filename in ('contest_streak.manual.json', 'contest_streak.json', 'season_contract.json', 'saver_state.json'):
            for kind in ('directory', 'invalid JSON', 'list', 'FIFO'):
                directory = root / filename / kind / 'picks'
                model_and_switch(directory, DAY, 57)
                path = directory / 'account_state' / filename
                path.unlink(missing_ok=True)
                if kind == 'FIFO':
                    os.mkfifo(path)
                elif kind == 'directory':
                    path.mkdir()
                else:
                    path.write_text('[' if kind == 'invalid JSON' else '[]')
                previous = signal.signal(signal.SIGALRM, time_limit)
                signal.setitimer(signal.ITIMER_REAL, .05)
                try:
                    print('malformed live', filename, kind, call(lambda: (state(directory).streak, state(directory).saver_flag_state)))
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)
                print('malformed historical', filename, kind, [(a.level, a.message.split(' This check')[0]) for a in
                      health.check(directory, expected=True, now=at(DAY), historical_day=DAY)])
                if filename == 'contest_streak.json' and kind == 'list':
                    dm = Mock(return_value='id')
                    with patch('bts.dm.send_dm', dm):
                        print('fetch over list', fetch(directory, DAY, 57, 57), 'alerts', dm.call_args_list)


def delivery():
    for operation in ('failed', 'unconfirmed', 'no recipient failure', 'R33 correction'):
        with TemporaryDirectory() as tmp, w.offline(at(DAY)), ExitStack() as st:
            directory = Path(tmp) / 'picks'
            dm = Mock(return_value='local-id')
            _, games = inputs(st, DAY, dm=dm)
            cfg = cfg_for(directory)
            model_and_switch(directory, DAY, prior=57)
            if operation == 'no recipient failure':
                result = fetch(directory, DAY, 401, 401, recipient=False)
                print('over cap no recipient', result, 'live', state(directory).streak, 'DMS', dm.call_args_list)
                continue
            if HIGH:
                assert fetch(directory, DAY, 58, 58)[0] == 0
            work = select(DAY, directory, cfg, games)
            saved = picks.load_pick(str(DAY), directory)
            dm.reset_mock()
            if operation == 'failed':
                dm.side_effect = OSError('synthetic send failed')
            if operation == 'unconfirmed':
                saved.delivery_attempted = True
                picks.save_pick(saved, directory)
                saved = picks.load_pick(str(DAY), directory)
            if operation == 'R33 correction':
                best = 58 if HIGH else 57
                correction = fetch(directory, DAY, 0, best)
                assert correction[0] == 0
                current = state(directory)
                ctx = strategy.DecisionContext(.96, .95, True, str(DAY), True,
                                               strategy._load_mdp(), current.best_streak, current.best_status)
                print('R33 new live', current.streak, current.best_streak,
                      'next fresh policy', strategy.resolve_policy_decision(ctx, current.streak, current.saver_available).action,
                      'reused refusal', picks.saved_pick_refusal(saved, state(directory)))
            ok = scheduler._deliver_and_lock_pick(saved, cfg, directory, sched_state(DAY), str(DAY),
                                                  operation, selection=work)
            print('standing delivery', operation, ok, 'DMs', dm.call_args_list, 'attempted',
                  picks.load_pick(str(DAY), directory).delivery_attempted,
                  'record', daily_decision.load_decision(str(DAY), directory))


def rollback(out):
    day = date(2027, 7, 15)
    with w.offline(at(day)), ExitStack() as st:
        dm = Mock(return_value='id')
        _, games = inputs(st, day, dm=dm)
        auto_dir = out / 'auto' / 'picks'
        dated_dir = out / 'dated' / 'picks'
        if HIGH:
            for kind, directory in [('automatic', auto_dir), ('dated', dated_dir)]:
                assert not w.install_state(directory, day, w.State(kind, 58, 58, 'used', .96))
                select(day, directory, cfg_for(directory), games)
            assert fetch(dated_dir, day, 5, 5, recipient=False)[0] == 0  # ledger-only under absent switch
        else:
            for directory in (auto_dir, dated_dir):
                saved = picks.load_pick(str(day), directory)
                print('rollback', directory, call(lambda: state(directory)), 'pick still loads', saved is not None,
                      'chosen_on', saved.chosen_on)
                assert saved.chosen_on is None
            print('rollback next-day enable and low fetch')
            next_day = day + timedelta(days=1)
            for directory in (auto_dir, dated_dir):
                assert not w.install_state(directory, next_day, w.State('automatic', 5, 5, 'used', .96))
                print(directory, call(lambda: state(directory)))
            assert state(auto_dir).streak == 5
            assert 'check 5' in call(lambda: state(dated_dir))


def wrong_best():
    original = scheduler._write_classification_decision

    def finalize(directory, day, *, selection=None, **kwargs):
        if selection is None:
            return original(directory, day, **kwargs)
        target = kwargs['state']
        scheduler._capture_fallback_skip(target, SimpleNamespace(selection=selection))
        return scheduler._write_endofday_skip(directory, day, target)

    with patch.object(scheduler, '_write_classification_decision', finalize):
        wrong_case('automatic', 5, 157, DAY)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['standing', 'rollback'])
    parser.add_argument('out', nargs='?', type=Path)
    args = parser.parse_args()
    print(cs.__file__)
    if args.mode == 'rollback':
        rollback(args.out)
    else:
        rollover()
        interruption()
        malformed()
        delivery()
        wrong_best()
