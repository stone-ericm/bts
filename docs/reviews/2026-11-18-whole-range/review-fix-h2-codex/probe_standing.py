import argparse
import json
import os
import signal
from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import contest_state as cs, daily_decision, orchestrator, picks, scheduler
from bts.health import contest_state as health
from support import DAY, at, counts, deliver, fetch, fixture, fresh, install, invoke, live, walk


HEAD = hasattr(cs, 'MAX_SEASON_STREAK')
HIGH = 58 if HEAD else 57


def files(directory):
    state = directory / 'account_state'
    return {'obs': state / 'contest_streak.json', 'ledger': state / 'contest_ledger.jsonl',
            'saver': state / 'saver_state.json', 'status': directory.parent / 'health_state/contest_streak_fetch_status.json'}


def interruption(root):
    for stage, changed in [('obs', set()), ('status', {'obs'}), ('ledger', {'obs', 'status'}),
                           ('saver', {'obs', 'status', 'ledger'})]:
        directory = root / stage / 'picks'
        with walk.offline(at()):
            install(directory, 'automatic', 5, 5, saver='not_earned')
            paths = files(directory)
            old = {name: path.read_bytes() for name, path in paths.items()}
            replace, open_path = os.replace, Path.open

            def change(src, dest, *args, **kwargs):
                if Path(dest) == paths[stage]:
                    raise KeyboardInterrupt('stop before replace')
                return replace(src, dest, *args, **kwargs)

            def opening(path, mode='r', *args, **kwargs):
                if stage == 'ledger' and path == paths[stage] and mode == 'a':
                    raise KeyboardInterrupt('stop before append')
                return open_path(path, mode, *args, **kwargs)

            walk._set_clock(at(hour=12, minute=1))
            with patch.object(os, 'replace', side_effect=change), patch.object(Path, 'open', opening):
                result = fetch(directory, counts(HIGH, HIGH), recipient=False)
            assert result[0] != 0, (stage, result)
            differences = {name for name, path in paths.items() if path.read_bytes() != old[name]}
            assert differences == changed, (stage, differences)
            assert live(directory).streak == (HIGH if 'obs' in changed else 5)
            assert not list(directory.parent.rglob('*.tmp'))
            print('interrupt before', stage, 'changed', sorted(differences), 'active', live(directory).streak)


def describe(fn):
    def expired(*_):
        raise TimeoutError('bounded file read')

    original = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, .08)
    try:
        return fn()
    except Exception as exc:
        return type(exc).__name__, str(exc).splitlines()[0]
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, original)


def malformed(root):
    names = ('contest_streak.manual.json', 'contest_streak.json', 'season_contract.json', 'saver_state.json',
             '../streak.json', '../' + str(DAY) + '.json', 'contest_ledger.jsonl',
             '../../health_state/contest_streak_fetch_status.json')
    for name in names:
        for shape in ('json', 'array', 'utf8', 'directory', 'fifo'):
            directory = root / name.replace('/', '_') / shape / 'picks'
            with walk.offline(at()):
                install(directory, 'automatic', HIGH, HIGH)
                path = (directory / 'account_state' / name).resolve()
                path.unlink(missing_ok=True)
                if shape == 'directory':
                    path.mkdir()
                elif shape == 'fifo':
                    os.mkfifo(path)
                else:
                    path.write_bytes({'json': b'{', 'array': b'[]', 'utf8': b'\xff'}[shape])
                reader = describe(lambda: (live(directory).saver_flag_state, live(directory).model_state_error))
                historical = describe(lambda: [(a.level, a.source) for a in
                                      health.check(directory, now=at(), expected=True, historical_day=DAY)])
                assert 'TimeoutError' not in historical, (name, shape, historical)
                if name == f'../{DAY}.json':
                    writer = describe(lambda: picks.load_pick(str(DAY), directory))
                else:
                    value = describe(lambda: fetch(directory, counts(HIGH, HIGH), recipient=False))
                    writer = (value[0], value[2]) if len(value) == 4 else value
                if name == 'contest_streak.json' and shape in {'array', 'utf8'}:
                    assert writer == (1, 'AttributeError' if shape == 'array' else 'UnicodeDecodeError'), writer
                print('malformed', name, shape, str(reader).replace(str(directory.parent), '<temp>'), historical, writer)


def rollover(root):
    day = date(2027, 12, 31)
    tomorrow = day + timedelta(days=1)
    for mode in ('automatic', 'dated'):
        directory = root / mode / 'picks'
        with walk.offline(at(day, 23, 59)):
            install(directory, 'automatic', 5, 5, day=day)
            walk._set_clock(at(day, 23, 59))
            if mode == 'automatic':
                assert fetch(directory, counts(HIGH, HIGH), day=day, recipient=False)[0] == 0
            else:
                assert invoke('set-contest-streak', '--picks-dir', directory, '--date', day,
                              '--streak', HIGH, '--best-streak', HIGH, '--saver', 'used')[0] == 0
            assert (live(directory).streak, live(directory).observation_kind) == (HIGH, mode)
            walk._set_clock(at(tomorrow, 0))
            error = describe(lambda: live(directory))
            assert error[0] == 'ContestStateError' and 'other_season' in error[1], error
            assert fetch(directory, counts(HIGH, HIGH), day=tomorrow, recipient=False)[0] == 0
            assert 'other_season' in describe(lambda: live(directory))[1]
            assert invoke('season-contract', 'enable', '--season', tomorrow.year, '--picks-dir', directory)[0] == 0
            assert 'automatic observation season is 2027' in describe(lambda: live(directory))[1]
            assert fetch(directory, counts(HIGH, HIGH), day=tomorrow, recipient=False)[0] == 0
            assert live(directory).streak == HIGH and live(directory).observation_kind == 'automatic'
            print('year rollover', mode, HIGH, 'accepted 23:59, refused 00:00, new activation + fetch restored eligibility')


def delivery(root):
    for mode in ('failed', 'unconfirmed', 'auto-corrected', 'dated-corrected'):
        directory = root / mode / 'picks'
        with walk.offline(at()), ExitStack() as stack:
            config, games, frame, dm = fixture(stack, directory)
            kind = 'dated' if mode == 'dated-corrected' else 'automatic'
            install(directory, kind, HIGH, HIGH)
            work = fresh(DAY, config, games)
            daily = picks.load_pick(str(DAY), directory)
            assert daily.chosen_on['streak'] == HIGH
            if mode == 'failed':
                dm.side_effect = OSError('definite transport failure')
                assert not deliver(directory, config, daily, work)
                assert daily_decision.load_decision(str(DAY), directory) is None
                daily = picks.load_pick(str(DAY), directory)
                assert not daily.delivery_attempted
                dm.side_effect = None
                dm.reset_mock()
                assert deliver(directory, config, daily, None)
                assert dm.call_count == 1
            elif mode == 'unconfirmed':
                dm.side_effect = KeyboardInterrupt('interrupt after fake transport received pick')
                try:
                    deliver(directory, config, daily, work)
                except KeyboardInterrupt:
                    pass
                else:
                    raise AssertionError('send was not interrupted')
                assert dm.call_count == 1
                daily = picks.load_pick(str(DAY), directory)
                assert daily.delivery_attempted and not picks.pick_was_delivered(daily)
                dm.reset_mock()
                dm.side_effect = None
                assert not deliver(directory, config, daily, None)
                assert scheduler.load_state(str(DAY), directory).pick_locked
                assert dm.call_count == 1 and 'DELIVERY OUTCOME UNKNOWN' in dm.call_args.args[1]
                assert daily_decision.load_decision(str(DAY), directory)['delivery_status'] == 'locked_unconfirmed'
            elif mode == 'auto-corrected':
                assert fetch(directory, counts(0, HIGH))[0] == 0
                assert fresh(DAY, config, games).action == 'skip'
                assert picks.saved_pick_refusal(daily, live(directory)) is None
                assert deliver(directory, config, daily, work)
                assert dm.call_count == 1
            elif HEAD:
                assert invoke('set-contest-streak', '--picks-dir', directory, '--date', DAY, '--streak', HIGH,
                              '--best-streak', HIGH + 1, '--saver', 'used')[0] == 0
                with patch.object(orchestrator, 'run_cascade', return_value=(frame.iloc[:0], 'empty')):
                    refresh = scheduler._refresh_pick_at_fallback_decision(config, str(DAY), daily, .01)
                assert refresh.daily is daily and 'best streak 58' in picks.saved_pick_refusal(daily, live(directory))
                assert not deliver(directory, config, daily, refresh.selection)
                assert daily_decision.load_decision(str(DAY), directory) is None
            else:
                continue
            record = daily_decision.load_decision(str(DAY), directory)
            print('delivery', mode, 'captured DMs', dm.call_count, 'status', record and record['delivery_status'])


def rollback(root):
    for mode in ('automatic', 'dated', 'best-only', 'committed'):
        directory = root / mode / 'picks'
        with walk.offline(at()), ExitStack() as stack:
            config, games, _, dm = fixture(stack, directory)
            kind = 'automatic' if mode in {'automatic', 'committed'} else 'dated'
            active = 12 if mode == 'best-only' else 58
            if HEAD:
                install(directory, 'automatic', 5, 5)
                install(directory, kind, active, 58)
                work = fresh(DAY, config, games)
                if mode == 'best-only':
                    # Obtain a genuinely selected pick earlier in the season under the same identity shape.
                    early = date(2027, 7, 15)
                    install(directory, 'dated', active, 58, day=early)
                    config, games, _, dm = fixture(stack, directory, early, dm=dm)
                    assert fresh(early, config, games).action != 'skip'
                elif mode == 'committed':
                    assert deliver(directory, config, picks.load_pick(str(DAY), directory), work)
                continue
            day = date(2027, 7, 15) if mode == 'best-only' else DAY
            walk._set_clock(at(day))
            daily = picks.load_pick(str(day), directory)
            assert daily.chosen_on is None
            error = describe(lambda: live(directory))
            assert error[0] == 'ContestStateError'
            if mode in {'dated', 'best-only'}:
                assert ('check 6' if mode == 'best-only' else 'check 5') in error[1]
                following = day + timedelta(days=1)
                install(directory, 'automatic', 5, 5, day=following)
                assert describe(lambda: live(directory))[0] == 'ContestStateError'
            elif mode == 'committed':
                assert daily_decision.is_scoreable_commit(str(DAY), directory, daily)
            else:
                install(directory, 'automatic', 5, 5)
                assert 'there is no valid record' in picks.saved_pick_refusal(daily, live(directory))
            print('rollback', mode, error[1], 'pending metadata absent, scoreable', mode == 'committed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollback', type=Path)
    args = parser.parse_args()
    if args.rollback:
        rollback(args.rollback)
    else:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            print(cs.__file__)
            interruption(root / 'interrupt')
            malformed(root / 'malformed')
            rollover(root / 'year')
            delivery(root / 'delivery')
