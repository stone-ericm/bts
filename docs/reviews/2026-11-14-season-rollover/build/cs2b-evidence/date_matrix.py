"""Recorded base/branch real-entry comparison. No network; all clocks pinned.
Run with .venv/bin/python .codex-review/cs2b/date_matrix.py.
Each worker imports only the selected package's src (tests supply fixture data).
"""
import contextlib
import datetime as dt
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
ROWS = [
    ('padded', '2027-03-24', 2027), ('unpadded-month', '2027-3-24', 2027),
    ('unpadded-day', '2027-03-4', 2027), ('unpadded-both', '2027-3-4', 2027),
    ('compact', '20270324', 2027), ('week', '2027-W12-3', 2027),
    ('leading-space', ' 2027-03-24', None), ('trailing-space', '2027-03-24 ', None),
    ('time', '2027-03-24T10:00', None), ('slash', '2027/03/24', None),
    ('dot', '2027.03.24', None), ('impossible', '2027-02-30', None),
    ('junk', 'junk', None), ('empty', '', None),
    ('date-object', dt.date(2027, 3, 24), None), ('none', None, None),
    ('bytes', b'2027-03-24', None), ('integer', 20270324, None),
]
PATHS = ['shadow', 'void', 'final_peer', 'midgame_peer', 'cli', 'strategy', 'update']


def plant_pick(p, directory, day):
    # Persistence interpolates the requested key into the filename. DailyPick.date is
    # JSON text even for non-string lookup keys; no date parser is mocked.
    from tests.test_scoring_lock import _plant_scoreable_pick
    (directory / f'{day}.json').parent.mkdir(parents=True, exist_ok=True)
    _plant_scoreable_pick(directory, str(day))
    daily = p.load_pick(day, directory)
    return daily


def drive(path, day, directory):
    from bts import picks as p, scheduler as s, daily_decision
    from zoneinfo import ZoneInfo
    from contextlib import ExitStack, nullcontext
    from pytest import MonkeyPatch
    fixed = dt.datetime(2027, 3, 24, 23, tzinfo=ZoneInfo('America/New_York'))
    seen = []
    with MonkeyPatch.context() as mp, ExitStack() as stack:
        class FrozenDatetime(dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)
        mp.setattr(p, 'datetime', FrozenDatetime)
        mp.setattr(daily_decision, 'datetime', FrozenDatetime)
        mp.setattr(s, '_now_et', lambda: fixed)
        mp.setattr(p, 'get_game_statuses_detailed', lambda *a: {})
        if path == 'update':
            return p.update_streak([True], directory, date=day)
        if path == 'shadow':
            mp.setattr(s, 'predict_local_shadow', lambda *a, **k: object())
            mp.setattr(s, 'heartbeat_watchdog', lambda *a, **k: nullcontext())
            mp.setattr(s, '_analytics_job_status', lambda *a: {})
            mp.setattr(s, '_update_analytics_job_status', lambda *a, **k: None)
            def select(*a, **kw):
                seen.append(kw['streak'])
                return SimpleNamespace(pick_result=None)
            mp.setattr(s, 'select_pick', select)
            result = s._run_shadow_prediction({'orchestrator': {'picks_dir': str(directory)}}, day, 'Prod')
            return {'return': result, 'streak': seen}
        if path == 'strategy':
            from bts import strategy
            from tests.test_strategy import _predictions
            mp.setattr(strategy, 'datetime', FrozenDatetime)
            stack.enter_context(patch.object(strategy, 'get_game_statuses', return_value={778899: 'P'}))
            stack.enter_context(patch.object(strategy, '_load_mdp', return_value=None))
            decision = stack.enter_context(patch.object(strategy, 'resolve_policy_decision', wraps=strategy.resolve_policy_decision))
            result = strategy.select_pick(_predictions([{'p_game_hit': .85}]), day, directory, streak=13)
            return {'selection': asdict(result), 'saver': decision.call_args.args[2] if decision.called else None}
        plant_pick(p, directory, day)
        if path == 'cli':
            from bts import cli as c
            from click.testing import CliRunner
            mp.setattr(c, '_now_et', lambda: fixed)
            mp.setattr(p, 'resolve_daily_slot_results', lambda *a: {'pick': 'void'})
            # Click's date argument is text; non-string values cannot enter this command.
            if not isinstance(day, str):
                return c.check_results.callback(date=day, picks_dir=str(directory), shadow_status_output=None, wait_deadline_et=None, wait_interval_min=1, allow_stale_scoring=False)
            r = CliRunner().invoke(c.cli, ['check-results', '--date', day, '--picks-dir', str(directory)])
            return {'exit': r.exit_code, 'exception': type(r.exception).__name__ if r.exception else None, 'output': r.output}
        mp.setattr(s, 'poll_game_result', lambda *a: 'final')
        def peer(*a):
            daily = p.load_pick(day, directory)
            daily.result = 'hit'
            p.save_pick(daily, directory)
            return [True]
        if path == 'midgame_peer':
            mp.setattr(s, '_check_hits_midgame', peer)
            mp.setattr(s, '_poll_interval_sleep', lambda *a, **k: None)
            polls = iter(['live', 'final'])
            mp.setattr(s, 'poll_game_result', lambda *a: next(polls))
            daily = p.load_pick(day, directory)
            daily.bluesky_uri = 'at://test'
            p.save_pick(daily, directory)
        def resolve(*a):
            if path == 'final_peer':
                peer()
            return {'pick': 'void'}
        mp.setattr(p, 'resolve_daily_slot_results', resolve)
        reply = stack.enter_context(patch('bts.posting.format_result_reply', return_value='reply'))
        stack.enter_context(patch('bts.posting.reply_to_bluesky', return_value='at://reply'))
        result = s.run_result_polling(111, day, directory)
        return {'return': result, 'reply': reply.call_args.args if reply.called else None}


def worker():
    from bts import picks as p, strategy
    results = []
    for name, day, season in ROWS:
        # Independently validate the mandated parser table on this interpreter.
        actual = None
        if isinstance(day, str):
            try:
                actual = dt.date.fromisoformat(day).year
            except ValueError:
                try:
                    actual = dt.datetime.strptime(day, '%Y-%m-%d').year
                except ValueError:
                    pass
        assert actual == season, (name, actual, season)
        for path in PATHS:
            with tempfile.TemporaryDirectory(dir=ROOT / '.codex-review/cs2b') as temp:
                directory = Path(temp)
                state = directory / 'streak.json'
                state.write_text(json.dumps({'season': 2026, 'streak': 13, 'saver_available': False, 'applied_dates': []}))
                before = state.read_text()
                output = io.StringIO()
                reads = []
                def observe(reader, name):
                    def wrapped(*args, **kwargs):
                        value = reader(*args, **kwargs)
                        reads.append({'reader': name, 'season': kwargs.get('season'), 'value': value})
                        return value
                    return wrapped
                with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output), \
                     patch.object(p, 'load_streak', observe(p.load_streak, 'streak')), \
                     patch.object(p, 'load_saver_available', observe(p.load_saver_available, 'saver')), \
                     patch.object(strategy, 'load_saver_available', observe(strategy.load_saver_available, 'saver')):
                    try:
                        value = drive(path, day, directory)
                        outcome = {'value': value}
                    except Exception as exc:
                        outcome = {'exception': type(exc).__name__, 'message': str(exc)}
                daily = p.load_pick(day, directory)
                results.append({'row': name, 'path': path, 'season': season, 'outcome': outcome,
                                'output': output.getvalue(), 'reads': reads, 'pick_result': daily.result if daily else None,
                                'before': before, 'after': state.read_text(),
                                'before_hex': before.encode().hex(), 'after_hex': state.read_bytes().hex()})
    print(json.dumps(results, default=str))


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        for label, src in [('base', ROOT / '.codex-review/cs2b/base/src'), ('branch', ROOT / 'src')]:
            env = dict(os.environ, PYTHONPATH=f'{src}:{ROOT}', TZ='America/New_York', PYTHONDONTWRITEBYTECODE='1')
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'], env=env, capture_output=True, text=True)
            if run.returncode:
                raise SystemExit(run.stderr)
            (ROOT / f'.codex-review/cs2b/matrix-{label}.json').write_text(run.stdout)
