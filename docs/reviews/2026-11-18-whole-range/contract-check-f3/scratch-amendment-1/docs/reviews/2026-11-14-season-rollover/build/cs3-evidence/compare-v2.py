"""Recorded base/branch matrix. Parent invokes isolated workers, never swaps tracked code."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / '.codex-review/cs3'


def worker():
    import datetime as dt
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from click.testing import CliRunner
    from bts.cli import cli
    from bts.contest_state import load_decision_streak_state
    from bts.health.contest_state import check
    from bts.web import saver_dashboard_context, saver_transition_response
    import bts.cli as cm
    import bts.contest_fetch as cf
    import bts.leaderboard.auth as auth
    from zoneinfo import ZoneInfo
    rows = {}
    flags = ['missing', 'stale', 'not_earned', 'active', 'used']

    def setup(folder, flag, year, streak=7, source=None):
        d = folder / 'account_state'
        d.mkdir(parents=True)
        if flag != 'missing':
            (d / 'saver_state.json').write_text(json.dumps(dict(
                state='active' if flag == 'stale' else flag,
                season=year - 1 if flag == 'stale' else year)))
        if source != 'model':
            (d / 'contest_streak.json').write_text(json.dumps(dict(
                schema_version='bts_contest_streak_auto_v1', active_streak=streak,
                best_streak=streak, source='mlb_bts_profile', source_date=source)))
        (folder / 'streak.json').write_text(json.dumps(dict(season=year, streak=streak)))
        return d

    for year in [2027, 2028]:
        now = dt.datetime(year, 7, 15, 16, tzinfo=dt.timezone.utc)
        for source in [None, f'{year-1}-06-18', f'{year}-06-18', 'model']:
            for flag in flags:
                for streak in [0, 9, 10, 15, 16]:
                    with TemporaryDirectory(dir=SCRATCH) as t:
                        folder = Path(t)
                        setup(folder, flag, year, streak, source)
                        key = f'{year}/{source}/{flag}/{streak}'
                        decision = load_decision_streak_state(folder, now=now)
                        rows['decision/' + key] = decision.saver_available
                        if source != 'model':
                            for expected in [False, True]:
                                rows[f'health/{key}/{expected}'] = [
                                    [a.level, a.source, a.message.replace(str(folder), "<picks>")] for a in check(
                                        folder, expected=expected, now=now)]
                        ctx = saver_dashboard_context(folder, now=now)
                        rows['dashboard/' + key] = [ctx.state, ctx.warning, ctx.nudge]
                        status, _ = saver_transition_response(
                            folder, expected_prior='active', new_state='used',
                            same_origin=True, now=now)
                        f = folder / 'account_state/saver_state.json'
                        body = json.loads(f.read_text()) if f.exists() else None
                        rows['post/' + key] = [status, body and body['state'], body and body['season']]
        for flag in flags:
            for option in ['--use', '--undo']:
                with TemporaryDirectory(dir=SCRATCH) as t:
                    folder = Path(t)
                    setup(folder, flag, year)
                    result = CliRunner().invoke(cli, ['saver-state', '--picks-dir', t,
                        '--season', str(year), option])
                    f = folder / 'account_state/saver_state.json'
                    rows[f'cli/{year}/{flag}/{option}'] = [result.exit_code,
                        json.loads(f.read_text())['state'] if f.exists() else 'missing']

    year = dt.datetime.now(ZoneInfo('America/New_York')).year
    class Session:
        xsid = 'x'; user_id = 50311; username = 'fixture'
    for command in ['fetch-contest-streak', 'set-contest-streak']:
        for source in [None, f'{year-1}-06-18', f'{year}-06-18']:
            # Manual CLI supplies today's date when --source-date is absent (no null mode).
            for best in [0, 7, 10, 12]:
                for flag in flags:
                    with TemporaryDirectory(dir=SCRATCH) as t:
                        folder = Path(t)
                        setup(folder, flag, year)
                        if command == 'fetch-contest-streak':
                            preds = [] if source is None else [{'roundId': 1, 'result': 'hit'}]
                            profile = dict(activeStreak=0, seasonBestStreak=best, predictions=preds)
                            rounds = {} if source is None else {1: dt.date.fromisoformat(source)}
                            with patch.object(auth, 'load_session_cookies', return_value={}), \
                                 patch.object(auth, 'extract_uid', return_value='u'), \
                                 patch.object(auth, 'fetch_login_session', return_value=Session()), \
                                 patch.object(cf, 'fetch_profile', return_value=profile), \
                                 patch.object(cm, '_fetch_rounds', return_value=rounds):
                                result = CliRunner().invoke(cli, [command, '--picks-dir', t,
                                    '--expected-username', 'fixture'])
                        else:
                            args = [command, '--picks-dir', t, '--streak', '0', '--best-streak', str(best)]
                            if source is not None:
                                args += ['--source-date', source]
                            result = CliRunner().invoke(cli, args)
                        f = folder / 'account_state/saver_state.json'
                        body = json.loads(f.read_text()) if f.exists() else None
                        rows[f'earn/{command}/{source}/{best}/{flag}'] = [result.exit_code,
                            body and body['state'], body and body['season']]
    print(json.dumps(rows, sort_keys=True))


if __name__ == '__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        results = {}
        for label, package in [('base', SCRATCH / 'base/src'), ('branch', ROOT / 'src')]:
            env = dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE='1', TZ='America/New_York')
            run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                 env=env, text=True, capture_output=True, check=True)
            (SCRATCH / f'matrix-v2-{label}.json').write_text(run.stdout)
            results[label] = json.loads(run.stdout)
        # Check every branch outcome against its required row, and ensure health's
        # other alerts have not changed. Thus every recorded difference is attributable.
        from datetime import datetime
        from zoneinfo import ZoneInfo
        current_year = datetime.now(ZoneInfo('America/New_York')).year
        for key, actual in results['branch'].items():
            parts = key.split('/')
            family = parts[0]
            if family in {'decision', 'health', 'dashboard', 'post'}:
                _, year_s, source, flag, streak_s, *rest = parts
                year, streak = int(year_s), int(streak_s)
                state = 'uninitialized' if flag in {'missing', 'stale'} else flag
                if family == 'decision':
                    assert actual == (state == 'active' or (state == 'not_earned' and streak < 10)), key
                elif family == 'health':
                    wanted = (state == 'uninitialized' and (rest[0] == 'True' or 10 <= streak <= 15)
                              or state == 'not_earned' and 10 <= streak <= 15)
                    saver = [a for a in actual if a[1] == 'saver_state']
                    assert len(saver) == int(wanted), key
                    for level, src, msg in saver:
                        assert level == 'WARN' and f'--season {year}' in msg and str(streak) in msg
                        assert 'counting the saver as spent' in msg
                    other_before = [a for a in results['base'][key] if 'Streak Saver' not in a[2]]
                    other_after = [a for a in actual if a[1] != 'saver_state']
                    assert other_before == other_after, key
                elif family == 'dashboard':
                    assert actual == [state, state == 'active' and streak > 15 and source != 'model', False], key
                else:
                    if state == 'active':
                        assert actual == [200, 'used', year], key
                    else:
                        prior = None if flag == 'missing' else ('active' if flag == 'stale' else flag)
                        prior_year = None if flag == 'missing' else (year - 1 if flag == 'stale' else year)
                        assert actual == [409, prior, prior_year], key
            elif family == 'cli':
                _, year_s, flag, option = parts
                state = 'uninitialized' if flag in {'missing', 'stale'} else flag
                raw = 'missing' if flag == 'missing' else ('active' if flag == 'stale' else flag)
                if option == '--use':
                    expected = [0, 'used'] if state in {'active', 'not_earned', 'used'} else [1, raw]
                else:
                    expected = [0, 'active'] if state == 'used' else [1, raw]
                assert actual == expected, (key, actual, expected)
            elif family == 'earn':
                _, command, source, best_s, flag = parts
                best = int(best_s)
                year = current_year
                raw = None if flag == 'missing' else ('active' if flag == 'stale' else flag)
                raw_year = None if flag == 'missing' else (year - 1 if flag == 'stale' else year)
                eligible = ((command == 'set-contest-streak' and source == 'None')
                            or source.startswith(str(year)))
                if eligible:
                    if flag in {'missing', 'stale'} and best < 10:
                        raw, raw_year = 'not_earned', year
                    elif flag == 'not_earned' and best >= 10:
                        raw, raw_year = 'active', year
                assert actual == [0, raw, raw_year], (key, actual, [0, raw, raw_year])
            else:
                raise AssertionError(key)
        diffs = {k: [v, results['branch'][k]] for k, v in results['base'].items()
                 if v != results['branch'][k]}
        (SCRATCH / 'matrix-v2-differences.json').write_text(json.dumps(diffs, indent=2))
        from collections import Counter
        print('Total cases:', len(results['base']))
        print('Differences by family:', dict(Counter(k.split('/')[0] for k in diffs)))
