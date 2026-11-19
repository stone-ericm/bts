"""Pinned year boundaries and scoring through Click and synthetic MLB boxscores."""

import io
import json
import re
import subprocess
from urllib.parse import urlsplit

from bts import cli, contest_fetch, contest_state, picks, strategy
from bts.health import tail_policy
from bts.leaderboard import auth
from types import SimpleNamespace

from offline import HERE, ROOT, instant, offline


def inventory():
    diff = subprocess.check_output(
        ['git', 'diff', '--no-ext-diff', '-U0', '619abda7', 'HEAD', '--',
         'src', 'scripts', 'config', 'deploy', '.github/workflows'], cwd=ROOT, text=True,
    )
    needle = re.compile(r'now\(|today\(|%Y|2026|2027|SEASON_|date\.fromisoformat')
    hunk = re.compile(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@')
    filename, line = '', 0
    for record in diff.splitlines():
        if record.startswith('+++ b/'):
            filename = record[6:]
            continue
        match = hunk.match(record)
        if match:
            line = int(match[1])
            continue
        if record.startswith(('+', ' ')) and not record.startswith('+++ '):
            if record.startswith('+') and needle.search(record):
                print(f'{filename}:{line}: {record[1:]}')
            line += 1


def fetch_fixture(w, mp, *, rounds, streak=8, rows=None):
    mp.setattr(auth, 'load_session_cookies', lambda: {})
    mp.setattr(auth, 'extract_uid', lambda _: 'synthetic-uid')
    mp.setattr(auth, 'fetch_login_session', lambda **_: SimpleNamespace(user_id=42, username='offline', xsid='none'))
    mp.setattr(contest_fetch, 'fetch_profile', lambda *_: {
        'activeStreak': streak, 'seasonBestStreak': streak,
        'predictions': [{'roundId': 9, 'result': 'hit'}] if rows is None else rows,
    })
    mp.setattr(cli, '_fetch_rounds', lambda: rounds)


def live(w, *, kind=None, why=None):
    try:
        result = contest_state.load_decision_streak_state(w.picks, require_contest_state=True)
    except contest_state.ContestStateError as exc:
        assert why and why in str(exc), exc
        print(w.now.isoformat(), str(exc).split(' To ')[0])
    else:
        assert why is None and result.observation_kind == kind, (why, result)
        print(w.now.isoformat(), result.observation_kind, 'streak', result.streak,
              'model', result.model_streak, 'saver', result.saver_flag_state)
        return result


def dates():
    print('\n=== clock and year matrix ===')
    for day, days_left in [('2027-03-24', 186), ('2027-03-25', 185),
                           ('2027-12-31', 0), ('2028-01-01', 271)]:
        with offline(instant(day, 0)) as (w, _):
            assert cli._today_et() == day
            assert picks.season_of_date(day) == w.now.year
            assert strategy._days_remaining(day) == days_left
            warnings = tail_policy.season_end_alerts(today=w.now.date())
            assert bool(warnings) == (w.now.year == 2028)
            print(day, 'tomorrow', cli._tomorrow_et(), 'days remaining', days_left, 'tail WARN', bool(warnings))
    with offline(instant('2027-03-24', 23, 59, 59)) as (w, _):
        w.dated(8, saver='not_earned')
        live(w, kind='dated')
        w.advance(1)
        live(w, why='switch is absent')
        w.dated(8, day='2027-03-24', code=2)
        w.dated(0)
        live(w, kind='dated')
    with offline(instant('2027-12-31', 23, 59, 59)) as (w, mp):
        w.automatic(8, source_date='2027-09-26')
        w.write_account('saver_state.json', {'season': 2027, 'state': 'used'})
        picks.save_streak(12, w.picks, False, season=2027)
        live(w, kind='automatic')
        fetch_fixture(w, mp, rounds={9: w.now.date().replace(month=9, day=26)})
        old = (w.account / 'contest_streak.json').read_bytes()
        w.advance(1)
        state = w.call('season-contract', 'status', '--picks-dir', str(w.picks)).stdout
        assert '2028' in state and 'live decision now: refused' in state
        live(w, why='other_season')
        w.call('fetch-contest-streak', '--picks-dir', str(w.picks), '--dm-recipient', 'offline.invalid')
        assert (w.account / 'contest_streak.json').read_bytes() == old and not w.sends('dm')
        ledger = json.loads((w.account / 'contest_ledger.jsonl').read_text())
        assert ledger['season'] == 2028 and ledger['settled_by_year'] == {'2027': 1} and not ledger['switched']
        w.dated(0)
        result = live(w, kind='dated')
        assert result.model_streak == 0 and result.saver_flag_state == 'used'


def boxscore(w, mp):
    hit = {'value': True}

    def response(url, **kwargs):
        if '/feed/live' not in urlsplit(url).path:
            return w.http(url, **kwargs)
        batting = {'hits': int(hit['value']), 'atBats': 4}
        feed = {'gameData': {'status': {'abstractGameCode': 'F'}}, 'liveData': {'boxscore': {'teams': {
            'away': {'players': {}}, 'home': {'players': {'ID601': {'stats': {'batting': batting}}}},
        }}}}
        return io.BytesIO(json.dumps(feed).encode())

    mp.setattr(picks, 'retry_urlopen', response)
    return hit


def delivered(w, day):
    daily = picks.DailyPick(day, instant(day, 18).isoformat(), picks.Pick(
        'Synthetic Batter', 601, 'ABC', 1, 'Synthetic Pitcher', 901, 0.9, [], False, 101,
        instant(day, 19).isoformat(),
    ), None, None)
    daily.notification_sent = True
    daily.notification_id = 'previous-confirmed-dm'
    picks.save_pick(daily, w.picks)
    return daily


def grade(w, day, **kwargs):
    return w.call('check-results', '--date', day, '--picks-dir', str(w.picks), **kwargs)


def reconcile(w):
    return w.call('reconcile', '--picks-dir', str(w.picks))


def replay_status(w):
    [receipt] = (w.health / 'reconcile_receipts' / w.now.date().isoformat()).glob('*.json')
    return json.loads(receipt.read_text())['replay']['state']


def raw_streak(w):
    return json.loads((w.picks / 'streak.json').read_text())


def scoring_boundaries():
    print('\n=== 01:00 scoring and 02:00 reconciliation ===')
    day = '2027-03-24'
    with offline(instant('2027-03-25', 1)) as (w, mp):
        picks.save_streak(12, w.picks, False, applied_date='2026-09-28', season=2026)
        hit = boxscore(w, mp)
        delivered(w, day)
        assert 'Streak: 1' in grade(w, day).stdout
        first = raw_streak(w)
        assert (first['season'], first['streak'], first['saver_available'], first['applied_dates']) == (2027, 1, True, [day])
        w.advance(3600)
        hit['value'] = False
        assert 'CORRECTIONS FOUND (1)' in reconcile(w).stdout
        assert picks.load_pick(day, w.picks).result == 'miss' and raw_streak(w)['streak'] == 0
        assert raw_streak(w)['applied_dates'] == [day]
    with offline(instant('2028-01-01', 1)) as (w, mp):
        # Synthetic Dec 31 game isolates file semantics; MLB has no contest game then.
        old = '2027-12-31'
        picks.save_streak(12, w.picks, False, season=2027)
        delivered(w, old)
        hit = boxscore(w, mp)
        assert 'Streak: 13' in grade(w, old).stdout
        assert raw_streak(w)['season'] == 2027 and raw_streak(w)['applied_dates'] == [old]
        w.advance(3600)
        hit['value'] = False
        assert 'CORRECTIONS FOUND (1)' in reconcile(w).stdout
        assert picks.load_pick(old, w.picks).result == 'miss'
        assert raw_streak(w)['season'] == 2028 and raw_streak(w)['streak'] == 0
    with offline(instant('2028-01-01', 2)) as (w, mp):
        # Cron's wait deadline allows scoring to succeed *after* the 02:00 replay.
        old = '2027-12-31'
        picks.save_streak(12, w.picks, False, season=2027)
        delivered(w, old)
        boxscore(w, mp)
        reconcile(w)
        assert raw_streak(w)['season'] == 2028
        w.advance(60)
        assert 'Streak: 1' in grade(w, old).stdout
        assert raw_streak(w)['season'] == 2027 and raw_streak(w)['streak'] == 1
        assert raw_streak(w)['saver_available'] is True
        print('reconcile-before-grading rolls the sole file back to 2027 and loses its starting 12')


def refused_preview_blocks_replay():
    print('\n=== refused preview permanently blocks season replay ===')
    from bts import scheduler

    day = '2027-03-24'
    follow = '2027-03-25'
    with offline(instant(day, 18, 20)) as (w, mp):
        w.schedule[day] = w.schedule[day][:1]
        w.dated(0)
        w.call('preview', '--date', day, '--picks-dir', str(w.picks), '--models-dir', str(w.models))
        w.dated(30)
        w.predictions = lambda: None
        scheduler.run_day(day, w.config)
        assert not w.sends('dm') and picks.load_pick(day, w.picks).result is None
        assert any('chosen without the live check' in body for body in w.sends('health-dm'))
        w.now = instant('2027-03-25', 2)
        assert 'not scoring' in grade(w, day).output.lower()
        picks.save_streak(19, w.picks, False, season=2027)
        reconcile(w)
        assert raw_streak(w)['streak'] == 19
        w.now = instant('2027-03-26', 1)
        boxscore(w, mp)
        delivered(w, follow)
        grade(w, follow)
        assert raw_streak(w)['streak'] == 20
        w.now = instant('2027-03-26', 2)
        assert 'No scoring changes detected. Streak: 20' in reconcile(w).stdout
        assert raw_streak(w)['streak'] == 20
        assert replay_status(w) == 'unavailable'
        w.now = instant('2027-07-01', 2)
        assert 'No scoring changes detected. Streak: 20' in reconcile(w).stdout
        path = w.picks / f'{day}.json'
        path.rename(path.with_suffix('.refused.json'))
        assert 'Streak: 1' in reconcile(w).stdout and raw_streak(w)['streak'] == 1


if __name__ == '__main__':
    import sys
    import traceback

    failed = False
    for check in (inventory, dates, scoring_boundaries, refused_preview_blocks_replay):
        if len(sys.argv) == 1 or check.__name__ in sys.argv[1:]:
            try:
                check()
            except Exception:
                traceback.print_exc()
                failed = True
    raise SystemExit(failed)
