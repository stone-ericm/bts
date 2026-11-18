"""Exercise delivery, the scheduler, and the real fetch CLI with offline inputs."""

import contextlib
import datetime as calendar
import io
import json
import logging
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock


def reject_network(event, args):
    if event in {'socket.connect', 'socket.sendto', 'socket.getaddrinfo',
                 'socket.gethostbyname', 'socket.gethostbyaddr', 'socket.getnameinfo'}:
        raise AssertionError(f'network forbidden: {event}')


sys.addaudithook(reject_network)

from click.testing import CliRunner
from pytest import MonkeyPatch

from bts import cli, contest_fetch, contest_state, dm, orchestrator, picks, posting, scheduler
from bts.daily_decision import load_decision
from bts.health import alert, contest_state as health, runner
from bts.leaderboard import auth
from tests.test_pick_chosen_on_cs5a import DAY, NOW, account_file, set_automatic
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


def read(path):
    return json.loads(path.read_text())


def run(root, mode, outcomes, *, full_day=False):
    with MonkeyPatch.context() as mp:
        config, directory, clock, _ = world.__wrapped__(root, mp)
        prepare_run_day(mp)
        config['scheduler'].update(pick_delivery=mode, entry_intent='enter')
        frame, tier = orchestrator.run_cascade.return_value
        orchestrator.run_cascade.return_value = (frame.assign(p_game_hit=[0.99, 0.81]), tier)
        set_automatic(directory, streak=30)
        observation = directory / 'account_state/contest_streak.json'
        auto = read(observation)
        auto.update(recorded_at=(NOW - calendar.timedelta(hours=20)).astimezone(
            calendar.timezone.utc).isoformat().replace('+00:00', 'Z'),
            source='mlb_bts_profile', source_date='2027-07-12')
        observation.write_text(json.dumps(auto))
        for index, result in enumerate(outcomes, start=13):
            (directory / f'2027-07-{index}.json').write_text(json.dumps({'result': result}))
        account_file(directory, 'saver_state.json', {'schema_version': 'bts_saver_state_v1',
                      'season': 2027, 'state': 'used', 'updated_at': NOW.isoformat()})
        decision_state = contest_state.load_decision_streak_state(directory, require_contest_state=True)
        print(f'{mode} outcomes={outcomes}: {decision_state.status}: {decision_state.message}')
        print('delivery-time health:', [(a.level, a.message) for a in health.check(
            directory, expected=True, now=NOW)])
        console = io.StringIO()
        handler = logging.StreamHandler(console)
        logging.getLogger('bts.health.alert').addHandler(handler)
        try:
            with contextlib.redirect_stderr(console):
                if full_day:
                    config['health_checks']['enabled'] = True
                    real_check = health.check
                    real_safe_run = runner._safe_run

                    def check(*args, **kwargs):
                        print('EOD contest check at', clock[0].isoformat())
                        return real_check(*args, **kwargs, now=clock[0])

                    def poll(*args, **kwargs):
                        clock[0] = NOW.replace(hour=23, minute=0)
                        return 'final'

                    mp.setattr(health, 'check', check)
                    mp.setattr(runner, '_safe_run', lambda name, fn: real_safe_run(name, fn)
                               if name == 'contest_state' else [])
                    mp.setattr(runner, 'read_systemd_nrestarts', lambda: None)
                    mp.setattr(scheduler, 'run_result_polling', Mock(side_effect=poll))
                    mp.setattr(scheduler, '_watchdog_ping_sleep', lambda seconds: clock.__setitem__(
                        0, clock[0] + calendar.timedelta(seconds=seconds)))
                    scheduler.run_day(DAY, config)
                    state = scheduler.load_state(DAY, directory)
                else:
                    _, selection, _ = orchestrator.run_and_pick(config, DAY)
                    daily = selection.pick_result.daily
                    state = scheduler_state()
                    assert scheduler._deliver_and_lock_pick(
                        daily, config, directory, state, DAY, 'probe', selection=selection)
        finally:
            logging.getLogger('bts.health.alert').removeHandler(handler)
        assert state.pick_locked and load_decision(DAY, directory)['delivery_status'] == 'delivered'
        assert dm.send_dm.call_count == (mode == 'dm')
        assert posting.post_to_bluesky.call_count == (mode == 'public')
        assert alert.send_dm.call_count == 0
        assert 'STALE CONTEST READING' not in console.getvalue()
        assert (picks.load_pick(DAY, directory).notification_sent if mode == 'dm'
                else picks.load_pick(DAY, directory).bluesky_posted)
        print(console.getvalue().rstrip())
        print('pick transport:', dm.send_dm.call_args or posting.post_to_bluesky.call_args)


def fetch_sequence(root):
    with MonkeyPatch.context() as mp:
        _, directory, clock, _ = world.__wrapped__(root, mp)
        set_automatic(directory, streak=30)
        observation = directory / 'account_state/contest_streak.json'
        old = read(observation)
        old.update(source='mlb_bts_profile', source_date='2027-07-13',
                   recorded_at='2027-07-14T17:30:00Z')
        observation.write_text(json.dumps(old))
        before = observation.read_bytes()
        status = root / 'health_state/contest_streak_fetch_status.json'
        status.parent.mkdir()
        status.write_text(json.dumps({'last_success_at': '2027-07-14T17:30:02Z', 'last_error': None}))
        mp.setattr(calendar, 'datetime', contest_state.datetime)
        mp.setattr(auth, 'load_session_cookies', lambda: {})
        mp.setattr(auth, 'extract_uid', lambda _: 'offline')
        login = Mock(side_effect=auth.TransientAuthError('offline upstream outage'))
        mp.setattr(auth, 'fetch_login_session', login)
        mp.setattr(contest_fetch, 'fetch_profile', lambda *_: {
            'activeStreak': 30, 'seasonBestStreak': 30,
            'predictions': [{'roundId': 1, 'result': 'hit'}]})
        mp.setattr(cli, '_fetch_rounds', lambda: {1: calendar.date(2027, 7, 13)})
        command = ['fetch-contest-streak', '--picks-dir', str(directory),
                   '--expected-username', 'offline', '--dm-recipient', 'offline.test']
        runner = CliRunner()
        for hour, minute, count in [(1, 10, 1), (2, 10, 1), (10, 30, 2), (13, 30, 2)]:
            clock[0] = NOW.replace(hour=hour, minute=minute)
            res = runner.invoke(cli.cli, command, catch_exceptions=False)
            assert res.exit_code == 2 and dm.send_dm.call_count == count
            assert observation.read_bytes() == before and 'last_success_at' not in read(status)
            print(f'fetch {hour:02}:{minute:02}: {res.output.strip()}, total DMs {count}, {read(status)}')
        for call in dm.send_dm.call_args_list:
            print('fetch DM:', call.args[1])

        login.side_effect = None
        login.return_value = SimpleNamespace(user_id=1, username='offline', xsid='offline')
        account_dir = directory / 'account_state'
        (account_dir / 'season_contract.json').unlink()
        assert runner.invoke(cli.cli, command, catch_exceptions=False).exit_code == 0
        assert observation.read_bytes() == before and read(status)['last_error'] is None
        ledger = account_dir / 'contest_ledger.jsonl'
        ledger_row = json.loads(ledger.read_text().splitlines()[-1])
        assert ledger_row['switched'] is False and ledger_row['activation_id'] is None
        with contextlib.suppress(contest_state.ContestStateError):
            contest_state.load_decision_streak_state(directory, require_contest_state=True)
            raise AssertionError('ledger-only fetch unexpectedly covered')
        print('ledger-only:', read(status), 'auto unchanged:', observation.read_bytes() == before)

        for action, extra, refusal in [('enable', [], 'activation_id'),
                                       ('revoke', ['--reason', 'offline'], 'revoked'),
                                       ('enable', [], 'activation_id')]:
            res = runner.invoke(cli.cli, ['season-contract', action, '--season', '2027',
                                          '--picks-dir', str(directory), *extra], catch_exceptions=False)
            assert res.exit_code == 0, res.output
            try:
                contest_state.load_decision_streak_state(directory, require_contest_state=True)
            except contest_state.ContestStateError as exc:
                assert refusal in str(exc)
                print(f'{action}: {str(exc).split(". ")[0]}')
            else:
                raise AssertionError(f'{action} unexpectedly covered old observation')
            assert observation.read_bytes() == before
        res = runner.invoke(cli.cli, command, catch_exceptions=False)
        assert res.exit_code == 0, res.output
        (directory / '2027-07-14.json').write_text('{"result":"miss"}')
        s = contest_state.load_decision_streak_state(directory, require_contest_state=True)
        assert s.status == 'stale' and read(observation)['source_date'] == '2027-07-13'
        print('covered refresh after re-enable:', read(observation), s.status, s.message)


if __name__ == '__main__':
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
        os.chdir(tmp)
        root = Path(tmp)
        for i, (mode, outcomes, full_day) in enumerate([
            ('dm', [], False), ('public', ['hit'], False), ('dm', ['miss'], False),
            ('dm', ['hit', 'miss'], True),
        ]):
            run(root / str(i), mode, outcomes, full_day=full_day)
        fetch_sequence(root / 'fetch')
