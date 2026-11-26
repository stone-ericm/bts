import datetime as dt
import json
import os
from pathlib import Path
import shlex
from unittest.mock import Mock

from click.testing import CliRunner
import httpx
import pytest

from probe_support import freeze, offline_probe
from review_runner import dump, environment

RECIPIENT = ' --dm-recipient stonehengee.bsky.social'
FETCH = 'bts fetch-contest-streak'
UTC = dt.timezone.utc
NOW = dt.datetime(2027, 7, 15, 23, 0, tzinfo=UTC)


def cron(root, source):
    from tests import test_entry_intent as fixture

    folder = root / 'box'
    folder.mkdir()
    original_exe = fixture._exe

    def guarded_exe(path, body):
        if path.name == 'uv':
            body = body.replace('exec ', 'export PYTHONPATH=' + shlex.quote(environment(source)['PYTHONPATH'])
                                + '\nexport PYTHONDONTWRITEBYTECODE=1\nexec ')
        original_exe(path, body)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fixture, 'SCRIPT', source / 'scripts/cron-setup-hetzner.sh')
        patch.setattr(fixture, '_exe', guarded_exe)
        box = fixture.box.__wrapped__(folder)
        for intent, body in (
            ('enter', 'entry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'),
            ('research', 'entry_intent = "research"\npick_delivery = "private"\n'),
        ):
            box.cron.write_text('0 8 * * * echo foreign\n')
            box.config.write_text('[scheduler]\n' + body)
            for action in ('show', 'install'):
                response = box.run(action)
                assert response.returncode == 0, (response.stdout, response.stderr)
                assert response.stderr == ''
                (root / f'{intent}.{action}.txt').write_text((response.stdout if action == 'show'
                   else response.stdout + box.cron.read_text()).replace(str(folder), '<box>'))
                if action == 'show':
                    assert box.cron.read_text() == '0 8 * * * echo foreign\n'


def probe(root, source):
    from bts import cli, contest_fetch, dm
    from bts.health import alert, contest_state, runner
    from bts.leaderboard import auth
    from tests.test_contest_fetch import _mock_auth_failure, _mock_auth_success

    cron(root, source)
    failures = {
        'cookie': auth.AuthError('expired'), 'transient-auth': auth.TransientAuthError('outage'),
        'rate-limited-auth': auth.RateLimitedLoginError('429'),
        'auth-network': httpx.ConnectError('offline auth'),
    }
    comparison = {}
    for case in (*failures, 'wrong-identity', 'prior-identity', 'profile-403', 'profile-429',
                 'profile-503', 'profile-network', 'bad-shape', 'corrupt-switch', 'empty-rounds',
                 'rounds-network', 'prior-nonobject', 'dm-send-fails'):
        picks = root / case / 'data/picks'
        account = picks / 'account_state'
        account.mkdir(parents=True)
        with pytest.MonkeyPatch.context() as patch:
            clock = freeze(NOW)
            patch.setattr(dt, 'datetime', clock)
            for mod in (alert, contest_state):
                patch.setattr(mod, 'datetime', clock)
            _mock_auth_success(patch)
            sender = Mock(side_effect=RuntimeError('captured Bluesky transport failure') if case == 'dm-send-fails' else None)
            patch.setattr(dm, 'send_dm', sender)
            patch.setattr(cli, '_fetch_rounds', Mock(return_value={1: NOW.date()}))
            patch.setattr(contest_fetch, 'fetch_profile', Mock(return_value={
                'activeStreak': 0, 'seasonBestStreak': 0, 'predictions': [],
            }))
            if case in failures:
                _mock_auth_failure(patch, failures[case])
            elif case == 'prior-identity':
                dump(account / 'contest_streak.json', {'username': 'another'})
            elif case == 'corrupt-switch':
                dump(account / 'season_contract.json', {})
            elif case == 'prior-nonobject':
                dump(account / 'contest_streak.json', [])
            elif case == 'bad-shape':
                contest_fetch.fetch_profile.return_value = {}
            elif case.startswith('profile-'):
                code = case.split('-')[-1]
                request = httpx.Request('GET', 'https://example.invalid')
                contest_fetch.fetch_profile.side_effect = (httpx.ConnectError('offline profile') if code == 'network'
                    else httpx.HTTPStatusError(code, request=request, response=httpx.Response(int(code), request=request)))
            elif case == 'empty-rounds':
                cli._fetch_rounds.return_value = {}
            elif case == 'rounds-network':
                cli._fetch_rounds.side_effect = httpx.ConnectError('offline rounds')
            elif case == 'dm-send-fails':
                _mock_auth_failure(patch, auth.AuthError('expired'))
            args = ['fetch-contest-streak', '--picks-dir', str(picks), '--expected-username',
                    'another' if case == 'wrong-identity' else 'stonehengee']
            if case == 'dm-send-fails':
                args += ['--dm-recipient', 'offline.invalid']
            response = CliRunner().invoke(cli.cli, args)
            status_path = picks.parent / 'health_state/contest_streak_fetch_status.json'
            status = json.loads(status_path.read_text()) if status_path.exists() else None
            stderr = response.stderr.replace(str(root), '<work>')
            if case == 'prior-nonobject':
                assert response.exit_code == 1 and isinstance(response.exception, AttributeError), response.output
                assert not status and not stderr
            else:
                assert response.exit_code == 2 and status['last_alert_at'] is None, (case, response.output, response.exception)
                assert status['last_error'].replace(str(root), '<work>') in stderr
            if case == 'dm-send-fails':
                sender.assert_called_once()
            else:
                sender.assert_not_called()
            comparison[case] = dict(exit=response.exit_code, stderr=stderr,
                                    exception=repr(response.exception), dm_count=sender.call_count, status=status)
            if case in {'cookie', 'corrupt-switch'}:
                health_send = Mock()
                patch.setattr(alert, 'send_dm', health_send)
                # Research entry intent does not require an automatic contest observation.
                alerts = runner.run_all_checks(picks, root / 'empty-models', 'offline.invalid',
                                               today=NOW.date(), contest_state_expected=False)
                comparison[case]['research_health'] = [(a.level, a.source, a.message) for a in alerts
                    if a.source == 'contest_state' or 'fetch-contest-streak' in a.message]
                comparison[case]['health_dm'] = str(health_send.call_args_list).replace(str(root), '<work>')
                if case == 'cookie':
                    forced = contest_state.check(picks, expected=True, now=NOW)
                    comparison[case]['expected_control'] = [a.message.replace(str(root), '<work>') for a in forced]
    dump(root / 'failures.json', comparison)


def compare(root):
    for intent in ('enter', 'research'):
        for action in ('show', 'install'):
            path = f'{intent}.{action}.txt'
            before, after = [(root / label / path).read_text() for label in ('base', 'head')]
            expected = before
            if intent == 'research':
                assert before.count(FETCH) == after.count(FETCH) == 4
                expected = ''.join(line.replace(RECIPIENT, '') if FETCH in line else line
                                   for line in before.splitlines(keepends=True))
            assert expected == after, path
            print(f'{path}: ' + ('identical' if intent == 'enter' else 'only four recipient deletions'))
    def normalized(label):
        return (root / label / 'failures.json').read_text().replace(str(root / label), '<work>')

    a, b = [json.loads(normalized(label)) for label in ('base', 'head')]
    for case in a:
        assert a[case] == b[case], (case, a[case], b[case])
    print(normalized('head'))


if __name__ == '__main__':
    raise SystemExit(offline_probe(probe, compare))
