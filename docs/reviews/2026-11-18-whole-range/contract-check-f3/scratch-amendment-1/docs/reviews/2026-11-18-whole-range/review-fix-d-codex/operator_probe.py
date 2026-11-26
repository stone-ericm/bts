import datetime as dt
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ET = ZoneInfo('America/New_York')
NOW = dt.datetime(2027, 7, 15, 9, tzinfo=ET)


class Clock(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


def no_network(*args, **kwargs):
    raise AssertionError('the operator probe has no network')


def previous_workflow_shim(name, args):
    place = Path(os.environ['WRD_REVIEW_CASE'])
    with (place / 'trace').open('a') as log:
        log.write(json.dumps([name, *args]) + '\n')
    if name == 'id':
        print(1000)
    elif name == 'sleep':
        pass
    elif name == 'curl':
        print('503')
    elif name == 'sudo' and {'git', 'rev-parse'} <= set(args):
        print('a' * (7 if '--short' in args else 40))
    elif name == 'sudo' and {'git', 'reset'} <= set(args):
        assert (place / 'hold').is_file()
        (place / 'head').write_text(args[-1])
    elif name == 'sudo' and 'is-active' in args:
        if '--quiet' not in args:
            print('active')
    elif name != 'sudo' or not any(token in args for token in ('git', 'tailscale', 'sync', 'restart', 'bash')):
        raise AssertionError((name, args))
    return 0


def previous_workflow(place):
    result = subprocess.run(['/usr/bin/git', 'show', 'b128854d:.github/workflows/deploy.yml'], cwd=REPO,
                            stdin=subprocess.DEVNULL, capture_output=True, timeout=5, check=True)
    start, = re.finditer(rb'^          script: \|\n', result.stdout, flags=re.M)
    lines = result.stdout[start.end():].splitlines(keepends=True)
    assert all(line == b'\n' or line.startswith(b' ' * 12) for line in lines)
    payload = b''.join(line[12:] if line != b'\n' else line for line in lines)
    (place / 'workflow').write_bytes(payload)
    (place / 'hold').write_text('pending fixture')
    (place / 'bin').mkdir()
    for command in ('sudo', 'id', 'sleep', 'curl'):
        shim = place / 'bin' / command
        shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{__file__}" --old-deploy-shim {command} "$@"\n')
        shim.chmod(0o700)
    for command in ('cut', 'head', 'grep'):
        (place / 'bin' / command).symlink_to('/usr/bin/' + command)
    (place / 'bash-env').write_text('cd() { builtin cd "$WRD_REVIEW_CASE"; }\n'
        '[() { if [[ "$2" == /etc/systemd/system/bts-scheduler.service ]]; then return 1; fi; builtin [ "$@"; }\n')
    ended = subprocess.run(['/bin/bash', str(place / 'workflow')], cwd=place,
        env={'WRD_REVIEW_CASE': str(place), 'PATH': str(place / 'bin'), 'BASH_ENV': str(place / 'bash-env'),
             'TARGET_SHA': 'b' * 40, 'PYTHONDONTWRITEBYTECODE': '1'},
        stdin=subprocess.DEVNULL, capture_output=True, timeout=10, text=True)
    calls = [json.loads(line) for line in (place / 'trace').read_text().splitlines()]
    assert ended.returncode == 1 and (place / 'head').read_text() == 'a' * 40, (ended, calls)
    resets = [call[-1] for call in calls if 'reset' in call]
    assert resets == ['b' * 40, 'a' * 40]
    assert (place / 'hold').read_text() == 'pending fixture'
    print('Base workflow in a revert: reset history', resets, 'despite a pending hold')
    print('Base workflow rollback output:', ended.stdout.strip().replace('\n', ' | '))


def main():
    from click.testing import CliRunner
    from bts.cli import cli

    runner = CliRunner()
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        fixture = Path(directory)
        configured = fixture / 'data' / 'live-account'
        conventional = fixture / 'data' / 'picks'
        configured.mkdir(parents=True)
        conventional.mkdir()
        cfg = fixture / 'orchestrator.toml'
        cfg.write_text(f'[orchestrator]\npicks_dir={json.dumps(str(configured))}\n'
                       '[scheduler]\nentry_intent="enter"\npick_delivery="dm"\n'
                       'contest_state_required=true\n[health_checks]\ncontest_state_expected=true\n')
        origin = Path.cwd()
        try:
            os.chdir(fixture)
            with patch('datetime.datetime', Clock), patch('bts.cli._now_et', return_value=NOW), \
                 patch.object(socket.socket, 'connect', no_network):
                initial = runner.invoke(cli, ['deploy-canary', '--config', str(cfg)])
                assert initial.exit_code == 1 and 'contest state refused' in initial.output
                result = runner.invoke(cli, ['set-contest-streak', '--date', '2027-07-15',
                                             '--streak', '12', '--best-streak', '14', '--saver', 'not_earned'])
                assert result.exit_code == 0, (result.output, result.exception)
                wrong_status = runner.invoke(cli, ['season-contract', 'status'])
                actual_status = runner.invoke(cli, ['season-contract', 'status', '--picks-dir', str(configured)])
                repeat = runner.invoke(cli, ['deploy-canary', '--config', str(cfg)])
        finally:
            os.chdir(origin)
        assert list(conventional.rglob('*.json')) and not list(configured.rglob('*.json'))
        assert 'live decision now: accepted' in wrong_status.output
        assert 'live decision now: refused' in actual_status.output
        assert repeat.exit_code == 1 and 'contest state refused' in repeat.output
        print('initial:', initial.output.strip())
        print('operator wrote:', result.output.strip())
        print('default status:', next(line for line in wrong_status.output.splitlines() if 'live decision now' in line))
        print('configured status:', next(line for line in actual_status.output.splitlines() if 'live decision now' in line))
        print('retry:', repeat.output.strip())
        old = fixture / 'old-workflow'
        old.mkdir()
        previous_workflow(old)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--old-deploy-shim']:
        raise SystemExit(previous_workflow_shim(sys.argv[2], sys.argv[3:]))
    main()
