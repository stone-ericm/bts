from datetime import datetime
from importlib import import_module
import json
import os
from pathlib import Path
import re
import shlex
import socket
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

import bts
from bts import orchestrator, scheduler
from bts.cli import cli
from tests.test_contest_dated_cs4a import observation, write
from review_env import GOOD, HEAD, HERE, OLD, tree

REV = sys.argv[1]
ROOT = tree(REV)
ET = ZoneInfo('America/New_York')
NOW = datetime.fromisoformat(GOOD + 'T18:00:00').replace(tzinfo=ET)
CLI_MODULE = import_module('bts.cli')
RUNNER = CliRunner()


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


def disallowed(*args, **kwargs):
    raise AssertionError('unexpected network or scheduler fetch')


def snapshot(directory):
    return {str(p.relative_to(directory)): (p.stat().st_mtime_ns, p.read_bytes())
            for p in directory.rglob('*') if p.is_file()}


def canary(config, now=NOW):
    before = snapshot(config.parent)
    with patch.object(CLI_MODULE, '_now_et', return_value=now) as clock, \
         patch.object(scheduler, 'run_day', side_effect=disallowed), \
         patch('importlib.import_module', wraps=import_module) as imports:
        result = RUNNER.invoke(cli, ['deploy-canary', '--config', str(config)])
    assert clock.call_count == 1 and before == snapshot(config.parent), (result.output, result.exception)
    print(REV, config.name, result.exit_code, result.output.rstrip(), 'imports', [c.args[0] for c in imports.call_args_list])
    return result


def example_variants(fixture, example):
    def check(name, contents, new=0, old=0):
        config = fixture / (name + '.toml')
        config.write_text(contents)
        result = canary(config)
        assert result.exit_code == (new if REV == HEAD else old), (name, result.output, result.exception)
        return config

    public = check('example-as-written', example)
    assert public.read_bytes() == (ROOT / 'config/orchestrator.example.toml').read_bytes()
    dm = example.replace('pick_delivery = "public"', 'pick_delivery = "dm"')
    check('typical-dm', dm)
    research = example.replace('pick_delivery = "public"', 'pick_delivery = "private"').replace('entry_intent = "enter"', 'entry_intent = "research"')
    check('typical-private', research)
    check('private-legacy-alias', research.replace('pick_delivery = "private"', 'private_mode = true'))
    local = dm.split('[[tiers]]')[0] + '[[tiers]]\nname="local"\ntype="local"\n'
    check('local-compute-dm', local)
    for intent, content in (('dm', dm), ('public', example), ('research', research)):
        check(f'old-malformed-budget-{intent}', content.replace('[scheduler]', '[scheduler]\ncascade_budget_min="forty"'), new=1)
    check('old-missing-recipient', re.sub(r'^dm_recipient = .*\n', '', dm, flags=re.M), new=1)
    check('old-missing-tiers', dm.split('[[tiers]]')[0], new=1)

    # The schedule entry point fails in write_heartbeat before READY on this path.
    (fixture / 'regular-file').write_text('parent is a regular file')
    conf = check('accepted-unusable-heartbeat-parent', dm.replace('[orchestrator]',
                 '[orchestrator]\nheartbeat_path="regular-file/.heartbeat"'))
    before = snapshot(fixture)
    with patch.object(scheduler, 'fetch_schedule', side_effect=disallowed) as network, \
         patch.object(scheduler, 'notify_ready') as ready:
        actual = RUNNER.invoke(cli, ['schedule', '--config', str(conf), '--date', GOOD])
    assert actual.exit_code == 1 and isinstance(actual.exception, FileExistsError) and not ready.called and not network.called
    assert snapshot(fixture) == before
    print(REV, 'real scheduler before READY', repr(actual.exception))

    # Runtime treats all non-Windows platforms as POSIX, including Linux.
    linux = dm.replace('timeout_min = 5', 'timeout_min = 5\nplatform = "linux"', 1)
    conf = check('linux-ssh-tier', linux, new=1)
    argv = []

    def ssh(arguments, **options):
        assert arguments[0] == 'ssh' and options['timeout'] == 300 and arguments[-2] == 'mac'
        argv.append(arguments)
        return SimpleNamespace(returncode=0, stdout='[{"batter_id": 1}]', stderr='')

    with patch.object(orchestrator.subprocess, 'run', side_effect=ssh):
        frame, used = orchestrator.run_cascade(orchestrator.load_config(conf)['tiers'], GOOD)
    assert frame.iloc[0]['batter_id'] == 1 and used == 'mac' and len(argv) == 1
    print(REV, 'runtime used linux platform through POSIX SSH shim', used)


def day_boundaries(fixture, contents):
    picks = fixture / 'data/picks'
    config = fixture / 'boundary.toml'
    config.write_text(contents)
    for day, next_ in ((GOOD, '2027-07-16'), ('2027-12-31', '2028-01-01')):
        write(picks, observation(day, recorded_at=day + 'T12:00:00-05:00'))
        before = datetime.fromisoformat(day + 'T23:59:59').replace(tzinfo=ET)
        after = datetime.fromisoformat(next_ + 'T00:00:01').replace(tzinfo=ET)
        assert canary(config, before).exit_code == 0
        refused = canary(config, after)
        assert refused.exit_code == 1 and 'contest state refused' in refused.output


def operator(fixture, contents):
    configured = fixture / 'data/live-account'
    configured.mkdir()
    incident = (ROOT / 'INCIDENT.md').read_text()
    if REV == HEAD:
        status, dated = re.findall(r'`(uv run --no-sync bts (?:season-contract status|set-contest-streak) [^`]+)`', incident)[:2]
        assert 'Any ship, dispatch or\nrevert must carry this guarded workflow and the same guard protocol.' in incident
        replace = {'<the configured picks_dir>': shlex.quote(str(configured)), '<today ET>': GOOD,
                   '<active>': '12', '<best>': '14', '<state>': 'not_earned'}
        for key, value in replace.items():
            status, dated = status.replace(key, value), dated.replace(key, value)
        status_args, dated_args = shlex.split(status)[4:], shlex.split(dated)[4:]
    else:
        status_args = ['season-contract', 'status']
        dated_args = ['set-contest-streak', '--date', GOOD, '--streak', '12', '--best-streak', '14', '--saver', 'not_earned']
        assert 'same guard protocol' not in incident
    config = fixture / 'recovery.toml'
    config.write_text(contents.replace('picks_dir = "data/picks"', 'picks_dir = "data/live-account"'))
    assert canary(config).exit_code == 1
    with patch('datetime.datetime', Clock):
        written = RUNNER.invoke(cli, dated_args)
        assert written.exit_code == 0, (written.output, written.exception)
        inspected = RUNNER.invoke(cli, status_args)
    result = canary(config)
    assert result.exit_code == (0 if REV == HEAD else 1)
    assert bool(list(configured.rglob('*.json'))) == (REV == HEAD)
    print(REV, 'F7 literal docs recovery', written.output.splitlines()[0],
          next(line for line in inspected.output.splitlines() if 'live decision now' in line), result.exit_code)


if __name__ == '__main__':
    assert Path(bts.__file__).resolve().is_relative_to(ROOT / 'src'), bts.__file__
    contents = (ROOT / 'config/orchestrator.example.toml').read_text()
    with tempfile.TemporaryDirectory(dir=HERE) as tmp, patch.object(socket.socket, 'connect', disallowed):
        place = Path(tmp)
        cwd = Path.cwd()
        (place / 'data/picks').mkdir(parents=True)
        write(place / 'data/picks', observation(GOOD, recorded_at=GOOD + 'T12:00:00-05:00'))
        try:
            os.chdir(place)
            example_variants(place, contents)
            day_boundaries(place, contents)
            operator(place, contents)
        finally:
            os.chdir(cwd)
