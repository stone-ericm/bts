from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from owner_probe import sources

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = HERE / 'base/scripts/cron-setup-hetzner.sh'
HEAD = REPO / 'scripts/cron-setup-hetzner.sh'
JOBS = ('check-results', 'reconcile', 'fetch-contest-streak', 'nightly-preview')


def append(case, name, args):
    with (case / 'trace').open('a') as trace:
        trace.write(json.dumps({'command': name, 'args': args, 'cwd': os.getcwd(),
                                'export': os.environ.get('BTS_FIXTURE_EXPORTED')}) + '\n')


def shim(name, args):
    case = Path(os.environ['BTS_CRON_FIXTURE'])
    if name == 'crontab':
        current = case / 'tab'
        if args == ['-l']:
            if os.environ.get('BTS_CRON_READ_FAIL'):
                print('fixture permission refused', file=sys.stderr)
                return 2
            if current.exists():
                sys.stdout.buffer.write(current.read_bytes())
            else:
                print('no crontab for fixture', file=sys.stderr)
                return 1
        else:
            assert args == ['-']
            contents = sys.stdin.buffer.read(65537)
            assert len(contents) <= 65536
            current.write_bytes(contents)
        return 0
    if name == 'uv' and args[:3] == ['run', 'bts', 'entry-intent']:
        print(os.environ['BTS_CRON_INTENT'])
        return 0
    if name == 'date':
        append(case, name, args)
        date = datetime(2027, 7, 15, 3, 0)
        if args[:1] == ['-d']:
            date -= timedelta(days={'yesterday': 1, '7 days ago': 7}[args[1]])
            args = args[2:]
        assert len(args) == 1 and args[0].startswith('+')
        print(date.strftime(args[0][1:]))
        return 0
    if name == 'flock':
        append(case, name, args)
        assert args[0] == '-n' and Path(args[1]).is_relative_to(case)
        with Path(args[1]).open('ab') as descriptor:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 1
            result = subprocess.run(args[2:], stdin=subprocess.DEVNULL, timeout=3)
            return result.returncode
    assert name in ('uv', 'python3', 'curl'), name
    append(case, name, args)
    print('stdout from ' + name, flush=True)
    print('stderr from ' + name, file=sys.stderr)
    fail = os.environ.get('BTS_CRON_FAIL')
    if name == 'uv' and fail and fail in args:
        return 7
    return 0


def executable(path, name):
    path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{__file__}" --shim {name} "$@"\n')
    path.chmod(0o700)


def setup(case, intent):
    home = case / 'home'
    project = home / 'projects' / 'bts'
    project.mkdir(parents=True)
    (project / '.env').write_text('BTS_FIXTURE_EXPORTED="kept words"\nBTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.invalid/fixture\n')
    (home / 'logs').mkdir()
    uv = home / '.local' / 'bin' / 'uv'
    uv.parent.mkdir(parents=True)
    executable(uv, 'uv')
    bindir = case / 'bin'
    bindir.mkdir()
    for command in ('crontab', 'date', 'flock', 'curl', 'python3'):
        executable(bindir / command, command)
    root = case / 'guard'
    root.mkdir(mode=0o755)
    for filename in ('writers.lock', 'transaction.lock', 'state.lock'):
        (root / filename).touch(mode=0o644)
    tool = case / 'bts-deploy-hold.py'
    tool.write_text(sources()[0])
    guard = bindir / 'bts-deploy-hold'
    guard.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{tool}" "$@"\n')
    guard.chmod(0o700)
    env = dict(HOME=str(home), PATH=str(bindir) + ':/usr/bin:/bin', TZ='America/New_York',
               BTS_DEPLOY_HOLD_BIN=str(guard), BTS_TEST_DEPLOY_GUARD_DIR=str(root),
               BTS_CRON_FIXTURE=str(case), BTS_CRON_INTENT=intent, PYTHONDONTWRITEBYTECODE='1',
               HEALTHCHECKS_PING_URL='https://hc-ping.invalid/box-fixture')
    return home, project, root, env


def run_installer(script, action, case, env):
    return subprocess.run(['/bin/bash', str(script), action], env=env, cwd=case,
                          stdin=subprocess.DEVNULL, capture_output=True, timeout=5, text=True)


def rendered(result):
    assert result.returncode == 0, (result.stdout, result.stderr)
    headers, rows = result.stdout.split('Would install:\n', 1)
    scheduled = [row for row in rows.splitlines() if not row.startswith('#')]
    return headers, scheduled


def transform(cron):
    expression, input_parts, active = [], [], []
    active = expression
    escaped = False
    for char in cron:
        if escaped:
            if char != '%':
                active.append('\\')
            active.append(char)
            escaped = False
        elif char == '\\':
            escaped = True
        elif char == '%':
            if active is expression:
                active = input_parts
            else:
                active.append('\n')
        else:
            active.append(char)
    if escaped:
        active.append('\\')
    return ''.join(expression), ''.join(input_parts)


def replay(case, command, env, *, fail='', held=False):
    text, input_text = transform(command)
    assert not input_text
    # Boundary translations for paths unavailable on the workstation.
    for real, shim_path in (('/usr/bin/flock', case / 'bin/flock'),
                            ('/home/bts/.check-results.lock', case / 'grader-lock'),
                            ('/usr/bin/python3', case / 'bin/python3'),
                            ('/var/lib/bts-deploy/hold', case / 'guard/hold')):
        text = text.replace(real, str(shim_path))
    trace = case / 'trace'
    trace.unlink(missing_ok=True)
    for path in (case / 'home/logs').iterdir():
        path.unlink()
    result = subprocess.run(['/bin/sh', '-c', text], input=input_text.encode(), cwd=case,
                            env={**env, 'BTS_CRON_FAIL': fail}, capture_output=True, timeout=5)
    if held:
        assert result.returncode == 75 and not trace.exists()
        assert 'skipped: held' in (case / 'home/logs/cron.log').read_text()
    log_data = {f.name: f.read_bytes() for f in (case / 'home/logs').iterdir()}
    return (result.returncode, result.stdout, result.stderr,
            [json.loads(line) for line in trace.read_text().splitlines()] if trace.exists() else [], log_data)


def normalize(events):
    clean = []
    for event in events:
        args = event['args'][:]
        name = event['command']
        if name == 'uv' and args[:2] == ['run', '--no-sync']:
            del args[1]
        if name == 'python3':
            assert args[0] == 'scripts/check_heartbeat.py' and '--deploy-hold' in args
            i = args.index('--deploy-hold')
            del args[i:i + 2]
            args = ['run', 'python', *args]
            name = 'uv'
        clean.append(dict(event, command=name, args=args))
    return clean


def exercise(case, intent):
    home, project, guard, env = setup(case, intent)
    tab = case / 'tab'
    preserve = '0 9 * * * fixture-command\n'
    tab.write_text(preserve)
    old_show, old = rendered(run_installer(BASE, 'show', case, env))
    head_show, head = rendered(run_installer(HEAD, 'show', case, env))
    assert old_show == head_show and len(old) == len(head)
    counts = dict.fromkeys(JOBS, 0)
    pairs = []
    for before, after in zip(old, head):
        schedule = before.split(maxsplit=5)[:5]
        assert schedule == after.split(maxsplit=5)[:5]
        old_command = before.split(maxsplit=5)[5]
        new_command = after.split(maxsplit=5)[5]
        if new_command.startswith(env['BTS_DEPLOY_HOLD_BIN']):
            job = new_command.split()[2]
            counts[job] += 1
            pairs.append((job, old_command, new_command))
        elif before != after:
            pairs.append(('unheld', old_command, new_command))
        else:
            assert 'curl -fsS' in old_command
    assert tuple(counts.values()) == (1, 2, 4, 1)
    for job, before, after in pairs:
        old_effect = replay(case, before, env)
        new_effect = replay(case, after, env)
        assert old_effect[:3] == new_effect[:3] == (0, b'', b''), (job, old_effect, new_effect)
        assert normalize(old_effect[3]) == normalize(new_effect[3]), (job, old_effect, new_effect)
        for event in new_effect[3]:
            assert event['cwd'] == str(project) and event['export'] == 'kept words'
        if 'check_heartbeat.py' in before:
            assert new_effect[4]['heartbeat.log'] == b'stdout from python3\nstderr from python3\n'
        else:
            assert old_effect[4] == new_effect[4], (job, old_effect, new_effect)
        print(intent, job, [event['args'] for event in new_effect[3]], list(new_effect[4]), flush=True)
        if job != 'unheld':
            armed = int(time.time()) - 2
            (guard / 'hold').write_text(json.dumps({
                'schema': 'bts_deploy_hold_v1', 'deploy_id': '1795209800-cron',
                'owner_unit': 'bts-deploy-1795209800-cron.service', 'state': 'pending', 'step': 'arm',
                'target': 'b' * 40, 'previous': 'a' * 40, 'armed_at': armed, 'updated_at': armed,
            }))
            replay(case, after, env, held=True)
            (guard / 'hold').unlink()
    nightly = next(pair[2] for pair in pairs if pair[0] == 'nightly-preview')
    for fail in ('pull', 'build', 'preview', 'sync-to-r2'):
        effect = replay(case, nightly, env, fail=fail)
        steps = [event['args'][2:4] for event in effect[3] if event['command'] == 'uv']
        assert (['preview'] == steps[2][:1] and steps[3] == ['data', 'sync-to-r2']) if fail in ('preview', 'sync-to-r2') else len(steps) <= 2
        assert effect[0] == (0 if fail == 'preview' else 7)
    grade = next(pair[2] for pair in pairs if pair[0] == 'check-results')
    with (case / 'grader-lock').open('ab') as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        effect = replay(case, grade, env)
        assert effect[0] == 1 and [e['command'] for e in effect[3]] == ['date', 'flock']

    result = run_installer(HEAD, 'install', case, env)
    assert result.returncode == 0 and tab.read_text() == preserve + '# BTS-HETZNER\n' + '\n'.join(head) + '\n'
    tab.write_text(preserve)
    tool = Path(env['BTS_DEPLOY_HOLD_BIN'])
    executable_bytes = tool.read_bytes()
    for refusal in ('missing-lock', 'wrong-version', 'extra-version-line', 'nonzero-version', 'read-error', 'missing-env'):
        trial_env = dict(env)
        lock = guard / 'writers.lock'
        dotenv = project / '.env'
        if refusal == 'missing-lock':
            lock.unlink()
        elif refusal in ('wrong-version', 'extra-version-line', 'nonzero-version'):
            response = {'wrong-version': 'bts-deploy-hold 2\\n', 'extra-version-line': 'bts-deploy-hold 1\\n\\n',
                        'nonzero-version': 'bts-deploy-hold 1\\n'}[refusal]
            tool.write_text(f'#!/bin/sh\nprintf "{response}"\nexit {int(refusal == "nonzero-version")}\n')
        elif refusal == 'read-error':
            trial_env['BTS_CRON_READ_FAIL'] = '1'
        else:
            dotenv.rename(project / '.env.disabled')
        try:
            failed = run_installer(HEAD, 'install', case, trial_env)
            assert failed.returncode == 1 and tab.read_text() == preserve, (refusal, failed)
            print(intent, 'install refusal', refusal, failed.stderr.strip(), flush=True)
        finally:
            tool.write_bytes(executable_bytes)
            lock.touch(mode=0o644)
            if (project / '.env.disabled').exists():
                (project / '.env.disabled').rename(dotenv)
    print(intent, 'counts', counts, 'changed replays', len(pairs), 'independence and grader flock verified', flush=True)


def main():
    if sys.argv[1:2] == ['--shim']:
        return shim(sys.argv[2], sys.argv[3:])
    assert transform(r'date +\%Y%one%two') == ('date +%Y', 'one\ntwo')
    for intent in ('research', 'enter'):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            exercise(Path(tmp), intent)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
