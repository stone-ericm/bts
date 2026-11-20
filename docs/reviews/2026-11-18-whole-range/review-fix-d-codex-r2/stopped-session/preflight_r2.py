import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from review_env import HEAD, OLD, PREVIOUS, PY, RUN, TARGET, baseline_env, capture, fixture, guard_source

COMMAND = '/usr/bin/python3 -I /usr/local/bin/bts-deploy-hold check-clear --scheduler'
GUARD = '/usr/local/bin/bts-deploy-hold'
MONITOR = '/usr/bin/python3 scripts/check_heartbeat.py --heartbeat-path data/.heartbeat --deploy-hold /var/lib/bts-deploy/hold'
ENV = 'UV_CACHE_DIR=/tmp/uv-cache PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin'


def inventory():
    jobs = [('check-results', 'check-results'), ('reconcile', 'reconcile'), ('reconcile', 'reconcile'),
            *[('fetch-contest-streak', 'fetch-contest-streak')] * 4, ('nightly-preview', 'preview')]
    return [f'{i} 3 * * * {GUARD} run {job} -- /bin/sh -c '
            f"'uv run bts {cmd} --fixture' >> /home/bts/logs/cron.log 2>&1 # BTS-HETZNER"
            for i, (job, cmd) in enumerate(jobs)] + [f'*/5 * * * * cd /home/bts/projects/bts && {MONITOR}']


def structured(command, *, path=None, ignored='no'):
    actual = path or command.split()[0]
    return f'{{ path={actual} ; argv[]={command} ; ignore_errors={ignored} ; start_time=[n/a] ; }}'


def hold(state, step='arm', run=RUN, stamp=None):
    epoch = int(time.time()) - 20 if stamp is None else stamp
    return dict(schema='bts_deploy_hold_v1', deploy_id=run,
                owner_unit=f'bts-deploy-{run}.service', state=state, step=step,
                target=TARGET, previous=PREVIOUS, armed_at=epoch, updated_at=epoch)


def shim(tool, path, *, inventory_lines=None, fields=None, tailscale_ok=True):
    fields = fields or {}
    calls = []
    tool.os = SimpleNamespace(**vars(os))
    tool.os.path = SimpleNamespace(**vars(os.path))
    tool.os.path.lexists = lambda value: str(value).startswith(str(path)) and os.path.lexists(value)
    tool.os.readlink = lambda value: tool.PROJECT if value == '/proc/456/cwd' else os.readlink(value)
    def scoped_path(value):
        if value == '/proc/456/cgroup':
            return SimpleNamespace(read_text=lambda: '0::/fixture/bts-dashboard.service\n')
        assert not str(value).startswith('/proc/')
        return Path(value)

    tool.Path = scoped_path

    def call(argv, *, limit=10):
        calls.append(argv)
        if '/usr/bin/id' in argv:
            return 0, b'1000\n'
        if '/usr/bin/crontab' in argv:
            return 0, ('\n'.join(inventory() if inventory_lines is None else inventory_lines) + '\n').encode()
        if '/usr/bin/systemctl' in argv:
            assert 'show' in argv
            if '--user' not in argv:
                return 0, b'not-found\n'
            unit = argv[argv.index('show') + 1]
            suffix = f'bts schedule --config {tool.CONFIG}' if unit == 'bts-scheduler.service' else 'python3 -m bts.web'
            props = {'LoadState': 'loaded', 'NeedDaemonReload': 'no', 'ExecCondition': structured(COMMAND),
                     'ExecStart': structured(f'{tool.UV} run --no-sync {suffix}'), 'WorkingDirectory': tool.PROJECT,
                     'Environment': ENV, 'EnvironmentFiles': f'{tool.PROJECT}/.env (ignore_errors=no)',
                     'SuccessExitStatus': '143', 'ActiveState': 'active', 'ControlGroup': '/fixture/bts-dashboard.service'}
            if unit == 'bts-scheduler.service':
                props.update(fields)
            names = [value for i, value in enumerate(argv) if i and argv[i - 1] == '-p']
            return 0, ('\n'.join(props[n] if '--value' in argv else f'{n}={props[n]}' for n in names) + '\n').encode()
        if '/usr/bin/git' in argv:
            if 'rev-parse' in argv:
                return 0, PREVIOUS.encode() + b'\n'
            if 'cat-file' in argv:
                return 0, b'commit\n'
            if 'show' in argv:
                return 0, guard_source(REV)[0].read_bytes()
            assert 'config' in argv or 'fetch' in argv
            return 0, b''
        if '/usr/bin/tailscale' in argv:
            return (0 if tailscale_ok else 1), b''
        if argv[0] == '/usr/bin/ss':
            return 0, b'LISTEN 0 5 127.0.0.1:3003 0.0.0.0:* users:(("python3",pid=456,fd=3))\n'
        if argv[0] == '/usr/bin/curl':
            assert argv[-1] == 'http://127.0.0.1:3003/'
            return 0, b'200'
        raise AssertionError((argv, limit))

    tool.call = call
    return calls


def preflight(tool):
    try:
        return tool.preflight(1000, TARGET, fetch=False)[0]
    except tool.Refused as exc:
        return exc.step


def cron_case(tool, path):
    baseline = inventory()
    cases = [('canonical', baseline), ('quote', baseline + ["0 4 * * * uv run 'bts' 'reconcile'"]),
             ('indirect', baseline + ['0 4 * * * /bin/sh /fixture/maintenance']),
             ('python-module', baseline + ['0 4 * * * .venv/bin/python -m bts.cli reconcile']),
             ('wrapped', baseline + ["0 4 * * * /bin/sh -c 'uv run bts reconcile'"]),
             ('chained-ungated', [x.replace(' # BTS-HETZNER', '; uv run bts reconcile # BTS-HETZNER')
                                 if i == 1 else x for i, x in enumerate(baseline)]),
             ('monitor-comment', baseline[:-1] + ['# ' + baseline[-1]]),
             ('commented-writer-only', baseline + [' # uv run bts reconcile']),
             ('shell-escaped', baseline + [r'0 4 * * * uv run bts rec\oncile']),
             ('monitor-short-circuit', baseline[:-1] + [f'*/5 * * * * /usr/bin/true || {MONITOR}'])]
    results = {}
    for name, lines in cases:
        shim(tool, path, inventory_lines=lines)
        results[name] = preflight(tool)
    print(REV, 'inventory', results)
    assert results['canonical'] == PREVIOUS and results['commented-writer-only'] == PREVIOUS
    for name in ('quote', 'python-module', 'wrapped', 'chained-ungated', 'monitor-comment'):
        assert results[name] == ('crontab' if REV == HEAD else PREVIOUS), (name, results[name])
    assert results['shell-escaped'] == results['indirect'] == results['monitor-short-circuit'] == PREVIOUS

    bindir = path / 'bin'
    bindir.mkdir()
    uv = bindir / 'uv'
    uv.write_text('#!/bin/sh\n[ "$1" = run ] && [ "$2" = bts ] && [ "$3" = reconcile ] || exit 97\nprintf "ran\\n" >> "$BTS_FIXTURE_MARKER"\n')
    uv.chmod(0o700)
    monitor = bindir / 'monitor'
    monitor.write_text('#!/bin/sh\nprintf "ran\\n" >> "$BTS_FIXTURE_MARKER"\n')
    monitor.chmod(0o700)
    wrapper = path / 'maintenance'
    wrapper.write_text('uv run bts reconcile\n')
    writer_mark = path / 'writes'
    with tool.directory() as directory:
        tool.publish_json(directory, 'hold', hold('pending'))
    for command in (r'uv run bts rec\oncile', f'/bin/sh "{wrapper}"'):
        run = subprocess.run(['/bin/sh', '-c', command], cwd=path, env={'PATH': str(bindir),
            'BTS_FIXTURE_MARKER': str(writer_mark)}, stdin=subprocess.DEVNULL, capture_output=True, timeout=4)
        assert run.returncode == 0
    assert writer_mark.read_text() == 'ran\nran\n'
    check_mark = path / 'pings'
    run = subprocess.run(['/bin/sh', '-c', f'/usr/bin/true || "{monitor}"'], cwd=path,
                         env={'PATH': str(bindir), 'BTS_FIXTURE_MARKER': str(check_mark)},
                         stdin=subprocess.DEVNULL, capture_output=True, timeout=4)
    assert run.returncode == 0 and not check_mark.exists()
    print(REV, 'real /bin/sh, two accepted writer spellings wrote under pending; accepted heartbeat never ran')


def unit_case(tool, path):
    local = path / 'unit-recovery-root'
    local.mkdir()
    cases = [('canonical', {}), ('condition executable', {'ExecCondition': structured(COMMAND, path='/usr/bin/true')}),
             ('condition skip promoted', {'SuccessExitStatus': '143 1'}), ('working tree', {'WorkingDirectory': str(local)}),
             ('environment', {'Environment': ENV + ' BTS_TEST_DEPLOY_GUARD_DIR=/srv/stale'}),
             ('environment file', {'EnvironmentFiles': str(local / '.env') + ' (ignore_errors=no)'}),
             ('namespace root', {'RootDirectory': str(local)}), ('bind mount', {'BindPaths': '/srv/backup:/var/lib/bts-deploy'}),
             ('inherited test override', {})]
    for name, fields in cases:
        calls = shim(tool, path, fields=fields)
        result = preflight(tool)
        if name in ('namespace root', 'bind mount'):
            assert result == PREVIOUS and not any(set(fields) & set(args) for args in calls)
        print(REV, 'unit', name, result)

    script = path / 'staged-guard.py'
    script.write_text(guard_source(REV)[1])
    official_guard, alternate_guard = tool.ROOT, path / 'test-root'
    alternate_guard.mkdir(mode=0o755)
    outputs = []
    for directory in (official_guard, alternate_guard):
        env = {**baseline_env(REV), 'BTS_TEST_DEPLOY_GUARD_DIR': str(directory)}
        response = subprocess.run([str(PY), '-I', str(script), 'check-clear', '--scheduler'], cwd=path,
                                  env=env, timeout=4, stdin=subprocess.DEVNULL, capture_output=True)
        outputs.append(response.returncode)
    assert outputs == [1, 0]
    print(REV, 'same scheduler condition with held real-equivalent and empty inherited test root', outputs)

    shim(tool, path, tailscale_ok=False)
    try:
        tool.dashboard(1000)
        result = 'accepted loopback 200'
    except tool.Refused as exc:
        result = exc.step
    assert result == ('accepted loopback 200' if REV == HEAD else 'canary-http')
    print(REV, 'F15', result)

    shim(tool, path, fields={'WorkingDirectory': str(local), 'ExecCondition': structured(COMMAND, ignored='yes')})
    with tool.directory() as directory:
        os.unlink('hold', dir_fd=directory)
        replay = capture(lambda: tool.owned_deploy(directory, RUN, TARGET, ''))
    assert replay.rc == 1
    print(REV, 'F14', replay.stdout.strip())


if __name__ == '__main__':
    REV = sys.argv[1]
    with fixture(REV) as (tool, path):
        cron_case(tool, path)
        unit_case(tool, path)
