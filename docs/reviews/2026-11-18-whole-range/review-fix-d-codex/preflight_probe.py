from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from owner_probe import gates, guard, INSTALLER, PREVIOUS, TARGET

HERE = Path(__file__).resolve().parent
RUN = '1800000000-probe'
NOW = 1_800_000_000
COND = '/usr/bin/python3 -I /usr/local/bin/bts-deploy-hold check-clear --scheduler'
SCHED = '/home/bts/.local/bin/uv run --no-sync bts schedule --config /home/bts/.bts-orchestrator.toml'
DASH = '/home/bts/.local/bin/uv run --no-sync python3 -m bts.web'
SCHEDULER = 'bts-scheduler.service'
SYSTEMCTL = '/usr/bin/systemctl'
GATE = '/usr/local/bin/bts-deploy-hold run'


def cron_inventory():
    return '\n'.join(f"{n} 1 * * * {GATE} {job} -- /bin/sh -c 'cd /fixture && uv run bts {command} >> log 2>&1'"
                    for n, (job, command) in enumerate((('check-results', 'check-results'), ('reconcile', 'reconcile'),
                        ('reconcile', 'reconcile'), *(('fetch-contest-streak', 'fetch-contest-streak'),) * 4,
                        ('nightly-preview', 'preview')))) + '\n*/5 * * * * /usr/bin/python3 scripts/check_heartbeat.py --no-churn\n'


def initialize(case):
    tool = guard(case)
    tool.ROOT.mkdir(mode=0o755)
    for name in ('transaction.lock', 'writers.lock', 'state.lock'):
        (tool.ROOT / name).touch(mode=0o644)
    (tool.ROOT / 'guard-install.sha256').write_text(hashlib.sha256(INSTALLER.read_bytes()).hexdigest() + '\n')
    (Path(tool.PROJECT) / 'data').mkdir(parents=True)
    tool.time.time = lambda: NOW
    tool.os = SimpleNamespace(**vars(os))
    tool.os.path = SimpleNamespace(**vars(os.path))
    tool.os.path.lexists = lambda path: str(path).startswith(str(case)) and os.path.lexists(path)
    return tool


def shim(tool, *, cron=None, condition=COND, start=SCHED, reload='no', changed=False,
         candidate=True, head=PREVIOUS, dirty=False, uv_status=0, stop=True):
    calls = []

    def call(argv, *, limit=10):
        calls.append(argv)
        assert limit <= 480
        if argv == ['/usr/bin/id', '-u', 'bts']:
            return 0, b'1000\n'
        if argv[0] == '/usr/bin/crontab':
            assert argv[1:] == ['-u', 'bts', '-l']
            return 0, ((cron_inventory() if cron is None else cron).encode())
        if SYSTEMCTL in argv:
            if 'stop' in argv:
                return (0 if stop else 1), b''
            assert 'show' in argv, argv
            if '--user' not in argv:
                return 0, b'not-found\n'
            field = argv[argv.index('-p') + 1]
            vals = {'LoadState': 'loaded', 'NeedDaemonReload': reload, 'ExecCondition': condition,
                    'ExecStart': start if SCHEDULER in argv else DASH,
                    'ActiveState': 'inactive' if stop else 'active'}
            assert field in vals, argv
            return 0, (vals[field] + '\n').encode()
        if 'git' in ''.join(argv):
            if 'rev-parse' in argv:
                return 0, (head + '\n').encode()
            if 'cat-file' in argv:
                return 0, b'commit\n'
            if 'show' in argv:
                return 0, INSTALLER.read_bytes() + (b'# target changed\n' if changed else b'')
            if 'status' in argv:
                return 0, b' M tracked.py\n' if dirty else b''
            assert 'config' in argv or 'fetch' in argv, argv
            return 0, b''
        if 'deploy-canary' in argv:
            return int(not candidate), b'fixture classification\n'
        if 'sync' in argv:
            assert '--check' in argv and '--offline' in argv and 'UV_PYTHON_DOWNLOADS=never' in argv
            if uv_status is None:
                raise FileNotFoundError('fixture uv')
            return uv_status, b'fixture uv result'
        raise AssertionError(argv)

    tool.call = call
    return calls


def test_preflight(tool):
    inventory = cron_inventory()
    official = '{ path=/usr/bin/python3 ; argv[]=' + COND + ' ; ignore_errors=no ; start_time=[n/a] ; }'
    for label, options, expected in (
        ('production', {}, None),
        ('systemd show serialization', {'condition': official}, None),
        ('guard-stale', {'changed': True}, 'guard-stale'),
        ('bad ExecStart', {'start': SCHED.replace('--no-sync ', '')}, 'guard'),
        ('bad ExecCondition', {'condition': official.replace('ignore_errors=no', 'ignore_errors=yes')}, 'guard'),
        ('extra condition', {'condition': official * 2}, 'guard'),
        ('stale reload', {'reload': 'yes'}, 'guard'),
        ('stray exact command', {'cron': inventory + '0 4 * * * uv run bts reconcile\n'}, 'crontab'),
        ('quoted executable/subcommand', {'cron': inventory + "0 4 * * * uv run 'bts' 'reconcile'\n"}, None),
        ('indirect writer wrapper', {'cron': inventory + '0 4 * * * /bin/sh /fixture/maintenance\n'}, None),
        ('commented writer', {'cron': inventory + ' # uv run bts reconcile\n'}, None),
        ('quoted in comment on extra row', {'cron': inventory + '0 4 * * * /bin/true # bts reconcile\n'}, 'crontab'),
        ('no canonical preview', {'cron': inventory.replace('bts preview', 'bts broken')}, 'crontab'),
        ('commented monitor only', {'cron': inventory.replace('*/5 * * * * /usr/bin', '# */5 * * * * /usr/bin')}, None),
        ('missing monitor', {'cron': inventory.replace('/usr/bin/python3', '/usr/bin/python3.12')}, 'crontab'),
    ):
        calls = shim(tool, **options)
        try:
            assert tool.preflight(1000, TARGET)[0] == PREVIOUS
        except tool.Refused as exc:
            step = exc.step
        else:
            step = None
        assert step == expected, (label, step, calls)
        if label == 'guard-stale':
            assert not any(set(c) & {'fetch', 'config', 'reset', 'stop', 'restart'} for c in calls)
        assert not (tool.ROOT / 'hold').exists()
        print('preflight', label, step or 'accepted', flush=True)
    calls = shim(tool, condition=official.replace('path=/usr/bin/python3', 'path=/usr/bin/true'))
    assert tool.preflight(1000, TARGET)[0] == PREVIOUS
    print('preflight accepts condition with different path and expected argv; rejects neither WorkingDirectory nor SuccessExitStatus overrides', flush=True)
    assert not any(set(c) & {'WorkingDirectory', 'EnvironmentFiles', 'SuccessExitStatus', 'KillMode'} for c in calls)

    bind = tool.ROOT.parent / 'writer'
    bind.write_text('#!/bin/sh\nprintf "reconcile wrote on a pending hold\\n" >> "$BTS_FIXTURE_WRITE"\n')
    bind.chmod(0o700)
    out = bind.with_suffix('.out')
    fakebin = tool.ROOT.parent / 'bin'
    fakebin.mkdir()
    (fakebin / 'uv').write_text('#!/bin/sh\n[ "$1" = run ] && [ "$2" = bts ] && [ "$3" = reconcile ] || exit 97\nexec "$BTS_FIXTURE_WRITER"\n')
    (fakebin / 'uv').chmod(0o700)
    (tool.ROOT / 'hold').write_text(json.dumps(hold('pending')))
    with patch.object(sys, 'argv', ['bts-deploy-hold', 'check-clear', '--scheduler']):
        assert tool.main() == 1
    command = "uv run 'bts' 'reconcile'"
    result = subprocess.run(['/bin/sh', '-c', command], cwd=bind.parent,
                            env={'PATH': str(fakebin), 'BTS_FIXTURE_WRITER': str(bind), 'BTS_FIXTURE_WRITE': str(out)},
                            timeout=4, stdin=subprocess.DEVNULL, capture_output=True)
    assert result.returncode == 0 and 'pending hold' in out.read_text()
    (tool.ROOT / 'hold').unlink()
    with patch.object(sys, 'argv', ['bts-deploy-hold', 'check-clear', '--scheduler']):
        assert tool.main() == 0
    print('accepted quoted row executed through /bin/sh and wrote while the real guard held pending', flush=True)


def set_window(tool, state, age=0, wake=None):
    path = Path(tool.PROJECT) / 'data/.heartbeat'
    if state is None:
        path.unlink(missing_ok=True)
        return
    raw = {'state': state, 'timestamp': datetime.fromtimestamp(NOW - age, timezone.utc).isoformat()}
    if wake is not None:
        raw['sleeping_until'] = datetime.fromtimestamp(NOW + wake, timezone.utc).isoformat()
    path.write_text(json.dumps(raw))


def test_window(tool):
    for state, age, wake, recovery, expected in (
        ('running', 300, None, TARGET, False), ('running', 301, None, '', False),
        ('running', 301, None, TARGET, True), ('waiting_for_games', 600, None, TARGET, False),
        ('waiting_for_games', 601, None, TARGET.upper(), False),
        ('sleeping', 3, 1799, TARGET, False), ('sleeping', 3, 1800, '', True),
        (None, 0, None, '', False), (None, 0, None, TARGET, True),
        (None, 0, None, PREVIOUS, False), ('running', -1, None, '', False),
    ):
        set_window(tool, state, age, wake)
        try:
            tool.window(recovery, TARGET, NOW + 1200)
        except tool.Refused:
            accepted = False
        else:
            accepted = True
        assert accepted == expected
        print('window', state, age, wake, 'authorization', bool(recovery), accepted, flush=True)
    set_window(tool, 'sleeping', 2, 7200)


def hold(state):
    return {'schema': 'bts_deploy_hold_v1', 'deploy_id': RUN, 'owner_unit': f'bts-deploy-{RUN}.service',
            'state': state, 'step': 'sync', 'target': TARGET, 'previous': PREVIOUS,
            'armed_at': NOW - 20, 'updated_at': NOW - 10}


def contention(tool):
    calls = shim(tool)
    for filename, op, step in (('transaction.lock', fcntl.LOCK_EX, 'hold-busy'),
                               ('writers.lock', fcntl.LOCK_SH, 'writers-busy')):
        run = RUN + ('Owner' if step == 'hold-busy' else 'Writer')
        with tool.directory() as root, tool.lock(root, filename, op):
            assert tool.owned_deploy(root, run, TARGET, '') == 1
            assert tool.read_hold(root) is None
            assert tool.read_json(root, tool.run_name(run, 'result'))['step'] == step
    assert not any(set(c) & {'reset', 'stop', 'restart'} for c in calls)
    print('independent locks enforce owner and writer contention before arm', flush=True)


def dead_owner(tool):
    shim(tool)
    case = tool.ROOT.parent
    with tool.directory() as root:
        tool.publish_json(root, 'hold', hold('activating'))
        assert not gates(tool, case, 'deactivating')['scheduler_clear']
        with tool.lock(root, 'transaction.lock', fcntl.LOCK_EX):
            # In the model this is the next run's handle. The old MainPID is gone,
            # while ExecStopPost belongs to the old deactivating systemd unit.
            admission = gates(tool, case, 'deactivating')
            assert admission['scheduler_clear'] and len(admission['writers']) == 4
            try:
                tool.reconcile(root, RUN)
            except BlockingIOError:
                pass
            else:
                raise AssertionError('hook should fail to acquire the competing lock')
            assert not (tool.ROOT / tool.run_name(RUN, 'evidence')).exists()
        assert not gates(tool, case, 'inactive')['scheduler_clear']
        assert tool.read_hold(root)['state'] == 'activating'
        print('dead previous MainPID + old unit deactivating + next owner lock: real gate admitted scheduler and four writers; hook contended', flush=True)
        (tool.ROOT / 'hold').unlink()


def test_hand_release(case):
    for label, state, options, reason in (
        ('no hold', None, {}, 'verified'), ('pending', 'pending', {}, 'verified'),
        ('activating', 'activating', {}, 'verified'), ('target', 'failed', {}, 'verified'),
        ('dirty', 'failed', {'dirty': True}, 'verified'), ('uv unsynced', 'failed', {'uv_status': 1}, 'verified'),
        ('uv without check', 'failed', {'uv_status': 2}, 'verified'), ('uv absent', 'failed', {'uv_status': None}, 'verified'),
        ('reason blank', 'failed', {}, '   '), ('reason newline', 'failed', {}, 'x\ny'),
        ('reason over 200', 'failed', {}, 'a' * 201), ('busy', 'failed', {}, 'verified'),
        ('bad state', 'unreadable', {}, 'verified'), ('canary', 'failed', {'candidate': False}, 'verified'),
        ('audit fsync error', 'failed', {}, 'verified'), ('success', 'failed', {}, 'verified'),
    ):
        path = case / label.replace(' ', '-')
        path.mkdir()
        tool = initialize(path)
        set_window(tool, 'sleeping', 2, 7200)
        shim(tool, head=PREVIOUS if label == 'target' else TARGET, **options)
        clock = datetime.fromtimestamp(NOW, timezone.utc)

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock.astimezone(tz) if tz else clock.replace(tzinfo=None)

        tool.datetime = Clock
        events = []

        def sync(fd):
            if (tool.ROOT / 'releases.log').exists() and (tool.ROOT / 'hold').exists():
                events.append('audit fsync')
                if label == 'audit fsync error':
                    raise OSError('fixture failed fsync')
            os.fsync(fd)

        def unlink(name, **kwargs):
            if name == 'hold':
                events.append('unlink')
                assert 'audit fsync' in events
                assert json.loads((tool.ROOT / 'releases.log').read_text())['event'] == 'intent'
            return os.unlink(name, **kwargs)

        tool.os.fsync = sync
        tool.os.unlink = unlink
        with tool.directory() as root:
            if state == 'unreadable':
                (tool.ROOT / 'hold').write_text('bad JSON')
            elif state:
                tool.publish_json(root, 'hold', hold(state))
            before = {p.name: p.read_bytes() for p in tool.ROOT.iterdir()}
            if label == 'busy':
                with tool.lock(root, 'transaction.lock', fcntl.LOCK_EX):
                    code = tool.release_by_hand(root, reason)
            else:
                code = tool.release_by_hand(root, reason)
            after = {p.name: p.read_bytes() for p in tool.ROOT.iterdir()}
            if label == 'success':
                assert code == 0 and 'hold' not in after and 'unlink' in events
                assert [json.loads(l)['event'] for l in after['releases.log'].splitlines()] == ['intent', 'completion']
                assert tool.read_json(root, 'last-release')['kind'] == 'hand'
            elif label == 'audit fsync error':
                assert code == 1 and after['hold'] == before['hold'] and 'unlink' not in events
                assert 'last-release' not in after and 'releases.log' in after
            else:
                assert code == 1 and before == after
            print('hand release', label, code, events, flush=True)


def main():
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        directory = Path(tmp)
        tool = initialize(directory)
        test_preflight(tool)
        test_window(tool)
        contention(tool)
        dead_owner(tool)
        test_hand_release(directory)


if __name__ == '__main__':
    main()
