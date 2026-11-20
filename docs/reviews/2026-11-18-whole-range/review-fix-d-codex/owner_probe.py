"""Replay the exact YAML shell and embedded owner with a filesystem-only manager."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
INSTALLER = REPO / 'scripts/install-deploy-guard-hetzner.sh'
TARGET = 'b' * 40
PREVIOUS = 'a' * 40
NEXT = '1815674401-next'
JOBS = ('check-results', 'reconcile', 'fetch-contest-streak', 'nightly-preview')
BOUNDARIES = ('preflight', 'initial-window', 'armed', 'stop-old', 'drain', 'reset', 'head', 'sync',
              'restart', 'wait', 'candidate', 'dashboard', 'final-window', 'final-head', 'final-candidate',
              'activating', 'production-started', 'activation-verified', 'released', 'reported', 'page-claim')


def sources():
    installer = INSTALLER.read_text()
    tool = installer.split("<<'BTS_DEPLOY_HOLD_PYTHON'\n", 1)[1].split('\nBTS_DEPLOY_HOLD_PYTHON\n', 1)[0] + '\n'
    workflow = (REPO / '.github/workflows/deploy.yml').read_bytes()
    matches = list(re.finditer(rb'^          script: \|\n', workflow, re.M))
    assert len(matches) == 1
    literal = workflow[matches[0].end():]
    lines = literal.splitlines(keepends=True)
    assert all(line == b'\n' or line.startswith(b' ' * 12) for line in lines)
    payload = b''.join(line[12:] if line != b'\n' else line for line in lines)
    phase = payload.split(b"<<'BTS_DEPLOY_PHASE'\n", 1)[1].removesuffix(b'BTS_DEPLOY_PHASE\n')
    return tool, payload, phase


def log(case, event, **data):
    with (case / 'trace').open('a') as stream:
        stream.write(json.dumps(dict(event=event, **data), sort_keys=True) + '\n')


def read(case, name, fallback=None):
    try:
        return json.loads((case / name).read_bytes())
    except FileNotFoundError:
        return fallback


def wait_until(test, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = test()
        if value:
            return value
        time.sleep(0.004)
    raise TimeoutError('fixture wait deadline')


def append_file(case, name, value):
    # Single small root-local write, used only to exchange data in the model.
    temp = case / (name + '.writing')
    temp.write_text(json.dumps(value))
    temp.replace(case / name)


def guard(case):
    tool = ModuleType('review_guard')
    source, _, _ = sources()
    exec(compile(source, str(INSTALLER), 'exec'), vars(tool))
    tool.ROOT = case / 'guard'
    tool.PROJECT = str(case / 'project')
    tool.UID = os.geteuid()
    assert tool.UID != 0
    tool.time = SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda n: time.sleep(min(n, 0.01)))
    return tool


def gates(tool, case, owner_active):
    real_os = tool.os
    real_subprocess = tool.subprocess

    class Dispatched(Exception):
        pass

    def execute(argv0, argv):
        assert argv0 == '/bin/sh' and argv[:2] == ['/bin/sh', '-c']
        raise Dispatched()

    def show(argv, **kwargs):
        assert argv[:2] == ['/usr/bin/systemctl', 'show'] and kwargs['timeout'] <= 3
        state = owner_active if isinstance(owner_active, str) else ('active' if owner_active else 'inactive')
        return SimpleNamespace(returncode=0, stdout=(state + '\n').encode())

    tool.os = SimpleNamespace(**vars(real_os))
    tool.os.execv = execute
    tool.subprocess = SimpleNamespace(run=show, DEVNULL=subprocess.DEVNULL,
                                      TimeoutExpired=subprocess.TimeoutExpired)
    try:
        with tool.directory() as root, redirect_stdout(StringIO()):
            hold = tool.read_hold(root)
            status = None if hold is None else (hold['state'], hold['step'])
            scheduler = tool.clear(root)
            writers = []
            for job in JOBS:
                try:
                    assert tool.writer(job, ['/bin/sh', '-c', ':']) == 75
                except Dispatched:
                    writers.append(job)
            return {'hold': status, 'scheduler_clear': scheduler, 'writers': writers,
                    'scheduler_running': read(case, 'scheduler', False)}
    finally:
        tool.os = real_os
        tool.subprocess = real_subprocess


def configure(tool, case, selected=None, fault=None):
    head_reads = 0
    candidate_reads = 0
    window_reads = 0
    candidate_waited = False
    armed = False
    chosen = False
    original_publish = tool.publish_json
    original_release = tool.release
    original_lock = tool.lock

    def hit(name, step=None):
        nonlocal chosen
        log(case, 'boundary', phase=name)
        if name != selected or chosen:
            return
        chosen = True
        append_file(case, 'barrier', name)
        wait_until(lambda: (case / 'continue').exists())
        if fault == 'fail':
            raise tool.Refused(step or name)

    def call(argv, *, limit=10):
        assert limit <= 480
        log(case, 'command', argv=argv)
        nonlocal head_reads
        if argv == ['/usr/bin/id', '-u', 'bts']:
            return 0, b'1000\n'
        if 'stop' in argv and 'bts-scheduler.service' in argv:
            if read(case, 'scheduler', False):
                hit('stop-old')
            append_file(case, 'scheduler', False)
            return 0, b''
        if 'start' in argv and 'bts-scheduler.service' in argv:
            with tool.directory() as root:
                record = tool.read_hold(root)
                assert record['state'] == 'activating' and candidate_reads == 2
            append_file(case, 'scheduler', True)
            append_file(case, 'started', True)
            (Path(tool.PROJECT) / 'data/.heartbeat').write_text(json.dumps({
                'state': 'running', 'timestamp': datetime.now(timezone.utc).isoformat()}))
            hit('production-started', 'activation')
            return 0, b''
        if 'show' in argv and 'bts-scheduler.service' in argv:
            active = read(case, 'scheduler', False)
            if '--value' in argv:
                return 0, b'active\n' if active else b'inactive\n'
            return 0, b'ActiveState=active\nSubState=running\nResult=success\nExecMainStatus=0\n'
        if 'reset' in argv:
            hit('reset')
            append_file(case, 'head', TARGET)
        elif 'rev-parse' in argv:
            head_reads += 1
            hit('head' if head_reads == 1 else 'final-head', 'head')
            return 0, read(case, 'head', PREVIOUS).encode() + b'\n'
        elif 'sync' in argv:
            hit('sync')
        elif 'restart' in argv:
            hit('restart')
        else:
            raise AssertionError(argv)
        return 0, b''

    def window(*args, **kwargs):
        nonlocal window_reads
        window_reads += 1
        hit('initial-window' if window_reads == 1 else 'final-window', 'window')

    def candidate():
        nonlocal candidate_reads
        candidate_reads += 1
        if selected == 'page-claim':
            raise tool.Refused('canary-contest-state')
        hit('candidate' if candidate_reads == 1 else 'final-candidate', 'canary-contest-state')

    def publish(root, name, payload):
        nonlocal armed
        original_publish(root, name, payload)
        if name == 'hold' and payload['state'] == 'pending' and not armed:
            armed = True
            hit('armed', 'arm')
        elif name == 'hold' and payload['state'] == 'activating':
            hit('activating', 'release')
        elif name.endswith('.evidence'):
            if payload.get('activation') == 'verified' and payload.get('page') == 'not-attempted':
                hit('activation-verified', 'release')
            elif payload.get('page') == 'attempted':
                hit('page-claim', 'interrupted')
        elif name.endswith('.result') and payload.get('outcome') == 'committed':
            hit('reported', 'release')

    def release(root, record, kind):
        original_release(root, record, kind)
        hit('released', 'release')

    def lock(root, name, op, timeout=0):
        if name == 'writers.lock' and timeout:
            hit('drain')
        return original_lock(root, name, op, timeout)

    def sleep(seconds):
        nonlocal candidate_waited
        if seconds == 30 and not candidate_waited:
            candidate_waited = True
            hit('wait')
        else:
            time.sleep(min(seconds, 0.01))

    tool.preflight = lambda *args, **kwargs: (hit('preflight', 'guard'), (PREVIOUS, ['fixture-git']))[1]
    tool.verified_target = lambda *args, **kwargs: (PREVIOUS, ['fixture-git'])
    tool.call = call
    tool.window = window
    tool.canary = candidate
    tool.dashboard = lambda uid: hit('dashboard', 'canary-dashboard')
    tool.page = lambda record, stopped: (log(case, 'page', step=record['step'], stopped=stopped), True)[1]
    tool.write_stamp = lambda: True
    tool.publish_json = publish
    tool.release = release
    tool.lock = lock
    tool.time.sleep = sleep


def handoff(case, cmd):
    assert cmd[:5] == ['-n', '--', '/usr/bin/python3', '-I', '/usr/local/bin/bts-deploy-hold']
    if cmd[5:] == ['version']:
        print('bts-deploy-hold 1')
        return 0
    assert cmd[5:] == ['handoff', TARGET, 'workflow_dispatch', TARGET]
    tool = guard(case)

    def launch(argv, *, limit):
        assert argv[0] == '/usr/bin/systemd-run' and limit == 15
        run_id, target, digest = argv[-3:]
        assert ('--property=RuntimeMaxSec=1200' in argv and '--property=TimeoutStopSec=120' in argv
                and '--property=KillMode=mixed' in argv and '--property=Type=exec' in argv)
        assert any(token.endswith(f'reconcile --deploy-id {run_id}') for token in argv)
        phase = Path(argv[-4]).read_bytes()
        assert phase == sources()[2] and hashlib.sha256(phase).hexdigest() == digest
        # Use a real /bin/bash to exec its supplied phase at a safe interpreter boundary.
        # The four fixed absolute executable tokens in the phase are translated only here.
        sh = case / 'execute_phase'
        prefix = f'exec {sys.executable} {__file__} --service {case}'
        assert phase.count(b'exec /usr/bin/python3 -I /usr/local/bin/bts-deploy-hold owner') == 1
        sh.write_bytes(phase.replace(b'exec /usr/bin/python3 -I /usr/local/bin/bts-deploy-hold owner', prefix.encode()))
        with (tool.ROOT / tool.run_name(run_id, 'log')).open('ab') as capture:
            child = subprocess.Popen(['/bin/bash', str(sh), run_id, target, digest],
                                     stdin=subprocess.DEVNULL, stdout=capture, stderr=capture,
                                     start_new_session=True, cwd=case)
        append_file(case, 'owner', [child.pid, run_id])
        return 0, b''

    tool.call = launch
    with tool.directory() as root:
        return tool.handoff(root, TARGET, 'workflow_dispatch', TARGET)


def service(case, run_id, target, digest):
    tool = guard(case)
    selected, fault = read(case, 'fixture')
    configure(tool, case, selected, fault)
    with tool.directory() as root:
        try:
            result = tool.run_owner(root, run_id, target, digest)
        except (OSError, tool.Refused) as exc:
            log(case, 'uncaught-owner', exception=type(exc).__name__)
            result = 1
    append_file(case, 'owner-exited', result)
    return result


def gone(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


def local_call(case, limit):
    tool = guard(case)

    def interrupted(signum, frame):
        raise InterruptedError()

    signal.signal(signal.SIGTERM, interrupted)
    try:
        status, output = tool.call([sys.executable, __file__, '--local-worker', str(case)], limit=limit)
    except (InterruptedError, subprocess.TimeoutExpired) as exc:
        append_file(case, 'call-error', type(exc).__name__)
        return 41
    else:
        assert status == 0 and output == b'before child wait\nafter child wait\n'
        return 0


def local_worker(case):
    append_file(case, 'command-child', os.getpid())
    print('before child wait', flush=True)
    wait_until(lambda: (case / 'resume-child').exists(), seconds=4)
    append_file(case, 'late-write', True)
    print('after child wait', flush=True)
    return 0


def command_probe():
    for mode in ('complete', 'timeout', 'TERM', 'KILL'):
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            case = Path(temp)
            pid = None
            runner = subprocess.Popen([sys.executable, __file__, '--local-call', temp,
                                       '0.35' if mode == 'timeout' else '3'],
                                      stdin=subprocess.DEVNULL, cwd=case, start_new_session=True)
            try:
                pid = wait_until(lambda: read(case, 'command-child'), seconds=2)
                if mode == 'complete':
                    (case / 'resume-child').touch()
                    assert runner.wait(timeout=2) == 0 and read(case, 'late-write')
                elif mode == 'timeout':
                    assert runner.wait(timeout=2) == 41 and read(case, 'call-error') == 'TimeoutExpired'
                    assert gone(pid) and not read(case, 'late-write')
                elif mode == 'TERM':
                    runner.send_signal(signal.SIGTERM)
                    assert runner.wait(timeout=2) == 41
                    assert read(case, 'call-error') in ('InterruptedError', 'TimeoutExpired')
                    assert gone(pid) and not read(case, 'late-write')
                else:
                    runner.kill()
                    assert runner.wait(timeout=2) == -signal.SIGKILL
                    assert not gone(pid), 'the separate command session should still need manager cleanup'
                    os.killpg(pid, signal.SIGKILL)
                    wait_until(lambda: gone(pid), seconds=2)
                    assert not read(case, 'late-write')
                print('real guard.call child session', mode, 'late write', bool(read(case, 'late-write')),
                      'reported', read(case, 'call-error'), flush=True)
            finally:
                if runner.poll() is None:
                    runner.kill()
                    runner.wait(timeout=2)
                if pid is not None and not gone(pid):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


def trial(case, phase, fault, recovery):
    tool = guard(case)
    tool.ROOT.mkdir(mode=0o755)
    for key in ('transaction.lock', 'writers.lock', 'state.lock'):
        (tool.ROOT / key).touch(mode=0o644)
    (Path(tool.PROJECT) / 'data').mkdir(parents=True)
    append_file(case, 'scheduler', True)
    append_file(case, 'fixture', [phase, fault])
    _, payload, _ = sources()
    (case / 'workflow').write_bytes(payload)
    shim = case / 'sudo'
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{__file__}" --sudo "{case}" "$@"\n')
    shim.chmod(0o700)
    env = dict(PATH=str(case), TARGET_SHA=TARGET, RECOVER_SHA=TARGET, DEPLOY_EVENT='workflow_dispatch',
               PYTHONDONTWRITEBYTECODE='1')
    pid = None
    with (case / 'follower.log').open('wb') as follower_log:
        follower = subprocess.Popen(['/bin/bash', str(case / 'workflow')], cwd=case, env=env,
                                    stdin=subprocess.DEVNULL, stdout=follower_log, stderr=follower_log,
                                    start_new_session=True)
        try:
            pid, run_id = wait_until(lambda: read(case, 'owner'))
            assert wait_until(lambda: read(case, 'barrier')) == phase
            before = gates(tool, case, True)
            history = []
            for key in ('result', 'evidence'):
                history.append(read(tool.ROOT, tool.run_name(run_id, key)))
            if fault == 'lost-ssh':
                # At the reported boundary the follower may already have read the result.
                if follower.poll() is None:
                    follower.send_signal(signal.SIGHUP)
                rc = follower.wait(timeout=2)
                assert rc == -signal.SIGHUP or (phase == 'reported' and rc == 0), rc
            elif fault == 'KILL':
                os.kill(pid, signal.SIGKILL)
            elif fault in ('TERM', 'RuntimeMaxSec'):
                os.kill(pid, signal.SIGTERM)
            (case / 'continue').touch()
            if fault == 'KILL':
                wait_until(lambda: not (g := gates(tool, case, False))['scheduler_clear']
                           or g['hold'] is None, seconds=2)
                with tool.directory() as root:
                    def lock_released():
                        try:
                            with tool.lock(root, 'transaction.lock', tool.fcntl.LOCK_SH):
                                return True
                        except BlockingIOError:
                            return False
                    wait_until(lock_released, seconds=2)
            else:
                wait_until(lambda: (case / 'owner-exited').exists())
            before_recovery = gates(tool, case, False)
            configure(tool, case)
            recovery_code = None
            with tool.directory() as root, redirect_stdout(StringIO()) as stdout:
                if recovery == 'hook':
                    recovery_code = tool.reconcile(root, run_id)
                elif before_recovery['hold'] is not None:
                    recovery_code = tool.owned_deploy(root, NEXT, TARGET, '')
            after = gates(tool, case, False)
            evidence = read(tool.ROOT, tool.run_name(run_id, 'evidence'))
            result = read(tool.ROOT, tool.run_name(run_id, 'result'))
            next_result = read(tool.ROOT, tool.run_name(NEXT, 'result'))
            events = [json.loads(line) for line in (case / 'trace').read_text().splitlines()]
            pages = [event for event in events if event['event'] == 'page']
            if before['hold'] and before['hold'][0] != 'activating':
                assert before['scheduler_clear'] is False and before['writers'] == []
            if before['hold'] and before['hold'][0] == 'activating':
                assert before['scheduler_clear'] and before['writers'] == list(JOBS)
            if before_recovery['hold'] is not None:
                assert not before_recovery['scheduler_clear'] and not before_recovery['writers']
            if after['hold']:
                assert after['hold'][0] == 'failed' and not after['scheduler_running']
            if fault == 'lost-ssh' and phase != 'page-claim':
                assert result['outcome'] == 'committed' and evidence['activation'] == 'verified' and after['hold'] is None
            print(json.dumps(dict(phase=phase, fault=fault, recovery=recovery, before=before,
                                  before_recovery=before_recovery, after=after, history=history,
                                  evidence=evidence, result=result, next_result=next_result, pages=pages,
                                  recovery_code=recovery_code, hook_text=stdout.getvalue()), sort_keys=True), flush=True)
        finally:
            if follower.poll() is None:
                follower.kill()
                follower.wait(timeout=2)
            if pid is not None and not (case / 'owner-exited').exists() and fault != 'KILL' and not gone(pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def main():
    if sys.argv[1] == '--command-probe':
        command_probe()
        return 0
    if sys.argv[1] == '--local-call':
        return local_call(Path(sys.argv[2]), float(sys.argv[3]))
    if sys.argv[1] == '--local-worker':
        return local_worker(Path(sys.argv[2]))
    if sys.argv[1] == '--sudo':
        return handoff(Path(sys.argv[2]), sys.argv[3:])
    if sys.argv[1] == '--service':
        return service(Path(sys.argv[2]), *sys.argv[3:])
    fault, recovery, *phases = sys.argv[1:]
    assert fault in ('fail', 'TERM', 'KILL', 'lost-ssh', 'RuntimeMaxSec')
    assert recovery in ('hook', 'preflight')
    _, payload, phase = sources()
    print('workflow', hashlib.sha256(payload).hexdigest(), 'phase', hashlib.sha256(phase).hexdigest(), flush=True)
    for boundary in phases or BOUNDARIES:
        with tempfile.TemporaryDirectory(dir=HERE) as temp:
            trial(Path(temp), boundary, fault, recovery)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
