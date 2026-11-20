from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Event, Thread, current_thread
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from preflight_r2 import hold
from review_env import GOOD, HEAD, OLD, PREVIOUS, RUN, TARGET, capture, fixture

ORIGIN = datetime.fromisoformat('2027-07-15T23:59:59.800-04:00').timestamp()
MIDNIGHT = ORIGIN + 1
ET = ZoneInfo('America/New_York')
FOLLOWUP = '1800000001-followup'
RECOVERY = '1800000002-recovery'


def model(tool, home, *, now=ORIGIN, trigger=lambda name: None, bad_target=False, old_dashboard=False):
    values = SimpleNamespace(now=now, starts=0, stops=0, dashboard=old_dashboard, canaries=0, events=[])
    values.scheduler, values.pager = home / 'scheduler', home / 'page'
    values.heartbeat = Path(tool.PROJECT) / 'data/.heartbeat'
    values.scheduler.touch()
    tool.time = SimpleNamespace(time=lambda: values.now, monotonic=time.monotonic, sleep=lambda seconds: None)
    tool.os = SimpleNamespace(**vars(os))

    def track(name):
        values.events.append(name)
        trigger(name)

    def previous(*args, **kwargs):
        track('target')
        if bad_target:
            raise tool.Refused('guard-stale', PREVIOUS)
        return PREVIOUS, ['fixture-git']

    def stop(uid):
        track('scheduler-stop')
        values.scheduler.unlink(missing_ok=True)
        values.stops += 1
        return True

    def stop_dashboard(uid):
        with tool.directory() as directory:
            assert tool.read_hold(directory)['state'] == 'pending'
            try:
                with tool.lock(directory, 'writers.lock', tool.fcntl.LOCK_SH):
                    raise AssertionError('dashboard stop outside writer exclusion')
            except BlockingIOError:
                pass
        track('dashboard-stop')
        values.dashboard = False
        return True

    def call(argv, *, limit=10):
        if 'deploy-canary' in argv:
            values.canaries += 1
            sampled = datetime.fromtimestamp(values.now, ET).isoformat()
            track(f'canary-{values.canaries}-sampled')
            valid = sampled.startswith(GOOD)
            data = f'deploy canary {sampled}: ' + ('live contest state accepted (dated, streak 12)' if valid else 'contest state refused')
            return int(not valid), (data + '\n').encode()
        if 'status' in argv or 'sync' in argv:
            return 0, b''
        raise AssertionError((argv, limit))

    def activate(uid, moment):
        track('activation-start-request')
        values.scheduler.touch()
        values.starts += 1
        heartbeat_time = datetime.fromtimestamp(values.now + 0.01, timezone.utc).isoformat()
        values.heartbeat.write_text(json.dumps({'state': 'sleeping', 'timestamp': heartbeat_time,
            'sleeping_until': datetime.fromtimestamp(values.now + 7200, timezone.utc).isoformat()}))
        values.now += 0.03
        track('activation-verified')

    def page(record, stopped):
        track('before-ping')
        with values.pager.open('a') as stream:
            stream.write(f'{record["deploy_id"]} {record["step"]} {stopped}\n')
        return 204

    original_transition, original_publish = tool.transition, tool.publish_json

    def transition(root, record, state, step):
        track(f'before-{state}')
        result = original_transition(root, record, state, step)
        track(state)
        return result

    def publish(root, name, payload):
        if name == 'last-release':
            track('before-release-marker')
        if name.endswith('.evidence') and payload.get('activation') == 'verified':
            track('activation-evidence')
        original_publish(root, name, payload)
        if name == 'last-release':
            track('release-marker')

    def unlink(name, **kwargs):
        if name == 'hold':
            track('before-unlink-system-call')
        result = os.unlink(name, **kwargs)
        if name == 'hold':
            track('after-unlink-system-call')
        return result

    def require(argv, step, expected=None, *, limit=10):
        track(step)
        if 'deploy-canary' in argv:
            code, output = call(argv, limit=limit)
            if code != 0:
                raise tool.Refused(step)
            return output
        return expected or b''

    tool.preflight = tool.verified_target = previous
    tool.bts_uid = lambda: 1000
    tool.verified_stop = stop
    if hasattr(tool, 'verified_stop_dashboard'):
        tool.verified_stop_dashboard = stop_dashboard
    tool.require, tool.call = require, call
    tool.dashboard = lambda uid: track('dashboard-verified')
    tool.window = lambda *args, **kw: track('window')
    tool.activate_scheduler, tool.page = activate, page
    tool.write_stamp = lambda: True
    tool.transition, tool.publish_json, tool.os.unlink = transition, publish, unlink
    return values


def liveness(rev):
    with fixture(rev) as (tool, home):
        data = hold('activating', 'activation')

        def query(argv, **kwargs):
            assert argv[:2] == ['/usr/bin/systemctl', 'show'] and kwargs['timeout'] == 3
            output = f'ActiveState={state}\nMainPID={manager_pid}\n' if 'MainPID' in argv else state + '\n'
            return SimpleNamespace(returncode=0, stdout=output.encode())

        tool.subprocess = SimpleNamespace(run=query, DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired)
        info = (tool.ROOT / 'transaction.lock').stat()
        outcomes = {}
        with tool.directory() as directory, tool.lock(directory, 'transaction.lock', tool.fcntl.LOCK_EX):
            tool.publish_json(directory, 'hold', data)
            for state, manager_pid, holder, inode in (
                    ('active', 456, 456, info.st_ino), ('active', 456, 457, info.st_ino),
                    ('deactivating', 456, 456, info.st_ino), ('active', 0, 456, info.st_ino),
                    ('active', 456, 456, info.st_ino + 1)):
                tool.PROC_LOCKS.write_text(f'9: FLOCK ADVISORY WRITE {holder} '
                                          f'{os.major(info.st_dev):x}:{os.minor(info.st_dev):x}:{inode} 0 EOF\n')
                outcomes[(state, manager_pid, holder, inode == info.st_ino)] = tool.clear(directory)
        expected = [True, False, False, False, False] if rev == HEAD else [True] * 5
        assert list(outcomes.values()) == expected, outcomes
        print(rev, 'F2 active/PID/lock matrix', outcomes)


def old_incidents(rev):
    for cause in ('stale-target', 'uid', 'stop'):
        with fixture(rev) as (tool, home):
            values = model(tool, home, bad_target=cause == 'stale-target')
            with tool.directory() as directory:
                tool.publish_json(directory, 'hold', hold('activating', 'activation', stamp=int(ORIGIN - 5)))
                if cause == 'stale-target':
                    result = tool.owned_deploy(directory, FOLLOWUP, TARGET, '')
                else:
                    def bad(*args):
                        raise tool.Refused('id') if cause == 'uid' else OSError('shim stop failure')
                    if cause == 'uid':
                        tool.bts_uid = bad
                    else:
                        tool.verified_stop = bad
                    try:
                        result = tool.reconcile(directory, RUN)
                    except (tool.Refused, OSError) as exc:
                        result = type(exc).__name__
                record = tool.read_hold(directory)
                original = tool.read_json(directory, tool.run_name(RUN, 'result'))
                sent = values.pager.read_text().splitlines() if values.pager.exists() else []
                print(rev, 'F2 original', cause, result, record['state'], 'scheduler', values.scheduler.exists(),
                      'old result', original, 'pages', sent)
                assert bool(sent) == (rev == HEAD)


def wait_killed(pid):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        child, status = os.waitpid(pid, os.WNOHANG)
        if child:
            assert os.waitstatus_to_exitcode(status) == -signal.SIGKILL
            return
        time.sleep(0.01)
    os.kill(pid, signal.SIGKILL)
    raise TimeoutError('hook fixture did not exit at boundary')


def kill_hook(rev, phase):
    with fixture(rev) as (tool, home):
        def kill(name):
            if name == phase:
                os.kill(os.getpid(), signal.SIGKILL)

        case = model(tool, home, trigger=kill, bad_target=True)
        with tool.directory() as directory:
            tool.publish_json(directory, 'hold', hold('activating', 'activation', stamp=int(ORIGIN - 5)))
            sys.stdout.flush()
            sys.stderr.flush()
            pid = os.fork()
            if pid == 0:
                os._exit(tool.reconcile(directory, RUN))
            wait_killed(pid)
            name = tool.run_name(RUN, 'evidence')
            before = tool.read_json(directory, name)
        with fixture(rev) as (fresh, _):
            fresh.ROOT, fresh.PROJECT = tool.ROOT, tool.PROJECT
            reset = model(fresh, home, bad_target=True)
            if phase == 'before-ping':
                reset.scheduler.unlink(missing_ok=True)
            with fresh.directory() as directory:
                assert fresh.owned_deploy(directory, FOLLOWUP, TARGET, '') == 1
                after = fresh.read_json(directory, name)
                result = fresh.read_json(directory, fresh.run_name(RUN, 'result'))
        sent = case.pager.read_text().splitlines() if case.pager.exists() else []
        if rev == HEAD:
            assert after['page'] == 'sent 204' and after['stop'] == 'verified' and result['outcome'] == 'failed' and len(sent) == 1
        else:
            assert not sent and result is None
        print(rev, 'F2/F11 actual hook SIGKILL', phase, before, after, result, sent)


def late_hook():
    with fixture(HEAD) as (tool, home):
        values = model(tool, home)
        entered, continue_hook = Event(), Event()
        trace = []
        original_lock, original_stop = tool.lock, tool.verified_stop

        def lock(root, name, mode, timeout=0):
            if current_thread().name == 'hook' and name == 'transaction.lock':
                assert timeout == 60
                raise BlockingIOError('modeled full 60s contention')
            return original_lock(root, name, mode, timeout)

        def stop(uid):
            if current_thread().name == 'hook':
                entered.set()
                assert continue_hook.wait(3)
            return original_stop(uid)

        tool.lock, tool.verified_stop = lock, stop
        with tool.directory() as directory:
            tool.publish_json(directory, 'hold', hold('activating', 'activation', stamp=int(ORIGIN - 5)))
            thread = Thread(target=lambda: trace.append(tool.reconcile(directory, RUN)), name='hook', daemon=True)
            thread.start()
            try:
                assert entered.wait(3)
                assert tool.owned_deploy(directory, FOLLOWUP, TARGET, '') == 1
                assert tool.owned_deploy(directory, RECOVERY, TARGET, '') == 0
                assert values.scheduler.exists() and tool.read_hold(directory) is None
            finally:
                continue_hook.set()
                thread.join(timeout=4)
            assert trace == [1] and not values.scheduler.exists() and tool.read_hold(directory) is None
            result = tool.read_json(directory, tool.run_name(RECOVERY, 'result'))
            assert result['outcome'] == 'committed'
            assert len(values.pager.read_text().splitlines()) == 1
        print(HEAD, 'F2 old hook stopped newer committed scheduler; hold clear, original incident only paged', result)


def midnight(rev):
    cuts = ('before-activating', 'activating', 'activation-start-request', 'activation-verified',
            'canary-3-sampled', 'activation-evidence', 'before-release-marker', 'release-marker',
            'before-unlink-system-call', 'after-unlink-system-call', None)
    for cut in cuts:
        if cut == 'canary-3-sampled' and rev == OLD:
            continue
        with fixture(rev) as (tool, home):
            def inject(name):
                if name == cut:
                    values.now = MIDNIGHT

            values = model(tool, home, trigger=inject)
            with tool.directory() as directory:
                result = capture(lambda: tool.owned_deploy(directory, RUN, TARGET, ''))
                record = tool.read_hold(directory)
                marker = tool.read_json(directory, 'last-release')
            want_fail = rev == HEAD and cut not in ('before-unlink-system-call', 'after-unlink-system-call', None)
            assert bool(result.rc) == want_fail, (rev, cut, result)
            if want_fail:
                assert record['state'] == 'failed' and record['step'] == 'activation' and not values.scheduler.exists()
                assert len(values.pager.read_text().splitlines()) == 1
            else:
                assert record is None and values.scheduler.exists() and marker is not None
            print(rev, 'F6 boundary', cut, 'rc', result.rc, 'start', values.starts, 'release', marker,
                  'classifications', [line for line in result.stdout.splitlines() if line.startswith('deploy canary')])


def window_release(rev):
    for label, state, age, wake, head_code in (
            ('stopped wake elapsed', 'sleeping', 7200, -901, 0), ('stopped idle', 'idle_end_of_day', 7200, None, 0),
            ('stopped missing', None, 0, None, 0), ('stopped stale running', 'running', 301, None, 0),
            ('fresh running', 'running', 30, None, 1), ('fresh waiting', 'waiting_for_games', 300, None, 1),
            ('near wake', 'sleeping', 3, 29, 1), ('safe manual wake', 'sleeping', 3, 31, 0)):
        with fixture(rev) as (tool, home):
            window = tool.window
            values = model(tool, home)
            tool.window = window
            tool.preflight = lambda *args, **kw: (TARGET, ['fixture-git'])
            if state is not None:
                row = dict(timestamp=datetime.fromtimestamp(values.now-age, timezone.utc).isoformat(), state=state)
                if wake is not None:
                    row['sleeping_until'] = datetime.fromtimestamp(values.now+wake, timezone.utc).isoformat()
                values.heartbeat.write_text(json.dumps(row))
            with tool.directory() as directory:
                tool.publish_json(directory, 'hold', hold('failed', 'sync', stamp=int(ORIGIN - 5)))
                result = capture(lambda: tool.release_by_hand(directory, 'verified stopped and observed MLB'))
                released = tool.read_hold(directory) is None
            assert result.rc == (head_code if rev == HEAD else 1) and released == (result.rc == 0)
            print(rev, 'F9', label, result.rc, 'held', not released, result.stdout.strip().replace('\n', ' | '))


def dashboard_arm(rev):
    for originally_stopped in (True, False):
        with fixture(rev) as (tool, home):
            seen = []

            def sample(name):
                if name == 'pending':
                    seen.append(values.dashboard)

            values = model(tool, home, trigger=sample, old_dashboard=not originally_stopped)
            with tool.directory() as directory:
                assert capture(lambda: tool.owned_deploy(directory, RUN, TARGET, '')).rc == 0
            assert ('dashboard-stop' in values.events) == (rev == HEAD)
            if rev == HEAD:
                assert values.events.index('pending') < values.events.index('dashboard-stop') < values.events.index('reset')
            assert seen == [not originally_stopped]
            print(rev, 'F4 manual stop verified', originally_stopped, 'old process serves at durable arm', seen[0], values.events)


if __name__ == '__main__':
    revision = sys.argv[1]
    chosen = sys.argv[2:] or ['owner', 'midnight', 'release', 'dashboard']
    if 'owner' in chosen:
        liveness(revision)
        old_incidents(revision)
        for boundary in ('scheduler-stop', 'before-ping'):
            kill_hook(revision, boundary)
        if revision == HEAD:
            late_hook()
    if 'midnight' in chosen:
        midnight(revision)
    if 'release' in chosen:
        window_release(revision)
    if 'dashboard' in chosen:
        dashboard_arm(revision)
