from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import fcntl
from importlib.util import module_from_spec, spec_from_file_location
from io import StringIO
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOW = datetime(2027, 7, 15, 18, tzinfo=timezone.utc)
URL = 'https://hc-ping.com/fixture-only'
RUN = '1815674400-review'


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)


class Response:
    status = 204

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return b''


def invoke(module, guard, heartbeat, *, ping=True, n=None, unit='inactive', body=None):
    output = StringIO()
    errors = StringIO()
    requests = []

    def ping_request(req, timeout):
        assert req.full_url in (URL, URL + '/fail') and timeout <= 10
        requests.append((req.get_method(), req.full_url[len(URL):], (req.data or b'').decode()))
        return Response()

    def subprocess_request(argv, **kwargs):
        assert kwargs['timeout'] <= 3
        if argv[0] == '/usr/bin/python3':
            assert argv[-2:] == ['status', '--json']
            return SimpleNamespace(returncode=1, stdout=json.dumps(body).encode())
        assert argv == ['/usr/bin/systemctl', 'show', f'bts-deploy-{RUN}.service', '-p', 'ActiveState', '--value']
        return SimpleNamespace(returncode=0, stdout=(unit + '\n').encode())

    args = ['monitor', '--heartbeat-path', str(heartbeat), '--churn-state', str(heartbeat.parent / 'churn')]
    if ping:
        args += ['--ping-url', URL]
    with patch.object(module, 'datetime', Clock), patch.object(module, 'read_nrestarts', return_value=n), \
         patch.dict(os.environ, {'BTS_TEST_DEPLOY_GUARD_DIR': str(guard)}), patch.object(sys, 'argv', args), \
         patch.object(module.subprocess, 'run', side_effect=subprocess_request), \
         patch('urllib.request.urlopen', side_effect=ping_request), \
         patch('urllib.request.build_opener', return_value=SimpleNamespace(open=ping_request)), \
         redirect_stdout(output), redirect_stderr(errors):
        try:
            module.main()
        except SystemExit as exc:
            code = exc.code
        except (ValueError, TypeError, AttributeError) as exc:
            code = ('raised', type(exc).__name__, str(exc))
        else:
            raise AssertionError('monitor returned without exit code')
    return code, output.getvalue(), errors.getvalue(), requests


def record(state, age=0, until=None):
    value = {'state': state, 'timestamp': (NOW - timedelta(seconds=age)).isoformat()}
    if until is not None:
        value['sleeping_until'] = (NOW + timedelta(seconds=until)).isoformat()
    return json.dumps(value)


def main():
    old = load('monitor_base', HERE / 'base/scripts/check_heartbeat.py')
    new = load('monitor_head', REPO / 'scripts/check_heartbeat.py')
    payloads = [None, '', '{garbage', '[]', 'null', '{}', '{"timestamp":13}',
                '{"timestamp":"2027-07-15T18:00:00"}', record('stalled'), record('teleporting')]
    payloads += [record(state, age) for state, age in itertools.product(
        ['running', 'waiting_for_games', 'idle_end_of_day'], [-1, 0, 299, 300, 301, 599, 600, 601, 5399, 5400, 5401, 5700])]
    payloads += [record('sleeping', age, wake) for age, wake in itertools.product(
        [0, 3600], [None, -901, -601, -600, -599, 0, 1800, 7200])]
    with tempfile.TemporaryDirectory(dir=HERE) as temp:
        root = Path(temp)
        guard = root / 'guard'
        guard.mkdir(mode=0o755)
        for name in ('state.lock', 'transaction.lock'):
            (guard / name).touch(mode=0o644)
        hb = root / 'fixture-hb'
        churn = root / 'churn'
        churn_seed = json.dumps({'samples': [
            {'ts': (NOW - timedelta(minutes=10)).isoformat(), 'n': 2}, {'ts': 'invalid'},
        ]})
        comparisons = 0
        for payload, ping, n in itertools.product(payloads, (True, False), (None, 2, 5)):
            if payload is None:
                hb.unlink(missing_ok=True)
            else:
                hb.write_text(payload)
            churn.write_text(churn_seed)
            before = invoke(old, guard, hb, ping=ping, n=n)
            old_churn = churn.read_bytes()
            churn.write_text(churn_seed)
            after = invoke(new, guard, hb, ping=ping, n=n)
            assert before == after, (payload, ping, n, before, after)
            assert churn.read_bytes() == old_churn
            comparisons += 1
        print('Python', sys.version.split()[0], 'no-hold comparisons', comparisons)

        hb.write_text(record('sleeping', 20, 7200))
        print('without guard, same fresh sleeping input')
        for module in (old, new):
            result = invoke(module, root / 'absent-guard', hb)
            print(module.__name__, result)
        (guard / 'state.lock').unlink()
        print('missing state.lock, with neither record', invoke(new, guard, hb))
        (guard / 'state.lock').touch(mode=0o644)

        hold = guard / 'hold'
        hold.write_text('{fixture; guard status shim decides}')
        for state, age, unit, locked in itertools.product(
                ('pending', 'activating', 'failed'), (30, 1380, 1381), ('active', 'deactivating', 'inactive'), (True, False)):
            data = {'state': state, 'step': 'arm', 'target': 'a' * 40, 'deploy_id': RUN,
                    'armed_at': int(NOW.timestamp()) - age}
            with (guard / 'transaction.lock').open('rb') as channel:
                if locked:
                    fcntl.flock(channel, fcntl.LOCK_EX | fcntl.LOCK_NB)
                for _ in range(2):
                    result = invoke(new, guard, hb, body=data, unit=unit)
                    suppress = state != 'failed' and age <= 1380 and locked and unit != 'inactive'
                    if suppress:
                        assert result[0] == 0 and not result[3] and 'no ping' in result[1], result
                    else:
                        assert result[0] == 1 and result[3][0][:2] == ('POST', '/fail'), result
        hold.unlink()
        os.mkfifo(hold)
        assert invoke(new, guard, hb)[3][0][:2] == ('POST', '/fail')
        hold.unlink()
        hold.symlink_to(root / 'missing')
        assert invoke(new, guard, hb)[3][0][:2] == ('POST', '/fail')
        hold.unlink()
        print('hold overlay: 108 hold runs plus FIFO and dangling symlink, using fresh sleeping control')

        released = NOW - timedelta(seconds=30)
        for kind in ('hand', 'deploy'):
            (guard / 'last-release').write_text(json.dumps({
                'released_at': int(released.timestamp()), 'kind': kind, 'deploy_id': RUN}))
            for age in (0, 29, 30, 31, 601):
                hb.write_text(record('running', age))
                result = invoke(new, guard, hb)
                assert result[0] == int(age >= 30), result
                assert result[3][0][:2] == (('POST', '/fail') if age >= 30 else ('GET', '')), result
            hb.write_text(record('sleeping', 30, 7200))
            assert invoke(new, guard, hb)[3][0][:2] == ('POST', '/fail')
            hb.write_text(record('sleeping', 29, 7200))
            assert invoke(new, guard, hb)[3][0][:2] == ('GET', '')
        print('release freshness: old/equal fail materially even asleep, newer fresh succeeds')


if __name__ == '__main__':
    main()
