from contextlib import ExitStack
from datetime import date, datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
from threading import Event, Thread
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DAY = '2026-06-18'
NOW = datetime.fromisoformat(DAY + 'T16:00:00+00:00')
REFUSED = b'deploy hold present: saver changes are refused'


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob('*') if p.is_file()}


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


class Calendar(date):
    @classmethod
    def today(cls):
        return NOW.date()


def blocked(*args, **kwargs):
    raise AssertionError('unexpected external call')


def request(web, method, path):
    body = b'expected_prior=active&new_state=used' if method == 'POST' else b''
    obj = object.__new__(web.Handler)
    obj.path = path
    obj.command = method
    obj.requestline = f'{method} {path} HTTP/1.1'
    obj.request_version = 'HTTP/1.1'
    obj.headers = {'Content-Length': str(len(body)), 'Host': 'fixture', 'Origin': 'http://fixture'}
    obj.client_address = ('127.0.0.1', 12345)
    obj.rfile = BytesIO(body)
    obj.wfile = BytesIO()
    getattr(obj, 'do_' + method)()
    return obj.wfile.getvalue()


def status(raw):
    return int(raw.split(b' ', 2)[1])


def prepare(place, variant):
    from bts.saver_state import transition_saver_state

    picks = place / 'picks'
    state = picks / 'account_state'
    state.mkdir(parents=True)
    (state / 'contest_streak.json').write_text(json.dumps({
        'schema_version': 'bts_contest_streak_auto_v1', 'active_streak': 10, 'best_streak': 10,
        'source': 'mlb_bts_profile', 'source_date': DAY}))
    transition_saver_state(picks, expected_prior='uninitialized', new_state='active', season=2026, source='fixture')
    guard = place / 'guard'
    guard.mkdir(mode=0o755)
    for name in ('writers.lock', 'transaction.lock', 'state.lock'):
        (guard / name).touch(mode=0o644)
    (place / 'leaderboard').mkdir()
    (place / 'health_state').mkdir()
    (place / 'validation').mkdir()
    (place / 'data' / 'hetzner_results' / 'valid').mkdir(parents=True)
    (place / 'data' / 'hetzner_results' / 'valid' / 'boxes.json').write_text('{}')
    if variant == 'populated':
        (place / 'heartbeat').write_text(json.dumps({'timestamp': NOW.isoformat(), 'state': 'running'}))
        (picks / (DAY + '.json')).write_text(json.dumps({'date': DAY, 'pick': {
            'batter_name': 'Fixture batter', 'team': 'NYY', 'p_game_hit': 0.82}, 'result': 'hit'}))
    return picks, guard


def exercise(root, implementation):
    import bts
    import bts.web as web

    print(str(Path(bts.__file__).resolve()), flush=True)
    paths = ('/', '/dashboard', '/health', '/api/live', '/api/live?date=2026-06-17',
             '/api/live-html', '/api/live-html?date=2026-06-17', '/api/audit-progress',
             '/api/audit-progress?provider=nonsense&dir=valid', '/api/audit-progress?provider=hetzner&dir=missing',
             '/api/audit-progress?provider=hetzner&dir=valid', '/api/leaderboard')

    def fake_games(pick_data):
        return [], {'game_status': 'final', 'fixture_batter': pick_data['pick']['batter_name']}

    for variant in ('missing', 'populated'):
        place = root / variant
        picks, guard = prepare(place, variant)
        hold = guard / 'hold'
        form = '/saver/transition'
        with ExitStack() as stack:
            for key, val in {'PICKS_DIR': picks, 'PROJECT_ROOT': place, 'LEADERBOARD_DIR': place / 'leaderboard',
                             'HEALTH_STATE_DIR': place / 'health_state', 'VALIDATION_DIR': place / 'validation',
                             'HEARTBEAT_PATH': place / 'heartbeat', 'datetime': Clock,
                             'fetch_bluesky_posts': lambda: [], 'load_orchestrator_config': lambda: {},
                             '_build_live_game_data': fake_games,
                             'scan_audit_progress': lambda *a, **kw: {'fixture': 1}}.items():
                stack.enter_context(patch.object(web, key, val))
            stack.enter_context(patch('datetime.date', Calendar))
            stack.enter_context(patch('bts.heartbeat.datetime', Clock))
            stack.enter_context(patch.dict(os.environ, {'BTS_TEST_DEPLOY_GUARD_DIR': str(guard)}))
            stack.enter_context(patch.object(web.Handler, 'date_time_string', return_value='Thu, 18 Jun 2026 16:00:00 GMT'))
            stack.enter_context(patch.object(socket.socket, 'connect', blocked))
            stack.enter_context(patch.object(web.subprocess, 'run', side_effect=blocked))
            before = snapshot(picks)
            clear_get = [request(web, 'GET', path) for path in paths]
            assert all(status(response) in (200, 400, 404, 503) for response in clear_get)
            assert status(clear_get[-2]) == 200
            print('audit-route fixture response', clear_get[-2].split(b'\r\n\r\n')[-1].decode(), flush=True)
            assert b'TODAY' in clear_get[0] or b'Streak' in clear_get[0]
            hold.write_text('durable hold fixture')
            held_get = [request(web, 'GET', path) for path in paths]
            assert clear_get == held_get and snapshot(picks) == before
            for path, raw in zip(paths, clear_get):
                print(variant, 'GET', path, status(raw), hashlib.sha256(raw).hexdigest(), flush=True)
            result = request(web, 'POST', form)
            if implementation == 'base':
                assert status(result) == 303 and snapshot(picks) != before
                print('old loaded dashboard while held: 303; saver state and audit changed', flush=True)
                continue
            assert status(result) == 409 and result.endswith(REFUSED) and snapshot(picks) == before
            hold.unlink()
            (guard / 'writers.lock').unlink()
            result = request(web, 'POST', form)
            assert status(result) == 409 and result.endswith(REFUSED) and snapshot(picks) == before
            (guard / 'writers.lock').touch(mode=0o644)
            with (guard / 'writers.lock').open('rb') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = request(web, 'POST', form)
                assert status(result) == 409 and result.endswith(REFUSED) and snapshot(picks) == before
                hold.write_text('arm wins')
            hold.unlink()
            release = Event()
            inside = Event()
            response = []
            original = web.saver_transition_response

            def delayed(*args, **kwargs):
                inside.set()
                assert release.wait(3)
                return original(*args, **kwargs)

            with patch.object(web, 'saver_transition_response', side_effect=delayed):
                worker = Thread(target=lambda: response.append(request(web, 'POST', form)), daemon=True)
                worker.start()
                try:
                    assert inside.wait(3)
                    with (guard / 'writers.lock').open('rb') as lock:
                        try:
                            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            pass
                        else:
                            raise AssertionError('arm acquired while saver was admitted')
                finally:
                    release.set()
                    worker.join(timeout=4)
            assert response and status(response[0]) == 303 and snapshot(picks) != before
            print('current dashboard: exact 409 for held, missing lock and arm wins; POST-first excludes arm', flush=True)


def main():
    if len(sys.argv) > 1:
        exercise(Path(sys.argv[2]), sys.argv[1])
        return
    gets = []
    with tempfile.TemporaryDirectory(dir=HERE) as directory:
        for version in ('base', 'head'):
            source = HERE / 'base/src' if version == 'base' else REPO / 'src'
            fixture = Path(directory) / 'fixture'
            fixture.mkdir()
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), version, str(fixture)],
                                    env={**os.environ, 'PYTHONPATH': str(source)}, cwd=fixture,
                                    stdin=subprocess.DEVNULL, capture_output=True, timeout=15, text=True)
            print(result.stdout, end='')
            assert result.returncode == 0, result.stderr
            assert Path(result.stdout.splitlines()[0]).is_relative_to(source), result.stdout
            gets.append([line for line in result.stdout.splitlines() if ' GET ' in line])
            shutil.rmtree(fixture)
    assert gets[0] == gets[1] and len(gets[0]) == 24
    print('24 raw HTTP GET byte hashes identical to base, for all six branches and default page aliases')


if __name__ == '__main__':
    main()
