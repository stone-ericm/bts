"""Run scorer commands and the scratch predicate independently on each input."""
from contextlib import contextmanager
import json
import os
import signal
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
if os.environ.get('F4_SCRATCH') == '1':
    sys.path.insert(0, str(ROOT / '.codex-review/scratch-f4/src'))
sys.path.insert(0, str(ROOT / 'docs/reviews/2026-11-18-whole-range/closing-test/codex-blind'))

from bts import day_outcome, picks, scheduler
import offline
from offline import World, at
from opening import OPEN, DAY2

offline.ROOT = ROOT


class Blocked(BaseException):
    pass


@contextmanager
def read_limit():
    def expire(*_):
        raise Blocked()

    previous = signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, .12)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def create(path, payload):
    if payload == 'absent':
        return
    if payload == 'directory':
        path.mkdir()
    elif payload == 'fifo':
        os.mkfifo(path)
    elif payload == 'dangling':
        path.symlink_to('no-such-file')
    elif payload == 'loop':
        path.symlink_to(path.name)
    elif payload == 'unreadable':
        path.write_text('{}')
        path.chmod(0)
    else:
        path.write_bytes(payload)


cases = {
    'absent': 'absent', 'empty': b'', 'array': b'[]', 'object': b'{}', 'string': b'"x"',
    'utf8': b'\xff', 'syntax': b'{', 'directory': 'directory', 'fifo': 'fifo', 'unreadable': 'unreadable',
    'dangling symlink': 'dangling', 'symlink loop': 'loop',
    'untagged string count': b'{"streak":"3"}', 'untagged bool count': b'{"streak":true}',
    'untagged negative count': b'{"streak":-1}', 'untagged float count': b'{"streak":3.0}',
    'untagged null count': b'{"streak":null}', 'raw bool': b'true', 'raw negative': b'-1',
    'raw float': b'3.0', 'raw null': b'null', 'raw huge allowed': b'1' + b'0' * 99,
    'raw huge rejected': b'1' + b'0' * 4999, 'object huge rejected': b'{"streak":' + b'9' * 5000 + b'}',
    'unknown season': b'{"season":"2027","streak":3}', 'prior season': b'{"season":2026,"streak":3}',
    'prior season string count': b'{"season":2026,"streak":"3"}',
    'same season count omitted': b'{"season":2027}',
    'same season string count': b'{"season":2027,"streak":"3"}',
    'same season bool count': b'{"season":2027,"streak":true}',
    'same season negative count': b'{"season":2027,"streak":-1}',
    'same season float count': b'{"season":2027,"streak":3.0}',
    'same season huge count': b'{"season":2027,"streak":' + b'1' + b'0' * 99 + b'}',
    'same season maximal digit carry': b'{"season":2027,"streak":' + b'9' * 4300 + b'}',
    'same season null count': b'{"season":2027,"streak":null}',
    'same season string count, false saver': b'{"season":2027,"streak":"3","saver_available":false}',
    'same season string count, already applied': b'{"season":2027,"streak":"3","applied_dates":["2027-03-24"]}',
    'wrong applied_dates': b'{"season":2027,"streak":3,"applied_dates":"2027-03-24"}',
    'duplicate marker elements': b'{"season":2027,"streak":3,"applied_dates":[null,"2027-03-24","2027-03-24"]}',
    'wrong saver object': b'{"season":2027,"streak":10,"saver_available":{}}',
    'deep': b'{"nested":' + b'[' * 2300 + b'0' + b']' * 2300 + b'}',
    'NaN in ignored member': b'{"season":2027,"streak":3,"other":NaN}',
    'Infinity as count': b'{"season":2027,"streak":1e400}',
    'duplicate streak keys': b'{"season":2027,"streak":"3","streak":3}',
}


def invoke_limited(call):
    try:
        with read_limit():
            return call()
    except Blocked:
        return None


def status(result):
    if result is None:
        return 'blocks'
    if result.exit_code:
        return f'{result.exit_code}/{type(result.exception).__name__}'
    return '0'


with TemporaryDirectory() as tmp, World(OPEN, Path(tmp)) as w:
    primary = picks.pick_from_row(w.frame().iloc[0])
    for index, (name, payload) in enumerate(cases.items()):
        results = {}
        for method in ('candidate', 'cron hit', 'cron miss', 'cron void', 'daemon hit', 'daemon miss', 'daemon void',
                       'replay blocked', 'replay full'):
            directory = w.root / str(index) / method.replace(' ', '-') / 'picks'
            directory.mkdir(parents=True)
            path = directory / 'streak.json'
            create(path, payload)
            if method == 'candidate':
                # Use the command so parse exceptions inherited from gather are visible, including exit 2.
                w.now = at(DAY2, '08:30')
                result = w.invoke('day-outcome', '--date', DAY2, '--picks-dir', str(directory),
                                  '--config', str(w.toml()), '--dm-recipient', 'offline.test')
                info = status(result)
                if result.exit_code in (0, 1) and result.stdout:
                    info += f'/{json.loads(result.stdout).get("streak_problem")}'
                results[method] = info
                continue
            terminal = method == 'replay full'
            daily = picks.DailyPick(OPEN, w.now.isoformat(), primary, None, None,
                                    notification_sent=True, notification_id='offline-confirmed',
                                    result='hit' if terminal else None)
            picks.save_pick(daily, directory)
            grade = 'miss' if 'miss' in method else 'void' if 'void' in method else 'hit'
            w.results[101] = grade
            if method.startswith('cron '):
                w.now = at(DAY2, '01:00')
                result = invoke_limited(lambda: w.invoke('check-results', '--date', OPEN, '--picks-dir', str(directory),
                                                          '--wait-deadline-et', '06:00'))
                info = status(result)
            elif method.startswith('daemon '):
                w.now = at(OPEN, '22:05')
                saved_err = sys.stderr
                sys.stderr = open(os.devnull, 'w')
                try:
                    info = invoke_limited(lambda: scheduler.run_result_polling(101, OPEN, directory)) or 'blocks'
                except Exception as exc:
                    info = type(exc).__name__
                finally:
                    sys.stderr.close()
                    sys.stderr = saved_err
            else:
                w.now = at(DAY2, '02:00')
                result = invoke_limited(lambda: w.invoke('reconcile', '--picks-dir', str(directory)))
                info = status(result)
            after = picks.load_pick(OPEN, directory).result
            results[method] = f'{info}:{after}'
        print(name, json.dumps(results), flush=True)
