import hashlib
import json
import logging
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from review_evidence import Clock, build, seed
from review_features import CACHE, HEAD, feed
from review_health import exact_text, row, basename, ROUTE

import bts
from bts import dm
from bts.features import probable_pitcher_cache as producer
from bts.health import alert, probable_pitcher_cache as health, runner


DAY = date(2026, 12, 31)
NEXT = DAY + timedelta(days=1)
SOURCE = 'probable_pitcher_cache'
STATUS = Path('data/health_state') / SOURCE
PICKS = Path('data/picks')
RECEIPT = STATUS.parent / 'health_dm_delivery_status.json'
FALLBACK = Path('data/health_state/probable_pitcher_cache')


class Wall(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2027, 1, 1, 5, 1, tzinfo=timezone.utc).astimezone(tz)


def key(day, *members):
    raw = json.dumps(sorted(members), separators=(',', ':')).encode('utf-8')
    return f'{SOURCE}:{day}:{hashlib.sha256(raw).hexdigest()[:16]}'


def expected(path, count, labels):
    noun = 'observation' if count == 1 else 'observations'
    return (f'Probable-pitcher cache: {count} {noun} ({", ".join(sorted(labels))}) recorded today or yesterday; '
            f'details in {path}. Keep the active cache, raw feeds, evidence copies and other copies intact. '
            'Send this alert promptly to the supervised BTS lead Claude session (bts-lead), or to Eric if it is unavailable, '
            'to review the latest preview and any pending pick before the next selection; for an unreadable cache, '
            'to secure a usable R2 copy before the next successful sync.')


@contextmanager
def captured():
    packets = []
    logs = StringIO()
    handler = logging.StreamHandler(logs)
    logger = logging.getLogger(health.__name__)
    logger.addHandler(handler)
    safe_run = runner._safe_run

    def api(request, **kwargs):
        url = request.full_url
        if url.endswith('createSession'):
            return BytesIO(b'{"accessJwt":"offline"}')
        if 'getConvoForMembers' in url:
            assert url.endswith('did:plc:fake')
            return BytesIO(b'{"convo":{"id":"offline"}}')
        assert url.endswith('chat.bsky.convo.sendMessage')
        assert kwargs == dict(timeout=15, idempotent=False)
        body = json.loads(request.data)['message']['text']
        assert body.isascii()  # ASCII characters are graphemes in this fixture.
        if len(body) > 1000:
            print('captured cap rejection:', len(body), 'characters,', body.splitlines()[0], flush=True)
            raise HTTPError(url, 400, 'synthetic chat maxGraphemes=1000', {}, None)
        packets.append(body)
        return BytesIO(b'{"id":"offline"}')

    try:
        with (patch.object(runner, '_safe_run', lambda name, call: safe_run(name, call) if name == SOURCE else []),
              patch.object(runner.realized_calibration, '_current_deploy_iso', return_value=None),
              patch.object(runner, 'datetime', Wall), patch.object(alert, 'datetime', Wall),
              patch.object(dm, 'get_dm_password', return_value='offline'), patch.object(dm, 'retry_urlopen', api)):
            yield packets, logs
    finally:
        logger.removeHandler(handler)


def run(day=DAY, picks=PICKS):
    return runner.run_all_checks(picks, picks.parent / 'models', 'did:plc:fake', today=day)


def records():
    return [json.loads(path.read_bytes()) for path in sorted(STATUS.glob('*.json'))]


def summary(result, day, labels, ids, shown=STATUS):
    assert (result.level, result.source) == ('WARN', SOURCE)
    assert result.message == expected(shown, len(ids), labels)
    assert result.incident_key == key(day, *ids) and len(result.message) <= 600


def one_packet(packets, result):
    assert packets[-1] == (f'BTS health WARN attention:\n- [{SOURCE}] {result.message} (1st observed day)')
    assert len(packets[-1]) < 1000


def persistent():
    seed(b'{"100":null}', feeds=False)
    Clock.at = datetime(2027, 1, 1, 4, 59, tzinfo=timezone.utc)
    build()
    Clock.at = datetime(2027, 1, 1, 5, 1, tzinfo=timezone.utc)
    build()
    first, second = records()
    assert first['identity'] == second['identity']
    identity = first['identity']
    labels = {'invalid entries'}
    with captured() as (packets, logs):
        result, = run(DAY)
        summary(result, DAY, labels, [identity])
        assert first['recorded_at'] in logs.getvalue() and second['recorded_at'] not in logs.getvalue()
        one_packet(packets, result)
        assert run(DAY) == [result] and len(packets) == 1
        tomorrow, = run(None)
        summary(tomorrow, NEXT, labels, [identity, identity])
        one_packet(packets, tomorrow)
        assert run(NEXT) == [tomorrow] and len(packets) == 2
        bad = STATUS / f'{NEXT}.broken.json'
        bad.write_bytes(b'\xff')
        changed, = run(NEXT)
        summary(changed, NEXT, labels | {'status unreadable'}, [identity, identity, bad.name])
        one_packet(packets, changed)
        assert run(NEXT) == [changed] and len(packets) == 3
        assert f'Probable-pitcher cache status unreadable at {bad}. Send this alert {ROUTE}' in logs.getvalue()
        assert json.loads(RECEIPT.read_bytes())['status'] == 'sent'
        print('persistent D/late D, D+1 and new bad entry: successful bodies', list(map(len, packets)),
              'keys', [result.incident_key, tomorrow.incident_key, changed.incident_key])
        print('late D text omits exact dates and next-day repeat has attention suffix:', packets[1])


def pair():
    Clock.at = datetime(2026, 12, 31, 23, 0, tzinfo=timezone.utc)
    seed(b'{"100":null}', feeds=False)
    build()
    seed(b'\xff', feeds=False)
    build()
    initial = records()
    with captured() as (packets, logs):
        keys = [record['identity'] for record in initial]
        labels = {'invalid entries', 'unreadable cache'}
        for day in (DAY, NEXT):
            result, = run(day)
            summary(result, day, labels, keys)
            one_packet(packets, result)
            assert run(day) == [result]
        assert len(packets) == 2
        assert all(exact_text(record) in logs.getvalue() for record in initial)
        assert all('current or previous R2 manifest' not in body for body in packets)
        print('same-day invalid/unreadable pair: successful bodies', list(map(len, packets)))
        print(packets[-1])


def unreadable_with_copy():
    Clock.at = datetime(2026, 12, 31, 20, 0, tzinfo=timezone.utc)
    seed(b'\xff', feeds=False)
    feed('100.json')
    build()
    record, = records()
    assert record['preserved'] and record['rescanned'] == 1
    with captured() as (packets, logs):
        result, = run()
        summary(result, DAY, {'unreadable cache'}, [record['identity']])
        one_packet(packets, result)
        rendered = exact_text(record).replace(chr(96), '')
        assert rendered in logs.getvalue() and record['preserved'] not in packets[0]
        print('logged R6/A7 (unreadable with copy):', rendered)


def changed_valid():
    Clock.at = datetime(2026, 12, 31, 20, 0, tzinfo=timezone.utc)
    seed(b'{"100":null}', feeds=False)
    build()
    with captured() as (packets, logs):
        first, = run()
        feed('100.json')
        build()
        second, = run()
        summary(second, DAY, {'invalid entries'}, [r['identity'] for r in records()])
        assert second.incident_key != first.incident_key and len(packets) == 2
        assert run() == [second] and len(packets) == 2
        one_packet(packets, second)
        for record in records():
            rendered = exact_text(record).replace(chr(96), '')
            assert rendered in logs.getvalue()
            print('logged R6/A7:', rendered)
        print('same-day material recovery: sent keys', first.incident_key, second.incident_key)


def many_and_long():
    real_dir = Path('q' * 135) / ('r' * 135) / 'health_state' / SOURCE
    real_dir.mkdir(parents=True)
    keys = []
    for i in range(15):
        record = row(DAY, 'invalid_entries' if i % 2 else 'unreadable_cache', i)
        keys.append(record['identity'])
        (real_dir / basename(record)).write_text(json.dumps(record))
    bad = real_dir / f'{DAY}.invalid.json'
    bad.write_bytes(b'{')
    keys.append(bad.name)
    with captured() as (packets, logs):
        result, = run(DAY, real_dir.parent.parent / 'picks')
        summary(result, DAY, {'invalid entries', 'unreadable cache', 'status unreadable'}, keys, shown=FALLBACK)
        one_packet(packets, result)
        assert len(logs.getvalue().splitlines()) == 16 and not FALLBACK.exists()
        print('16 findings, one bounded DM', len(result.message), len(packets[0]),
              'literal fallback exists', FALLBACK.exists(), 'actual details', str(real_dir))


def hostile_paths():
    tick = chr(96)
    STATUS.mkdir(parents=True)
    path = STATUS / f'{DAY}.broken{tick}.json'
    path.write_bytes(b'{')
    with captured() as (packets, logs):
        result, = run()
        summary(result, DAY, {'status unreadable'}, [path.name])
        assert tick in logs.getvalue() and tick not in packets[-1]
        print('R6 raw selected basename in health log:', repr(logs.getvalue()))

    path.unlink()
    STATUS.rmdir()
    os.mkfifo(STATUS)
    with captured() as (packets, logs):
        result, = run()
        summary(result, DAY, {'status unreadable'}, [STATUS.name])
        assert str(STATUS) in logs.getvalue() and STATUS.is_fifo()
        print('unreadable FIFO directory claim in DM:', result.message)

    other_root = Path(f'other{tick}data')
    copy_dir = other_root / 'health_state' / SOURCE
    copy_dir.mkdir(parents=True)
    record = row(DAY, 'invalid_entries', 20)
    record['cache_path'] = f'{other_root}/models/probable_pitcher_lookup.json'
    record['preserved'] = 'probable_pitcher_lookup.20261231T140000Z.bbbbbbbbbbbb.json'
    (copy_dir / basename(record)).write_text(json.dumps(record))
    with captured() as (packets, logs):
        result, = run(DAY, other_root / 'picks')
        assert tick in packets[-1] and tick in logs.getvalue()
        print('R6 raw root in DM and evidence log:', repr(result.message), repr(logs.getvalue()))


def oom_directory(prior=False):
    Clock.at = datetime(2026, 12, 31, 20, 0, tzinfo=timezone.utc)
    seed(b'{"100":null}', feeds=False)
    build()
    alias = Path('oom/data')
    alias.parent.mkdir()
    alias.symlink_to(PICKS.parent.resolve())
    assert (alias / 'health_state' / SOURCE).samefile(STATUS)
    with captured() as (packets, _):
        warnings = run(DAY, alias / 'picks')
        if prior:
            assert len(warnings) == len(packets) == 1 and warnings[0].level == 'WARN' and 'oom/' not in packets[0]
            print('round 1 same OOM-named path with one record: one ordinary WARN, accepted', len(packets[0]))
            return
        assert len(warnings) == 2 and not packets
        summary_result, critical = warnings
        assert summary_result.source == SOURCE and summary_result.level == 'WARN'
        assert critical.source == 'analytics_job_oom' and critical.level == 'CRITICAL'
        assert critical.message.endswith(summary_result.message)
        receipt = json.loads(RECEIPT.read_bytes())
        assert receipt['status'] == 'failed' and receipt['critical_count'] == 1
        print('only W15 input, same actual status directory via alias, misleading escalation:', critical.message)


def main():
    assert Path(bts.__file__).resolve() == HEAD / 'bts/__init__.py'
    producer.datetime = Clock
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger().addHandler(logging.NullHandler())
    with TemporaryDirectory(dir=HERE, prefix='health-') as name:
        work = Path(name)
        try:
            for scenario in (persistent, pair, unreadable_with_copy, changed_valid, many_and_long, hostile_paths, oom_directory):
                folder = work / scenario.__name__
                folder.mkdir()
                os.chdir(folder)
                scenario()
            folder = work / 'round1-oom'
            folder.mkdir()
            prior = HERE / 'r1/src'
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(HERE.parent / 'offline'), str(prior)]),
                       PYTHONDONTWRITEBYTECODE='1')
            replay = subprocess.run([sys.executable, str(Path(__file__).resolve()), 'round1-oom'],
                                    cwd=folder, check=True, env=env, capture_output=True, text=True, timeout=8)
            print(replay.stdout, end='')
        finally:
            os.chdir(HERE)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        main()
    else:
        assert Path(bts.__file__).resolve() == HERE / 'r1/src/bts/__init__.py'
        producer.datetime = Clock
        logging.getLogger().addHandler(logging.NullHandler())
        oom_directory(prior=True)
