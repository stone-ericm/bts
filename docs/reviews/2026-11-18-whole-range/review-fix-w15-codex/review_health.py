import hashlib
import json
import logging
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from urllib.error import HTTPError

from atproto_client.models.chat.bsky.convo.defs import MessageInput
from bts import dm
from bts.features import probable_pitcher_cache as producer
from bts.health import alert, probable_pitcher_cache as health, runner

from review_evidence import Clock, build, seed
from review_features import CACHE, HEAD, ROOT


DAY = date(2026, 12, 31)
SOURCE = 'probable_pitcher_cache'
STATUS = Path('data/health_state') / SOURCE
PICKS = Path('data/picks')
RECEIPT = STATUS.parent / 'health_dm_status.json'
ROUTE = 'to the supervised BTS lead Claude session (bts-lead), or to Eric if the session is unavailable.'
KEEP = 'Keep the active cache, raw feeds, any named evidence copy and other available copies intact for review.'
NO_COPY = 'This observation names no evidence copy. That run left the active cache unchanged.'


class Wall(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2027, 1, 1, 5, 1, tzinfo=timezone.utc).astimezone(tz)


def row(day=DAY, reason='invalid_entries', index=0):
    valid_count = reason == 'invalid_entries'
    return dict(
        schema_version='bts_probable_pitcher_cache_status_v1', reason=reason,
        recorded_at=f'{day}T14:00:00+00:00', recorded_date_et=str(day),
        cache_path=str(CACHE), dropped=1 if valid_count else None,
        recovered_from_cache=0 if valid_count else None, rescanned=0,
        unusable_feed=0 if valid_count else None, without_feed=1 if valid_count else None,
        game_ids=[100 + index] if valid_count else [],
        identity=hashlib.sha256(f'{day}/{reason}/{index}'.encode()).hexdigest(),
        preserved=None, original_sha256='b' * 64,
    )


def basename(payload):
    return f'{payload["recorded_date_et"]}.{payload["reason"]}.{payload["identity"][:12]}.json'


def write(payload, name=None):
    STATUS.mkdir(parents=True, exist_ok=True)
    path = STATUS / (name or basename(payload))
    path.write_text(json.dumps(payload))
    return path


def malformed(path):
    return f'{SOURCE}:status_unreadable:{path.name}'


def key(payload):
    return f'{SOURCE}:{payload["recorded_date_et"]}:{payload["identity"]}'


def check_one(path, expected):
    result, = health.check(STATUS, DAY)
    assert (result.level, result.source, result.incident_key) == ('WARN', SOURCE, expected)
    if expected == malformed(path):
        assert result.message == f'Probable-pitcher cache status unreadable at {path}. Send this alert {ROUTE}'
    return result


def valid_and_invalid_envelopes():
    sample = row()
    normal_name = basename(sample)
    mutations = [
        ('non-object', []), ('missing-field', {k: v for k, v in sample.items() if k != 'preserved'}),
        ('extra-field', sample | {'extra': 1}),
    ]
    variants = {
        'schema_version': ['wrong', 1, None], 'reason': ['new_reason', None, []],
        'recorded_at': ['2026-12-31T14:00:00', '2026-12-31T14:00:00Z', '2026-12-31T14:00:00+01:00',
                        '2027-01-01T05:01:00+00:00', '2026-12-31T99:00:00+00:00', None],
        'recorded_date_et': ['2027-01-01', '20261231', None], 'cache_path': ['', None, 1],
        'dropped': [0, True, -1, 1.0, 2], 'recovered_from_cache': [True, -1, 0.0],
        'rescanned': [True, -1, '0'], 'unusable_feed': [False, -1, 0.0],
        'without_feed': [True, 0, '1'],
        'game_ids': [[], [True], [100.0], ['100'], [100, 100], [101, 100], [100, 101], list(range(11)), {}],
        'identity': [sample['identity'].upper(), 'f' * 64, 'z' * 64, sample['identity'][:12], None],
        'preserved': ['../probable_pitcher_lookup.20261231T140000Z.aaaaaaaaaaaa.json',
                      'probable_pitcher_lookup.20261231T140000Z.AAAAAAAAAAAA.json',
                      'probable_pitcher_lookup.20261231T14000Z.aaaaaaaaaaaa.json', [], 'missing.json'],
        'original_sha256': ['A' * 64, 'z' * 64, '0' * 63, 1],
    }
    for field, values in variants.items():
        for value in values:
            candidate = sample | {field: value}
            mutations.append((f'{field}={value!r}', candidate))
    cases = []
    for label, payload in mutations:
        path = write(payload, normal_name)
        check_one(path, malformed(path))
        path.unlink()
        cases.append(label)
    unreadable = row(reason='unreadable_cache')
    for field, value in [('dropped', 0), ('recovered_from_cache', 0), ('unusable_feed', 0),
                         ('without_feed', 0), ('rescanned', None), ('rescanned', False), ('game_ids', [100])]:
        path = write(unreadable | {field: value})
        check_one(path, malformed(path))
        path.unlink()
        cases.append(f'unreadable {field}={value!r}')
    for name in (f'{DAY}.unknown.json', f'{DAY}.invalid_entries.extra.{sample["identity"][:12]}.json',
                 f'{DAY}.unreadable_cache.{sample["identity"][:12]}.json'):
        path = write(sample, name)
        check_one(path, malformed(path))
        path.unlink()
        cases.append(name)
    for item in (sample, sample | dict(dropped=30, without_feed=30, game_ids=list(range(-3, 7))),
                 sample | dict(original_sha256=None), unreadable, unreadable | dict(rescanned=100)):
        path = write(item)
        check_one(path, key(item))
        path.unlink()
    first = write(sample)
    candidate = sample | dict(identity=sample['identity'][:12] + 'd' * 52)
    first.unlink()
    second = write(candidate)
    check_one(second, key(candidate))
    assert key(sample) != key(candidate)
    second.unlink()
    for name in (f'{DAY}.json', f'{DAY}.invalid_entries.txt', 'empty.json'):
        write(sample, name)
    assert health.check(STATUS, DAY) == []
    print('A6 rejected', len(cases), 'bad envelopes/basenames:', ', '.join(cases))
    print('A6 accepted both reasons, null digest, 10 signed IDs with 30 drops, full key for equal-prefix identities')


def files(form):
    if form == 'missing':
        assert health.check(STATUS, DAY) == []
        return
    sample = row()
    if form.startswith('dir-'):
        path = STATUS
        path.parent.mkdir(parents=True)
    else:
        path = STATUS / basename(sample)
        path.parent.mkdir(parents=True)
    bare = form.removeprefix('dir-')
    try:
        if bare == 'fifo':
            os.mkfifo(path)
        elif bare == 'symlink':
            target = Path('elsewhere').resolve()
            if form.startswith('dir-'):
                target.mkdir()
            else:
                target.write_text(json.dumps(sample))
            path.symlink_to(target)
        elif bare == 'directory' or bare == 'unreadable-dir':
            path.mkdir()
            if bare == 'unreadable-dir':
                path.chmod(0)
        elif bare == 'undecodable':
            path.write_bytes(b'\xff')
        elif bare == 'unreadable':
            path.write_text(json.dumps(sample))
            path.chmod(0)
        elif bare in ('limit', 'oversized'):
            payload = json.dumps(sample).encode()
            size = 65536 + int(bare == 'oversized')
            path.write_bytes(payload.ljust(size, b' '))
        else:
            raise ValueError(form)
        result = check_one(path, key(sample) if form == 'limit' else malformed(path))
        print('bounded reader', form, result.incident_key)
    finally:
        if bare in ('unreadable', 'unreadable-dir'):
            path.chmod(0o700)


def exact_text(payload):
    evidence = NO_COPY
    if payload['preserved']:
        evidence_path = Path(payload['cache_path']).parent / 'rejected' / payload['preserved']
        evidence = f'That run recorded an evidence copy at `{evidence_path}`.'
    if payload['reason'] == 'unreadable_cache':
        k = payload['rescanned']
        return ' '.join([
            f'Probable-pitcher cache was unreadable at {payload["recorded_at"]}.',
            f'That run started from an empty lookup and built usable entries from raw feeds for {k} ' + ('game.' if k == 1 else 'games.'),
            KEEP, "Other games may have been missing from that run's lookup.", evidence,
            f'Send this alert immediately {ROUTE}',
            'Ask them to preserve and verify any usable cache copy in the current or previous R2 manifest before the next successful sync.',
            'An unreferenced content-addressed cache object can be pruned once it is at least seven days old.',
        ])
    n = payload['dropped']
    ids = ', '.join(map(str, payload['game_ids']))
    return ' '.join([
        f'Probable-pitcher cache at {payload["recorded_at"]} had {n} ' + ('invalid entry' if n == 1 else 'invalid entries') + f' (first game IDs {ids}).',
        f'Of those entries, {payload["recovered_from_cache"]} had a valid cache alias and {payload["rescanned"]} had a usable raw feed.',
        f'No usable lookup was found for {payload["unusable_feed"]} with unusable or unverified feeds and {payload["without_feed"]} with missing feeds.',
        evidence, 'This describes the cache at the recorded time; health may show this observation again on the next processed day.',
        KEEP, f'Send this alert promptly {ROUTE}',
        'Ask them to review the latest preview and any pending pick before the next scheduled selection.',
        'If a pick was already sent or entered, report that and preserve its records.',
    ])


def texts():
    producer.datetime = Clock
    cache_broken = b'\xff'
    for reason, payload in [('invalid_entries', None), ('unreadable_cache', cache_broken)]:
        for with_scan in (False, True):
            if payload is None:
                seed(feeds=with_scan)
            else:
                seed(payload, feeds=with_scan)
            result, lines = build()
            record, = STATUS.glob('*.json')
            data = json.loads(record.read_bytes())
            result = check_one(record, key(data))
            assert data['reason'] == reason and bool(data['preserved']) == with_scan
            assert result.message == exact_text(data)
            formatted = alert.format_health_dm_body([], [result])
            assert formatted == f'BTS health WARN attention:\n- [{SOURCE}] {exact_text(data)}'
            print(f'P6/A7 {reason} copy={with_scan}\n{formatted}\n' + '\n'.join(lines))
            record.unlink()
            for path in Path('data/raw').glob('*/*.json'):
                path.unlink()
    record = write(row(), f'{DAY}.bad.json')
    result = check_one(record, malformed(record))
    print('P6 status unreadable\n' + alert.format_health_dm_body([], [result]))


def run_eod(day):
    return runner.run_all_checks(PICKS, CACHE.parent, 'did:plc:fake', today=day,
                                 thresholds_overrides={'health_dm_delivery_status': RECEIPT})


def delivery(scenario):
    seed_calls = []
    accepted = []
    safe_run = runner._safe_run

    def local_api(request, **kw):
        url = request.full_url
        if url.endswith('createSession'):
            return BytesIO(b'{"accessJwt":"synthetic"}')
        if 'getConvoForMembers' in url:
            assert url.endswith('did:plc:fake')
            return BytesIO(b'{"convo":{"id":"synthetic"}}')
        assert url.endswith('chat.bsky.convo.sendMessage') and kw == dict(timeout=15, idempotent=False)
        message = json.loads(request.data)['message']['text']
        seed_calls.append(message)
        if scenario == 'ascii-limit' and len(message) > 1000:
            assert message.isascii()
            raise HTTPError(url, 400, 'synthetic chat maxGraphemes=1000', {}, None)
        if scenario == 'sdk-limit':
            MessageInput(text=message)
        accepted.append(message)
        return BytesIO(b'{"id":"synthetic-dm"}')

    with (patch.object(runner, '_safe_run', lambda src, cb: safe_run(src, cb) if src == SOURCE else []),
          patch.object(runner.realized_calibration, '_current_deploy_iso', return_value=None),
          patch.object(runner, 'datetime', Wall), patch.object(alert, 'datetime', Wall),
          patch.object(dm, 'get_dm_password', return_value='synthetic'), patch.object(dm, 'retry_urlopen', local_api)):
        if scenario == 'normal':
            yesterday, today, tomorrow = [row(DAY + timedelta(days=offset)) for offset in (-1, 0, 1)]
            write(row(DAY - timedelta(days=2)))
            write(tomorrow)
            write(yesterday)
            write(today)
            assert {a.incident_key for a in run_eod(DAY)} == {key(yesterday), key(today)}
            run_eod(DAY)
            assert len(accepted) == 1
            assert {a.incident_key for a in run_eod(None)} == {key(today), key(tomorrow)}
            run_eod(DAY + timedelta(days=1))
            assert len(accepted) == 2 and '2026-12-31T14:' in accepted[0] and '2026-12-31T14:' in accepted[1]
            changed = row(DAY + timedelta(days=1), index=1)
            write(changed)
            assert len(run_eod(DAY + timedelta(days=1))) == 3 and len(accepted) == 3
            assert 'first game IDs 101' in accepted[-1]
            recorded = json.loads(RECEIPT.read_bytes())
            assert recorded['sent_day'] == '2027-01-01' and len(recorded['sent_sources']) == 3
            print('late D excludes D+1, implicit ET D+1, same-day suppression, changed key delivered:',
                  [len(body) for body in accepted], recorded['sent_sources'])
        else:
            for index in range(1 if scenario == 'ascii-limit' else 15):
                write(row(index=index))
            write(row(reason='unreadable_cache'))
            expected_count = 2 if scenario == 'ascii-limit' else 16
            for day in (DAY, DAY + timedelta(days=1)):
                assert len(run_eod(day)) == expected_count
                receipt = json.loads(RECEIPT.read_bytes())
                assert receipt['status'] == 'failed' and receipt['critical_count'] == 0
            assert not accepted and len(seed_calls) == 2
            print(scenario, 'rejected both processed days', [len(body) for body in seed_calls], 'valid WARNs', expected_count)


def main():
    with TemporaryDirectory(dir=ROOT, prefix='health-run-') as name:
        work = Path(name)
        cases = [(valid_and_invalid_envelopes, ()), (texts, ()),
                 *((delivery, (scenario,)) for scenario in ('normal', 'ascii-limit', 'sdk-limit'))]
        try:
            for case, args in cases:
                folder = work / f'{case.__name__}{args[0] if args else ""}'
                folder.mkdir()
                os.chdir(folder)
                case(*args)
            for form in ('missing', 'fifo', 'directory', 'symlink', 'undecodable', 'unreadable', 'limit',
                         'oversized', 'dir-fifo', 'dir-symlink', 'dir-undecodable', 'dir-oversized', 'dir-unreadable-dir'):
                if form == 'dir-oversized':
                    continue  # The directory check is type-based, as shown by dir-undecodable.
                folder = work / form
                folder.mkdir()
                result = subprocess.run([sys.executable, str(Path(__file__).resolve()), form], cwd=folder,
                                        env=dict(os.environ, PYTHONPATH=str(HEAD), PYTHONDONTWRITEBYTECODE='1'),
                                        capture_output=True, text=True, timeout=6)
                assert result.returncode == 0, (form, result.stdout, result.stderr)
                print(result.stdout, end='')
        finally:
            os.chdir(ROOT)


if __name__ == '__main__':
    logging.disable(logging.CRITICAL)
    if len(sys.argv) == 1:
        main()
    else:
        files(sys.argv[1])
