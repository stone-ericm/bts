from datetime import date, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo
import json
import logging
import os
import re
import subprocess
import sys

import pandas as pd

from probe_features import ROOT, SIG, compute, feature, feed, pa, put
from probe_health import alert, attention


CACHE = Path('data/models/probable_pitcher_lookup.json')
REJECTED = CACHE.parent / 'rejected'
STATUS = Path('data/health_state/probable_pitcher_cache')
RAW = Path('data/raw/2025/109.json')
FOR_D = '2027-01-01T04:59:00+00:00'
FOR_NEXT = '2027-01-01T05:01:00+00:00'
D = date(2026, 12, 31)
SOURCE = 'probable_pitcher_cache'
ORIGINAL = {'999': None, **{str(game): SIG for game in range(100, 111) if game != 109}}
REAL_REPLACE = os.replace


def observation(at=FOR_D, *, reason='invalid_entries', counts=(1, 0, 0, 0, 1), affected=(999,), preserved=None):
    dropped, recovered, rescanned, unusable, absent = counts
    payload = (reason, str(CACHE), dropped, recovered, rescanned, unusable, absent, sorted(affected))
    identity = sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()
    return dict(schema_version='bts_probable_pitcher_cache_status_v1', reason=reason, recorded_at=at,
                recorded_date_et=datetime.fromisoformat(at).astimezone(ZoneInfo('America/New_York')).date().isoformat(),
                cache_path=str(CACHE), dropped=dropped, recovered_from_cache=recovered, rescanned=rescanned,
                unusable_feed=unusable, without_feed=absent, game_ids=sorted(set(affected))[:10],
                identity=identity, preserved=preserved)


def status_path(root, row):
    return root / f"{row['recorded_date_et']}.{row['reason']}.{row['identity'][:12]}.json"


def evidence_path(original):
    return REJECTED / f'probable_pitcher_lookup.20270101T045900Z.{sha256(original).hexdigest()[:12]}.json'


def commit(staged, destination, label, stop):
    if stop == f'before-{label}':
        os._exit(87)
    REAL_REPLACE(staged, destination)
    if stop == f'after-{label}':
        os._exit(87)


def atomic_bytes(path, data, label, stop):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / 'staged'
        staged.write_bytes(data)
        commit(staged, path, label, stop)


def policy_lookup(stop):
    # Project v3's three renames around the real scanner. The live cache here is a fixture.
    # Use the exact original fixture bytes for evidence and the real scanner's staged bytes for the live cache.
    original = CACHE.read_bytes()
    decoded = json.loads(original)
    builder = compute._build_probable_pitcher_lookup
    if '999' not in decoded:
        return builder()
    usable = {k: v for k, v in decoded.items() if k != '999'}
    committed = []

    def publish(preserved):
        row = observation(preserved=preserved)
        file = status_path(STATUS, row)
        if not file.exists():
            atomic_bytes(file, json.dumps(row).encode(), 'record', stop)

    def gated_replace(staged, destination, *args, **kwargs):
        assert destination == scratch
        evidence = evidence_path(original)
        if not evidence.exists():
            atomic_bytes(evidence, original, 'evidence', stop)
        publish(evidence.name)
        commit(staged, CACHE, 'cache', stop)
        committed.append(True)

    with TemporaryDirectory(dir=CACHE.parent) as directory:
        scratch = Path(directory) / 'working.json'
        put(scratch, usable)
        with patch.object(compute, '_probable_pitcher_cache_path', scratch), patch.object(os, 'replace', gated_replace):
            lookup = builder()
    if not committed:
        publish(None)
    return lookup


def child(stop):
    dates = pd.date_range('2025-04-01', periods=11)
    rows = [pa(100 + i, day, 3 if i < 10 else 1, True, i % 2) for i, day in enumerate(dates)]
    compute._probable_pitcher_cache_path = CACHE
    with patch.object(compute, '_build_probable_pitcher_lookup', wraps=compute._build_probable_pitcher_lookup) as original:
        # The feature hook calls the projection; route its scanner call through to the actual copied function.
        def read_projected():
            with patch.object(compute, '_build_probable_pitcher_lookup', original):
                return policy_lookup(stop)

        with patch.object(compute, '_build_probable_pitcher_lookup', read_projected):
            _, result, frame = feature(rows)
    assert result == 'ok', result
    print(frame.iloc[-1]['opp_bullpen_hr_30g'])


def run_child(work, stop='none'):
    return subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', stop], cwd=work,
                          text=True, capture_output=True, timeout=7)


def seed(work, with_feed=True):
    put(work / CACHE, ORIGINAL)
    if with_feed:
        feed(work / RAW, SIG)
    return (work / CACHE).read_bytes()


def test_commits(root):
    cases = [f'{edge}-{name}' for name in ('evidence', 'record', 'cache') for edge in ('before', 'after')]
    for stop, expected in zip(cases, [(False, False, False), (True, False, False), (True, False, False),
                                      (True, True, False), (True, True, False), (True, True, True)], strict=True):
        work = root / stop
        original = seed(work)
        result = run_child(work, stop)
        assert result.returncode == 87, result.stderr
        preserved = work / evidence_path(original)
        published = list((work / STATUS).glob('*.json'))
        current = json.loads((work / CACHE).read_bytes())
        actual = (preserved.is_file(), bool(published), '109' in current)
        assert actual == expected
        assert (preserved.read_bytes() == original) if actual[0] else True
        assert (work / CACHE).read_bytes() == original if not actual[2] else '999' not in current
        if published:
            assert json.loads(published[0].read_bytes())['preserved'] == preserved.name
        print('process stop', stop, 'evidence / observation / new cache', actual)
        retry = run_child(work)
        assert retry.returncode == 0 and retry.stdout.strip() == '0.5', retry.stderr
        assert '109' in json.loads((work / CACHE).read_bytes())

    work = root / 'feed-removed'
    original = seed(work)
    first = run_child(work)
    assert first.returncode == 0 and first.stdout.strip() == '0.5', first.stderr
    (work / RAW).unlink()
    retry = run_child(work)
    assert retry.returncode == 0 and retry.stdout.strip() == '0.5', retry.stderr
    assert json.loads((work / CACHE).read_bytes()) == {str(game): SIG for game in [*range(100, 109), 110, 109]}
    assert (work / evidence_path(original)).read_bytes() == original
    print('v3 fresh process after dropping unrelated 999 and archiving feed 109: value still', retry.stdout.strip())


def test_coalescing(root):
    work = root / 'same-outcomes'
    original = seed(work, with_feed=False)
    first = run_child(work)
    assert first.returncode == 0 and first.stdout.strip() == 'nan', first.stderr
    file = status_path(work / STATUS, observation())
    assert json.loads(file.read_bytes())['preserved'] is None
    feed(work / RAW, SIG)
    second = run_child(work)
    assert second.returncode == 0 and second.stdout.strip() == '0.5', second.stderr
    assert (work / evidence_path(original)).read_bytes() == original
    assert '999' not in json.loads((work / CACHE).read_bytes())
    assert json.loads(file.read_bytes())['preserved'] is None and len(list((work / STATUS).glob('*.json'))) == 1
    print('same-name coalescing after unrelated usable scan: live cache replaced, evidence exists, observation still null')

    chosen = evidence_path(original)
    for shape in ('wrong-content', 'fifo', 'symlink'):
        fake = root / shape / chosen
        fake.parent.mkdir(parents=True)
        if shape == 'fifo':
            os.mkfifo(fake)
        elif shape == 'symlink':
            target = fake.with_name('real.json')
            target.write_bytes(original)
            fake.symlink_to(target)
        else:
            fake.write_text('{}')
        assert list(fake.parent.glob(f'*.{sha256(original).hexdigest()[:12]}.json')) == [fake]
        print('matching preservation suffix is present for', shape, 'without establishing a verified evidence copy')

    group = (120, 121)
    left = {120: 'rescanned', 121: 'without_feed'}
    right = {120: 'without_feed', 121: 'rescanned'}
    common = observation(counts=(2, 0, 1, 0, 1), affected=group)
    assert left != right
    print('P4 inputs for swapped per-game outcomes', left, right, 'same prefix:', common['identity'][:12])
    corrupted = (b'{"100":', b'[]')
    same_count = observation(reason='unreadable_cache', counts=(None, None, 1, None, None), affected=())
    assert len({evidence_path(blob).name for blob in corrupted}) == 2
    print('distinct unreadable bytes and evidence names, same published identity at same scan count:', same_count['identity'][:12])


def test_health(root):
    contract = (ROOT / 'reference/kickoff-fix-w15.md').read_text()
    line = next(line for line in contract.splitlines() if line.startswith('| P6 |'))
    text = dict(zip(['invalid_entries', 'unreadable_cache', 'bad_status'], re.findall(r'`([^`]+)`', line), strict=True))
    preserved = evidence_path(json.dumps(ORIGINAL).encode()).name

    def render(row):
        tokens = {'recorded_at': row['recorded_at'], 'n': row['dropped'], 'ids': ', '.join(map(str, row['game_ids'])),
                  'a': row['recovered_from_cache'], 'k': row['rescanned'], 'u': row['unusable_feed'], 'm': row['without_feed']}
        message = text[row['reason']].replace('entr(y|ies)', 'entry' if row['dropped'] == 1 else 'entries')
        message = re.sub(r'<preserved or "([^"]+)">', lambda match: row['preserved'] or match[1], message)
        for token, value in tokens.items():
            message = message.replace(f'<{token}>', str(value))
        return alert.Alert('WARN', SOURCE, message,
                           f"{SOURCE}:{row['recorded_date_et']}:{row['identity'][:12]}")

    def read(day):
        # These files have valid envelopes by construction. This tests dated discovery and the real dispatcher.
        files = [file for target in (day, day - timedelta(days=1)) for file in sorted(root.glob(f'{target}.*.json'))]
        return [render(json.loads(file.read_bytes())) for file in files]

    first = observation()
    second = observation(counts=(1, 0, 1, 0, 0), affected=(998,), preserved=preserved)
    next_day = observation(FOR_NEXT)
    unreadable_empty = observation(reason='unreadable_cache', counts=(None, None, 0, None, None), affected=())
    unreadable_preserved = observation(reason='unreadable_cache', counts=(None, None, 1, None, None),
                                       affected=(), preserved=preserved)
    sent = []

    def transport(recipient, body):
        sent.append(body)
        return 'fake-delivered-id'

    def dispatch(day):
        alerts = read(day)
        _, warns = attention.apply_warn_attention_policy(alerts, state_path=root / 'attention.json', today=day)
        return alert.dispatch_dm_for_health_alerts(alerts, 'offline', warns, root / 'delivery.json', day)

    put(status_path(root, first), first)
    put(status_path(root, next_day), next_day)
    assert [row.incident_key for row in read(D)] == [render(first).incident_key]
    print('late processed D with D+1 already present:', [row.incident_key for row in read(D)])
    with patch.object(alert, 'send_dm', transport), patch.object(
        attention, 'ALWAYS_ATTENTION_WARN_SOURCES', attention.ALWAYS_ATTENTION_WARN_SOURCES | {SOURCE}
    ):
        assert dispatch(D)
        put(status_path(root, second), second)
        assert dispatch(D) and sent[-1].count(f'[{SOURCE}]') == 2
        assert not dispatch(D)
        assert dispatch(D + timedelta(days=1)) and sent[-1].count(f'[{SOURCE}]') == 3
        assert not dispatch(D + timedelta(days=1))
        print('same-day materially changed counts/game IDs get new digest; repeats covered entry; D+1 catches both D records')
    for label, row in [('invalid null', first), ('invalid preserved', second), ('unreadable null', unreadable_empty),
                       ('unreadable preserved', unreadable_preserved)]:
        print(f'P6 {label}\n' + alert.format_health_dm_body([], [render(row)]))
    broken = text['bad_status'].replace('<path>', str(status_path(STATUS, first)))
    print('P6 status unreadable\n' + alert.format_health_dm_body([], [alert.Alert('WARN', SOURCE, broken)]))


def main():
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=ROOT / 'runs') as name:
        for test in (test_commits, test_coalescing, test_health):
            work = Path(name) / test.__name__
            work.mkdir()
            test(work)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        child(sys.argv[2])
    else:
        main()
