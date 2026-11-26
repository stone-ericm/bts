from contextlib import redirect_stderr
from copy import deepcopy
from datetime import date, datetime, timedelta
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch
from zoneinfo import ZoneInfo
import json
import logging
import os
import signal
import stat
import sys

from probe_features import ROOT, SIG, compute, feature, load_copy, pa, put


def offline_transport(*args):
    raise AssertionError('network transport not available in probe')


dm = ModuleType('bts.dm')
dm.send_dm = offline_transport
sys.modules['bts.dm'] = dm
alert = load_copy('bts.health.alert', 'alert.py')
attention = load_copy('bts.health.attention', 'attention.py')
schema = ModuleType('bts.data.schema')
schema.SCHEMA_VERSION = 'probe'
sys.modules['bts.data.schema'] = schema
sync = load_copy('w15_base_sync', 'sync.py')
SOURCE = 'probable_pitcher_cache'
SCHEMA = 'bts_probable_pitcher_cache_status_v1'
STATUS = 'probable_pitcher_cache_status.json'
DAY = date(2026, 12, 31)
STAMP = '2027-01-01T04:59:00+00:00'
ROUTE = ('Paste this alert as written into the supervised BTS lead Claude session (`bts-lead`), '
         'or keep it for the next such session if none is running.')
INVALID_TEXT = ('Probable-pitcher cache: {n} invalid {entries} dropped at {stamp} ({k} rescanned from raw feeds, '
                '{m} without a feed on this machine; games {ids}). Features for those games use the rescanned '
                'probable pitchers, or none where the feed is missing. No action is needed for rescanned games. '
                'Do not delete or edit the cache by hand: entries whose feeds are not on this machine cannot be '
                'rebuilt. ' + ROUTE + ' It is not time-bound: the 03:00 preview and every lineup check already use '
                'the lookup without these entries, waiting does not change what they compute, and the alert '
                'repeats daily while the cause remains.')
UNREADABLE = (f'Probable-pitcher cache was unreadable at {STAMP} and was rebuilt from the raw feeds present; '
              'entries for games whose feeds are not on this machine were lost. Do not delete or edit the cache '
              'by hand. ' + ROUTE + ' It is not time-bound: the lost entries cannot be recovered by acting sooner, '
              'and the rebuilt cache is what the 03:00 preview and every lineup check now use.')


def invalid_alert(n=2, k=1, m=1, ids='100, 101'):
    return alert.Alert('WARN', SOURCE, INVALID_TEXT.format(
        n=n, entries='entry' if n == 1 else 'entries', stamp=STAMP, k=k, m=m, ids=ids))


def writer_reader_paths(work):
    live = work / 'box/projects/bts'
    other = work / 'box/projects/backtest'
    test = work / 'box/tmp/test-one'
    cache_rel = Path('data/models/probable_pitcher_lookup.json')
    reader = live / 'data/picks'
    expected = reader.parent / 'health_state' / STATUS
    for name, cwd, cache, picks in (
        ('service', live, cache_rel, reader), ('03:00 cron', live, cache_rel, reader),
        ('same-clone backtest', live, cache_rel, reader), ('separate checkout', other, cache_rel, reader),
        ('pytest tmp cache', live, test / 'models/cache.json', reader),
        ('flat patched cache', live, live / 'data/cache.json', reader),
        ('service custom picks', live, cache_rel, live / 'alternate/picks'),
    ):
        writer_status = (cwd / cache).parent.parent / 'health_state' / STATUS
        reader_status = picks.parent / 'health_state' / STATUS
        record_marker = name
        writer_status.parent.mkdir(parents=True, exist_ok=True)
        writer_status.write_text(record_marker)
        print('route', name, 'writer', writer_status.relative_to(work), 'matches reader', writer_status == reader_status,
              'can change live slot', expected.exists() and expected.read_text() == record_marker)


def end_of_day(work):
    receipt = work / 'delivery.json'
    warn_state = work / 'attention.json'
    issue = invalid_alert()
    unreadable = alert.Alert('WARN', SOURCE, UNREADABLE)
    other = alert.Alert('WARN', 'disk_fill', 'Other issue')
    calls = []

    def send(recipient, body):
        calls.append(body)
        return 'fake-ack'

    def dispatch(day, issues):
        _, selected = attention.apply_warn_attention_policy(issues, state_path=warn_state, today=day)
        return alert.dispatch_dm_for_health_alerts(issues, 'fake-recipient', selected, receipt, day)

    assert attention.build_policy_alerts([issue], streaks={}) == ([], [])
    et = ZoneInfo('America/New_York')
    for instant in (STAMP, '2027-01-01T05:01:00+00:00'):
        print('event UTC/ET', instant, datetime.fromisoformat(instant).astimezone(et).isoformat())
    status = work / STATUS
    status.write_text(json.dumps({'schema_version': SCHEMA, 'recorded_date_et': DAY.isoformat()}))

    def fresh(day):
        observed = date.fromisoformat(json.loads(status.read_text())['recorded_date_et'])
        return observed in (day, day - timedelta(days=1))

    with patch.object(alert, 'send_dm', send), patch.object(
        attention, 'ALWAYS_ATTENTION_WARN_SOURCES', attention.ALWAYS_ATTENTION_WARN_SOURCES | {SOURCE}
    ):
        assert fresh(DAY) and dispatch(DAY, [issue])
        assert not dispatch(DAY, [issue])
        assert not dispatch(DAY, [unreadable])  # Same key, different incident.
        assert dispatch(DAY, [issue, other])  # Re-includes the already sent issue.
        assert fresh(DAY + timedelta(days=1)) and dispatch(DAY + timedelta(days=1), [issue])
        assert not fresh(DAY + timedelta(days=2))
        print('P5 unchanged D event sends again for D+1; later unreadable same day suppressed; other source repeats it')
        print('P6 rendered mixed WARN digest:\n' + calls[0])
        for n, k, m, ids in [(1, 1, 0, '100'), (1, 0, 1, '101'), (10, 0, 10, ', '.join(str(100000 + i) for i in range(10)))]:
            candidate = invalid_alert(n, k, m, ids)
            rendered = alert.format_health_dm_body([], [candidate])
            print(f'P6 rendered invalid only (characters={len(rendered)}):\n{rendered}')
        print('P6 rendered unreadable rebuild/no-feed/no-write/refill (same body):\n' +
              alert.format_health_dm_body([], [unreadable]))
        print('P6 unreadable status:\n' + alert.format_health_dm_body([], [
            alert.Alert('WARN', SOURCE, f'probable-pitcher cache status unreadable at {status.name}')]))
        status.write_text(json.dumps({'schema_version': SCHEMA, 'recorded_date_et': (DAY + timedelta(days=1)).isoformat()}))
        assert not fresh(DAY)
        print('single slot D+1 observation overwrites and hides D from a late D run')

        for label, outcome in [('exception', RuntimeError('fake outage')), ('blank response', '')]:
            trial = work / f'{label}.json'
            warn = [issue]
            with patch.object(alert, 'send_dm', side_effect=outcome if isinstance(outcome, Exception) else None,
                              return_value=outcome):
                alert.dispatch_dm_for_health_alerts(warn, 'fake-recipient', warn, trial, DAY)
            state = json.loads(trial.read_text())
            print('send', label, state['status'], 'H6 note', alert._resurface_note(state),
                  'empty future run attempted', alert.dispatch_dm_for_health_alerts([], 'fake-recipient', [], trial, DAY))
            if isinstance(outcome, Exception):
                assert dispatch(DAY + timedelta(days=1), [issue]) is False  # Other receipt already sent this day.
                assert alert.dispatch_dm_for_health_alerts(warn, 'fake-recipient', warn, trial, DAY)
            else:
                assert not alert.dispatch_dm_for_health_alerts(warn, 'fake-recipient', warn, trial, DAY)
    assert len(calls) == 4


def filesystem(work):
    def interrupt(signum, frame):
        raise KeyboardInterrupt('probe timeout')

    def time_limit(label, function):
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        try:
            function()
        except KeyboardInterrupt:
            print('blocked until 200 ms probe timer', label)
        else:
            raise AssertionError(f'expected to block: {label}')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    signal.signal(signal.SIGALRM, interrupt)
    cache = work / 'models/probable_pitcher_lookup.json'
    cache.parent.mkdir()
    compute._probable_pitcher_cache_path = cache
    raw = work / 'raw'
    raw_feed = raw / '2025/100.json'
    raw_feed.parent.mkdir(parents=True)
    os.mkfifo(cache)
    time_limit('base cache FIFO', lambda: compute._build_probable_pitcher_lookup(str(raw)))
    cache.unlink()
    os.mkfifo(raw_feed)
    time_limit('base raw FIFO', lambda: compute._build_probable_pitcher_lookup(str(raw)))
    raw_feed.unlink()

    target = work / STATUS
    temporary = target.with_suffix('.tmp')
    os.mkfifo(temporary)
    time_limit('insecure fixed-name status temp FIFO', lambda: temporary.write_text('{}'))
    temporary.unlink()
    os.mkfifo(target)
    temporary.write_text('{}')
    os.replace(temporary, target)
    assert stat.S_ISREG(target.stat().st_mode)
    print('blind atomic rename replaces a FIFO destination without error')
    target.unlink()
    os.mkfifo(target)
    try:
        alert._read_regular_file_text(target)
    except OSError as error:
        print('existing fd-checked reader rejects FIFO:', error)
    read_fd, write_fd = os.pipe()
    os.close(read_fd)
    with os.fdopen(write_fd, 'w', buffering=1) as output:
        try:
            print('WARNING: could not write probable-pitcher cache status', file=output)
        except BrokenPipeError:
            print('warning emission can raise BrokenPipeError on closed stderr')
            # Discard the buffered retry by replacing the underlying descriptor in-place.
            with (work / 'discarded-warning.txt').open('w') as discard:
                os.dup2(discard.fileno(), output.fileno())


class MemoryR2:
    def __init__(self):
        self.objects = {}

    def upload_file(self, source, key):
        self.objects[key] = source.read_bytes()

    def get_object_json(self, key):
        return deepcopy(self.objects.get(key))

    def put_object_json(self, key, value):
        self.objects[key] = deepcopy(value)

    def copy_object(self, source, target):
        self.objects[target] = deepcopy(self.objects[source])

    def delete_object(self, key):
        del self.objects[key]


def backup(work):
    client = MemoryR2()
    processed = work / 'processed'
    processed.mkdir()
    models = work / 'models'
    cache = models / 'probable_pitcher_lookup.json'
    key = 'models/probable_pitcher_lookup.json'
    put(cache, {'100': SIG, '999': SIG})
    with patch.object(sync, '_current_git_sha', return_value='test'), patch.object(sync, '_current_git_branch', return_value='test'):
        original = sync.sync_to_r2(client, processed, models)
        cached_object = original['files'][key]['key']
        cache.write_text('{')
        compute._probable_pitcher_cache_path = cache
        stderr = StringIO()
        with redirect_stderr(stderr):
            assert compute._build_probable_pitcher_lookup(str(work / 'absent')) == {}
        print('unreadable P6 case, no rewrite:', cache.read_text(), stderr.getvalue().strip())
        put(cache, {'100': SIG})  # Rebuilt copy now lacks cache-only 999.
        sync.sync_to_r2(client, processed, models)
        first_prev = client.objects[sync.PREV_MANIFEST_KEY]['files'][key]['key']
        assert first_prev == cached_object and '999' in json.loads(client.objects[cached_object])
        sync.sync_to_r2(client, processed, models)
        assert client.objects[sync.PREV_MANIFEST_KEY]['files'][key]['key'] != cached_object
        print('cache-only game 999 recoverable via outgoing R2 manifest after first sync; second sync advances previous pointer')


def active_nested_team(work):
    old = Path.cwd()
    os.chdir(work)
    try:
        compute._probable_pitcher_cache_path = Path('data/models/cache.json')
        rows = [pa(100, '2026-04-01', 3, True), pa(100, '2026-04-01', 22, False)]
        for value in ([51], {'id': 51}):
            put(compute._probable_pitcher_cache_path, {'100': dict(SIG, away_tid=value)})
            print('active nested team group key:', *feature(rows)[:2])
    finally:
        os.chdir(old)


def main():
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(prefix='health-', dir=ROOT / 'runs') as name:
        work = Path(name)
        for probe in (writer_reader_paths, end_of_day, filesystem, backup, active_nested_team):
            probe_dir = work / probe.__name__
            probe_dir.mkdir()
            probe(probe_dir)


if __name__ == '__main__':
    main()
