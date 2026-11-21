from contextlib import redirect_stderr
from datetime import date, datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo
import json
import logging
import os
import re
import signal
import stat
import subprocess
import sys

import pandas as pd

from probe_features import ROOT, SIG, compute, feature, feed, pa, put
from probe_health import MemoryR2, alert, attention, sync


CACHE = Path('data/models/probable_pitcher_lookup.json')
RAW_109 = Path('data/raw/2025/109.json')
HEALTH = Path('data/health_state')
STATUS = {
    'invalid_entries': 'probable_pitcher_cache_invalid_entries.json',
    'unreadable_cache': 'probable_pitcher_cache_unreadable.json',
}
DAY = date(2026, 12, 31)
BEFORE = '2027-01-01T04:59:00+00:00'
AFTER = '2027-01-01T05:01:00+00:00'
SOURCE = 'probable_pitcher_cache'


def writer_path(cache, reason):
    return cache.parent.parent / 'health_state' / STATUS[reason]


def frozen_child(mode):
    disk = json.loads(CACHE.read_text())
    dropped = [k for k, v in disk.items() if not isinstance(v, dict)]
    writes = []
    feed_reads = []
    compute._probable_pitcher_cache_path = CACHE
    real_builder = compute._build_probable_pitcher_lookup
    actual_read = Path.read_text
    dates = pd.date_range('2025-04-01', periods=11)
    rows = [pa(100 + i, day, 3 if i < 10 else 1, True, i % 2) for i, day in enumerate(dates)]

    def monitor_read(path, *args, **kwargs):
        if path == RAW_109:
            feed_reads.append(str(path))
        return actual_read(path, *args, **kwargs)

    def projected_builder():
        if mode == 'base' or not dropped:
            return real_builder()
        # Project the contract's drop/no-write rules. The real scanner uses a scratch copy
        # with the rejected member removed, then the real feature code consumes its result.
        with TemporaryDirectory(prefix='scan-', dir=Path.cwd()) as name:
            scratch = Path(name) / 'cache.json'
            put(scratch, {k: v for k, v in disk.items() if k not in dropped})
            with patch.object(compute, '_probable_pitcher_cache_path', scratch):
                result = real_builder()
            return result

    def monitor_replace(source, target, *args, **kwargs):
        if Path(target) == CACHE:
            writes.append(str(target))
        return actual_replace(source, target, *args, **kwargs)

    actual_replace = os.replace
    with patch.object(Path, 'read_text', monitor_read), patch.object(os, 'replace', monitor_replace), patch.object(
        compute, '_build_probable_pitcher_lookup', projected_builder
    ):
        _, outcome, frame = feature(rows)
    assert outcome == 'ok', outcome
    field = float(frame.iloc[-1]['opp_bullpen_hr_30g'])
    after = json.loads(CACHE.read_text())
    print(json.dumps({'feature': field, 'disk_new': '109' in after, 'read_bad': dropped,
                      'would_report': dropped if mode == 'v2' else [], 'cache_replaces': len(writes),
                      'feed_reads': len(feed_reads)}))


def cache_freeze(work):
    initial = {'999': None, **{str(game): SIG for game in range(100, 111) if game != 109}}
    for mode in ('base', 'v2'):
        cwd = work / mode
        put(cwd / CACHE, initial)
        feed(cwd / RAW_109, SIG)

        def fresh_process(label):
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', mode],
                                    capture_output=True, text=True, check=True, cwd=cwd, timeout=7)
            print('cache', mode, label, result.stdout.strip())
            return json.loads(result.stdout)

        first = fresh_process('first')
        second = fresh_process('restart with feed')
        assert first['feature'] == second['feature'] == 0.5
        assert first['feed_reads'] == 1
        assert second['feed_reads'] == (mode == 'v2')
        (cwd / RAW_109).unlink()
        third = fresh_process('restart after archiving feed')
        assert third['disk_new'] == (mode == 'base')
        if mode == 'base':
            assert third['feature'] == 0.5
        else:
            assert pd.isna(third['feature']) and third['read_bad'] == ['999']
            put(cwd / CACHE, {k: v for k, v in initial.items() if k != '999'})
            fourth = fresh_process('targeted repair after feed was archived')
            assert pd.isna(fourth['feature']) and not fourth['read_bad']


def unreadable_commit(work):
    raw = work / 'raw'
    feed(raw / '2025/100.json', SIG)
    real_replace = os.replace
    for case in ('directory', 'fifo', 'symlink', 'unwritable', 'interrupt', 'works'):
        cache = work / case / CACHE
        target = writer_path(cache, 'unreadable_cache')
        cache.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True)
        cache.write_text('{')
        if case == 'directory':
            target.mkdir()
        elif case == 'fifo':
            os.mkfifo(target)
        elif case == 'symlink':
            target.symlink_to(cache)

        def replace_with_v2_gate(source, destination, *args, **kwargs):
            assert Path(destination) == cache and cache.read_text() == '{'
            try:
                if os.path.lexists(target) and not stat.S_ISREG(target.lstat().st_mode):
                    raise OSError('special destination')
                if case == 'unwritable':
                    raise PermissionError('injected access failure')
                with TemporaryDirectory(prefix='status-', dir=target.parent) as name:
                    staged = Path(name) / 'observation.json'
                    staged.write_text(json.dumps({'reason': 'unreadable_cache', 'rescanned': 1}))
                    real_replace(staged, target)
            except OSError as error:
                print(f'WARNING: could not write probable-pitcher cache status at {target} ({type(error).__name__})',
                      file=sys.stderr)
                return
            if case == 'interrupt':
                raise KeyboardInterrupt('after status, before cache replacement')
            real_replace(source, destination, *args, **kwargs)

        stderr = StringIO()
        compute._probable_pitcher_cache_path = cache
        with redirect_stderr(stderr), patch.object(os, 'replace', replace_with_v2_gate):
            try:
                assert compute._build_probable_pitcher_lookup(str(raw)) == {100: SIG}
            except KeyboardInterrupt:
                assert case == 'interrupt'
        committed = cache.read_text() != '{'
        assert committed == (case == 'works')
        expected_lines = 1 if case in ('works', 'interrupt') else 2
        assert len(stderr.getvalue().splitlines()) == expected_lines
        print('unreadable v2 gate', case, 'published:', target.is_file() and not target.is_symlink(),
              'cache replaced:', committed, 'stderr lines:', expected_lines)


def record(reason, when, game=999, **extras):
    return {'schema_version': 'bts_probable_pitcher_cache_status_v1', 'recorded_at': when,
            'recorded_date_et': datetime.fromisoformat(when).astimezone(ZoneInfo('America/New_York')).date().isoformat(),
            'cache_path': str(CACHE), 'reason': reason, 'dropped': 1 if reason == 'invalid_entries' else None,
            'recovered_from_cache': 0 if reason == 'invalid_entries' else None,
            'rescanned': 0, 'unusable_feed': 0 if reason == 'invalid_entries' else None,
            'without_feed': 1 if reason == 'invalid_entries' else None,
            'game_ids': [game] if reason == 'invalid_entries' else [], **extras}


def split_status(work):
    contract = (ROOT / 'reference/kickoff-fix-w15.md').read_text()
    line = next(line for line in contract.splitlines() if line.startswith('| P6 |'))
    templates = dict(zip([*STATUS, 'status_unreadable'], re.findall(r'`([^`]+)`', line), strict=True))
    receipt = work / 'receipt.json'
    attention_state = work / 'attention.json'
    health = work / 'data/health_state'
    calls = []

    def render(row):
        reason = row['reason']
        replacements = {'recorded_at': row['recorded_at'], 'n': row['dropped'],
                        'ids': ', '.join(map(str, row['game_ids'])), 'a': row['recovered_from_cache'],
                        'k': row['rescanned'], 'u': row['unusable_feed'], 'm': row['without_feed']}
        body = templates[reason].replace('entr(y|ies)', 'entry' if row['dropped'] == 1 else 'entries')
        for name, val in replacements.items():
            body = body.replace(f'<{name}>', str(val))
        return alert.Alert('WARN', SOURCE, body, f'{SOURCE}:{reason}')

    def projected_health(day):
        found = []
        for name in STATUS.values():
            path = health / name
            if path.is_file():
                row = json.loads(path.read_bytes())
                recorded = date.fromisoformat(row['recorded_date_et'])
                if recorded in (day, day - timedelta(days=1)):
                    found.append(render(row))
        return found

    def send(recipient, body):
        calls.append(body)
        return 'offline-id'

    def dispatch(day, selected):
        _, warns = attention.apply_warn_attention_policy(selected, state_path=attention_state, today=day)
        return alert.dispatch_dm_for_health_alerts(selected, 'offline-recipient', warns, receipt, day)

    older = record('invalid_entries', BEFORE, 999)
    newer = record('invalid_entries', AFTER, 998, rescanned=1, without_feed=0)
    unreadable = record('unreadable_cache', BEFORE, rescanned=0)
    put(health / STATUS['invalid_entries'], older)
    with patch.object(alert, 'send_dm', send), patch.object(
        attention, 'ALWAYS_ATTENTION_WARN_SOURCES', attention.ALWAYS_ATTENTION_WARN_SOURCES | {SOURCE}
    ):
        assert dispatch(DAY, projected_health(DAY))
        assert not dispatch(DAY, projected_health(DAY))
        put(health / STATUS['unreadable_cache'], unreadable)
        assert dispatch(DAY, projected_health(DAY))
        assert calls[-1].count(f'[{SOURCE}]') == 2
        print('same day later different reason: digest resent the covered invalid WARN and added unreadable')
        assert dispatch(DAY + timedelta(days=1), projected_health(DAY + timedelta(days=1)))
        assert not dispatch(DAY + timedelta(days=1), projected_health(DAY + timedelta(days=1)))
        print('D and D+1 both send D reports, even if there was an intervening repair and no new drop')

        backtest = work / CACHE
        dashboard_reader = work / 'data/picks'
        assert writer_path(backtest, 'invalid_entries') == health / STATUS['invalid_entries']
        assert dashboard_reader.parent / 'health_state' == health
        put(writer_path(backtest, 'invalid_entries'), newer)
        late = projected_health(DAY)
        assert len(late) == 1 and 'was unreadable' in late[0].message
        assert not dispatch(DAY + timedelta(days=1), projected_health(DAY + timedelta(days=1)))
        print('same-checkout backtest updates invalid slot at 00:01 ET; late D omits D invalid event, D+1 same key hides update')
        custom = work / 'operations/picks'
        assert writer_path(backtest, 'invalid_entries').parent != custom.parent / 'health_state'
        print('supported custom picks_dir points the reader at a different directory')
        # A D call finishing after a D+1 caller and writing a timestamp it sampled earlier.
        put(health / STATUS['invalid_entries'], older)
        assert '999' in projected_health(DAY + timedelta(days=1))[0].message
        print('older timestamp finishing last overwrote the newer record under physical last-replace-wins')

    for label, trigger in (
        ('D absent feed', older), ('D+1 late rescan', newer), ('unreadable no usable scans', unreadable),
        ('unreadable rebuilt', record('unreadable_cache', BEFORE, rescanned=2)),
        ('ten-ID sample with additional games', record('invalid_entries', BEFORE, dropped=11,
                                                       without_feed=9, unusable_feed=1, recovered_from_cache=1,
                                                       game_ids=list(range(100, 110)))),
    ):
        print('P6', label, '\n' + alert.format_health_dm_body([], [render(trigger)]))
    body = templates['status_unreadable'].replace('<path>', str(HEALTH / STATUS['unreadable_cache']))
    print('P6 broken status\n' + alert.format_health_dm_body([], [alert.Alert('WARN', SOURCE, body)]))


class AgedMemoryR2(MemoryR2):
    def __init__(self):
        super().__init__()
        self.timestamps = {}

    def upload_file(self, source, key):
        super().upload_file(source, key)
        self.timestamps[key] = datetime.now(timezone.utc)

    def list_objects(self, prefix):
        return [{'key': key, 'last_modified': self.timestamps[key]} for key in self.objects if key.startswith(prefix)]


def manifest_rotation(work):
    processed = work / 'processed'
    models = work / 'models'
    processed.mkdir()
    cache = models / CACHE.name
    logical = f'models/{CACHE.name}'
    remote = AgedMemoryR2()
    with patch.object(sync, '_current_git_sha', return_value='offline'), patch.object(sync, '_current_git_branch', return_value='offline'):
        put(cache, {'100': SIG, '109': SIG})
        before = sync.sync_to_r2(remote, processed, models)['files'][logical]['key']
        # The good cache object can be older than the age guard even on its first day out of a manifest.
        remote.timestamps[before] -= timedelta(days=8)
        remote.objects['models/legacy.json'] = b'legacy'
        young = 'objects/young/probable_pitcher_lookup.json'
        remote.objects[young] = b'young'
        remote.timestamps[young] = datetime.now(timezone.utc)
        put(cache, {'999': None, '100': SIG})
        poisoned = sync.sync_to_r2(remote, processed, models)['files'][logical]['key']
        first = sync.prune_unreferenced(remote)
        assert before in remote.objects and remote.get_object_json(sync.PREV_MANIFEST_KEY)['files'][logical]['key'] == before
        assert '109' not in json.loads(remote.objects[poisoned])
        assert before not in first['deleted']
        print('R2 first sync of frozen corrupt cache publishes invalid member and omits 109; good cache in prev manifest')
        sync.sync_to_r2(remote, processed, models)
        assert before in remote.objects  # Sync only changes manifests.
        second = sync.prune_unreferenced(remote)
        assert second['deleted'] == [before] and second['kept_recent'] == [young]
        assert 'models/legacy.json' in remote.objects
        print('R2 next sync advances prev; prune deletes eight-day-old good object, retains young and legacy objects')

        cache.unlink()
        os.mkfifo(cache)

        def stop_read(signum, frame):
            raise KeyboardInterrupt('bounded probe interrupted FIFO')

        signal.signal(signal.SIGALRM, stop_read)
        signal.setitimer(signal.ITIMER_REAL, 0.2)
        try:
            sync.sync_to_r2(remote, processed, models)
        except KeyboardInterrupt:
            print('R2 cache FIFO still blocked base hashing until the probe interrupted it')
        else:
            raise AssertionError('FIFO open should have blocked')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


def main():
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(prefix='v2-', dir=ROOT / 'runs') as name:
        root = Path(name)
        for scenario in (cache_freeze, unreadable_commit, split_status, manifest_rotation):
            directory = root / scenario.__name__
            directory.mkdir()
            scenario(directory)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--child':
        frozen_child(sys.argv[2])
    else:
        main()
