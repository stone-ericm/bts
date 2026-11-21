from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os

from probe_features import ROOT, SIG, compute, feed, put
from probe_v3 import CACHE, RAW, STATUS, atomic_bytes, evidence_path, observation, status_path


def attempt(work, failure):
    cache = work / CACHE
    raw = work / RAW.parent.parent
    feed(work / RAW, SIG)
    cache.parent.mkdir(parents=True, exist_ok=True)
    original = b'\xff\xfe' + json.dumps({'777': SIG}).encode('utf-16-le')
    assert 777 == int(next(iter(json.loads(original))))
    cache.write_bytes(original)
    evidence = work / evidence_path(original)
    health = work / STATUS
    if failure in ('evidence', 'both'):
        evidence.parent.write_bytes(b'file blocking evidence directory')
    if failure in ('status', 'both'):
        health.parent.mkdir(parents=True)
        health.write_bytes(b'file blocking status directory')

    source_replace = os.replace
    errors = []
    row = None

    def projected_gate(staged, destination):
        nonlocal row
        assert destination == cache
        preserved = None
        try:
            atomic_bytes(evidence, original, 'evidence', 'none')
            preserved = evidence.name
        except OSError as error:
            errors.append(('evidence', type(error).__name__))
        row = observation(reason='unreadable_cache', counts=(None, None, 1, None, None), affected=(), preserved=preserved)
        try:
            atomic_bytes(status_path(health, row), json.dumps(row).encode(), 'record', 'none')
        except OSError as error:
            errors.append(('status', type(error).__name__))
        if not errors:
            source_replace(staged, destination)

    compute._probable_pitcher_cache_path = cache
    stderr = StringIO()
    with redirect_stderr(stderr), patch.object(os, 'replace', projected_gate):
        lookup = compute._build_probable_pitcher_lookup(str(raw))
    assert lookup == {109: SIG}
    assert 'UnicodeDecodeError' in stderr.getvalue()
    assert (cache.read_bytes() == original) == (failure != 'none')
    status = status_path(health, row)
    assert status.is_file() == (failure not in ('status', 'both'))
    if evidence.is_file():
        assert evidence.read_bytes() == original and row['preserved'] == evidence.name
    print('unreadable UTF-16 bytes / gate', failure, 'errors', errors, 'evidence', evidence.is_file(),
          'published', status.is_file(), 'original still active', cache.read_bytes() == original)


def raw_only(work):
    cache = work / CACHE
    put(cache, {'100': SIG})
    feed(work / RAW, dict(SIG, away='1'))
    compute._probable_pitcher_cache_path = cache
    lookup = compute._build_probable_pitcher_lookup(str(work / RAW.parent.parent))
    assert lookup[109]['away'] == '1'
    print('all-valid initial cache + raw-only rejected candidate: base includes and persists game 109')


def main():
    with TemporaryDirectory(dir=ROOT / 'runs') as name:
        directory = Path(name)
        for case in ('evidence', 'status', 'both', 'none'):
            attempt(directory / case, case)
        raw_only(directory / 'raw-only')


if __name__ == '__main__':
    main()
