"""Check the installed early-only window with a nonregular units snapshot."""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'docs/reviews/2026-11-18-whole-range/review-fix-f5b-codex'))

from bts import entry_receipt
from review_support import DAY, RIGHT, ReadTimedOut, World, assert_source, deadline, load_line, row
from probe_support import clock_slots


def suspect_capture(root, version, use_fifo, present):
    w = World(root, version)
    w.commit_at('06:10', at='04:55')
    capture = w.capture([RIGHT], f'{DAY}T085000Z')
    if use_fifo:
        capture.unlink()
        os.mkfifo(capture)

    def watch_read(path):
        if path == capture:
            print('reading', path.name, 'fifo' if use_fifo else 'regular file', flush=True)
        return read_bytes(path)

    at = '05:00' if version == 'f5' else '10:00'
    assert (at == '05:00') is ('05:00' in {t.strftime('%H:%M') for t in clock_slots(load_line(version))})
    read_bytes = Path.read_bytes
    with patch.object(Path, 'read_bytes', watch_read), deadline(0.35):
        result = w.invoke(at, pending=[row()] if present else [])
    sealed = entry_receipt.discover(w.picks, DAY)
    print(at, 'fifo', use_fifo, 'present', present, 'exit', result['exit'], 'stdout', result['stdout'].strip(),
          'stderr', result['stderr'].strip(), 'receipt count', len(sealed), 'early marker', w.early_state(), flush=True)
    if version == 'base':
        assert not result['calls'] and result['receipt']['outcome'] == 'outside_window'
    elif use_fifo:
        assert not sealed and 'ReadTimedOut' in result['stderr']
        if present:
            assert 'game check unavailable' in result['stderr']
    else:
        assert result['receipt']['qualification']['all_confirmed'] and result['exit'] == 0


version, scratch = sys.argv[1:]
assert_source((Path(scratch) / 'base' if version == 'base' else ROOT) / 'src')
with TemporaryDirectory() as folder:
    root = Path(folder)
    if version == 'base':
        suspect_capture(root / 'fifo', version, True, True)
    else:
        suspect_capture(root / 'control', version, False, True)
        suspect_capture(root / 'fifo-presence', version, True, True)
        suspect_capture(root / 'fifo-absence', version, True, False)
