"""Check how deep JSON fails and what reconcile already wrote before its exit."""
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.codex-review/scratch-f4/src'),
                str(ROOT / 'docs/reviews/2026-11-18-whole-range/closing-test/codex-blind')]

from bts import picks
import offline
from offline import World, at
from opening import DAY2, OPEN

offline.ROOT = ROOT


with TemporaryDirectory() as tmp, World(OPEN, Path(tmp)) as w:
    path = w.directory / 'streak.json'
    w.now = at(DAY2, '08:30')
    for depth in (6000, 12000, 25000, 60000, 100000):
        path.write_text('{"ignored":' + '[' * depth + '0' + ']' * depth + '}')
        print('nesting test:', depth, flush=True)
        try:
            count = picks.load_streak(w.directory, season=2027)
            loaded = f'accepted, streak {count}'
        except Exception as exc:
            loaded = type(exc).__name__
        result = w.outcome(OPEN)
        print('loaded:', loaded, 'candidate status and exception:', result.exit_code,
              type(result.exception).__name__, 'candidate JSON:', result.stdout.strip(), flush=True)
        if loaded == 'RecursionError':
            break

with TemporaryDirectory() as tmp, World(OPEN, Path(tmp)) as w:
    primary = picks.pick_from_row(w.frame().iloc[0])
    path = w.directory / 'streak.json'
    before_open = '2027-03-23'
    old = picks.DailyPick(before_open, w.now.isoformat(), primary, None, None)
    scored = picks.DailyPick(OPEN, w.now.isoformat(), primary, None, None, result='hit',
                             slot_results={'pick': 'hit'}, notification_sent=True)
    picks.save_pick(old, w.directory)
    picks.save_pick(scored, w.directory)
    path.write_text('[]')
    # The 01:00 cron exits early for the terminal production result.
    w.now = at(DAY2, '01:00')
    print('already resolved, corrupt file:', w.grade_command(expect=0).stdout.strip())
    w.results[101] = 'miss'
    w.now = at(DAY2, '02:00')
    result = w.invoke('reconcile')
    assert isinstance(result.exception, AttributeError)
    assert picks.load_pick(OPEN, w.directory).result == 'miss' and path.read_text() == '[]'
    print('reconcile error with correction already saved:', result.exit_code, repr(result.exception), result.output)
    receipts = sorted(w.directory.parent.rglob('*reconcile_receipts/*/*.json'))
    print('receipt before failed CLI last read:', [json.loads(f.read_text())['outcome'] for f in receipts])

    # An otherwise identical run with a complete history can rebuild, even after the correction cutoff.
    old.result = 'miss'
    picks.save_pick(old, w.directory)
    w.now = at(DAY2, '08:20')
    result = w.invoke('reconcile', expect=0)
    assert picks.load_streak(w.directory, season=2027) == 0 and path.read_text() != '[]'
    print('complete replay after cutoff:', result.stdout.strip(), path.read_text())
