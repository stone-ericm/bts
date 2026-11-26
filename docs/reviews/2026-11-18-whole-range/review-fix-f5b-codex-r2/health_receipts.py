"""Recheck B2's sealed inventory and ordering with a pinned elapsed clock."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'docs/reviews/2026-11-18-whole-range/review-fix-f5b-codex'))

from bts.cli import _atomic_write_json
from health_probe import input_matrix
from receipt_inventory_probe import health_with_sealed_recovery, precedence_without_seal
from review_support import DAY, RIGHT, World, assert_source, health, row


def displaced_by_yesterday(w):
    previous = World(w.root, day='2027-03-23')
    previous.commit_at('19:10', at='17:50')
    result = previous.invoke('18:15', manual=True, pending=[row()])
    assert result['marker']['date'] == previous.day


def paused_manual_write(root):
    w = World(root)
    w.commit_at('06:10', at='04:40')
    w.capture([RIGHT], f'{DAY}T084000Z')
    tick = 1000

    def later_write(path, marker):
        nonlocal tick
        if path == w.marker:
            missed = w.cron('05:00', dm='failed')
            assert missed['receipt']['marker_status'] == 'dm_failed' and not w.marker.exists()
            tick += 6 * 60
        _atomic_write_json(path, marker)

    recovery = w.invoke('04:55', manual=True, pending=[row()], patches={
        'bts.cli._atomic_write_json': later_write, 'time.monotonic': lambda: tick,
    })
    receipt = recovery['receipt']
    assert receipt['marker_status'] == 'confirmed'
    early = w.early_state()['observed_at']
    assert receipt['observation']['response_completed_at'] < early < receipt['published_at']
    displaced_by_yesterday(w)
    print('paused before write', receipt['observation']['response_completed_at'], early, receipt['published_at'], health(w))
    assert health(w) == []


def delayed_after_write(root, delay):
    w = World(root)
    w.commit_at('06:10', at='04:40')
    w.capture([RIGHT], f'{DAY}T084000Z')
    assert len(w.invoke('04:55', manual=True)['sent']) == 1
    tick = 1000
    output = []

    def early_failure():
        w.commit_at('06:10', at='04:59:55', game=102, batter=1020)
        missed = w.cron('05:00', dm='failed')
        assert missed['receipt']['marker_status'] == 'dm_failed' and len(missed['attempts']) == 1

    def send_after_marker(_, msg):
        nonlocal tick
        assert w.read_marker()['status'] == 'confirmed' and not w.early.exists()
        output.append(msg)
        if delay:
            early_failure()
            tick += delay

    recovery = w.invoke('04:59:50', manual=True, pending=[row()], patches={
        'bts.dm.send_dm': send_after_marker, 'time.monotonic': lambda: tick,
    })
    assert len(output) == 1 and output[0].startswith('✅')
    if not delay:
        early_failure()
    r = recovery['receipt']
    assert r['marker_status'] == 'confirmed' and r['selection']['slots'][0]['game_pk'] == 101
    assert w.early_state()['slots'][0]['game_pk'] == 102
    displaced_by_yesterday(w)
    alerts = health(w)
    print(f'delay {delay}s after marker write', r['observation']['response_completed_at'],
          w.early_state()['observed_at'], r['published_at'], 'alerts', alerts)
    return [level for level, _ in alerts]


mode, scratch = sys.argv[1:]
assert_source((Path(scratch) / 'r1' if mode == 'limit-round-one' else ROOT) / 'src')
with TemporaryDirectory() as folder:
    root = Path(folder)
    if mode == 'regression':
        health_with_sealed_recovery(root / 'short', 1)
        health_with_sealed_recovery(root / 'full', 325)
        precedence_without_seal(root / 'unsealed')
        paused_manual_write(root / 'paused')
    elif mode == 'inputs':
        input_matrix(root)
    else:
        assert delayed_after_write(root / 'no-delay', 0) == ['CRITICAL']
        for elapsed in (10.01, 30):
            alerts = delayed_after_write(root / str(elapsed), elapsed)
            assert alerts == (['CRITICAL'] if mode == 'limit-round-one' else [])
