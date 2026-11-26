"""Replay B1, then exercise absent, failed and already-cleared warning history."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'docs/reviews/2026-11-18-whole-range/review-fix-f5b-codex'))

from bts.picks import save_pick
from review_support import DAY, RIGHT, WRONG, World, assert_source, row


def warning_then_correction(root, wrong, manual):
    w = World(root)
    pick = w.commit_at('06:10', at='04:55')
    if wrong:
        w.capture([WRONG], f'{DAY}T085000Z')
    start = w.invoke('05:00', pending=[row()] if wrong else [], manual=manual)
    assert len(start['sent']) == 1 and start['sent'][0].startswith('⚠')
    w.edit_pitch(pick, '06:20')
    w.capture([{**RIGHT, 'id': 78}], f'{DAY}T091000Z')
    recovery = w.invoke('05:15', pending=[row(78)], manual=manual)
    assert recovery['receipt']['qualification']['all_confirmed'] and len(recovery['sent']) == 1
    assert recovery['sent'][0].startswith('✅')
    repeat = w.invoke('05:30', pending=[row(78)], manual=manual)
    assert repeat['receipt']['outcome'] == 'already_confirmed' and not repeat['calls'] and not repeat['sent']
    if not manual:
        w.edit_pitch(pick, '06:15')
        reopened = w.cron('05:45', pending=[row(78)])
        assert reopened['calls'] and not reopened['sent']
    print('wrong game' if wrong else 'missing entry', 'manual' if manual else 'early cron', 'one all-clear')


def equivalent_pitch(root):
    w = World(root)
    pick = w.commit_at('06:10', at='04:55')
    w.capture([RIGHT], f'{DAY}T085000Z')
    assert w.cron('05:00', pending=[row()])['receipt']['qualification']['all_confirmed']
    for time, pitch in [('05:15', f'{DAY}T10:10:00Z'), ('05:30', f'{DAY}T10:10:00')]:
        pick.pick.game_time = pitch
        save_pick(pick, w.picks)
        repeat = w.cron(time, pending=[row()])
        assert repeat['receipt']['outcome'] == 'already_confirmed' and not repeat['calls'] and not repeat['sent']
        print(pitch, 'terminal, zero account calls')


def no_warning_sent(root, kind):
    w = World(root)
    pick = w.commit_at('06:10', at='04:55')
    if kind == 'late':
        start = w.cron('06:00', elapsed=5 * 60)
        w.edit_pitch(pick, '06:30')
        recovery_at = '06:15'
        assert not start['attempts'] and start['receipt']['marker_status'] == 'alerted'
    else:
        if kind == 'without-recipient':
            index = w.args.index('--dm-recipient')
            args = w.args[:]
            del w.args[index:index + 2]
            start = w.cron('05:00')
            w.args = args
            assert not start['attempts'] and w.early_state()['escalations'] == ['initial']
        else:
            start = w.cron('05:00', dm='failed')
            assert len(start['attempts']) == 1 and start['receipt']['marker_status'] == 'dm_failed'
        w.edit_pitch(pick, '06:20')
        recovery_at = '05:15'
    assert not start['sent'] and w.early_state()['pending_all_clear'] is False
    w.capture([RIGHT], f'{DAY}T085000Z')
    recovery = w.cron(recovery_at, pending=[row()])
    assert recovery['receipt']['qualification']['all_confirmed'] and not recovery['attempts']
    print(kind, 'no warning sent and no all-clear attempted')


def legacy_transition(root, mode):
    w = World(root)
    if mode == 'save-round-one':
        w.commit_at('06:10', at='04:55')
        index = w.args.index('--dm-recipient')
        del w.args[index:index + 2]
        result = w.cron('05:00')
        ledger = w.early_state()
        assert not result['attempts'] and 'pending_all_clear' not in ledger and ledger['escalations'] == ['initial']
        print('round-one ledger with no send', json.dumps(ledger, sort_keys=True))
    else:
        w.capture([RIGHT], f'{DAY}T091000Z')
        result = w.cron('05:15', pending=[row()])
        assert result['receipt']['qualification']['all_confirmed'] and len(result['sent']) == 1
        print('round-two DMs after round-one invocation without a recipient', result['sent'])


if len(sys.argv) > 1:
    mode, sandbox = sys.argv[1:]
    sandbox = Path(sandbox)
    assert_source((sandbox / 'r1' if mode == 'save-round-one' else ROOT) / 'src')
    legacy_transition(sandbox / 'older-marker', mode)
else:
    assert_source(ROOT / 'src')
    with TemporaryDirectory() as folder:
        root = Path(folder)
        for wrong in (False, True):
            for manual in (False, True):
                warning_then_correction(root / f'{wrong}-{manual}', wrong, manual)
        equivalent_pitch(root / 'utc')
        for kind in ('failed', 'late', 'without-recipient'):
            no_warning_sent(root / kind, kind)
