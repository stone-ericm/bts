"""Replay the invalid early confirmations and an unrepresentable cutoff."""

from datetime import date
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / 'docs/reviews/2026-11-18-whole-range/review-fix-f5b-codex'))

from bts.picks import earliest_pick_game_et, save_pick
from review_support import DAY, RIGHT, World, assert_source, health, row


def bad_marker(root):
    for field, label, value in (
        ('observed_at', 'unparseable', 'not a datetime'),
        ('observed_at', 'offset missing', f'{DAY}T05:00:00'),
        ('observed_at', 'prior ET date', '2027-03-24T03:30:00+00:00'),
        ('receipt', 'list-valued', ['not', 'an', 'id']),
    ):
        w = World(root / label)
        w.commit_at('06:10', at='04:55')
        first = w.cron('05:00', pending=[row()])
        assert first['receipt']['outcome'] == 'observed' and w.early_state()['status'] == 'confirmed'
        marker = w.early_state()
        marker[field] = value
        w.early.write_text(json.dumps(marker))
        repeat = w.cron('05:15')
        assert repeat['receipt']['outcome'] == 'observed' and repeat['calls'] and len(repeat['sent']) == 1
        assert repeat['exit'] == 1 and repeat['receipt']['references'] is None
        print(field, label, 'ignored invalid confirmation, checked the account, sent initial warning')


def unrepresentable_cutoff(root, failed):
    w = World(root)
    daily = w.commit_at('06:10', at='04:55')
    if not failed:
        w.capture([RIGHT], f'{DAY}T085000Z')
    result = w.cron('05:00', dm='failed' if failed else 'sent', pending=[] if failed else [row()])
    assert result['receipt']['outcome'] == 'observed'
    daily.pick.game_time = '0001-01-01T05:00:00Z'
    assert earliest_pick_game_et(daily).date() == date.min
    save_pick(daily, w.picks)
    if failed:
        (level, message), = health(w)
        assert level == 'CRITICAL' and 'current selection is unknown' in message
        print('health', level, message)
    else:
        exit_code, report, messages = w.morning()
        assert exit_code == 1 and report['state'] == 'entry_unverifiable'
        assert len(messages) == 1 and 'ENTRY UNVERIFIABLE' in messages[0]
        print('morning', exit_code, report['state'], messages)
    repeat = w.cron('05:15')
    assert repeat['exit'] == 0 and repeat['receipt']['outcome'] == 'no_pick_file' and not repeat['calls']


assert_source(ROOT / 'src')
with TemporaryDirectory() as folder:
    root = Path(folder)
    bad_marker(root)
    unrepresentable_cutoff(root / 'morning', False)
    unrepresentable_cutoff(root / 'health', True)
