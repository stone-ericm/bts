from datetime import date, datetime, timedelta
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from bts import entry_receipt
from bts.picks import earliest_pick_game_et, save_pick
from replay_archives import activate
from review_support import DAY, ET, ROOT, RIGHT, WRONG, World, assert_source, deadline, health, row


def check_final_pitch_formats(root):
    slots = [datetime.fromisoformat(f'{DAY}T10:00').replace(tzinfo=ET) + timedelta(minutes=15 * index)
             for index in range(56)]
    for label, pitch, has_slot in (
        ('naive UTC', f'{DAY}T14:05:00', False), ('Z', f'{DAY}T14:05:01Z', True),
        ('explicit offset', f'{DAY}T11:15:00-04:00', True),
        ('late same date', f'{DAY}T23:50:00-04:00', True),
        ('last slot + 75', '2027-03-25T01:00:00-04:00', True),
        ('last slot + 75 + 1s', '2027-03-25T01:00:01-04:00', False),
    ):
        w = World(root / label)
        daily = w.commit_at('10:05', at='09:40')
        w.capture([RIGHT], f'{DAY}T134000Z')
        assert w.cron('09:45', pending=[row()])['receipt']['qualification']['all_confirmed']
        daily.pick.game_time = pitch
        save_pick(daily, w.picks)
        at = earliest_pick_game_et(daily)
        independent = any(5 < (at - slot).total_seconds() / 60 <= 75 for slot in slots)
        assert independent is has_slot and entry_receipt.has_original_slot(date.fromisoformat(DAY), at) is has_slot
        morning = w.morning()
        expected = 'entry_unverifiable' if has_slot else 'entered'
        assert morning[1]['state'] == expected, (label, morning)
        print(label, at, 'original slot', independent, 'morning', expected)


def expected_amended_sequences(root):
    w = World(root / 'manual-after-last-early')
    w.commit_at('10:05', at='09:40')
    w.capture([WRONG], f'{DAY}T134000Z')
    early = w.cron('09:45', pending=[row()])
    w.capture([{**RIGHT, 'id': 78}], f'{DAY}T135000Z')
    manual = w.invoke('09:50', manual=True, pending=[row(78)])
    assert len(early['sent']) == 1 and not manual['sent']
    assert w.cron('10:00')['receipt']['outcome'] == 'outside_window'
    print('manual fix after final early slot', len(early['sent']), len(manual['sent']))

    w = World(root / 'interrupted-after-confirm')
    w.commit_at('10:05', at='09:00')
    w.cron('09:15')
    w.capture([RIGHT], f'{DAY}T132000Z')
    interrupted = w.cron('09:30', pending=[row()], dm='interrupt')
    assert interrupted['attempts'] and not interrupted['sent'] and w.early_state()['status'] == 'confirmed'
    assert not w.cron('09:45', pending=[row()])['calls'] and not health(w)
    print('interrupt after marker replacement before confirmed all-clear', interrupted['receipt']['detail'])

    from standing import fifo_with_live, skip_and_streak
    fifo_with_live(root / 'fifo-with-live', 'f5')
    skip_and_streak(root / 'skip-and-streak', 'f5')


def unrepresentable_cutoff(root):
    for failed in (False, True):
        w = World(root / str(failed))
        daily = w.commit_at('06:10', at='04:55')
        if not failed:
            w.capture([RIGHT], f'{DAY}T085000Z')
        result = w.cron('05:00', dm='failed' if failed else 'sent', pending=[] if failed else [row()])
        assert result['receipt']['outcome'] == 'observed'
        daily.pick.game_time = '0001-01-01T05:00:00Z'
        assert earliest_pick_game_et(daily).date() == date.min
        save_pick(daily, w.picks)
        try:
            actual = health(w) if failed else w.morning()
        except (AssertionError, OverflowError) as exc:
            actual = type(exc).__name__, str(exc)
        print('unrepresentable cutoff', 'health' if failed else 'morning', actual)


if __name__ == '__main__':
    assert_source(ROOT / 'src')
    activate(Path(sys.argv[1]))
    with TemporaryDirectory() as dirname:
        root = Path(dirname)
        check_final_pitch_formats(root)
        expected_amended_sequences(root)
        unrepresentable_cutoff(root / 'malformed')
