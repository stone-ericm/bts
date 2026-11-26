from datetime import datetime
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from bts.picks import save_pick
from review_support import DAY, ROOT, RIGHT, World, assert_source, deadline, health, row


def gate(root):
    cases = (
        ('inclusive-75', '06:15', '05:00', 'observed'),
        ('outside-75', '06:15:01', '05:00', 'outside_window'),
        ('strict-cutoff', '06:05', '06:00', 'outside_window'),
        ('before-cutoff', '06:05:01', '06:00', 'observed'),
        ('no-original-1000', '10:05', '09:45', 'observed'),
        ('sole-original-1000', '10:05:01', '09:45', 'base_slot_available'),
        ('starts-at-1000', '10:05:01', '10:00', 'observed'),
        ('one-second-late', '10:05:01', '10:00:01', 'outside_window'),
        ('original-75', '11:15', '10:00', 'observed'),
    )
    for name, pitch, at, expected in cases:
        w = World(root / name)
        w.commit_at(pitch, at='04:30')
        trace = w.invoke(at)
        assert trace['receipt']['outcome'] == expected
        assert bool(trace['calls']) == bool(trace['sent']) == (expected == 'observed')
        assert bool(trace['receipt'].get('early_cron_run')) == (at < '10:00' and expected == 'observed')
        assert (w.early.exists() or w.marker.exists()) == (expected == 'observed')
        print(name, at, pitch, expected, trace['calls'], trace['receipt'].get('early_cron_run'))
    w = World(root / 'double')
    w.commit_at('13:10', at='04:55', dd=(1020, 202, '06:10'))
    trace = w.cron('05:00')
    assert trace['receipt']['outcome'] == 'observed' and 'DD Batter 1020' in trace['sent'][0]
    assert trace['receipt']['cutoff_at'].startswith(DAY + 'T06:05')
    print('double-down earlier leg', trace['receipt']['cutoff_at'])


def morning(root):
    cases = ('unchanged', 'late', 'gain-slot', 'lose-slot-to-live-absence', 'changed-leg', 'fifo',
             'wrong-date', 'invalid-time', 'oversize', 'invalid-encoding', 'non-object')
    for case in cases:
        w = World(root / case)
        daily = w.commit_at('10:05', at='09:40')
        w.capture([RIGHT], f'{DAY}T134000Z')
        trace = w.cron('09:45', pending=[row()], elapsed=360 if case == 'late' else 0)
        assert trace['receipt']['qualification']['all_confirmed'] and trace['receipt']['early_cron_run'] is True
        expected = 'entry_unverifiable'
        pick = w.picks / f'{DAY}.json'
        if case == 'unchanged':
            expected = 'entered'
        elif case == 'late':
            expected = 'entered_late'
            w.edit_pitch(daily, '09:55')
        elif case == 'gain-slot':
            w.edit_pitch(daily, '11:15')
        elif case == 'lose-slot-to-live-absence':
            w.edit_pitch(daily, '10:20')
            live = w.cron('10:00')
            assert live['sent'] and not live['receipt'].get('early_cron_run')
            w.edit_pitch(daily, '10:05')
            expected = 'not_entered'
        elif case == 'changed-leg':
            daily.pick.game_pk = 102
            save_pick(daily, w.picks)
        elif case == 'fifo':
            pick.unlink()
            os.mkfifo(pick)
        else:
            data = pick.read_bytes()
            pick.write_bytes({'wrong-date': data.replace(DAY.encode(), b'2027-03-23'),
                              'invalid-time': data.replace(b'T10:05:00-04:00', b'garbage'),
                              'oversize': data + b' ' * (1024 * 1024),
                              'invalid-encoding': b'\xff', 'non-object': b'[]'}[case])
        with deadline(1):
            result = w.morning()
        assert result[1]['state'] == expected, (case, result)
        assert result[0] == int(expected != 'entered') and len(result[2]) == result[0]
        print('morning', case, result[0], result[1]['state'], result[2])


def rollover(root):
    old = World(root, day='2027-12-31')
    old.commit_at('06:10', at='04:55')
    old.cron('05:00', dm='failed')
    previous = old.early.read_bytes()
    new = World(root, day='2028-01-01')
    midnight = new.cron('00:00')
    assert midnight['receipt']['outcome'] == 'no_pick_file' and not midnight['calls']
    assert not new.early.exists() and old.early.read_bytes() == previous
    new.commit_at('06:10', at='04:55')
    new.cron('05:00', pending=[row()])
    assert new.early_state()['date'] == new.day and old.early.read_bytes() == previous
    assert health(old)[0][0] == 'CRITICAL' and health(new) == []
    print('midnight and year boundary', old.early.name, new.early.name)


if __name__ == '__main__':
    assert_source(ROOT / 'src')
    with TemporaryDirectory() as dirname:
        root = Path(dirname)
        gate(root)
        morning(root)
        rollover(root / 'year')
