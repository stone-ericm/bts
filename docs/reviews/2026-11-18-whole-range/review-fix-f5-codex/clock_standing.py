"""Exercise generated clock slots and interrupted, malformed, and restarted runs."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
import signal
from tempfile import TemporaryDirectory

from probe_support import ET, Replay, clock_slots, load_line
from bts import entry_receipt
from bts.health import pick_entry
from tests.test_wr_f1_wrong_game import DAY


@contextmanager
def bounded_fifo():
    def interrupted(*_):
        raise TimeoutError('synthetic read bound')

    previous = signal.signal(signal.SIGALRM, interrupted)
    signal.setitimer(signal.ITIMER_REAL, 0.15)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def time_changes(root):
    base, f5 = load_line('base'), load_line('f5')
    wrapper = r'/usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout --verbose -k (\d+) (\d+)'
    assert re.findall(wrapper, base) == re.findall(wrapper, f5) == [('30', '840')]
    grace, limit = map(int, re.findall(wrapper, f5)[0])
    for day, count in (('2027-03-14', 92), (DAY, 96), ('2027-11-07', 100)):
        world = Replay(root / day, day=day)
        slots = list(clock_slots(f5, day))
        old = list(clock_slots(base, day))
        assert len(slots) == count and len(old) == 56
        gap = min((b.astimezone(timezone.utc) - a.astimezone(timezone.utc)).total_seconds()
                  for a, b in zip(slots, slots[1:]))
        assert gap == 900 and limit + grace < gap
        for instant in slots:
            result = world.invoke(instant.strftime('%H:%M'))
            assert result['receipt']['outcome'] == 'no_pick_file' and not result['calls']
        records = entry_receipt.discover(world.picks, day)
        assert len(records) == count
        counts = Counter(instant.strftime('%H:%M') for instant in slots)
        if count == 92:
            assert all(counts[f'02:{minute:02d}'] == 0 for minute in (0, 15, 30, 45))
        if count == 100:
            assert all(counts[f'01:{minute:02d}'] == 2 for minute in (0, 15, 30, 45))
            assert {slot.fold for slot in slots if slot.hour == 1} == {0, 1}
        print(day, f'{len(old)}/{count} launch projections; {count} local sealed receipts; {gap:.0f}s gap')
    before = list(clock_slots(f5, '2027-12-31'))[-1]
    after = list(clock_slots(f5, '2028-01-01'))[0]
    assert after - before == timedelta(minutes=15)

    spring = Replay(root / 'spring-window', day='2027-03-14')
    spring.commit_at('03:05', at='01:30')
    result = spring.invoke('01:45')
    saved = result['receipt']
    elapsed = (datetime.fromisoformat('2027-03-14T03:05').replace(tzinfo=ET).astimezone(timezone.utc)
               - datetime.fromisoformat('2027-03-14T01:45').replace(tzinfo=ET).astimezone(timezone.utc))
    assert elapsed == timedelta(minutes=20) and saved['minutes_to_pitch'] == 80
    assert saved['outcome'] == 'outside_window'
    print('spring wall-time limit: twenty elapsed minutes, eighty by the watchdog; no attempt')

    from bts.picks import save_pick
    fall = Replay(root / 'fall-window', day='2027-11-07')
    pick = fall.commit_at('01:55', at='00:50')
    pick.pick.game_time = '2027-11-07T01:55:00-05:00'
    save_pick(pick, fall.picks)
    result = fall.invoke('01:00')  # the first fold under the CLI pin
    assert result['receipt']['minutes_to_pitch'] == 55 and result['calls']
    print('fall wall-time limit: 115 elapsed minutes, 55 by the watchdog; account read')


def midnight(root):
    old_day, new_day = '2027-12-31', '2028-01-01'
    yesterday = Replay(root, day=old_day)
    pick = yesterday.commit_at('23:55', at='22:00')
    from bts.picks import save_pick
    pick.pick.game_time = f'{new_day}T00:35:00-05:00'
    save_pick(pick, yesterday.picks)
    yesterday.set_marker({'date': old_day, 'status': 'alerted'})
    today = Replay(root, day=new_day)
    for hm in ('00:00', '00:15', '00:30', '00:45'):
        result = today.invoke(hm)
        assert result['receipt']['et_date'] == new_day and result['receipt']['season'] == 2028
        assert result['receipt']['outcome'] == 'no_pick_file' and not result['calls']
    with_pick = Replay(root / 'with-today', day=new_day)
    with_pick.commit_at('00:50', at='00:01')
    result = with_pick.invoke('00:15')
    assert result['receipt']['outcome'] == 'observed' and 'pending' in result['calls']
    print('Dec 31 next-midnight pitch ignored on Jan 1; a Jan 1 pick was read as season 2028')


def failures(root):
    for kind in ('malformed', 'directory', 'fifo'):
        world = Replay(root / kind)
        path = world.picks / f'{DAY}.json'
        if kind == 'malformed':
            path.write_bytes(b'{bad JSON')
        elif kind == 'directory':
            path.mkdir()
        else:
            os.mkfifo(path)
        expected = {'malformed': 'JSONDecodeError', 'directory': 'IsADirectoryError', 'fifo': 'TimeoutError'}
        for hm, manual in (('05:00', False), ('05:15', True)):
            with bounded_fifo():
                result = world.invoke(hm, manual=manual)
            assert result['exception'] == expected[kind] and not result['calls'] and not result['marker_changed']
            assert result['receipt']['outcome'] == 'incomplete'
        print('pick input', kind, expected[kind], 'identical CLI behavior with and without --cron')

    world = Replay(root / 'retry')
    world.commit_at('06:10', at='04:50')
    interrupted = world.invoke('05:00', dm='interrupt')
    assert interrupted['exit'] != 0 and interrupted['attempts'] and not interrupted['marker_changed']
    assert interrupted['receipt']['detail'] == 'raised KeyboardInterrupt'
    restarted = Replay(root / 'retry')
    sent_without_confirmation = []

    def accepted_then_failed(recipient, message):
        sent_without_confirmation.append(message)
        raise RuntimeError('response lost after remote acceptance')

    failed = restarted.invoke('05:15', patches={'bts.dm.send_dm': accepted_then_failed})
    assert len(sent_without_confirmation) == 1 and failed['marker']['status'] == 'dm_failed'
    retry = Replay(root / 'retry').invoke('05:30')
    assert retry['marker']['status'] == 'alerted' and len(retry['sent']) == 1
    assert 'NOT entered' in sent_without_confirmation[0] and 'NOT entered' in retry['sent'][0]
    alerts = pick_entry.check(restarted.picks, datetime.fromisoformat(DAY).date(),
                             datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert [item.level for item in alerts] == ['WARN']
    print('interrupted send left no marker; uncertain send wrote dm_failed; restart retried and EOD retained WARN')


if __name__ == '__main__':
    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        time_changes(root)
        midnight(root / 'midnight')
        failures(root)
