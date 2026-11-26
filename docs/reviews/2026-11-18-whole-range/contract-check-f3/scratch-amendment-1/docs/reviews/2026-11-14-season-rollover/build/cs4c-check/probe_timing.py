"""Replay planned scheduler checks around the July 2026 break and timing controls."""

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import re

import probe_main  # Installs the network denial hook before application imports.

from bts import scheduler


ROOT = Path(__file__).resolve().parents[2]
ET = scheduler.ET
UTC = timezone.utc


def instant(day, clock):
    return datetime.fromisoformat(f'{day}T{clock}').replace(tzinfo=ET)


cron = (ROOT / 'scripts/cron-setup-hetzner.sh').read_text()
fetch_times = [f'{int(hour):02d}:{int(minute):02d}' for minute, hour in re.findall(
    r'^(\d+) (\d+) \* \* \* .* bts fetch-contest-streak ', cron, re.M)]
assert fetch_times == ['01:10', '02:10', '10:30', '13:30']


def evaluate(day, starts, *, same_pair=False):
    games = [{'gamePk': i, 'gameDate': instant(day, start).astimezone(UTC).isoformat(),
              'teams': {side: {'team': {'name': side if same_pair else f'{side}{i}'}}
                        for side in ['home', 'away']}}
             for i, start in enumerate(starts, start=1)]
    wake = scheduler.compute_wakeup_time(games, now_et=instant(day, '00:00'))
    checks = scheduler.plan_run_times(games, now=wake)
    slots = [instant(day + timedelta(days=offset), t) for offset in [-1, 0] for t in fetch_times]
    print(day, starts, 'wake', wake.strftime('%H:%M'),
          'DH2', scheduler.detect_doubleheader_game2s(games), 'checks', [
              (r['time_et'].strftime('%H:%M'),
               str(r['time_et'].astimezone(UTC) - max(
                   f.astimezone(UTC) for f in slots if f <= r['time_et']))) for r in checks])


# The 7/12 afternoon slate, 7/13–15 break, and 7/16 moved NYM–PHI first pitch
# are recorded in the eve-of-break audit and incident I-077. Other slot times
# below are chosen timing controls, not a recreation of every MLB game.
for number, starts in [(12, ('13:35', '14:10', '16:10')), (13, ()), (14, ()),
                       (15, ()), (16, ('18:10',)), (17, ('19:05', '22:10', '22:15')),
                       (18, ('13:05', '19:05'))]:
    evaluate(date(2026, 7, number), starts, same_pair=number == 18)
evaluate(date(2025, 3, 18), ('06:10',))  # Tokyo opener ET; first retained check at 05:25.
evaluate(date(2026, 8, 30), ('12:00', '13:20', '13:40', '13:55'))  # I-084 timing control.
day = date(2026, 7, 18)
evaluate(day, ('13:05',))  # Thin getaway slate; 10:05 and 11:05 precede regular 12:20.
prev = instant(day - timedelta(days=1), '13:30')
for clock in ['01:10', '02:10', '03:30', '03:31', '05:25', '10:05', '12:20', '21:25']:
    print('no overnight/AM write', clock, instant(day, clock) - prev)
for late in ['16:10', '16:11', '21:25', '23:50']:
    print('only 02:10 succeeded', late, instant(day, late) - instant(day, '02:10'))
for day in [date(2026, 3, 8), date(2026, 11, 1)]:
    a = instant(day - timedelta(days=1), '13:30')
    b = instant(day, '03:30')
    next_fetch = instant(day, '01:10')
    print('DST', day, 'elapsed 13:30–01:10 and 13:30–03:30',
          next_fetch.astimezone(UTC) - a.astimezone(UTC), b.astimezone(UTC) - a.astimezone(UTC),
          'wall subtraction', b - a)
