import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from timeline import H, at, games, run
from bts import scheduler
from tests.test_pick_chosen_on_cs5a import account_file


DAY = '2027-07-15'


def ready(root):
    (root / 'models').mkdir(parents=True)
    (root / 'models' / 'blend_2027-07-16.pkl').touch()
    for name in ('mdp_policy.npz', 'mdp_tail_policy.npz'):
        shutil.copyfile(Path('data/models') / name, root / 'models' / name)
    account_file(root, 'streak.json', {'season': 2027, 'streak': 0, 'saver_available': True})
    shutil.move(root / 'account_state' / 'streak.json', root / 'picks' / 'streak.json')


with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
    base = Path(tmp)
    for seed in ('dated', 'fresh', None, 'cached'):
        root = base / str(seed)
        (root / 'picks').mkdir(parents=True)
        ready(root)
        _, _, dms = run(root, f'operational files present: {seed}', DAY,
                        starts=('19:10', '19:40'), seed=seed)
        health_dms = [body for channel, body in dms if channel == 'health']
        keys = [part for body in health_dms for part in ('contest_state', 'missed_pick')
                if f'[{part}]' in body]
        print('control DMs:', seed, [channel for channel, _ in dms], keys)
        if seed in ('dated', 'fresh'):
            assert [channel for channel, _ in dms] == ['pick']
            assert not (root / 'health_state' / H).exists()
        if seed in (None, 'cached'):
            assert keys == (['contest_state'] if seed is None else ['contest_state', 'missed_pick'])

    for label, start, starts in [
        ('night', '10:00:30', ('19:10', '19:40', '22:10')),
        ('getaway', '10:00:30', ('13:05', '13:10', '13:15')),
        ('thin', '10:00:30', ('13:05',)), ('early', '05:10:30', ('06:10',)),
        ('opening', '10:00:30', ('19:10',)),
    ]:
        slate = games('2027-03-24', starts)
        now = at('2027-03-24', start)
        planned = scheduler.plan_run_times(slate, offset_min=60, now=now)
        print(label, 'production offset=60', [(r['time_et'].strftime('%H:%M'),
              int((r['time_et'] - now).total_seconds())) for r in planned])

    evening = at('2027-03-23', '23:50')
    with patch.object(scheduler, '_now_et', lambda: evening), \
            patch.object(scheduler, 'fetch_schedule', lambda _: []):
        assert scheduler._next_day_wakeup('2027-03-23', {}) == at('2027-03-24', '10:00')
    later = at('2027-03-24', '05:11')
    with patch.object(scheduler, '_now_et', lambda: later), \
            patch.object(scheduler, 'fetch_schedule', lambda day: games(day, ('06:10',))):
        assert scheduler._next_day_wakeup('2027-03-23', {}) == at('2027-03-24', '05:12')
    print('empty lookahead: next day 10:00; late lookahead with real games: now + 1 minute')
