"""Confirm the deep object on real scorer paths and check what the morning command sends."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / '.codex-review/scratch-f4/src'),
                str(ROOT / 'docs/reviews/2026-11-18-whole-range/closing-test/codex-blind')]

from bts import picks, scheduler
import offline
from offline import World, at
from opening import DAY2, OPEN

offline.ROOT = ROOT

with TemporaryDirectory() as tmp, World(OPEN, Path(tmp)) as world:
    path = world.directory / 'streak.json'
    content = '{"ignored":' + '[' * 12000 + '0' + ']' * 12000 + '}'
    path.write_text(content)
    primary = picks.pick_from_row(world.frame().iloc[0])
    day = picks.DailyPick(OPEN, world.now.isoformat(), primary, None, None, notification_sent=True)
    picks.save_pick(day, world.directory)
    world.now = at(OPEN, '22:05')
    try:
        scheduler.run_result_polling(101, OPEN, world.directory)
    except RecursionError:
        print('daemon: RecursionError')
    else:
        raise AssertionError('daemon accepted over-deep object')
    for hour, args in (('01:00', ('check-results', '--date', OPEN, '--wait-deadline-et', '06:00')),
                       ('02:00', ('reconcile',))):
        world.now = at(DAY2, hour)
        result = world.invoke(*args)
        assert isinstance(result.exception, RecursionError) and path.read_text() == content
        assert picks.load_pick(OPEN, world.directory).result is None
        print(hour, args[0], result.exit_code, type(result.exception).__name__, result.output.strip())
    world.now = at(DAY2, '08:30')
    result = world.outcome(OPEN)
    assert isinstance(result.exception, RecursionError) and not result.stdout
    assert not any(event[1] == 'send' for event in world.events)
    print('08:30: no output or DM; exception', type(result.exception).__name__)
    day.result = 'hit'
    picks.save_pick(day, world.directory)
    world.now = at(DAY2, '07:40')
    result = world.invoke('reconcile', expect=0)
    assert picks.load_streak(world.directory, season=2027) == 1
    print('complete replay:', result.output.strip())
