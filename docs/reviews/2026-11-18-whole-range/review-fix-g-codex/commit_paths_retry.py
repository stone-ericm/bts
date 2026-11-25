"""Supply the full team fields to the scheduler-day fixture in commit_paths."""
from unittest.mock import patch
import sys

import commit_paths as probe

original = probe.setup


def setup(stack, *args, **kwargs):
    result = original(stack, *args, **kwargs)
    games = probe.scheduler.fetch_schedule(probe.DAY)
    for game in games:
        game['teams'] = {side: {'team': {'name': f'{side}-{game["gamePk"]}'}}
                         for side in ('home', 'away')}
    stack.enter_context(patch.object(probe.scheduler, 'fetch_schedule', lambda *_: games))
    return result


probe.setup = setup
side = sys.argv[1]
print(side, probe.bts.__file__)
probe.lifecycle(side)
probe.corrupt_record(side)
if side == 'head':
    probe.fresh_and_cached()
