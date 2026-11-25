from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import os
import sys


here = Path(__file__).resolve().parent
sys.path[:0] = [str(here / 'scratch' / 'src'), str(here / 'scratch')]
os.chdir(here / 'scratch')
from bts import orchestrator, picks
from scripts.audit.season_walk import walk


real = orchestrator.run_and_pick
valid = []


def run(*args, **kwargs):
    output = real(*args, **kwargs)
    sel = output[1]
    if sel.pick_result is not None:
        daily = sel.pick_result.daily
        assert picks._valid_chosen_on(daily.chosen_on) is not None
        assert daily.chosen_on['required'] and daily.chosen_on['dated']['effective_date'] == daily.date
        assert (daily.double_down is not None) == (sel.action == 'double')
        valid.append(daily.date)
    return output


with patch.object(orchestrator, 'run_and_pick', side_effect=run):
    walk.walk(states=(walk.State('dated', 57, 57, 'used', .96),))
print('valid chosen_on and matching executed slot counts', len(valid))
with TemporaryDirectory() as folder:
    output = Path(folder) / 'walk.json'
    status = walk.main(['--start', '2027-03-24', '--end', '2027-03-24', '--out', str(output)])
    print('walk CLI exit with existing findings', status)
    assert status == 1
