from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import signal
import sys

import numpy as np


here = Path(__file__).resolve().parent
sys.path[:0] = [str(here / 'base' / 'src'), str(here / 'scratch')]
os.chdir(here / 'scratch')
from bts import strategy
from bts.health import tail_policy as health
from bts.simulate import mdp, tail_policy
from scripts.audit.season_walk import walk


class Deadline(BaseException):
    pass


def deadline(signum, frame):
    raise Deadline('read still blocked')


day = date(2027, 7, 15)
frame = walk._predictions(day, .96)
statuses = {int(pk): {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame['game_pk']}
base, tail = Path(mdp.DEFAULT_POLICY_PATH), Path(tail_policy.DEFAULT_TAIL_POLICY_PATH)
loaded = strategy._load_mdp()
with TemporaryDirectory() as folder:
    root = Path(folder)
    regular_bad = root / 'tail-bad.npz'
    with np.load(tail) as source:
        contents = dict(source)
    contents['manifest'] = np.array(json.dumps(json.loads(str(contents['manifest'])) | {'bin_counts': [None]}))
    np.savez_compressed(regular_bad, **contents)
    pipe = root / 'fifo'
    os.mkfifo(pipe)
    for base_path, tail_path in ((base, regular_bad), (pipe, tail), (base, pipe)):
        with patch.object(mdp, 'DEFAULT_POLICY_PATH', base_path), \
             patch.object(tail_policy, 'DEFAULT_TAIL_POLICY_PATH', tail_path), \
             patch.object(strategy, '_mdp_cache', None):
            calls = {'select': lambda: strategy.select_pick(frame, str(day), root, streak=57,
                                                               saver_available=False, game_statuses_detailed=statuses),
                     'health': lambda: health.check(base_path=base_path, tail_path=tail_path, today=day)}
            for label, call in calls.items():
                previous = signal.signal(signal.SIGALRM, deadline)
                signal.setitimer(signal.ITIMER_REAL, .2)
                try:
                    print(base_path.name, tail_path.name, label, call())
                except (Exception, Deadline) as exc:
                    print(base_path.name, tail_path.name, label, type(exc).__name__, exc)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, previous)

for kind in (KeyboardInterrupt, SystemExit, ValueError):
    with patch.object(strategy, 'lookup_tail_action', side_effect=kind('test')):
        try:
            print('existing tail exception', kind.__name__, strategy._tail_action(loaded, .96, 5, 10, False, 5))
        except (KeyboardInterrupt, SystemExit) as exc:
            print('existing tail propagated', type(exc).__name__)
