from dataclasses import asdict
from datetime import date, timedelta
from hashlib import sha256
import json
import logging
from pathlib import Path
import sys

import numpy as np

CHECK = Path(__file__).resolve().parent
mode = sys.argv[1]
sys.path.insert(0, str(CHECK / ('base' if mode == 'base' else 'scratch') / 'src'))
import bts
from bts import strategy


logging.disable(logging.CRITICAL)
loaded = strategy._load_mdp()
barriers = loaded['boundaries'] + loaded['tail'].boundaries
probs = [0.01, 0.99, 0.8]
for edge in barriers + [0.8]:
    probs.extend([np.nextafter(edge, 0.0), edge, np.nextafter(edge, 1.0)])
states = [None, loaded, loaded | {'tail': None, 'tail_error': 'synthetic absence'}]
digest = sha256()
cases = 0


def sample(mdp, streak, day, saver, p, raw_best, status, double, second):
    global cases
    ctx = strategy.DecisionContext(p, .75 if second else None, second, day, double, mdp, raw_best, status)
    out = strategy.resolve_policy_decision(ctx, streak, saver)
    bar = strategy.effective_pick_bar(streak, day, saver, mdp=mdp, best_streak=raw_best, best_status=status)
    digest.update(json.dumps([asdict(out), bar], sort_keys=True).encode())
    cases += 1


end = date(2027, 9, 27)
for mdp in states:
    for streak in range(57):
        for days in range(201):
            day = (end - timedelta(days=days)).isoformat()
            for saver in (False, True):
                i = (days + streak * 3 + saver) % len(probs)
                best_choices = ((None, None), (streak, 'trusted'), (57, 'trusted'), (57, 'untrusted'))
                sample(mdp, streak, day, saver, float(probs[i]), *best_choices[i % len(best_choices)],
                       bool(i % 2), bool(i % 3))
    for streak in (0, 10, 43, 55, 56):
        for days in (0, 1, 3, 7, 28, 29, 180, 181, 200):
            day = (end - timedelta(days=days)).isoformat()
            for p in probs:
                for best, status in ((None, None), (streak, 'trusted'), (57, 'trusted'), (57, 'untrusted')):
                    sample(mdp, streak, day, True, float(p), best, status, True, True)
print(bts.__file__, 'cases', cases, 'sha256', digest.hexdigest())
