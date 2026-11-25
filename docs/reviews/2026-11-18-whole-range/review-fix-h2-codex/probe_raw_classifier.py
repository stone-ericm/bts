import argparse
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import bts
from bts import contest_state as cs
from bts.health import contest_state as health
from probe_bounds import GRID, outcome
from support import ABSENT, DAY, at, install, live, walk


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--base', type=Path)
    args = parser.parse_args()
    limit = getattr(cs, 'MAX_SEASON_STREAK', 57)
    print(bts.__file__)
    assert cs.TARGET_STREAK == 57
    with TemporaryDirectory() as tmp, walk.offline(at()):
        directory = Path(tmp) / 'picks'
        install(directory, 'automatic', 0, 0)
        path = directory / 'account_state/contest_streak.json'
        sample = cs._parse_state_file(path)
        observed = []
        for value in GRID:
            put = None if value is ABSENT else value
            state = replace(sample, best_streak=put)
            observed.append(outcome(lambda: cs.classify_best_streak(state, now=at(), now_year=DAY.year), directory))
        args.out.write_text(json.dumps(observed, indent=1))
        if args.base:
            old = json.loads(args.base.read_text())
            for before, after in zip(old, observed, strict=True):
                if before != after:
                    assert before[0] == after[0] == 'return'
                    value = before[1][0]
                    assert 58 <= value <= 400 and before[1][1] == 'untrusted' and after[1] == [value, 'trusted']
        snapshot = json.loads(path.read_text())
        del snapshot['active_streak']
        snapshot.update(streak=limit, best_streak=limit)
        path.write_text(json.dumps(snapshot))
        assert live(directory).streak == limit
        assert not [a for a in health.check(directory, now=at(), expected=True, historical_day=DAY) if a.level == 'CRITICAL']
        path.write_text(json.dumps(dict(snapshot, active_streak=None)))
        assert outcome(lambda: live(directory), directory)[1] == 'ContestStateError'
        print(len(observed), 'raw classifier inputs, target is 57, active alias reads at', limit,
              'explicit null active is refused even with alias present')
