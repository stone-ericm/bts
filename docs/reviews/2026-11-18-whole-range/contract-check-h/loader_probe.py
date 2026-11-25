from datetime import date
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from bts import picks, strategy
from bts.health import tail_policy as health
from bts.simulate import mdp, tail_policy


container = Path(__file__).resolve().parent
case = sys.argv[1]
shipped_base = Path(mdp.DEFAULT_POLICY_PATH)
shipped_tail = Path(tail_policy.DEFAULT_TAIL_POLICY_PATH)
with TemporaryDirectory(dir=container) as temp:
    directory = Path(temp)
    base = directory / 'base.npz'
    tail = directory / 'tail.npz'
    if case != 'all-missing':
        shutil.copyfile(shipped_base, base)
        shutil.copyfile(shipped_tail, tail)
        target = base if case.startswith('base-') else tail
        issue = case.split('-', 1)[1]
        if issue == 'missing':
            target.unlink()
        elif issue == 'directory':
            target.unlink()
            target.mkdir()
        elif issue == 'fifo':
            target.unlink()
            os.mkfifo(target)
        elif issue == 'oversize':
            with target.open('ab') as f:
                f.truncate(tail_policy.MAX_ARTIFACT_BYTES + 1)
        elif issue == 'corrupt':
            target.write_bytes(target.read_bytes()[:70])
        elif issue == 'wrong-schema':
            np.savez(target, foo=[1])
        elif issue == 'shape':
            np.savez(target, policy_table=[0], boundaries=[0.8], season_length=180)
        elif issue == 'negative-horizon':
            np.savez(target, policy_table=[0], boundaries=[0.8], season_length=-1)
        elif issue == 'sha':
            with base.open('ab') as f:
                f.write(b'a harmless trailer')
    print('start', case, flush=True)
    if '--health' in sys.argv:
        print('health alerts', health.check(base_path=base, tail_path=tail, today=date(2027, 9, 10)))
    else:
        with patch.object(mdp, 'DEFAULT_POLICY_PATH', base), patch.object(tail_policy, 'DEFAULT_TAIL_POLICY_PATH', tail):
            loaded = strategy._load_mdp()
            print('loaded:', {k: None if v is None else str(v)[:160] for k, v in (loaded or {}).items() if k not in {'tail', 'policy_table'}})
            for day in ('2027-09-26', '2027-09-27'):
                for p in (.01, .99):
                    ctx = strategy.DecisionContext(p, .79, True, day, True, loaded, 57, 'trusted')
                    print(day, p, strategy.resolve_policy_decision(ctx, 57, False))
            print('provenance', picks.compute_provenance(policy_path=base, tail_path=tail))
