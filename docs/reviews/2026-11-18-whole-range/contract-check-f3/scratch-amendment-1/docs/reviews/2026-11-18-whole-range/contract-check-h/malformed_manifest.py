from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from bts import strategy
from bts.health import tail_policy as health
from bts.simulate import mdp, tail_policy


with TemporaryDirectory() as folder, np.load(tail_policy.DEFAULT_TAIL_POLICY_PATH) as source:
    destination = Path(folder) / 'tail.npz'
    fields = {name: source[name] for name in source.files}
    manifest = json.loads(str(fields['manifest']))
    manifest['bin_counts'] = [None]
    fields['manifest'] = np.array(json.dumps(manifest))
    np.savez_compressed(destination, **fields)
    with patch.object(tail_policy, 'DEFAULT_TAIL_POLICY_PATH', destination):
        calls = {
            'tail loader': lambda: tail_policy.load_tail_policy(destination),
            'strategy loader': strategy._load_mdp,
            'effective pick bar': lambda: strategy.effective_pick_bar(57, '2027-07-15', False),
            'health': lambda: health.check(base_path=mdp.DEFAULT_POLICY_PATH, tail_path=destination,
                                           today=date(2027, 9, 10)),
        }
        for name, call in calls.items():
            try:
                print(name, call())
            except Exception as exc:
                print(name, type(exc).__name__, str(exc))
