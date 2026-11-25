"""Give the alternate 240-day test horizon enough actual table rows."""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

import probe_parity


if __name__ == "__main__":
    if len(sys.argv) > 1:
        from bts import strategy

        policy = strategy._load_mdp()
        table = policy["policy_table"]
        assert table.shape[1] == 181
        extended = np.pad(table, ((0, 0), (0, 60), (0, 0), (0, 0)), mode="edge")
        with patch.object(strategy, "_load_mdp", return_value=policy | {"policy_table": extended}):
            probe_parity.worker(sys.argv[2], Path(sys.argv[3]), bool(int(sys.argv[4])))
    else:
        probe_parity.__file__ = __file__
        probe_parity.main()
