from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import probe_parity


def replace_with_solved_policy(artifact, **fields):
    if 'bin_freq' in fields:
        from bts.simulate import tail_policy as tp

        fields['policy_table'] = tp.solve_emax_season_best(
            fields['bin_freq'], fields['bin_p_hit'], fields['bin_p_both']).policy
    return replace(artifact, **fields)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        probe_parity.__file__ = __file__
        probe_parity.main()
    else:
        with patch.object(probe_parity, 'replace', replace_with_solved_policy):
            probe_parity.worker(Path(sys.argv[1]), Path(sys.argv[2]), bool(int(sys.argv[3])))
