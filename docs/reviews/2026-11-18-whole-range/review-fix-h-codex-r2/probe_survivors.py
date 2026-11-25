from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np

from probe_mutations import MUTATIONS, REVIEW, SOURCE, change, child


class TableLike:
    ndim = 4
    dtype = np.dtype('int8')
    shape = (58, 181, 2, 5)

    def min(self):
        return 0

    def max(self):
        return 2


def inspect(which, mutated):
    from probe_artifacts import assert_forced, choose
    from bts import strategy

    policy = strategy._load_mdp()
    table = policy['policy_table']
    if which == 'drop-positive-base-horizon':
        policy |= {'season_length': 0, 'policy_table': table[:, :1]}
    elif which == 'drop-list-type':
        policy['boundaries'] = tuple(policy['boundaries'])
    elif which == 'drop-int-horizon-type':
        policy['season_length'] = 180.0
    elif which == 'drop-array-type':
        policy['policy_table'] = TableLike()
    else:
        raise AssertionError(which)
    decision = choose(policy)
    if mutated:
        assert decision.action == 'double' and decision.tail_sha256 and not decision.degraded_reason
    else:
        assert_forced(decision)
    print(which, 'mutated' if mutated else 'reviewed', decision.action, decision.degraded_reason)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        inspect(sys.argv[1], sys.argv[2] == 'after')
    else:
        with TemporaryDirectory() as tmp:
            src = Path(tmp).resolve() / 'src'
            shutil.copytree(SOURCE, src, ignore=shutil.ignore_patterns('__pycache__'))
            for name in ('drop-positive-base-horizon', 'drop-list-type', 'drop-int-horizon-type', 'drop-array-type'):
                path = change(src, *MUTATIONS[name])
                for before, source in [('before', SOURCE), ('after', src)]:
                    name_log = f'mutation-example-{name}-{before}.log'
                    assert child(source, [__file__, name, before], name_log) == 0
                    print((REVIEW / name_log).read_text(), end='')
                shutil.copyfile(SOURCE / path.relative_to(src), path)
