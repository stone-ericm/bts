from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from bts.simulate import mdp, tail_policy as tail


def steps(p, pb):
    gain = 0.0
    for day in range(1, 181):
        values = (float(p) * (1.0 + gain), float(pb) * (2.0 + gain), gain)
        gain = max(values)
        yield day, ('single', 'double', 'skip')[values.index(gain)], values


def show(p, pb):
    data = list(steps(p, pb))
    ties = [(i + 1, j + 1, [day for day, _, v in data if v[i] == v[j]])
            for i, j in ((0, 1), (0, 2), (1, 2))]
    print(p, pb, Counter(action for _, action, _ in data), 'equalities',
          [(a, b, days[:1], days[-1:], len(days)) for a, b, days in ties if days])
    assert all(action != 'skip' for _, action, _ in data)


base_sha = tail.sha256_file(mdp.DEFAULT_POLICY_PATH)
shipped = tail.load_tail_policy(expected_base_sha=base_sha)
print('artifact SHA', shipped.sha256)
p, pb = shipped.bin_p_hit[0], shipped.bin_p_both[0]
show(p, pb)
assert [action for _, action, _ in steps(p, pb)] == ['double'] * 3 + ['single'] * 177
with TemporaryDirectory() as folder:
    for ph, joint, hits, both in (([0.], [0.], 0, 0), ([.75], [0.], 3, 0),
                                  ([1.], [.75], 4, 3), ([1.], [1.], 4, 4),
                                  ([.75, .75], [.5, .5], 3, 2)):
        n = len(ph)
        freq = [1 / n] * n
        table = tail.solve_emax_season_best(freq, ph, joint).policy
        model = replace(shipped, policy_table=table, boundaries=[.8] if n == 2 else [],
                        bin_freq=freq, bin_p_hit=ph, bin_p_both=joint,
                        manifest=tail.tail_manifest(n_bins=n, hits=hits, both=both,
                                                    late_seed_days=4, bin_counts=[4 // n] * n))
        path = Path(folder) / 'edge.npz'
        tail.save_tail_policy(model, path)
        artifact = tail.load_tail_policy(path, expected_base_sha=base_sha)
        print('valid loaded bin count', artifact.n_bins)
        if n == 1:
            show(artifact.bin_p_hit[0], artifact.bin_p_both[0])
