from collections import Counter
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from bts.simulate import tail_policy as tail


def values(p, pb, days=180):
    gain = 0.0
    for day in range(1, days + 1):
        actions = dict(single=p * (1 + gain), double=pb * (2 + gain), skip=gain)
        gain = max(actions.values())
        ties = [action for action, value in actions.items() if value == gain]
        equal = [(a, b) for i, a in enumerate(actions) for b in list(actions)[i+1:] if actions[a] == actions[b]]
        yield day, ties[0], ties, equal


shipped = tail.load_tail_policy(expected_base_sha=tail.sha256_file(Path('data/models/mdp_policy.npz')))
print('shipped', shipped.bin_p_hit, shipped.bin_p_both, shipped.sha256)
cases = [(1, [1.0], [0.0], [0.0], 0, 0), (1, [1.0], [0.75], [0.0], 3, 0),
         (1, [1.0], [1.0], [0.75], 4, 3), (1, [1.0], [1.0], [1.0], 4, 4),
         (2, [0.5, 0.5], [0.0, 1.0], [0.0, 1.0], 2, 2)]
production = list(values(shipped.bin_p_hit[0], shipped.bin_p_both[0]))
print('shipped actions:', Counter(r[1] for r in production),
      'double dates:', [r[0] for r in production if r[1] == 'double'])
print('shipped equalities:', [(d, ties, equal) for d, _, ties, equal in production if equal])
with TemporaryDirectory() as folder:
    for n, freq, ph, pb, hits, both in cases:
        solution = tail.solve_emax_season_best(freq, ph, pb)
        artifact = replace(shipped, policy_table=solution.policy, boundaries=[0.8] if n == 2 else [],
                           bin_freq=freq, bin_p_hit=ph, bin_p_both=pb,
                           manifest=tail.tail_manifest(n_bins=n, hits=hits, both=both, late_seed_days=4,
                                                       bin_counts=[2, 2] if n == 2 else [4]))
        path = Path(folder) / 'test.npz'
        tail.save_tail_policy(artifact, path, expected_base_sha=shipped.base_policy_sha256)
        loaded = tail.load_tail_policy(path, expected_base_sha=shipped.base_policy_sha256)
        actions = [list(values(p, q)) for p, q in zip(loaded.bin_p_hit, loaded.bin_p_both)]
        print('valid artifact', n, ph, pb, [(Counter(r[1] for r in a), a[0][2]) for a in actions])

for n in range(1, 181):
    p, pb = shipped.bin_p_hit[0], shipped.bin_p_both[0]
    prev = 0.0
    for _ in range(n - 1):
        prev = max(prev, p * (1 + prev), pb * (2 + prev))
    if n <= 5 or n >= 177:
        print('day gains', n, tuple(repr(v) for v in (prev, p * (1 + prev), pb * (2 + prev))))
