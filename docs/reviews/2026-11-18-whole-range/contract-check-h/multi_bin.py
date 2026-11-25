from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from bts.simulate import tail_policy as tail


artifact = tail.load_tail_policy(expected_base_sha=tail.sha256_file(Path('data/models/mdp_policy.npz')))
for ph, pb in (([.7146, .8097], [.5403, .5944]), ([.75, .75], [.5, .5]),
               ([.75, .875], [.5, .5]), ([.75, .875], [.5, .625])):
    freq = [.5, .5]
    policy = tail.solve_emax_season_best(freq, ph, pb).policy
    pair = replace(artifact, bin_freq=freq, bin_p_hit=ph, bin_p_both=pb, boundaries=[.82], policy_table=policy,
                   manifest=tail.tail_manifest(n_bins=2, hits=0, both=0, late_seed_days=4, bin_counts=[2, 2]))
    try:
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'tail.npz'
            tail.save_tail_policy(pair, path)
            observed = tail.load_tail_policy(path, expected_base_sha=artifact.base_policy_sha256)
            print('loads', observed.n_bins, ph, pb)
    except tail.TailPolicyError as exc:
        print('refused', ph, pb, exc)
