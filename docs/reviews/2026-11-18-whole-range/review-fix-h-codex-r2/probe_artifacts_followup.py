from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

import numpy as np

from bts import orchestrator, strategy
from bts.simulate import mdp, tail_policy as tp
from probe_artifacts import DAY, assert_forced, check_errors, check_paired, files, paired, slate


def remaining(root, payload, shipped):
    # Identical bin rates give a strictly validated two-bin comparison file.
    rates = [0.75, 0.75]
    both = [0.5, 0.5]
    two_bin = replace(shipped, boundaries=[.82], bin_p_hit=rates, bin_p_both=both, bin_freq=[.5, .5],
                      policy_table=tp.solve_emax_season_best([.5, .5], rates, both).policy,
                      manifest=tp.tail_manifest(n_bins=2, hits=3, both=2, late_seed_days=4, bin_counts=[2, 2]))
    for label, change, validated in [
        ('two-bin', two_bin, True),
        ('invalid-count', replace(shipped, manifest=shipped.manifest | {'bin_counts': [None]}), False),
        ('invalid-rate', replace(shipped, bin_p_both=[2.0]), False),
        ('neither', None, False),
        ('oversize', None, False),
    ]:
        folder = root / label
        folder.mkdir()
        base, tail, _ = paired(folder, payload, shipped)
        if change:
            base_sha = sha256(base.read_bytes()).hexdigest()
            tp.save_tail_policy(replace(change, base_policy_sha256=base_sha), tail, validate=validated)
            if validated:
                assert tp.load_tail_policy(tail, expected_base_sha=base_sha).n_bins == 2
        elif label == 'neither':
            base.unlink()
            tail.unlink()
        with (files(base, tail),
              patch.object(tp, 'MAX_ARTIFACT_BYTES', 2 if label == 'oversize' else tp.MAX_ARTIFACT_BYTES),
              slate(folder / 'picks', DAY) as (cfg, _, _)):
            _, selected, _ = orchestrator.run_and_pick(cfg, str(DAY))
            assert_forced(selected.decision)
            assert strategy.effective_pick_bar(57, str(DAY), False) == 0.
            print(label, selected.decision.degraded_reason)


if __name__ == '__main__':
    with np.load(mdp.DEFAULT_POLICY_PATH) as data:
        fixture = dict(data)
    artifact = strategy._load_mdp()
    with TemporaryDirectory() as tmp:
        work = Path(tmp)
        if '--paired' in sys.argv:
            assert not check_paired(work, fixture, artifact['tail'])
        else:
            remaining(work, fixture, artifact['tail'])
            check_errors(work, artifact)
