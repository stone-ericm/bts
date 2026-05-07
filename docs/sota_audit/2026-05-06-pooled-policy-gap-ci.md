# Pooled-Policy Gap Screen

- **Generated**: 2026-05-06
- **Artifact**: `data/validation/pooled_policy_gap_ci_2026-05-06.json`
- **Generator**: `scripts/pooled_policy_gap_ci.py`
- **Input**: `data/validation/pooled_policy_ab_24seed_consolidated.json`

## Verdict

The saved 24-seed pooled-policy A/B artifact survives an artifact-level paired-seed uncertainty check. The leave-one-out pooled policy beats the production policy on all 24 saved seed gaps.

This is an asymmetric falsification check, not deployment evidence. Because paired-seed resampling ignores shared day-level outcomes, its interval is narrower than a proper profile block-bootstrap. If this interval had crossed zero, the saved gap would be downgraded. Since it excludes zero, the positive screen remains standing, but profile-level uncertainty and provenance work are still required before any deployment claim.

## Results

| Comparison | Mean P(57) gap | 95% paired-seed bootstrap CI | Positive seeds | Exact sign-test p |
| --- | ---: | ---: | ---: | ---: |
| Within-pool | `+0.020816` | `[+0.016026, +0.025737]` | `24/24` | `1.19e-07` |
| Leave-one-out | `+0.019290` | `[+0.014468, +0.024308]` | `24/24` | `1.19e-07` |

The table reports two-sided sign-test p-values. The one-sided directional p-value for `24/24` positive gaps is `5.96e-08`. The leave-one-out gap range is `[+0.002351, +0.039286]`; no saved seed has a negative pooled-policy gap.

## Interpretation

The artifact-level seed bootstrap answers a narrower question than the v2.6 harness:

- It supports the claim that the saved 24 seed-level policy gaps pass a basic iid-seed falsification check.
- It does not address temporal dependence inside the raw profile surfaces.
- It does not remove the provenance caveat from `2026-05-06-pooled-seed-inventory.md`: the raw profile parquets do not embed determinism metadata, and seed identity is path-derived.
- It does not resolve the chronological replay caveat from `data/validation/pooled_policy_mc_replay_ab.json`: in 80 saved replay trajectories, production reached 57 once and pooled reached 57 zero times, despite pooled having higher mean replay max streak and higher MC P(57).

## Next Work

1. If this claim becomes deploy-relevant, implement the heavier profile-level bootstrap: resample day blocks, tag seed from path, recompute holdout bins, and rerun `evaluate_mdp_policy` per replicate.
2. Before any DR-MDP pooled-surface run, add seed tagging to `scripts/dr_mdp_gap_measure.py` or feed it a seed-tagged profile frame. The current raw parquet glob path is unsafe for multi-seed data.
3. Keep pooled prediction separate. The 2026-04-29 pooled-prediction Brier failure still blocks a production pooled-prediction cutover unless a new proper-scoring surface reverses it.
