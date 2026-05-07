# Pooled-Policy Gap Determinism Bound Screen

## Verdict

Existing artifacts do not provide a direct paired bound on nondeterminism inside the saved pooled-policy seed gaps.

The deterministic cutover evidence is still useful: the n=100 deterministic baseline shows no detectable distribution shift versus the prior non-deterministic baseline summary. But that is not the same estimand as the pooled-policy A/B gap, and the raw pooled-policy artifacts still do not embed deterministic/provider metadata.

The C0 pooled-policy screen therefore remains unchanged, not upgraded:

- C0 positive screen: still standing under iid seed-level resampling.
- Determinism caveat: still open.
- `iid_seed_assumption_verdict`: `not_evaluable_from_existing_artifacts`.

## Inputs

Artifact: `data/validation/pooled_gap_determinism_bound_2026-05-06.json`

Inputs:

- `data/validation/pooled_policy_gap_ci_2026-05-06.json`
- `data/validation/baseline_n100_deterministic_2026-04-27.json`
- `data/validation/screen_pooled_n10_2026-04-28.json`

## Numerical Screen

| Quantity | Value |
| --- | ---: |
| LOO mean pooled-policy gap | `+0.019290` |
| LOO seed-gap std | `0.012666` |
| LOO smallest positive gap | `+0.002351` |
| Within-pool mean gap | `+0.020816` |
| Within-pool seed-gap std | `0.012600` |
| Deterministic n=100 P(57) mean | `0.033647` |
| Prior non-deterministic P(57) mean | `0.033650` |
| P(57) distribution mean delta | `-0.000003` |
| P(57) distribution z vs prior std | `-0.000204` |
| Deterministic n=100 P(57) seed std | `0.016061` |

The deterministic n=100 baseline does not show a mean shift from the prior non-deterministic distribution. The saved LOO gap std is `0.7886x` the deterministic n=100 P(57) seed std.

## Deterministic-Only Proxy

The post-cutover deterministic feature screen has 10 seeds across 32 feature experiments. Its per-experiment deterministic `delta_p_57_mdp` seed std has:

- median `0.015771`
- max `0.034543`
- `21/32` experiments with std at least the C0 LOO gap std `0.012666`

This is not a pooled-policy A/B bound. It shows that substantial seed variation remains under deterministic training, so the observed C0 seed-gap variance cannot be interpreted as provider/model nondeterminism just because it exists. As a soft signal, C0's gap variance is not anomalously high relative to deterministic-only seed variation.

## What This Does Not Prove

- It does not decompose `sigma_gap^2` into seed-bootstrap variance and nondeterminism variance.
- It does not certify the C0 iid-seed assumption.
- It does not resolve raw-surface provenance, because the pooled raw parquets still rely on path-derived seed identity and do not embed deterministic flags.
- It does not justify any production policy or solver change.

## Required Evidence For A Direct Bound

A direct bound needs a paired same-seed deterministic versus non-deterministic rerun on the same pooled-policy A/B estimand, or embedded deterministic/provider metadata that lets the existing raw surfaces be partitioned by determinism state.

Until then, the correct status is: distribution shift not detected, direct bound missing, C0 screen unchanged.

## Verification

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/determinism_gap_bound.py \
  --out data/validation/pooled_gap_determinism_bound_2026-05-06.json
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/scripts/test_determinism_gap_bound.py
```
