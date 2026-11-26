# DR-MDP Gap Screen on 24-Seed Pooled Raw Surface

## Verdict

The 24-seed raw pooled-bin surface is now measurable by the finite-candidate DR-MDP screen after deriving `seed` from each parquet path. This closes the main ingestion blocker identified in the pooled-seed inventory.

The measured robust gap is larger than the single canonical-surface screen, but it remains inside the inherited v2.6 uncertainty width. This is evidence for continued monitoring and stronger profile-level uncertainty work, not a production solver change.

## Input Surface

- Source roots: `data/hetzner_results/pooled_bins_run` and `data/hetzner_results/pooled_bins_run_trackc`
- Matched parquet files: `120`
- Path-derived seeds: `24`
- Profile rows: `218750`
- Rank-1/rank-2 pair rows: `21888`
- Seed identity: parsed from `simulation_seedN` path segments

The raw parquet payloads still do not embed determinism metadata. Seed identity is therefore path-derived and should be treated as provenance-sensitive.

## Result

Artifact: `data/validation/dr_mdp_gap_pooled_24seed_raw_2026-05-06.json`

| Construction | Point P(57) | Robust P(57) | Delta | Exceeds 8.333pp CI width? | Policy disagreement |
| --- | ---: | ---: | ---: | --- | ---: |
| Wilson simplex | 0.052882 | 0.034458 | 0.018424 | no | 0.016581 |
| Paired-day bootstrap multinomial | 0.052882 | 0.009665 | 0.043218 | no | 0.105240 |

This run used the script default `250` bootstrap candidates. The earlier canonical-surface DR-MDP evidence run used `500`, so compare the bootstrap rows as same-method screens with different replicate counts, not as exact matched-replicate estimates.

## Interpretation

- The 24-seed raw surface produces a higher point P(57) than the explicit 2021-2025 canonical surface measured earlier.
- The paired-day bootstrap construction is the binding finite-candidate ambiguity screen, with max delta `0.043218`.
- The gap does not exceed the inherited v2.6 half-width `0.083333`, so the current production recommendation remains unchanged.
- This does not resolve the pooled-policy deployment question. It measures solver-side robustness on the pooled raw surface; it does not provide a profile block-bootstrap over the pooled-policy A/B gap.

## Verification

Command:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/dr_mdp_gap_measure.py \
  --profiles-glob 'data/hetzner_results/pooled_bins_run/*/simulation_seed*/backtest_*.parquet' \
  --profiles-glob 'data/hetzner_results/pooled_bins_run_trackc/*/simulation_seed*/backtest_*.parquet' \
  --derive-seed-from-path \
  --ci-half-width 0.083333 \
  --out data/validation/dr_mdp_gap_pooled_24seed_raw_2026-05-06.json \
  --pretty
```

Local validation also covered the loader path:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/scripts/test_dr_mdp_gap_measure.py tests/simulate/test_pooled_policy.py
```
