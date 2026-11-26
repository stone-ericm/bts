# DR-MDP gap screen result

**Date**: 2026-05-06
**Predecessors**: [MDP objective audit](./2026-05-06-mdp-objective-audit.md), [DR-MDP gap measurement plan](./2026-05-06-dr-mdp-gap-measure.md)
**Artifact**: `data/validation/dr_mdp_gap_2021_2025.json`

## Question

The objective audit ruled out CVaR-over-streak as the wrong topology for BTS and identified one solver-side candidate worth measuring first: a distributionally robust MDP over the empirical bin manifold. The decision gate was deliberately conservative:

If a rectangular finite-candidate robust screen reports a max point-vs-robust P(57) gap smaller than the falsification-harness block-bootstrap CI half-width, then a production DR-MDP solver is not justified from that surface.

The gating bar is borrowed from the v2.6 harness CI half-width as a conservative same-order-of-magnitude check; this screen's estimand is the bin-point-estimate optimum, not the harness's LOSO-corrected pipeline value.

## Run

The intended 24-seed pivoted profile files (`data/simulation/profiles_seed*_season*.parquet`) are not present in this worktree. This run therefore uses the explicit canonical single-seed 2021-2025 profile surface that is tracked locally:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python scripts/dr_mdp_gap_measure.py \
  --profiles-glob 'data/simulation/backtest_202[1-5].parquet' \
  --season-length 153 \
  --n-bins 5 \
  --n-bootstrap-candidates 500 \
  --ci-half-width 0.08333333333333334 \
  --out data/validation/dr_mdp_gap_2021_2025.json \
  --pretty
```

The CI half-width is inherited from `data/validation/v2_6_n500_summary.json`: cell 111 has `ci_lower=0` and `ci_upper=0.16666666666666669`, so the half-width convention is `(0.16666666666666669 - 0) / 2 = 0.08333333333333334`.

## Result

| Surface | point P(57) | Robust construction | robust P(57) | gap | Exceeds CI half-width? |
| --- | ---: | --- | ---: | ---: | --- |
| 2021-2025 canonical profiles | 0.039960 | Wilson hit/both + simplex frequency | 0.004282 | 0.035678 | no |
| 2021-2025 canonical profiles | 0.039960 | paired-day bootstrap hit/both + multinomial frequency | 0.004799 | 0.035161 | no |

Additional artifact details:

- `n_profile_rows=9120`
- `n_pair_rows=912`
- five fixed bins, with `n` between 182 and 183 per bin
- max policy-disagreement rate across the two constructions: `0.132187`
- `max_delta_p57=0.035678`
- `max_delta_exceeds_ci_half_width=false`

## Interpretation

On this explicit single-seed 2021-2025 surface, the rectangular robust gap is below the harness CI half-width. Because the rectangular construction is pessimistic relative to a future jointly constrained empirical-process robust solver, this result does not motivate production DR-MDP solver work.

The screen is still sensitive at the policy level: 11.8-13.2% of nonterminal state-cells flip action under robust evaluation, but those states contribute negligibly to initial-state P(57) on this surface.

This does not close the stronger 24-seed question. The expected pivoted profile surface is absent locally, so a stronger result would require regenerating or locating those profiles and rerunning the same script. Until that exists, the better next BTS-winning track is bin-side and multi-seed pooling rather than solver-objective work.

## Decision

- Do not change the production MDP solver from this run.
- Keep `scripts/dr_mdp_gap_measure.py` as the reusable screen for future bin manifolds.
- Queue bin-side / multi-seed pooling as the next likely policy-improvement lever.
- If the 24-seed profile surface is regenerated, rerun this exact screen before reopening DR-MDP solver scoping.
