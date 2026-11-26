# DR-MDP gap measurement plan

**Date**: 2026-05-06
**Branch**: `feature/dr-mdp-gap-measure`
**Predecessor**: [MDP objective audit (2026-05-06)](./2026-05-06-mdp-objective-audit.md)
**Scope**: measurement harness only. No production solver, strategy, or deploy change.

## Why this exists

The MDP objective audit concluded that `solve_mdp` already optimizes the BTS win metric: reachability probability P(57) under the supplied bin manifold. The remaining solver-side question is narrower:

> Are plausible uncertainty sets around `(p_hit, p_both, freq)` large enough to change the point-estimate optimum by more than the validation uncertainty we already tolerate?

`scripts/dr_mdp_gap_measure.py` answers that as a screen, not as a new policy. If the measured point-vs-robust gap is small relative to the falsification-harness block-bootstrap CI half-width, a production DR-MDP is likely not worth its complexity. If the gap is large under a defensible construction, that motivates a separate scoping PR.

## Implemented measurement

The script:

1. Loads profile parquet files from one or more paths/globs.
2. Normalizes either supported profile schema:
   - direct: `top1_p, top1_hit, top2_hit`
   - ranked: `date, rank, p_game_hit, actual_hit`, with optional `season` and `seed` included in the rank-1/rank-2 join key.
3. Computes fixed-count quality bins from top-1 confidence.
4. Solves the current point-estimate reachability MDP.
5. Builds finite ambiguity candidates for two constructions:
   - `wilson_simplex`: per-bin Wilson intervals for `U_hit`; Wilson cell bounds projected into simplex-respecting candidates for `U_freq`.
   - `paired_day_bootstrap_multinomial`: paired-day bootstrap quantiles for `U_hit`; bootstrap frequency quantiles projected into simplex-respecting candidates for `U_freq`.
6. Runs a finite-candidate rectangular robust DP.
7. Emits JSON with point P(57), robust P(57), delta, max delta, policy-disagreement rate, bin stats, candidate counts, and an optional `--ci-half-width` gate.

The robust DP is exact for the finite candidate grid:

```
V_robust(s,d,saver,q) =
  max_a min_{hit candidate for q} min_{freq candidate}
    E[V_robust(next state, d-1, next saver, q')]
```

This is intentionally not a continuous ambiguity-set optimizer. The JSON includes an explicit note that robust values are finite-grid measurements only.

## Acceptance bar

Use this script to produce an evidence table, not a deploy decision. The intended decision rule is:

- If every construction reports `delta_p57 <= ci_half_width`, close the solver-side DR-MDP track for now and redirect to bin-side work.
- If any construction reports `delta_p57 > ci_half_width`, inspect the construction for statistical defensibility, then open a follow-up design PR before any production solver change.

`ci_half_width` should come from the current falsification-harness block-bootstrap uncertainty for the same evaluation surface. Do not invent a new threshold inside this script.

## Known limitations

- The frequency candidates are deterministic simplex projections from marginal lower/upper bounds. They are a screen, not a full Dirichlet credible region or continuous multinomial uncertainty set.
- The bootstrap construction resamples profile-pair rows by `(season, date)` when those columns exist, preserving all seeds on the sampled day. If only `date` exists, the date is the block. If no date column exists, it falls back to row bootstrap.
- The robust solve is rectangular: hit-parameter and frequency candidates can be chosen independently at each state. This is conservative and may overstate pessimism relative to a jointly constrained empirical process, so the rectangular `max_delta_p57` is best read as an upper-bound screen for a future joint-constrained robust solver.
- Empty bins are retained for shape compatibility. If a profile surface creates empty bins through tied quantile boundaries, treat that output as a binning-quality warning before interpreting the robust gap.
- The default profile glob may pick up local scratch parquets. For a publishable measurement, pass explicit profile paths and record them in the output artifact.

## Verification

Local verification for this PR:

```
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/scripts/test_dr_mdp_gap_measure.py -q
UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/scripts/test_dr_mdp_gap_measure.py tests/scripts/test_run_falsification_harness.py tests/simulate -q
```

The test suite covers singleton equivalence to `solve_mdp`, robust monotonicity under lower hit candidates, simplex preservation, ranked-schema joins without cross-seed leakage, JSON output, and requested bin-count retention.

## Next steps

1. Run the script on the intended locked profile surface with an explicit profile path list.
2. Compare `max_delta_p57` to the block-bootstrap CI half-width from the relevant harness run.
3. If the gap clears the bar, add a follow-up memo that names the exact ambiguity construction to productionize and why it is statistically defensible.
4. If it does not clear the bar, queue bin-side work next: multi-seed bin pooling, cross-fitted calibration, or drift-aware re-binning.
