# Leaderboard mechanism-mining pre-registration

**Date**: 2026-05-10 ET
**Scope**: research-only mechanism mining on captured public BTS leaderboard
behavior versus locked production picks and ranked model surfaces.
**Status**: pre-registered analysis plan; no production policy, model,
threshold, feature, scheduler, or dashboard edit is supported by this memo.

This memo must remain upstream of the first disagreement-unit artifact run. It
locks the unit definition, ordered decomposition variables, primary estimand,
FDR rule, and mechanism threshold before historical subgroup outcomes are
inspected.

## Question

The prior leaderboard audits showed that fixed-cohort public consensus
outperformed our historical locked picks on overlapping resolved date-slot
units. The next question is not whether to copy the leaderboard. The question
is whether leaderboard consensus exposes a reproducible mechanism that our
model or decision protocol is missing.

The mechanism-mining lane asks:

1. Are fixed-cohort consensus picks usually inside the model's top candidate
   pool, or are they off-surface misses?
2. When consensus disagrees with our locked production pick, which pre-locked
   batter/game contexts explain the disagreement?
3. Do any such contexts survive multiplicity control strongly enough to justify
   a fresh prospective candidate, rather than a post-hoc story?

## Comparison Units

Primary unit: one resolved `(pick_date, pick_number)` date-slot.

Locked production pick: the realized production row for that date and slot
from the canonical realized-picks parquet. Slot mapping is `primary -> 1` and
`double_down -> 2`.

Leaderboard consensus pick: the modal captured public pick for the same date
and slot within the fixed cohort from the latest `active_streak` snapshot,
unless an explicit cohort JSON is supplied before running the artifact.

Void, pending, null, or otherwise unresolved production slots are preserved in
the unit-level artifact but excluded from hit-rate, delta, bootstrap, and FDR
denominators. They must not be coerced to misses.

## Primary Estimand

The primary estimand is fixed-cohort consensus top-N coverage by ranked model
surfaces:

- `N = 1, 2, 5, 10`
- unit: resolved `(pick_date, pick_number)`
- read: whether the fixed-cohort consensus batter appears in the model surface
  top-N for that date
- primary surface class: at-lock or manifest-proven ranked surfaces only

For realized-production surfaces, which contain only the locked slots rather
than a full ranking, the primary read degrades to same-slot agreement:
`consensus_batter_id == locked_production_batter_id`. That is a production
decision diagnostic, not a top-N model coverage estimate.

## Secondary Estimands

Secondary read 1: on resolved same-slot disagreement units,
`consensus_hit - production_hit`.

Secondary read 2: conditional miscalibration of model probabilities for
leaderboard-consensus batters after they are joined to ranked surfaces. This is
descriptive unless the surface provenance is at-lock or otherwise
manifest-proven.

Secondary read 3: consensus concentration. High consensus share may indicate a
publicly obvious batter class or a stale blind spot in our candidate universe,
but it is not a success metric by itself.

## Locked Decomposition Variable Order

The script output schema and any subgroup table must use this exact ordered
variable list:

1. `cohort`
2. `pick_number`
3. `consensus_pick_share_bin`
4. `production_p_game_hit_bin`
5. `agreement_state`
6. `production_batter_skill_quartile`
7. `production_batter_skill_prior_pa_bin`
8. `production_projected_lineup`
9. `production_regime`
10. `production_is_park_driven`
11. `production_is_indoor`
12. `production_weather_temp_bin`
13. `consensus_model_rank_bin`
14. `consensus_model_probability_bin`

Variables 1-12 are available from the realized-production surface and
leaderboard consensus artifact when the canonical realized picks include the
existing context columns. Variables 13-14 require a ranked model surface and
must remain null when no at-lock or manifest-proven ranked surface is supplied.

Bin definitions:

| Variable | Bins |
|---|---|
| `consensus_pick_share_bin` | `<0.15`, `0.15-0.25`, `>=0.25` |
| `production_p_game_hit_bin` | `<0.68`, `0.68-0.74`, `0.74-0.80`, `>=0.80` |
| `production_batter_skill_prior_pa_bin` | `<100`, `100-299`, `300-599`, `>=600` |
| `production_weather_temp_bin` | `indoor_or_missing`, `<60`, `60-74`, `75-84`, `>=85` |
| `consensus_model_rank_bin` | `rank1`, `rank2`, `rank3_5`, `rank6_10`, `off_top10`, `missing_surface` |
| `consensus_model_probability_bin` | `<0.68`, `0.68-0.74`, `0.74-0.80`, `>=0.80`, `missing_surface` |

## FDR Rule

The tested mechanism family is the cross-product of the locked decomposition
variables after empty and underpowered cells are removed. A cell is testable
only when it has at least `15` resolved disagreement units and both comparison
outcomes have non-null hit indicators.

For each testable cell, compute a one-sided positive test for
`mean(consensus_hit - production_hit) > 0`. The first implementation may use a
day-block bootstrap p-value with expected block length `7`, `2000` bootstrap
replicates, and seed `20260510`, or an exact paired sign/randomization test if
cell counts are too small for stable bootstrap interpretation. The chosen test
must be recorded per row.

Apply BH and BY q-values across the complete testable cell family. A historical
mechanism candidate may be named only if it survives BH at `q <= 0.10`; BY must
be reported as the arbitrary-dependence sensitivity check. A BY failure means
the mechanism remains BH-only exploratory, not a robust claim.

True online FDR or e-BH is not claimed here. Valid e-values or e-processes
would require a separate pre-registration amendment before prospective use.

No interim inspection of cell-level deltas, p-values, or hit rates is
permitted between artifact generation and pre-registered analysis completion.
Re-running the script with additional dates is exploration, not a significance
test, unless that rerun rule is separately pre-registered before outcomes are
inspected.

## Mechanism-Found Threshold

Historical mechanism mining can nominate, but not deploy, a candidate rule only
when all of the following hold:

1. At least `30` resolved disagreement units in the candidate mechanism cell.
2. Fixed-cohort consensus hit rate beats locked production hit rate by at least
   `0.05` absolute on those units.
3. The cell survives BH at `q <= 0.10` under the pre-registered FDR family.
4. The effect direction is not contradicted by the all-tracked cohort on the
   same cell.
5. The mechanism is stated in terms of variables available at lock time.

Even if all five conditions hold, the result is a hypothesis for a fresh
candidate protocol. It does not support a production pick-rule edit until a
prospective evaluation or a separately validated at-lock mechanism confirms it.

If no cell satisfies all five mechanism-found conditions, the mining produced
no actionable mechanism for this iteration. The leaderboard-mining-leverage
hypothesis is falsified for the current cohort and data window; future
iterations require either new data, new decomposition variables separately
pre-registered before inspection, or a different cohort definition.

## Methodology Constraints

Historical leaderboard mining is post-hoc. The 45-day backfill uses a latest
snapshot cohort retrospectively, not a cohort known on each historical date.
That creates survivorship and right-truncation bias and must be described in
every generated report.

Captured public pick logs are treated as observations of public behavior, not
as proof that a user had a pick locked before our decision cutoff. Any
pre-lock-visibility claim requires an explicit decision cutoff and capture
timestamp filter.

Backfilled ranked model surfaces are not at-lock unless their manifest proves
the candidate universe, lineup assumptions, feature computation, and
prediction timestamp were available before lock. Retrospective PA-row surfaces
may be useful mechanism diagnostics but cannot support deployment claims.

At the current sample size, most cross-product cells are expected to fall below
the `n >= 15` testability gate. The mining is power-limited. Absence of a found
mechanism at this size falsifies current-data nomination capacity, not the
possibility that leaderboard behavior contains useful signal under a larger or
better-instrumented sample.

## Artifact Contract

The first implementation artifact must write:

- a schema-versioned JSON report,
- a unit-level parquet of production-vs-consensus date-slots,
- `research_only = true`,
- `production_deploy_claim = false`,
- `no_policy_edit_supported = true`,
- the exact ordered decomposition variable list from this memo,
- `methodology_constraints` containing the post-hoc/latest-cohort caveat,
- `fdr_method` and bootstrap/randomization seed metadata.

No real-data artifact should be generated before this memo is committed.
