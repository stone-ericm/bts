# Leaderboard vs backfilled model audit pre-registration

**Date**: 2026-05-10 ET
**Scope**: research-only comparison of captured public BTS leaderboard picks
against one or more backfilled ranked model surfaces.
**Script**: `scripts/leaderboard_backfilled_model_audit.py`
**Status**: pre-registered implementation path; no production policy edit is
supported by this memo.

**Smoke-stage info-set verdict**: the copied `backtest_2026.parquet` is not a
valid at-lock production-decision surface. The backtest path is walk-forward for
model training and shifted rolling features, but it ranks over realized PA rows
and aggregates over actual `n_pas`. That is retrospective lineup/exposure
information.

## Question

Can we learn whether leaderboard players are concentrating on batter-days that
our model surfaces systematically underrank?

This is different from wiring leaderboard data into the dashboard. The goal is
to mine clues for better pick methodology, not to create a user-facing feature.

## Pre-registered Primary Read

Primary cohort: the fixed cohort from the latest `active_streak` leaderboard
snapshot, unless an explicit cohort JSON is supplied.

Primary unit: a resolved `(pick_date, pick_number)` date-slot.

Primary outcome: for each named backfilled surface, evaluate
`consensus_hit - model_rank_slot_hit` on units where fixed-cohort public
consensus disagrees with the surface's rank-matched pick (`rank=1` for pick 1,
`rank=2` for pick 2).

For realized-production surfaces, the primary historical-production read is
`fixed_cohort_consensus_hit_rate - realized_production_hit_rate` over resolved
date-slot units. Locked slots with null, pending, or void outcomes remain null
and are excluded from resolved outcome denominators; they are not coerced to
misses.

Primary descriptive check: fixed-cohort consensus top-k coverage and individual
tracked-pick top-k share for the configured `k` values. This answers whether
leaderboard behavior is mostly inside the model's top candidate pool or points
to batters the model often misses.

No production pick-rule edit is supported until this produces either:

1. at least 30 future resolved disagreement date-slot units under a fixed
   prospective cohort protocol, or
2. a separate, validated mechanism that explains and reproduces the historical
   gap without relying on survivor-biased leaderboard outcomes.

## Four Anchors

The audit must keep these anchors separate:

| Anchor | Meaning | Valid use |
|---|---|---|
| Leaderboard consensus | Modal captured public pick by date and slot | Policy-mining signal |
| Realized production pick | What the live system actually locked | Historical production truth |
| Production backfill | Current production-like code run over past dates | Current-model diagnostic |
| Candidate backfill | Candidate code/model run over past dates | Candidate diagnostic if cutoff is verified |

Backfilled production is not the same as realized production. Production code,
parameters, gating, and available data have changed during the season.

## Info-Set Verdict

The canonical backtest command path is `bts simulate backtest`, which calls
`src/bts/simulate/backtest_blend.py`. The relevant code path is designed to be
walk-forward:

- `src/bts/features/compute.py` states the temporal guarantee: PA features for
  date `D` use only dates strictly before `D`, with date-level rolling or
  expanding features shifted by one date.
- `blend_walk_forward` trains each test date on `train_pool` plus
  `test_data[test_data["date"] < day]`, then predicts only that day's rows.

This supports treating the training and rolling-feature side of a freshly
generated `backtest_2026.parquet` as walk-forward if it came from that command
path. It does **not** support treating the surface as a decision-time BTS
surface. `blend_walk_forward` predicts on `day_data` from the realized PA table,
then ranks batter-games after aggregating over actual PA rows:

- the candidate universe is restricted to batters who actually appeared,
- scratches and lineup misses are removed by construction,
- `p_game_hit` is aggregated over actual `n_pas`, not an at-lock PA forecast.

So the copied `backtest_2026.parquet` is an oracle-exposure diagnostic surface.
It is not historical-production truth, and it is not a valid proof that the
model would have beaten public consensus at lock time. Candidate artifacts need
their own provenance or manifest before any SOTA claim.

## Output Contract

`scripts/leaderboard_backfilled_model_audit.py` writes:

- schema-versioned JSON under `data/validation/`,
- joined individual pick rows as parquet,
- joined consensus date-slot rows as parquet.

The JSON embeds:

- `pre_registered_primary_comparison`,
- surface inventory and duplicate-collapse counts,
- fixed and all-tracked cohort metadata,
- individual pick overlap summaries,
- consensus rank coverage,
- consensus-vs-model outcome summaries with day-block bootstrap metadata,
- methodology constraints and falsification gates.

The script can ingest two surface types:

- `--surface NAME=PATH`: a ranked model surface with explicit `rank`.
- `--realized-production-surface NAME=PATH`: a canonical realized-picks parquet
  where locked production `slot` values are converted to ranks
  (`primary` -> 1, `double_down` -> 2). This is a true historical production
  decision surface, not a full model top-N ranking. Void, pending, and
  unresolved slots preserve null `actual_hit` values and remain outside
  resolved metric denominators.

## Initial Smoke Result

After the pre-registration above was written, the script was smoke-tested
against the remote leaderboard store using a temporary copy of
`backtest_2026.parquet`.

Inputs:

- leaderboard store: `/home/bts/projects/bts/data/leaderboard/`
- surface: `/tmp/bts_backtest_2026_for_leaderboard_audit.parquet`
- surface coverage: 350 rows, 35 dates, 2026-03-25 through 2026-04-28,
  top 10 ranks, no null `actual_hit`
- leaderboard coverage: 577 user-pick files, 515 non-empty users, 28,568
  deduped pick rows, 45 pick dates

Fixed-cohort consensus versus the copied oracle-exposure backfill:

| Metric | Value |
|---|---:|
| Resolved date-slot units with model surface | 70 |
| Disagreement units | 66 |
| Agreement rate | 0.057 |
| Backfilled model hit rate* | 0.857 |
| Fixed-cohort consensus hit rate | 0.757 |
| Mean delta, consensus minus model | -0.100 |
| Disagreement mean delta | -0.106 |
| Disagreement day-block bootstrap 95% CI | [-0.271, 0.043] |
| Bootstrap `p_mean_le_zero` | 0.923 |

*This is the hit rate of an oracle-exposure backfill, not an at-lock
production-decision surface.

Top-k overlap on dates covered by the surface:

| Coverage read | Top 1 | Top 2 | Top 5 | Top 10 |
|---|---:|---:|---:|---:|
| Fixed-cohort consensus in model top-k | 0.043 | 0.071 | 0.229 | 0.314 |
| Individual fixed-cohort picks in model top-k | 0.039 | 0.067 | 0.138 | 0.214 |

Read: on this partial copied surface, leaderboard consensus is mostly outside
the model top 10, but the surface itself has retrospective exposure
information. The model's apparent win on resolved rank-matched disagreement
units cannot be read as a valid at-lock production result. The defensible clue
is narrower: leaderboard behavior often points to batters outside the model's
retrospective top 10, so it may be useful as a policy-mining and
candidate-generation target. It does not support a direct "copy public
consensus" rule, and it does not support the backfilled model as an oracle for
future production decisions.

## Realized Production Smoke

After adding `--realized-production-surface`, the same audit was smoke-tested
against the canonical locked production pick artifact:

- leaderboard store: `/home/bts/projects/bts/data/leaderboard/`
- realized production surface:
  `data/validation/realized_picks_canonical_2026-05-10_p1.parquet`
- surface coverage: 80 rows, 42 dates, 2026-03-29 through 2026-05-09,
  ranks 1 and 2 only, 4 pending `actual_hit` rows

Fixed-cohort consensus versus true realized production picks:

| Metric | Value |
|---|---:|
| Resolved date-slot units with realized production pick | 75 |
| Disagreement units | 72 |
| Agreement rate | 0.040 |
| Realized production hit rate | 0.693 |
| Fixed-cohort consensus hit rate | 0.827 |
| Mean delta, consensus minus production | +0.133 |
| Disagreement mean delta | +0.139 |
| Disagreement day-block bootstrap 95% CI | [-0.050, +0.238] |
| Bootstrap `p_mean_le_zero` | 0.105 |

Top-k overlap on dates covered by the realized production surface:

| Coverage read | Top 1 | Top 2 |
|---|---:|---:|
| Fixed-cohort consensus equals realized production pick | 0.048 | 0.071 |
| Individual fixed-cohort picks equal realized production pick | 0.036 | 0.056 |

Read: this is the valid historical-production anchor. It points the same way
as the original leaderboard clue memo: public fixed-cohort consensus was usually
different from our locked production picks and had a higher hit rate on the
overlap. This is a directional signal consistent with the May 9 exploratory
leaderboard memo, but `n=75` is not enough for a 95% statistical conclusion.
The uncertainty interval is still wide and crosses zero, so this is not a
deployment authorization.

The repeated `n=75` should not be treated as two independent samples. Both reads
share the same canonical realized-production pick source and overlapping
date-slot unit definition. The current smoke adds a separate fixed-cohort code
path and at-lock surface abstraction, while the May 9 memo used all-tracked
consensus directly. That is useful corroboration of direction, not independent
confirmatory evidence.

`p_mean_le_zero = 0.105` is a day-block bootstrap tail fraction for the
disagreement mean delta, not a frequentist p-value. Subgroup reads from the
joined rows remain diagnostic unless separately pre-registered or FDR
controlled.

The stronger reason to continue is mechanism mining: which batter classes does
public consensus like that realized production underrates, and can that
mechanism be reproduced prospectively without survivorship bias?

## Constraints

This remains post-hoc and survivor-biased. The leaderboard pick logs record
scrape time, not user decision time. A user observed after lock may still have
made the pick before lock, but this artifact cannot prove pre-lock visibility.

The leaderboard rows carry `batter_id` but not `game_pk`, so same-date
doubleheader context is collapsed to the best-ranked batter-date row from the
model surface. That is acceptable for policy-mining, but it is not a
game-specific attribution proof.

Subgroup reads are diagnostic unless separately pre-registered or controlled
with an appropriate multiple-testing procedure.
