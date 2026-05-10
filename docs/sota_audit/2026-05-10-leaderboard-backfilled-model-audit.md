# Leaderboard vs backfilled model audit pre-registration

**Date**: 2026-05-10 ET
**Scope**: research-only comparison of captured public BTS leaderboard picks
against one or more backfilled ranked model surfaces.
**Script**: `scripts/leaderboard_backfilled_model_audit.py`
**Status**: pre-registered implementation path; no production policy edit is
supported by this memo.

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

This supports treating a freshly generated `backtest_2026.parquet` as a
leak-aware current-model backfill if it came from that command path. It still
does not prove the artifact is historical-production truth, and it does not
prove candidate artifacts were trained without 2026 leakage. Each surface needs
its own provenance or manifest before any SOTA claim.

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

Fixed-cohort consensus versus the copied production-like backfill:

| Metric | Value |
|---|---:|
| Resolved date-slot units with model surface | 70 |
| Disagreement units | 66 |
| Agreement rate | 0.057 |
| Backfilled model hit rate | 0.857 |
| Fixed-cohort consensus hit rate | 0.757 |
| Mean delta, consensus minus model | -0.100 |
| Disagreement mean delta | -0.106 |
| Disagreement day-block bootstrap 95% CI | [-0.271, 0.043] |
| Bootstrap `p_mean_le_zero` | 0.923 |

Top-k overlap on dates covered by the surface:

| Coverage read | Top 1 | Top 2 | Top 5 | Top 10 |
|---|---:|---:|---:|---:|
| Fixed-cohort consensus in model top-k | 0.043 | 0.071 | 0.229 | 0.314 |
| Individual fixed-cohort picks in model top-k | 0.039 | 0.067 | 0.138 | 0.214 |

Read: on this partial copied surface, leaderboard consensus is mostly outside
the model top 10, but it did not beat the backfilled model on resolved
rank-matched disagreement units. This argues for treating leaderboard behavior
as a policy-mining clue and candidate-generation target, not as a direct
"copy public consensus" rule.

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
