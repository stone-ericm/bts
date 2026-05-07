# Experiment Runner Nested Split Design

## Verdict

Structured verdict: `design_locked_ready_for_implementation`.

Do not retrofit `bts experiment screen/select` directly without a contract. The current runner is walk-forward inside each requested test season, but feature/model selection decisions are still made on the same `--test-seasons` span used for scoring.

That is not the same as leakage inside `blend_walk_forward`: for a given game day, training uses prior data only. The risk is one level higher: audit decisions can be selected on the evaluation seasons and then reported as if those seasons were held out.

The first implementation should be a season-level opt-in split, not full day-level nested CV. It should remove the direct selection-on-evaluation-span failure mode while preserving existing workflows by default.

## Current Behavior

`bts experiment screen`:

- loads all processed `pa_*.parquet`
- computes features once
- builds a baseline scorecard over `--test-seasons` (default `2024,2025`)
- runs each Phase 1 experiment over the same `--test-seasons`
- computes `diff_scorecards(baseline, experiment)` on that same span
- calls `evaluate_pass_fail(diff)` to choose Phase 1 winners

`bts experiment select`:

- reads Phase 1 winners from `experiments/results/phase1`
- runs forward selection and backward elimination on `--test-seasons`
- in multi-seed mode, keeps/drops candidates using paired seed deltas on that same span
- writes the final selected set and final scorecard from that same span

The existing multi-seed logic addresses provider/seed variance. It does not create a temporal holdout.

`runner_factored.py` is not a separate semantic runner. It is an acceleration path called from `run_single_screening` when eligible. The split contract should stay in `experiment/cli.py` and `runner.py`; factored paths should receive whichever season list the caller designates for the current role.

## Required Contract

The runner needs three distinct roles:

- Selection span: dates/seasons used to choose features, hyperparameters, thresholds, and candidate ordering.
- Outer evaluation span: later dates/seasons used once to estimate the selected stack's effect.
- Lockbox: a final future segment reserved for go/no-go claims, never used for selection.

These roles should be explicit in outputs. Every Phase 1/2 artifact should record:

- `selection_dates` or `selection_seasons`
- `outer_evaluation_dates` or `outer_evaluation_seasons`
- whether a #5 manifest/lockbox was used
- whether the artifact is selection-only, outer-eval, or lockbox-certification

The first implementation does not consume a #5 manifest. The manifest path remains deferred because `experiment` currently works at season granularity, while the #5 manifest is day/fold oriented.

## Minimal Retrofit

Start with an opt-in mode rather than changing the default:

1. Add CLI flags to `bts experiment screen` and `bts experiment select`:
   - `--selection-seasons`
   - `--outer-eval-seasons`
   - optionally `--manifest` once day-level fold support is wired through
2. Fail closed if the two season sets overlap.
3. Run Phase 1/2 keep/drop decisions only on `selection_seasons`.
4. Re-evaluate the final selected stack exactly once on `outer_eval_seasons`.
5. Save separate artifacts for selection and outer evaluation, with split metadata copied into both.

This is a season-level P1. It is not the final #5 lockbox certification, but it removes the most direct selection-on-evaluation-span failure mode.

Selected on the selection span:

- Phase 1 pass/fail
- Phase 2 forward-selection keep/drop decisions
- Phase 2 backward-elimination keep/drop decisions
- any threshold/keep-rule choice provided through existing CLI flags

Estimated on the outer span:

- baseline scorecard
- final selected stack scorecard
- final selected stack diff vs baseline

The outer span must not feed back into candidate ordering, keep/drop rules, or threshold choice.

## Public API

Proposed CLI surface:

```bash
bts experiment screen \
  --selection-seasons 2023,2024 \
  --outer-eval-seasons 2025

bts experiment select \
  --selection-seasons 2023,2024 \
  --outer-eval-seasons 2025
```

Rules:

- `--test-seasons` remains the legacy default path.
- `--selection-seasons` and `--outer-eval-seasons` must be supplied together.
- Legacy `--test-seasons` must not be combined with the new split flags.
- The two season sets must be disjoint.
- If `--seeds` or `--seed-set` is supplied, the same temporal split applies inside each seed run.

## Seed Pool

Seed pooling remains an outer robustness dimension, not part of the inner temporal split. The split contract is calendar-time first:

- within each seed, selection decisions use only `selection_seasons`
- within each seed, the final selected stack is evaluated on `outer_eval_seasons`
- pooled seed summaries aggregate after those per-seed temporal roles are respected

This keeps the first implementation small and avoids mixing two axes of validation in one API. Revisit only if the season-level split shows seed-dependent candidate order instability.

## Source Scope

Implementation slice should change:

- `src/bts/experiment/cli.py`: parse new flags, reject overlaps, route selection vs outer-eval seasons.
- `src/bts/experiment/runner.py`: accept split metadata in result payloads and expose a helper for final outer-eval of a selected stack.
- `tests/experiment/`: add CLI and runner tests for disjointness, routing, and output metadata.

Implementation slice should not change:

- `src/bts/experiment/runner_factored.py`, unless a test proves the existing caller-level season routing is insufficient.
- `src/bts/simulate/backtest_blend.py`.
- production prediction, scheduler, or deployment code.

## Risk / Rollback

Existing `experiments/results/phase1` and `experiments/results/phase2` artifacts should be treated as legacy selection-on-evaluation-span evidence. They are not deleted or rewritten by the retrofit.

Rollback path is simple: the new split flags are opt-in. If the implementation regresses existing workflows, users can omit the new flags and get the current `--test-seasons` behavior.

## Deferred Work

- Day-level #5 manifest integration for experiment runner folds.
- Aggregate fold uncertainty for the selected stack.
- Explicit lockbox certification artifact for any candidate production change.
- A richer nested CV mode where each outer fold has an inner time-respecting tuning split.

## Acceptance Criteria For The Implementation Slice

- Existing default CLI behavior remains unchanged unless opt-in flags are supplied.
- Tests prove selection and outer evaluation season sets cannot overlap.
- Tests prove Phase 1 pass/fail uses selection seasons, not outer-eval seasons.
- Tests prove Phase 2 keep/drop uses selection seasons, then writes a separate outer-eval result.
- Output JSON records split metadata and labels artifacts as `selection_only` or `outer_evaluation`.
- No deploy claim is made from the new outer-eval result.
