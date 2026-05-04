# SOTA #13 — Offline Policy Evaluation (OPE) — Phase 0 design memo (revised v2)

**Date**: 2026-05-04
**Branch**: `feature/sota-13-ope-design`
**Base**: main `da4d4d3` (post-PR #11 conformal gate v2 merge)
**Tracker entry**: §13 (full SOTA OPE — DR/FQE; foundation for #1 and #16)
**Author**: Claude (read-only inspection + design)
**Reviewer**: Codex (pre-implementation; bus #95–#99 sequencing)

This memo is a Phase 0 design/preflight. **No implementation code is in this PR.**
Writing this memo is the entire deliverable; an implementation PR is gated on
Codex sign-off.

**Revision note (v2, post Codex #99)**: the v1 draft proposed wiring
`bts.validate.ope.fitted_q_evaluation` directly into the audit path. Codex
flagged a column-contract mismatch: FQE expects integer `(t, s, a, sn, r)`
columns, but `_trajectory_dataframe_from_profiles` emits decomposed
`s_streak, s_days, s_saver, s_qbin, a, sn_streak` (partial next-state, no
encoder). Wiring FQE in directly would require building a state encoder,
emitting full s/sn transitions, and adding round-trip tests — substantial
upstream work. This v2 instead chooses **path B** per his recommendation: use
the existing `bts.simulate.pooled_policy.evaluate_mdp_policy` as the P0/P1
policy-value direct method. FQE is deferred to a future cycle that does the
state-encoder work.

## TL;DR

- **Estimand**: full P(57) policy value `V^π(s_0)` over the BTS MDP — the
  P(57) achieved by policy `π` starting at `(streak=0, day=0, saver=1)`,
  expectation taken over the **holdout** bin distribution.
- **First honest estimator**: **`evaluate_mdp_policy`** (already in
  `bts.simulate.pooled_policy`, docstring calls it "the honest A/B primitive").
  Wraps a fold-train-derived policy table and evaluates it forward against
  fold-holdout bin distributions.
- **Cross-check**: report terminal-reward MC replay on the same holdout
  profiles alongside the policy-value estimate. Disagreement is a
  **diagnostic flag**, not a gate.
- **Defer to P1.5+**: FQE (needs state encoder), full sequential DR-OPE (needs
  propensity model — pathological for deterministic policies), per-decision
  IS, multi-policy regret, robust-MDP variants.

## Background: what already exists on main

After grepping post-#11 main, BTS already has substantial policy-value
infrastructure:

| Function | Module | Status | Used where |
|---|---|---|---|
| `evaluate_mdp_policy(policy_table, early_bins, …)` | `bts.simulate.pooled_policy` | implemented; docstring calls it "honest A/B primitive" | called from various scripts; NOT yet from a #5-manifest-aware audit path |
| `solve_mdp(bins, …)` | `bts.simulate.mdp` | implemented | used by audit pipelines |
| `_run_terminal_r_mc_bootstrap(traj_df, …)` | `bts.validate.ope` | implemented | the actual estimator behind `corrected_audit_pipeline` |
| `paired_hierarchical_bootstrap_sample` / `stationary_bootstrap_indices` | `bts.validate.ope` | implemented | dependence-aware CI |
| `fitted_q_evaluation` | `bts.validate.ope` | implemented | NOT used; column contract mismatches the trajectory builder (see revision note) |
| `dr_ope_full_information` | `bts.validate.ope` | implemented | NOT used; same state-encoding gap |

The v1 falsification harness's "v1 simplification" (per
`project_bts_2026_05_03_v2.5_attribution.md`: "terminal-reward MC, not full
sequential DR") corresponds concretely to `corrected_audit_pipeline` calling
`_run_terminal_r_mc_bootstrap` instead of any direct policy-value method.
**`evaluate_mdp_policy` exists outside the audit-pipeline path** and is the
right primitive to bring into a manifest-aware audit.

## State / action / reward inventory

From `bts/simulate/mdp.py`, `bts/simulate/strategies.py`,
`bts/simulate/quality_bins.py`:

- **State**: `(streak ∈ [0, 57], days_remaining ∈ [0, season_length], saver ∈ {0,1}, quality_bin ∈ [0, n_bins])`
- **Actions**: `{skip, single, double}`
- **Reward**: terminal `+1` when `streak` first reaches 57; `0` otherwise
- **Horizon**: `season_length` (default 180)
- **Trajectory**: one per `(season, seed)` pair in backtest profiles
- **Bin distribution**: `QualityBins` with `(frequency, p_hit, p_both)` per
  bin; this is the input that `evaluate_mdp_policy` consumes

## Codex's #96 questions, answered (per #99 corrections)

### 1. Estimand

**Full P(57) policy value only.** Daily proper-scoring belongs to #12;
do NOT add it to the policy-value-eval schema.

`V^π(s_0) = E_holdout_bin_dist[ Σ_t r_t | π, s_0 ]` where `s_0 = (streak=0,
day=0, saver=1)` and the expectation is over the **holdout's quintile bin
frequencies** — the same convention `evaluate_mdp_policy` already uses, which
is the same convention `solve_mdp` uses for its `optimal_p57`.

### 2. Logged policy / action support

Logging policies (heuristic, MDP-optimal) are **deterministic** functions of
state. Action support is *thin*. **Practical implication for P0/P1**:

- **`evaluate_mdp_policy` (path B)** does NOT use IS weights. It forward-
  evaluates the fixed policy against the holdout's bin distribution. Robust
  to deterministic policies because it doesn't need action propensities at all.
- **Terminal MC replay** is unbiased for the *logged* policy's value but
  cannot do counterfactual.
- **Sequential DR-OPE / FQE on encoded transitions (path A)**: deferred. FQE
  needs a state encoder *and* gives degenerate IS weights for deterministic
  policies. P1.5+ work.

### 3. First honest estimator

**`evaluate_mdp_policy`** (path B per Codex #99). Reasons:

- Already implemented; its docstring calls it "the honest A/B primitive."
- Doesn't need a state encoder.
- Composes with #5 manifest naturally: train → solve → policy_table; holdout
  → bins → evaluate.
- Doesn't pretend tabular FQE over ~100K cells with ~120 trajectories per
  fold is adequate. (This was a real concern Codex flagged on the v1 draft;
  see q4 below.)
- Robust to deterministic policies.

**FQE / DR-OPE** are deferred to P1.5+ pending the state-encoder work
described in path A.

### 4. Sample-size adequacy + sparse-support diagnostics

Per Codex #99: 120 trajectories per fold is NOT adequate for naive 100K-cell
tabular FQE. With path B (model-based forward eval) we don't have this
problem because evaluate_mdp_policy operates on the bin-frequency distribution
(~5 bins × ~3 actions × ~58 streaks × ~180 days), which is ~150K cells but
is filled deterministically by Bellman backward induction over the bin
frequencies — no per-state count requirements.

Still, sparse-support diagnostics are required in the output schema. Per
fold, report:
- `n_holdout_dates`
- `n_holdout_trajectories` (`n_seeds × n_seasons` represented in this fold's
  holdout)
- `n_terminal_successes` (count of trajectories that reached 57 in the
  realized holdout)
- For target policy: `target_action_coverage` — fraction of (state) cells
  the policy visits where holdout has at least one observation contributing
  to that bin's `p_hit` / `p_both` estimate
- `holdout_bin_min_n`: smallest holdout bin's count (small bins → unreliable
  `p_hit`/`p_both`)
- `verdict_flag`: `"OK"` or `"SPARSE_HOLDOUT_SUPPORT"` if the smallest bin
  count falls below a configurable threshold (suggested default: `min_bin_n=200`)

The `SPARSE_HOLDOUT_SUPPORT` flag is **diagnostic only**, not a gate.

### 5. #5 manifest contract

For each fold:

```
fold_train_df  →  compute bins / corrected bins  →  solve_mdp  →  policy_table_train
fold_holdout_df →  compute holdout bins (early/late)  →  evaluate_mdp_policy(policy_table_train, holdout_bins)  →  V^π
fold_holdout_df →  terminal_MC_replay(holdout_profiles, policy_table_train)  →  V_replay (cross-check)
```

This is the **explicit "honest holdout policy-value evaluation"** form per
Codex #99 blocker 2. The fixed policy is trained on fold train; the
transition / bin model is estimated from fold holdout; evaluation reports
both the model-based V^π and the empirical replay V_replay.

The lockbox is held out of every fold and reserved for an explicit
end-of-cycle cert run (P1.5+).

### 6. #14 / #15 consumption

- **#14 (rare-event MC, full SOTA, CE-IS)**: parallel to #13. Both estimate
  V^π(s_0) but from different angles. Cross-method disagreement is the
  audit-relevant signal. P0/P1 #13 doesn't consume #14.
- **#15 (full SOTA dependence)**: feeds the trajectory-bootstrap CI when (and
  if) one is added to #13's pipeline. Per Codex #99 blocker 3, in P0/P1
  block-bootstrap CI on `evaluate_mdp_policy` is non-trivial because each
  bootstrap replicate must recompute holdout bins from the resampled holdout
  profiles, then re-call evaluate_mdp_policy. Either include the full
  procedure or report `aggregate_deferred=True` (matching the #5 P0/P1
  pattern) and present the per-fold distribution as the uncertainty surface.
  **P0/P1 lean: `aggregate_deferred=True`** — keep the scope tight; CI
  procedure is its own design problem and composes with #15 anyway.

## Proposed P0/P1 boundaries

**In scope (P0/P1)**:
1. New `compute_policy_value_eval_over_manifest(profiles_df, manifest_path,
   *, target_policy_name) -> dict` in `bts.validate.ope_eval`. Contract
   mirrors `compute_scorecard_over_manifest` from #5 P0/P1.
2. Per fold:
   - solve target policy on fold train (using existing `solve_mdp`)
   - compute holdout bins from fold holdout (early + optional late)
   - call `evaluate_mdp_policy(policy_table_train, holdout_bins)` for V^π
   - compute terminal-MC replay V_replay for cross-check
   - report sparse-support diagnostics + verdict flag
3. Per-fold reporting: `V_pi`, `V_replay`, `disagreement_abs = |V_pi - V_replay|`,
   `disagreement_rel = disagreement_abs / max(V_pi, eps)`. **Disagreement is
   diagnostic only, not a gate** (per Codex #99 q3).
4. Aggregate: `aggregate_deferred=True` in P0/P1. Per-fold distribution is
   the uncertainty surface.
5. Output schema: `policy_value_eval_v1` (per Codex #99 q5 — name the
   estimand, not the estimator) with structure:
   ```json
   {
     "schema_version": "policy_value_eval_v1",
     "estimand": {"name": "p57", "horizon": 180, ...},
     "estimator": {"primary": "evaluate_mdp_policy", "cross_check": "terminal_mc_replay"},
     "manifest_metadata": {...},
     "lockbox_held_out": true,
     "fold_results": [{"fold_idx": 0, "V_pi": 0.0833, "V_replay": 0.0750, ...}],
     "aggregate_deferred": true,
     "thresholds": {...}
   }
   ```
6. CLI: `bts validate policy-value-eval --manifest <path> --target-policy <name>
   --output <path>`.
7. Tests:
   - `test_solve_then_evaluate_round_trip`: solve on synthetic bins, evaluate
     same policy against same bins, V^π should match `optimal_p57` from solver.
   - `test_evaluate_against_different_holdout_bins`: V^π differs when holdout
     distribution differs from train.
   - `test_manifest_integration`: `compute_policy_value_eval_over_manifest`
     produces 5 fold_results, lockbox_held_out=true.
   - `test_disagreement_reported_continuously`: V_replay vs V_pi reported as
     scalars; SPARSE_HOLDOUT_SUPPORT flag fires below configurable threshold.
   - `test_target_policy_parameter`: `mdp_optimal` (default) vs at least one
     baseline (`always_skip` if cheap to express as a policy_table).
8. Real-data smoke against `data/simulation/backtest_*.parquet` using the
   existing manifest from #5 (lockbox 2025-08-30..09-28).

**Implementation-contract note (per Codex #101)**: the CLI/manifest path
consumes the repo's standard rank-row backtest profile format (`date, rank,
p_game_hit, actual_hit, seed/season`). Fold train/holdout bins must be built
through the existing rank-row helpers — `compute_pooled_bins` and the
`split_by_phase_pooled` / `build_pooled_policy` conventions — NOT through the
direct top1_p/top2_p format that the older `bts.validate.ope` helpers
(`_trajectory_dataframe_from_profiles`, `_compute_bins_from_direct_profiles`)
expect. The `data/simulation/backtest_*.parquet` files and the #5 manifest
flow are rank-row. P0/P1 implementation should either:
1. Reuse `compute_pooled_bins` directly on the (train_df, holdout_df) outputs
   from `apply_fold` — preferred, no new code surface.
2. Introduce one small `rank_row_to_top1_top2(df) -> df` adapter with
   round-trip tests if a top1/top2 representation is needed downstream.

The implementation tests must include a smoke test that `bts validate
policy-value-eval` runs successfully against the existing
`data/simulation/backtest_*.parquet` files via the #5 manifest, with bins
built through `compute_pooled_bins`.

**Out of scope (deferred — explicit follow-ups, not "forgotten")**:
- **#13 P1.5: encoded-transition FQE / DR-OPE** — requires a state encoder,
  full s/sn transition records from the trajectory builder, round-trip
  encoding tests, next-state correctness for skip/single/double/saver/terminal.
  Once landed, the existing `fitted_q_evaluation` and `dr_ope_full_information`
  in `bts.validate.ope` become directly callable. Sequential DR-OPE further
  needs propensity modeling or epsilon-greedy logging redesign.
- **#13 P1.5 (uncertainty)**: block-bootstrap CI on `V_pi` (recompute-bins-per-
  replicate procedure). Composes naturally with #15 dependence work.
- Per-decision IS estimators (depends on epsilon-greedy logging or propensity
  estimation)
- Multi-policy regret comparison (depends on `Strategy → policy_table` adapter)
- Named-strategy `Strategy → policy_table` adapter (only baselines that fit
  existing table cheaply in P0/P1: `always_skip`, `always_rank1`)
- Robust MDP / CVaR-MDP (#1; needs #13 first)
- **Lockbox cert run** — separate PR after the per-fold evaluator is trusted,
  matching the #5 P0/P1 deferral pattern

## Open questions for Codex (red-team this scope)

1. **Schema name `policy_value_eval_v1`**: per your #99 q5 guidance. Does
   the `{estimand: {…}, estimator: {…}}` block structure work for you, or
   prefer a different shape?

2. **Baselines in P0/P1**: `mdp_optimal` (default) is unambiguous. `always_skip`
   is trivial (policy_table = zeros). `always_rank1` would be `policy_table = ones`
   (always single). Are those two enough as included baselines? Named-threshold
   strategies (heuristic, streak_aware) need a `Strategy → policy_table` adapter
   which is its own design — defer per your #99 q2.

3. **`SPARSE_HOLDOUT_SUPPORT` threshold**: `min_bin_n=200` for the smallest
   holdout bin. Reasonable, or different cut?

4. **No aggregate CI in P0/P1** (`aggregate_deferred=true`): you said either
   that or a clearly-defined recompute-bins-per-replicate block-bootstrap. I
   lean `aggregate_deferred=true` for P0/P1 to keep scope tight and let the
   CI design land with #15 (which is the dependence-aware uncertainty
   surface). Agree, or push for the bootstrap procedure now?

5. **Path A (FQE on encoded transitions) deferred**: my implicit position
   is "P1.5 or later, requires state encoder + round-trip tests." Should
   the deferral be explicit in the memo as a named follow-up tracker item,
   or just informal "deferred"?

6. **Lockbox cert run**: I'm assuming this is a separate PR after we have
   confidence in the per-fold pattern. Same shape as #5 P0/P1's deferral.
   Agree?

## Phase 0 acceptance (per Codex #96 + #99)

- [x] This design memo with API boundaries (revised v2)
- [x] Implementation plan (in scope / out of scope)
- [x] Path B explicit; FQE deferred per Codex #99 blocker 1
- [x] Train/holdout contract explicit per Codex #99 blocker 2
- [x] Bootstrap scope explicit per Codex #99 blocker 3 (`aggregate_deferred=true`)
- [x] Sparse-support diagnostics per Codex #99 q4
- [x] Disagreement non-gating per Codex #99 q3
- [x] Schema name `policy_value_eval_v1` per Codex #99 q5
- [ ] Codex review of this revised memo
- [ ] (Optional, after sign-off) skeletal tests if interfaces are obvious
- [ ] (Implementation PR) gated on Codex sign-off

## What this memo does NOT propose

- No DR/FQE math change in `fitted_q_evaluation`.
- No new state encoder.
- No new strategy / action set.
- No new policy-class search.
- No replacement of `corrected_audit_pipeline`. P0/P1 ADDS a parallel
  policy-value path (using `evaluate_mdp_policy`); it does not replace the
  v2.6 reference.
- No code changes outside `docs/superpowers/specs/`.
