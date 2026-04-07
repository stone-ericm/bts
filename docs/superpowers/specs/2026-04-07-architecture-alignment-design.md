# BTS Architecture Alignment

**Date:** 2026-04-07
**Status:** Design approved, pending implementation

## Motivation

The BTS system was built incrementally: PA-level modeling, then 12-model blend, then MDP strategy, then scheduler. Some design decisions made early may no longer hold. This spec covers four sequential investigations to verify and align the architecture.

The primary metric for all decisions is **MDP P(57)** computed from 5-season walk-forward backtest profiles with phase-aware quality bins. P@1 per season serves as a diagnostic. Changes must improve P(57) or maintain it while reducing complexity.

## Phase 1: Game-Level vs PA-Level Modeling

### Problem

The model predicts P(hit|PA), then aggregates to P(game hit). But all 14 baseline features are date-level — identical across PAs within the same game. The only PA-level variation comes from pitcher changes mid-game, which the live pipeline can't exploit anyway (it uses league-average reliever features).

This PA-level framing creates a structural divergence: the backtest aggregates using actual PAs and actual pitcher features, while the live pipeline estimates PAs and uses a starter/reliever split. The MDP is trained on backtest profiles but receives live predictions.

### What We're Testing

Train and evaluate a game-level model alongside the current PA-level model:

- **Game-level dataset:** One row per batter-game, target = `max(is_hit)`. Aggregate PA-level features to game level (they're already date-level, so this is mostly deduplication). Add `lineup_position` as a feature (proxy for PA count).
- **Same everything else:** LGB_PARAMS, 12-model BLEND_CONFIGS structure, walk-forward with retrain every 7 days.
- **Evaluation:** 5-season walk-forward, quality bins, MDP solve. Compare P(57), P@1, and quality bin distributions.

### Decision Criteria

- If game-level P(57) >= PA-level P(57): **adopt game-level**. It eliminates the backtest/live divergence by construction, removes PA estimation, starter/reliever split, and aggregation logic. Skip Phase 2 entirely.
- If PA-level P(57) > game-level by a meaningful margin (>10% relative): **keep PA-level**, proceed to Phase 2.

### Implementation

Standalone experiment script. Does not modify production code. Loads existing PA parquets, aggregates to game-level, trains, evaluates.

## Phase 2: Align Backtest Aggregation (Conditional)

**Only proceeds if PA-level wins in Phase 1.**

### Problem

The backtest and live pipeline aggregate blend predictions differently:
- **Backtest:** Average PA-level predictions across 12 models, then aggregate to game level via `1 - prod(1-p)`
- **Live:** Each model aggregates to game level independently, then average game-level predictions

These give different results because `E[f(X)] != f(E[X])`.

### What Changes

Switch the backtest's blend averaging order from `game_agg(avg(models))` to `avg(game_agg(models))`, matching the live pipeline.

The backtest continues to use actual PAs and actual pitcher features (ground truth for evaluation). The starter/reliever split and PA estimation remain live-only approximations.

### What Does Not Change

Model, features, strategy, MDP structure. This is purely a measurement correction.

### Evaluation

Re-run 5-season scorecard with corrected aggregation. The new baseline may differ from the current one (86.51% P@1, 4.84% MDP P(57) on the current backtest). Whatever the new number is becomes the baseline for subsequent phases.

## Phase 3: Ablate Densest Bucket

**Independent of Phase 1 outcome. Uses whichever modeling level won.**

### Problem

The densest bucket strategy filters picks to the time window (early/prime/west) with the most games, with a 78% override for strong picks outside that window. This was designed pre-blend when ranking was less reliable. With the 12-model blend providing better tie-breaking, it may no longer add value.

### What We're Testing

Run the walk-forward backtest and evaluate P@1 / P(57) two ways:
1. **With densest bucket:** Current behavior (filter, then rank within window)
2. **Without densest bucket:** Rank all batters by blend P(game hit) directly

### Implementation

Requires adding `game_time` to backtest profile output (currently missing). The densest bucket filter is applied post-hoc to the saved profiles, not during model evaluation.

### Decision Criteria

- If removing densest bucket improves or maintains P(57): **remove it**. Less complexity, fewer arbitrary parameters (the time-window boundaries, the 78% override).
- If removing it hurts P(57): **keep it**.

## Phase 4: Alt-Params Blend Member

**Independent of Phase 1 outcome. Uses whichever modeling level won.**

### Problem

The 12-model blend's diversity is one-dimensional: all models share the same feature values and differ only in which Statcast columns they include. Adding a model with different feature computation parameters would add orthogonal diversity.

### What We're Testing

Add a 13th blend model trained on features computed with alternate parameters:
- `platoon_hr` with PA threshold = 40 (baseline: 30)
- `batter_gb_hit_rate` with expanding min_periods = 15 (baseline: 20)
- `park_factor` with venue expanding min_periods = 30 (baseline: 20)
- Batter Statcast features with rolling min_periods = 3 (baseline: 10)

### Implementation

Two approaches considered:

**Chosen: Duplicate columns.** Compute `platoon_hr_alt`, `batter_gb_hit_rate_alt`, `park_factor_alt`, and 4 alt Statcast columns alongside originals in `compute_all_features`. Define `ALT_FEATURE_COLS` that swaps baseline columns for alt versions. Add `("alt_params", ALT_FEATURE_COLS)` to BLEND_CONFIGS.

This keeps everything in one DataFrame and fits the existing blend infrastructure. No second feature computation pass needed at prediction time.

**Rejected: Two DataFrames.** Would require computing features twice and managing parallel data flows. More memory, more complexity, same result.

### Decision Criteria

- 13-model blend must improve P(57) on the 5-season walk-forward.
- P@1 must not degrade on either test season (both-seasons test).

## Execution Order

```
Phase 1: Game-level vs PA-level
    |
    |-- game-level wins --> Phase 3, Phase 4 (Phase 2 skipped)
    |-- PA-level wins   --> Phase 2 --> Phase 3, Phase 4
    |
Phase 5: Team bullpen composite (after architecture settled)
    |
After all phases: re-solve MDP with final quality bins
```

Each phase produces a 5-season walk-forward backtest + MDP P(57). Approximately 50 minutes per variant.

## Phase 5: Team Bullpen Composite Feature

**Depends on Phase 1 outcome. Runs after Phases 1-4 are resolved.**

### Problem

The live pipeline uses league-average reliever features for late-inning PAs. This treats all bullpens as identical — a batter facing the Yankees' elite relievers gets the same P(hit) adjustment as one facing the Rockies' bullpen. We have the data to do better.

### What We're Testing

Compute a rolling team bullpen composite: average pitcher_hr, entropy, and Statcast features across each team's recent relievers (identified from roster/usage data).

- **If game-level model won:** Add `opp_bullpen_quality` as a feature. The model learns that facing a strong bullpen reduces P(game hit).
- **If PA-level model won:** Replace league-average reliever features with team-specific bullpen composite in the starter/reliever split.

### Decision Criteria

Must improve MDP P(57) on 5-season walk-forward. Both-seasons test on P@1.

## Success Criteria

The final system should have:
1. Higher or equal MDP P(57) compared to current baseline
2. Equal or fewer arbitrary parameters
3. Backtest that accurately predicts live performance (no structural divergence)

## Out of Scope

- Changing LGB_PARAMS (hyperparameter tuning was tested and rejected)
- Changing the MDP state space (quality bin count, saver range)
- Starter/reliever split and PA estimation parameters (eliminated if game-level wins, or left as-is if PA-level wins)
