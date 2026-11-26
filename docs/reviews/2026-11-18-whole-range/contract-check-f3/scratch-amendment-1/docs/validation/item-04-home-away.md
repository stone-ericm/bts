# Item 4: Home/Away Analysis

**Date:** 2026-04-02
**Verdict:** Signal already captured (or negligible) — no action needed

## Question

Does the visiting team's structural advantage (guaranteed bottom of 9th = extra PA opportunity) show up as residual signal in rank-1 pick accuracy, or is it already absorbed by the model?

## Setup

- Backtest profiles: `data/simulation/backtest_{season}.parquet` (2021–2025, 912 rank-1 picks)
- PA data: `data/processed/pa_{season}.parquet` — joined on `(batter_id, date)` to get `is_home`
- Method: two-proportion z-test on P@1 for home vs away rank-1 picks

## Findings

### Pick composition

The model picks away batters at roughly 2:1 over home batters (67% away, 33% home). This skew is expected: away batters get more PAs per game.

| Group | Picks | P@1   |
|-------|-------|-------|
| Home  | 300   | 0.850 |
| Away  | 612   | 0.873 |
| Diff  |       | −0.023 (home − away) |

- z-stat: −0.937
- p-value: 0.349 (not significant)

### Average PAs per game (all batters, 2021–2025)

| Group | Avg PAs/game |
|-------|-------------|
| Home  | 3.634       |
| Away  | 3.783       |
| Diff  | +0.149 (away advantage) |

Away batters average ~0.15 more PAs/game — consistent with the bottom-of-9th structural advantage (visiting team guaranteed to bat in every inning; home team wins walk-off and skips).

### Model's predicted probability by group

The model already assigns slightly higher predicted probabilities to away batters:

| Group | Avg p_game_hit |
|-------|---------------|
| Home  | 0.814         |
| Away  | 0.820         |

The 0.6% gap in predictions roughly tracks the PA advantage, suggesting `park_factor` and the PA-level aggregation math (`P(>=1 hit) = 1 − ∏(1−p_pa)`) implicitly capture the extra at-bat opportunity.

### Per-season breakdown

| Season | Home P@1 | (n) | Away P@1 | (n) | Diff  |
|--------|----------|-----|----------|-----|-------|
| 2021   | 0.792    | 72  | 0.927    | 110 | −0.136 |
| 2022   | 0.877    | 65  | 0.877    | 114 | 0.000  |
| 2023   | 0.889    | 54  | 0.859    | 128 | +0.030 |
| 2024   | 0.836    | 55  | 0.846    | 130 | −0.010 |
| 2025   | 0.870    | 54  | 0.862    | 130 | +0.009 |

No consistent directional pattern across seasons. 2021 shows a large away advantage (−0.136) but 2023 slightly reverses it. Noise dominates.

## Architecture Note

`is_home` was previously tested as a direct PA-level feature and dropped (ARCHITECTURE.md: "Noise at PA level"). This analysis confirms: the structural away advantage is real (+0.15 PAs/game) but already encoded implicitly in the game-level PA aggregation formula, and the residual P@1 difference is not statistically significant (p=0.35).

## Verdict

**SIGNAL CAPTURED.** No action needed. The model's game-level aggregation (`1 − ∏(1−p_pa)`) naturally accounts for extra away PAs, and the model already skews its top picks toward away batters at the right 2:1 ratio. There is no statistically significant residual home/away signal in rank-1 accuracy.
