# Why the bullpen fix lowers the scorecard's mean max streak (R59 final test, reported-only metric), 2026-11-25

Asked by Eric ("yes, run it"). Outcomes 2021 to 2025 only; read from `~/bts-runs/bullpen-bt/out/` after the R59 final test (PASS, `docs/reviews/2026-11-18-whole-range/bullpen-r59-result/`).

## The gap
Mean max streak (180-day bootstrap under the scorecard's `combined` strategy, 2,000 to 10,000 trials): old 12.21, new 11.47; per-seed paired difference -0.74 (sd 0.95, se 0.19, 21 of 24 seeds lower).

## Hypothesis 1 (the lead's, 2026-11-25): the old arm's leak told the strategy WHEN to play. REFUTED.
The old feature was missing when the opposing team used no reliever. `check.py` marks a pick "no-reliever" when the batter's side faced exactly one pitcher (PA files): those are 0.5 to 0.7 % of team-games, and **no top-10 candidate of any day, in any seed or season, was in one** (the pitchers who throw complete games are not the ones the model's top batters face). The leak cannot have steered the top picks' play/skip timing; removing "those dates" changes nothing (the per-run gaps with and without differ only by bootstrap noise). Note: this proxy covers complete games, not the missing-probable cases of the leak.

## What the outputs do show (`check2.py`, per-seed paired means, new minus old)
| | change | seeds lower |
|---|---|---|
| rank-1 realized hit rate | +0.251 pt (se 0.175) | 11 of 24 |
| rank-2 realized hit rate | +0.200 pt (se 0.309) | 12 of 24 |
| both hit (the double) | +0.376 pt (se 0.217) | 10 of 24 |
| rank-1 predicted p_game_hit | **-0.059 pt (se 0.007)** | **23 of 24** |
| rank-2 predicted p_game_hit | **-0.078 pt (se 0.004)** | **24 of 24** |

The fix does not pick worse: the picks hit at least as often, the pair for a double included. What changes, consistently, is that the new model states slightly LOWER probabilities (under a tenth of a point).

## Reading (reasoning, not tested)
The `combined` strategy decides doubles and skips with fixed probability cutoffs; a uniform downward shift of a few hundredths of a point moves some days below a cutoff, so the simulated strategy doubles a little less and its streaks come out shorter, while the outcomes behind them are no worse. The shipped reach-57 policy also bins days by fixed cutpoints (fitted on older, hindsight-inflated profiles), so the same shift makes live play very slightly more conservative. That belongs to the policy-calibration work (D7, the 11-24 top-bin note), not to the fix. A direct confirmation would count the simulated doubles and skips in each arm (not run).
