# BTS Strategy Simulator Design

## Problem

The model achieves 86.2% P@1 with the 12-model blend. Monte Carlo analysis shows this yields ~1% P(57-game streak) per season with selective doubling. The contest objective is maximizing P(streak reaches 57), not maximizing per-pick accuracy. These are fundamentally different optimization targets.

BTS rules confirm three strategic levers beyond prediction quality:
1. **Skip days** — streak holds if you don't pick. Play only high-confidence days.
2. **Double-down policy** — advance by 2 when both picks hit, but any No Hit resets. Both picks locked before earliest game.
3. **Streak Saver** — one free miss when streak is 10-15. Auto-activates on first reach of 10. Once used, gone for the season.

Small improvements in effective per-play accuracy have exponential impact: 0.86^57 ≈ 1/7,500 vs 0.90^57 ≈ 1/400 (19x better).

## Approach

Bootstrap Monte Carlo against real backtest predictions.

1. Run blend walk-forward backtest for 5 seasons (2021-2025) to extract daily prediction profiles
2. Bootstrap synthetic seasons by sampling daily profiles with replacement
3. Simulate strategy profiles against each synthetic season
4. Compare P(57), expected max streak, and other metrics across strategies

## Component 1: Blend Backtest with Prediction Export

**Module**: `src/bts/simulate/backtest_blend.py`

Extends the existing walk-forward evaluation to use the 12-model blend (same `BLEND_CONFIGS` from `predict.py`) instead of a single model.

Walk-forward loop (per test season):
- Retrain all 12 models every 7 days on all data before the current day
- For each test day, predict with all 12 models and average for blend ranking
- Aggregate PA-level predictions to game-level P(>=1 hit) per batter
- Save the top-10 batters per day with blend `p_game_hit` and `actual_hit`

**Output**: `data/simulation/backtest_{season}.parquet`

Schema per row:
- `date`: game date
- `rank`: 1-10 (blend ranking for that day)
- `batter_id`: int
- `p_game_hit`: float (blend average)
- `actual_hit`: int (0 or 1)
- `n_pas`: int (plate appearances)

**Seasons**: 2021, 2022, 2023, 2024, 2025. Retrain every 7 days. Training data starts at 2019.

## Component 2: Monte Carlo Strategy Simulator

**Module**: `src/bts/simulate/monte_carlo.py`

### Simulation Loop (one synthetic season)

1. Sample ~180 daily profiles with replacement from the pool of ~900 historical profiles
2. Initialize: streak=0, max_streak=0, streak_saver_available=True, streak_saver_used=False, play_days=0
3. For each day:
   - Read top-1 and top-2 from the daily profile
   - **Skip check**: if top-1 `p_game_hit` < skip_threshold (may vary by current streak), skip. Streak holds.
   - **Double check**: if playing, compute P(both) = top-1 `p_game_hit` * top-2 `p_game_hit`. If P(both) >= double_threshold (may vary by streak), double.
   - **Resolve outcome**: use `actual_hit` from the backtest profile
     - Single pick: Hit → streak+1; No Hit → reset (unless Streak Saver)
     - Double pick: both Hit → streak+2; either No Hit → reset (unless Streak Saver)
   - **Streak Saver**: if reset would happen and saver_available and 10 <= streak <= 15: streak holds, saver consumed
   - Update max_streak, play_days
4. Return max_streak, play_days, streak_saver_used

### Monte Carlo Wrapper

Run N simulated seasons (default 10,000) per strategy profile. Collect:
- P(reach 57): fraction of seasons where max_streak >= 57, with 95% CI
- P(reach 30+), P(reach 20+)
- Median max streak, 95th percentile
- Mean playing days per season

### Replay Mode

Also run each strategy against actual historical seasons (2021-2025) deterministically — no bootstrapping. Shows "what would have happened" as a sanity check alongside the probabilistic Monte Carlo results.

## Component 3: Strategy Profiles

**Module**: `src/bts/simulate/strategies.py`

Each strategy is a dataclass with:
- `name`: str
- `skip_threshold`: float or None (None = never skip)
- `double_threshold`: float or None (None = never double)
- `streak_saver`: bool (default True)
- `streak_config`: optional dict mapping streak ranges to (skip_threshold, double_threshold) overrides

Strategy resolution: if `streak_config` is set, look up current streak in the config to get thresholds. Otherwise use flat thresholds.

### Profiles to Test

| Profile | Skip | Double | Streak-Aware | Saver |
|---------|------|--------|-------------|-------|
| baseline | never | never | no | yes |
| current | never | P(both)>0.65 | no | yes |
| skip-conservative | <0.78 | P(both)>0.65 | no | yes |
| skip-aggressive | <0.82 | P(both)>0.65 | no | yes |
| sprint | never | P(both)>0.50 | no | yes |
| streak-aware | varies | varies | yes | yes |
| combined | varies | varies | yes | yes |

**Streak-aware config**:
```
streak 0-9:   skip=None, double=0.55  (aggressive start, less to lose)
streak 10-15: skip=None, double=0.60  (saver protects, moderate doubling)
streak 16-30: skip=0.78, double=0.65  (tighten up)
streak 31-45: skip=0.80, double=None  (singles only, skip bad days)
streak 46-56: skip=0.78, double=0.60  (sprint to finish)
```

**Combined config** (same as streak-aware — placeholder for iteration once we see results).

These are hypotheses. The simulator tells us if they actually help. We iterate on the thresholds based on results.

## Component 4: CLI Integration

**Module**: `src/bts/simulate/cli.py`

Two subcommands under `bts simulate`:

### `bts simulate backtest`
```
bts simulate backtest --seasons 2021,2022,2023,2024,2025
```
Runs blend walk-forward for each season, saves daily profiles to `data/simulation/`. Can parallelize via `--parallel N` (default 1).

### `bts simulate run`
```
bts simulate run --trials 10000
bts simulate run --strategy skip-aggressive --trials 50000
bts simulate run --replay-only
```
Loads daily profiles, runs Monte Carlo (or replay-only), prints Rich table:

```
Strategy Comparison (10,000 seasons, 900 daily profiles)
┌──────────────────┬─────────┬─────────┬─────────┬────────┬───────┬───────────┐
│ Strategy         │ P(57)   │ P(30+)  │ P(20+)  │ Median │ 95th  │ Play Days │
├──────────────────┼─────────┼─────────┼─────────┼────────┼───────┼───────────┤
│ baseline         │ 0.24%   │ 4.1%    │ 18.2%   │ 21     │ 38    │ 180       │
│ ...              │         │         │         │        │       │           │
└──────────────────┴─────────┴─────────┴─────────┴────────┴───────┴───────────┘
```

Also saves raw results to `data/simulation/results.json`.

## Project Structure

```
src/bts/simulate/
    __init__.py
    backtest_blend.py   — blend walk-forward, saves daily profiles
    monte_carlo.py      — simulation engine + Monte Carlo wrapper
    strategies.py       — strategy profile definitions
    cli.py              — Click commands
```

Wired into main CLI via existing `src/bts/cli.py` group.

## Testing

- Unit tests for strategy resolution (streak-aware threshold lookup)
- Unit tests for simulation loop (deterministic outcomes with known profiles)
- Integration test: run small Monte Carlo (100 trials) on synthetic profiles, verify output shape
- Backtest output validation: spot-check that saved profiles match known P@1 for 2024/2025

## Dependencies

No new dependencies. Uses existing: pandas, numpy, lightgbm, click, rich, parquet.
