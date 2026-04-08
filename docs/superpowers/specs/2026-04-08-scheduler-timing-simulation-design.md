# Scheduler Timing Simulation

**Date:** 2026-04-08
**Status:** Design spec
**Depends on:** Architecture alignment Phases 1-5 resolved

## Motivation

Removing the densest bucket improves P(57) in the backtest, but the backtest uses all-confirmed lineups. In production, lineups confirm throughout the day — the scheduler might lock a pick on an early-game batter before evening lineups reveal a better option.

We need to simulate the scheduler's real-time dynamics over historical data to measure how often this happens, and whether guardrails (like a minimum confirmed-game threshold) are worth adding.

## What We're Simulating

For each historical game day (912 days across 2021-2025):

1. **Game schedule**: actual game start times from raw feeds (already available)
2. **Lineup confirmation times**: simulated by sampling from a normal distribution around `game_time - T`, where T ~ N(60, 15) minutes. Lineups are "confirmed" once their confirmation time has passed.
3. **Check times**: computed the same way as the scheduler — `game_time - 45 min`, clustered within 10 min.
4. **At each check**:
   - Confirmed-lineup batters use their actual features
   - Projected-lineup batters use prior-day lineup with actual features (same as production fallback)
   - Model predicts P(game_hit) for all batters
   - `should_lock` logic: top pick has confirmed lineup AND gap >= `early_lock_gap` vs best projected pick
   - If locked → record the pick
   - If not → continue to next check
5. **Fallback**: if not locked by `game_time - 15 min` for the top pick's game, force-post
6. **Oracle**: what the pick would be with all lineups confirmed (the standard backtest result)

## Metrics

- **Lock accuracy**: how often does the locked pick match the oracle pick?
- **Early-lock penalty**: on days where the locked pick differs from oracle, what's the P@1 difference?
- **Lock timing**: distribution of when picks lock (what check number, how many games confirmed)
- **P(57) under simulation**: run the full MDP on simulated picks vs oracle picks

## Variants to Test

All variants use the blend ranking (no densest bucket):

1. **Baseline**: current `should_lock` with `early_lock_gap=0.03`, no min-game threshold
2. **Min 50% confirmed**: `should_lock` additionally requires ≥ 50% of games confirmed
3. **Min 75% confirmed**: ≥ 75% of games confirmed
4. **Higher gap**: `early_lock_gap=0.05` (5% instead of 3%)
5. **Combined**: `early_lock_gap=0.05` + min 50% confirmed

## Implementation

Standalone experiment script. Key components:

### Lineup Confirmation Simulator

```python
def simulate_confirmation_times(games: list[dict], seed: int = 42) -> dict[int, datetime]:
    """Simulate when each game's lineup would be confirmed.
    
    Returns {game_pk: confirmation_time_et}.
    Samples from N(game_time - 60min, 15min), clipped to [game_time - 120min, game_time - 15min].
    """
```

### Daily Scheduler Simulator

```python
def simulate_day(
    date: str,
    predictions_confirmed: pd.DataFrame,  # predictions with all lineups confirmed
    predictions_projected: pd.DataFrame,   # predictions with prior-day lineups
    game_times: dict[int, datetime],       # game_pk → game start time
    confirmation_times: dict[int, datetime],  # game_pk → lineup confirmation time
    check_offset_min: int = 45,
    early_lock_gap: float = 0.03,
    min_confirmed_pct: float = 0.0,        # guardrail: min % games confirmed to lock
) -> dict:
    """Simulate the scheduler's check-and-lock flow for one day.
    
    Returns {locked_pick_batter_id, lock_time, n_games_confirmed, oracle_pick_batter_id}.
    """
```

### Data Requirements

- **Backtest profiles with game_pk**: need to map predictions back to games (current profiles don't have game_pk — would need to add it)
- **Game times**: from raw feed lookup (already built for Phase 3)
- **Prior-day lineups**: for each batter-game, whether the batter was in the prior day's starting lineup for that team

### Simplification

Rather than re-running the full model for each simulated check, we can approximate:
- **Confirmed picks**: use the full backtest predictions (with actual features)
- **Projected picks**: use the same predictions but flag batters whose lineup isn't confirmed yet
- The `should_lock` gap check compares confirmed vs projected — we just vary which batters are "confirmed" at each simulated time

This avoids retraining and re-predicting, making the simulation fast (~seconds per day).

## Stochastic Element

Lineup confirmation times are random. Run 100 trials per day with different random seeds, report median and 95th percentile outcomes. This captures the variance in "how often does confirmation timing matter?"

## Decision Criteria

- If baseline (no guardrail) lock accuracy is > 95% — no guardrail needed, the early-lock scenario is rare enough
- If lock accuracy < 90% — add the guardrail variant with best P(57)
- Between 90-95% — judgment call based on P(57) impact

## Out of Scope

- Changing the scheduler's check timing logic (game_time - 45 min offset)
- Modeling bullpen-specific effects on late-game predictions
- Real API calls or live scheduler changes
