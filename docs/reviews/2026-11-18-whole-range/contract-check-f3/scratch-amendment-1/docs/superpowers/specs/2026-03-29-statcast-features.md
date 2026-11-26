# Statcast Feature Extraction — Design Spec

## Goal

Extract all available Statcast-level data from existing game feeds, build rolling features from each, and test which improve P@1 via walk-forward backtesting.

## Key Insight

Game feeds already contain Statcast data we don't extract (pitch velocity, spin rate, pitch movement, batted ball trajectory, total distance). Additionally, barrel rate and hard hit rate can be computed from `launch_speed` and `launch_angle` already in the PA table. Barrel rate stabilizes at ~50 BIP vs ~910 AB for batting average — potentially faster, more reliable signal.

## Schema Changes

### New fields in `parse_game_feed` from `hitData`:

| Field | Source | Type |
|-------|--------|------|
| `trajectory` | `hitData.trajectory` | str: ground_ball, line_drive, fly_ball, popup |
| `hardness` | `hitData.hardness` | str: hard, medium, soft |
| `total_distance` | `hitData.totalDistance` | float |

### New pitch-level fields from `pitchData` (stored as lists, like pitch_types):

| Field | Source | Type |
|-------|--------|------|
| `pitch_speeds` | `pitchData.startSpeed` | list[float] |
| `pitch_end_speeds` | `pitchData.endSpeed` | list[float] |
| `pitch_spin_rates` | `pitchData.breaks.spinRate` | list[float] |
| `pitch_extensions` | `pitchData.extension` | list[float] |
| `pitch_break_vertical` | `pitchData.breaks.breakVertical` | list[float] |
| `pitch_break_horizontal` | `pitchData.breaks.breakHorizontal` | list[float] |

### Computed at PA level during feature computation (not stored in parquet):

| Field | Formula |
|-------|---------|
| `is_barrel` | EV >= 98 and LA in sweet zone (widens with EV) |
| `is_hard_hit` | EV >= 95 |
| `is_sweet_spot` | LA between 8-32 degrees |

## New Rolling Features

All use the same temporal guard as existing features: date-level aggregation, shift(1), rolling window.

### Batter features (from batted ball quality):

| Feature | Window | Description |
|---------|--------|-------------|
| `batter_barrel_rate_30g` | 30 game-dates | Barrels / batted ball events |
| `batter_hard_hit_rate_30g` | 30 game-dates | Hard hit (EV >= 95) / batted ball events |
| `batter_sweet_spot_rate_30g` | 30 game-dates | Sweet spot LA / batted ball events |
| `batter_avg_ev_30g` | 30 game-dates | Mean exit velocity on contact |

### Pitcher features (from pitch quality):

| Feature | Window | Description |
|---------|--------|-------------|
| `pitcher_avg_velo_30g` | 30 game-dates | Mean pitch velocity |
| `pitcher_avg_spin_30g` | 30 game-dates | Mean spin rate |
| `pitcher_avg_extension_30g` | 30 game-dates | Mean release extension |
| `pitcher_break_total_30g` | 30 game-dates | Mean total break (sqrt(vert^2 + horiz^2)) |

### Pitch quality faced by batter:

| Feature | Window | Description |
|---------|--------|-------------|
| `batter_avg_velo_faced_30g` | 30 game-dates | Mean pitch velocity faced |

## Barrel Classification Formula

```
def is_barrel(ev, la):
    if ev < 98:
        return False
    # At 98 mph, LA must be 26-30
    # Each mph above 98 widens range by ~2 degrees each side
    # At 116+: LA range is 8-50
    bonus = (min(ev, 116) - 98) * 2
    la_min = max(8, 26 - bonus)
    la_max = min(50, 30 + bonus)
    return la_min <= la <= la_max
```

## Implementation Plan

### Task 1: Schema + extraction
- Add new fields to PA_COLUMNS in schema.py
- Update parse_game_feed in build.py to extract hitData and pitchData fields
- Update tests with new fields

### Task 2: Rebuild parquets
- Rebuild all 9 seasons (2019-2025) with new schema
- Verify row counts match existing parquets

### Task 3: Feature computation
- Add barrel/hard_hit/sweet_spot PA-level computation in compute.py
- Add 9 new rolling features with same temporal guards
- Update FEATURE_COLS

### Task 4: Validation
- A/B backtest: 13 features (baseline) vs 22 features (all new)
- If improvement: ablate each new feature individually
- Must improve P@1 on BOTH 2024 and 2025 to ship
- Any feature that doesn't help gets dropped

## Risk

- Parquet rebuild is ~15 min (one-time)
- New features may be redundant with existing ones (e.g., batter_hard_hit_rate vs batter_gb_hit_rate)
- 68% launch data coverage means 32% of PAs will have NaN for batted ball features — LightGBM handles this natively
