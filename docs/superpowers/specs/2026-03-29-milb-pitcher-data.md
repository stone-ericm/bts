# MiLB Pitcher Arsenal Data — Design Spec

## Goal

Use minor league pitch data to provide real pitcher features (entropy, arsenal profile) for MLB debut pitchers, replacing the current league-average fallback. Improves both backtesting accuracy and live predictions.

## Context

- ~10-12% of MLB PAs per season face debut pitchers with no MLB history
- Currently these get league-average pitcher_hr_30g and pitcher_entropy_30g
- The MLB Stats API provides full pitch-level MiLB data (same format as MLB feeds)
- A pitcher's arsenal (pitch types, distribution) transfers directly from MiLB to MLB
- Hit rates do NOT transfer (different competition level)

## What Transfers vs What Doesn't

| Feature | Transfers? | Why |
|---------|-----------|-----|
| Pitch type distribution | Yes | Arsenal is a physical property of the pitcher |
| pitcher_entropy | Yes | Derived from pitch type distribution |
| Pitch velocity | Partially | Velo increases slightly at MLB level but relative ranking holds |
| pitcher_hr_30g | No | Different batter quality in MiLB |
| K rate, BB rate | Partially | Partially transferable but noisy |

## Approach

### Data Pipeline

1. **Identify debut pitchers**: For each MLB season, find pitchers with zero prior MLB PAs in our data
2. **Pull their MiLB game feeds**: Query schedule with `sportId=11` (AAA) and `sportId=12` (AA). Only need games where these pitchers appeared.
3. **Extract pitch sequences**: Same `parse_game_feed` logic works — same JSON structure
4. **Compute MiLB arsenal features**: `pitcher_entropy` from their MiLB pitch type distribution
5. **Store as a lookup**: pitcher_id → MiLB-derived entropy, pitch type profile

### Feature Integration

For debut pitchers, instead of league-average fallback:
- `pitcher_entropy_30g` → computed from their most recent 30 MiLB game-dates
- `pitcher_hr_30g` → still use league-average (hit rates don't transfer)

### Scope Considerations

- **AAA only vs AAA+AA**: AAA is closest to MLB. AA pitchers might have different arsenals by the time they debut. Start with AAA only.
- **How far back**: Pull MiLB data for the same years as MLB data (2019+). Pitchers who were in AAA in 2023 and debuted in 2024 get their 2023 AAA arsenal.
- **Data volume**: ~15 AAA games per day × 180 days = ~2,700 games per season. Much less than MLB.
- **Pitcher identification**: Same pitcher_id across MiLB and MLB (confirmed — MLB API uses unified IDs).

### Impact Estimate

- Affects 10-12% of PAs per season
- Current fallback is league-average entropy (~2.0 bits). Actual pitcher entropies range from 1.2 to 2.5.
- A debut pitcher with a 2-pitch mix (entropy ~1.0) facing batters is very different from one with a 5-pitch mix (entropy ~2.3). The current model can't distinguish these.

## Implementation Plan

### Task 1: MiLB data pull
- Add `sportId` parameter to `discover_games` and `pull_feeds`
- Add CLI: `bts data pull-milb --seasons 2019,2020,...`
- Storage: `data/raw/milb/{season}/{gamePk}.json`

### Task 2: MiLB PA build
- Reuse `parse_game_feed` (same JSON format)
- Only extract pitcher_id and pitch_types (don't need full PA features)
- Storage: `data/processed/milb_pitchers_{season}.parquet`

### Task 3: Debut pitcher lookup
- For each MLB season, identify debut pitchers
- Look up their MiLB pitch data
- Compute MiLB-based pitcher_entropy
- Store as lookup: `data/processed/debut_pitcher_features.parquet`

### Task 4: Integration
- Update `_build_feature_lookups` to load debut pitcher features
- Update debut fallback: use MiLB entropy instead of league average
- Update `walk_forward_evaluate` to use MiLB features for proper backtesting

### Task 5: Validation
- Compare backtesting results with MiLB features vs league-average fallback
- Check: does MiLB entropy for debut pitchers correlate with their eventual MLB performance?
