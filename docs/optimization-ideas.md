# Optimization Ideas (Parking Lot)

## Known Edge Cases for `bts predict`

### Opener vs starter
Some teams use a reliever ("opener") for the first 1-2 innings, then a bulk pitcher for the rest. Our prediction grabs pitcher features from the first PA's matchup, which would be the opener — wrong for 3 out of 4 PAs. The bulk pitcher's features are what matter.

**Detection**: Check if the scheduled "starter" has a reliever profile (low innings/appearance, high appearances, bullpen usage pattern). Or pull from team-specific opener tendency data.

**Impact on backtesting**: None — backtesting uses actual PA-level pitcher_id.

### Players on the Injured List
A player on the IL can't play and shouldn't be picked. A player just activated from the IL has stale rolling features (batter_hr_7g from weeks ago) and high days_rest. The model has few training examples with 15+ day gaps, so predictions for IL returns are unreliable.

**Fix**: Check roster status via MLB API before generating picks. Flag any player with days_rest > 7 as unreliable. Consider a minimum recency threshold for rolling features.

### Other edge cases to handle
- **Postponed/suspended games**: Pick wasted if game is rained out
- **Late scratches**: Player removed from lineup after pick is locked
- **Doubleheaders**: Player could play in both games (pick applies to first)
- **Interleague DH rules**: Universal DH since 2022, so no longer an issue

## Feature Ideas
- Batter H2H history vs specific pitcher (sparse but might add on top of archetype)
- Pitcher recent workload (days since last start, innings in last 7/14 days)
- Batter launch speed/angle rolling averages (batted ball quality trends)
- Time-of-day effects (day games vs night games — dayNight field in game feed)
- Batter career PA count (experience proxy)
- Opposing team bullpen quality (for later PAs in game aggregation)

## Model Ideas
- Optuna hyperparameter search (systematic instead of manual grid)
- CatBoost as alternative to LightGBM (handles categoricals natively)
- Stacked ensemble: LightGBM predictions as features for a second-stage model
- Probability calibration post-hoc (isotonic on held-out fold)

## Architecture Ideas
- Pre-game PA estimation: use lineup + pitcher history to estimate number of PAs per batter
- Separate models for first PA vs later PAs (different pitcher fatigue context)
- Opponent-adjusted features (batter stats adjusted for quality of pitchers faced)
