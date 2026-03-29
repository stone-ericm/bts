# Optimization Ideas (Parking Lot)

Ideas to try after validation confirms the model is real.

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
