# BTS Architecture

## Overview

Two-stage hit prediction model for MLB's Beat the Streak contest.
Stage 1 predicts P(hit) per plate appearance using LightGBM.
Stage 2 aggregates to P(>=1 hit) per game via probability math.

Validated results (walk-forward, provably leak-free):
- 2025 (train 2019-24): P@1=87%, P@100=91%, P@500=87%
- Tested across 6 seasons (2020-2025): P@1 82-91%, P@500 beats SOTA every year
- SOTA benchmark: Garnett (2026), P@100=85%, P@500=77%

## Data

- **Source**: MLB Stats API v1.1 (`/api/v1.1/game/{gamePk}/feed/live`)
- **Scope**: 9 seasons (2017-2025), 1.5M plate appearances
- **Training window**: 2019 onward (2017-18 hurts — game changed too much)
- **Filters**: Regular season only (no spring training, postseason, exhibitions, 7-inning COVID doubleheaders)
- **Storage**: Raw JSON (`data/raw/{season}/{gamePk}.json`) → PA Parquet (`data/processed/pa_{season}.parquet`)

## Features (13, provably leak-free)

All features use date-level shift(1) — only data from dates strictly before the prediction date.
Verified by nuclear test: 260/260 manual spot checks passed.

| Feature | Type | Description |
|---------|------|-------------|
| batter_hr_7g | Rolling | Hit rate, last 7 game-dates |
| batter_hr_30g | Rolling | Hit rate, last 30 game-dates |
| batter_hr_60g | Rolling | Hit rate, last 60 game-dates |
| batter_hr_120g | Rolling | Hit rate, last 120 game-dates |
| batter_whiff_60g | Rolling | Whiff rate (swinging strikes / swings) |
| batter_count_tendency_30g | Rolling | Avg (balls - strikes) at PA end |
| batter_gb_hit_rate | Expanding | Ground ball hit rate (speed proxy) |
| platoon_hr | Expanding | Hit rate by batter × pitcher handedness |
| pitcher_hr_30g | Rolling | Pitcher's hit rate allowed |
| pitcher_entropy_30g | Rolling | Shannon entropy of pitch type distribution |
| weather_temp | Context | Game temperature from feed |
| park_factor | Expanding | Venue hit rate / league avg (expanding normalization) |
| days_rest | Context | Days since batter's last game |

### Dropped features (tested and rejected)
- **lineup_position**: Double-counts with PA aggregation (helps with leaky features, hurts or neutral with clean)
- **is_home**: Noise at PA level
- **pitcher_cluster / batter_vs_arch_hr**: K-Means clustering was 90.8% unstable across train/test
- **umpire zone tendency**: Zero predictive power (+0.005 correlation)
- **exit velocity, launch angle trends**: Redundant with rolling hit rate
- **wind, career PAs, day of week, month**: All noise

## Model

- **Algorithm**: LightGBM (default hyperparameters — robust to tuning)
- **Training**: PA-level binary classification (hit / no-hit)
- **MLP ensemble**: Tested, no improvement — trees handle our interaction features better
- **Calibration**: Underconfident at top (predicts 82%, actual 90%), but calibration methods hurt P@K

## Evaluation

- **Primary**: Walk-forward backtesting (retrain every 14 days)
- **Metrics**: Precision@K at game level, streak simulation
- **Nuclear leakage test**: Manual from-scratch feature computation for random test PAs
- **Multi-season**: Validated across 6 test seasons (2020-2025)

## Pipeline

```
bts data pull --start 2019-03-20 --end 2025-10-01    # Raw JSON from MLB API
bts data build --seasons 2019,...,2025                 # PA-level Parquet
compute_all_features(df)                               # 13 temporal features
walk_forward_evaluate(df, test_season=2025)            # Walk-forward P@K
```

## Key Learnings

1. **PA-level > game-level**: 2x more training data, lineup position emerges naturally
2. **Leakage is invisible**: Three separate leakage bugs found and fixed (static features, K-Means clusters, doubleheader shift). Each looked correct until tested.
3. **Feature selection is fragile**: Results flip when leakage is present vs absent. Always validate on held-out season.
4. **More data helps, to a point**: 2019+ is optimal. 2017-18 hurts. Expanding features need volume but the model needs relevance.
5. **YAGNI applies to ML**: 13 features beat 18. Simpler models with clean features beat complex models with noisy ones.
