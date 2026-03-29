# Beat the Streak v2 — Design Spec

## Goal

Build a hit prediction model that beats the current state of the art (85% Precision@100) on backtested MLB data, using a novel two-stage plate-appearance-level architecture.

This is a successor to [Counting to 57](https://github.com/stone-ericm/MLB_counting_to_57_hits) (2021), which achieved ~71.4% daily accuracy using game-level Decision Trees on Statcast data. The v2 story is one of advancement: better architecture, better features, better evaluation, informed by what the field has built in the intervening years.

## Context: State of the Art

| Project | Year | Approach | P@100 | Key Innovation |
|---------|------|----------|-------|----------------|
| Alceo & Henriques | 2020 | MLP | 85% | Academic benchmark, 155K samples |
| Garnett | 2026 | MLP+LightGBM ensemble | 85% | Matches Alceo, degrades slower (P@500=77%) |
| Globerman | 2023 | Logistic Regression | 82% | Generalized > player-specific models |

All existing approaches model at the **batter-game level** — one prediction per batter per day. They aggregate pitch-level data into game-level features, losing sequence and PA-level granularity.

Garnett's key finding: **lineup position is the single most important feature** (3.8x more important than #2). This is suspicious — it suggests lineup position proxies for "more plate appearances = more chances" rather than capturing intrinsic batter quality.

## Thesis: Two-Stage PA-Level Architecture

### Core Insight

The game-level question decomposes naturally:

```
P(>=1 hit in game) = 1 - PROD(P(no hit | PA_i))
```

If we model hit probability at the PA level and aggregate, we get several compounding advantages:

1. **Training data multiplies.** ~282K batter-game records become ~1M+ PA records from the same data.
2. **Lineup position stops being magic.** The PA model doesn't need it — the game model just aggregates more PAs for leadoff hitters. The effect emerges naturally.
3. **All feature edges plug in at the right level.** Umpire zone tendency, pitcher arsenal, count effects — these are PA-level phenomena, not game-level.
4. **Mid-game matchup changes are captured.** Facing an ace for 3 PAs then a bad reliever for 2 PAs is different from averaging across the whole game.

### Potential Edges

Validated via quick exploratory analysis (2025 season sample, 150+ games):

- **Pitch-sequence features (HIGH):** Count state has a 20+ point effect on hit rate (35.3% at 1-0 vs 14.5% at 0-2). Pitch-level features (chase rate, whiff rate, zone contact%) may add predictive power beyond simple batting averages. No BTS project models at this granularity.
- **Better engineering (MEDIUM):** Rolling windows at multiple scales, strict temporal leakage prevention, Precision@K evaluation. Table stakes to match SOTA.
- **Umpire zone tendency (SMALL):** Umpire called-strike rates vary ~3.6%. The transmission mechanism (zone -> counts -> hit rate) adds ~0.2-0.4% marginal improvement. Worth including as a feature but not the central thesis.
- **Air density (SMALL):** Free from Open-Meteo + venue coordinates. Physics says ~7% of weather effect beyond temperature (which is already in the game feed). Let EDA decide if it survives feature selection.

## Project Structure

```
bts/
  src/bts/
    __init__.py
    cli.py                # Click CLI entrypoint
    data/
      pull.py             # MLB API game feed downloader
      build.py            # Raw JSON -> PA-level Parquet
      schema.py           # Column definitions, dtypes
    features/
      batter.py           # Rolling batter stats, platoon splits, sprint speed
      pitcher.py          # Pitcher archetypes, arsenal features
      context.py          # Umpire, park factors, lineup position, weather, air density
      registry.py         # Feature registration and selection
    model/
      train.py            # PA-level model training
      predict.py          # Daily prediction generation
      aggregate.py        # PA predictions -> game-level P(>=1 hit)
      ensemble.py         # LightGBM + MLP combination (later)
    evaluate/
      backtest.py         # Walk-forward validation engine
      metrics.py          # P@K, calibration, streak simulation
    eda/
      reports.py          # Feature importance, correlation analysis
  data/
    raw/                  # Game feed JSON files (raw/{season}/{gamePk}.json)
    processed/            # PA-level Parquet files (pa_{season}.parquet)
    models/               # Trained model artifacts ({experiment_name}/)
  tests/
  pyproject.toml
  ARCHITECTURE.md
```

**Stack:** Python 3.12+, `uv` for packaging, `click` for CLI, LightGBM + scikit-learn for modeling, pandas + pyarrow for data.

## CLI Commands

```
bts data pull --start 2023-03-30 --end 2025-10-01
bts data pull --enrich-weather        # Open-Meteo atmospheric data
bts data build [--seasons 2023,2024,2025]
bts eda features                      # Feature importance report
bts train --model lgbm [--config experiments/baseline.toml]
bts predict --date 2025-09-15         # Who to pick today
bts evaluate --method walk-forward --test-season 2025 --metric p@100
bts evaluate compare exp1 exp2        # Side-by-side experiment comparison
```

## Data Pipeline

### Stage 1: `bts data pull`

Downloads game feeds from the MLB Stats API and stores as raw JSON.

- **Endpoint:** `/api/v1.1/game/{gamePk}/feed/live` (must use v1.1, not v1)
- **Discovery:** `/api/v1/schedule?sportId=1&date={date}` to find game PKs
- **Storage:** `data/raw/{season}/{gamePk}.json` — one file per game
- **Incremental:** Skips games already downloaded. Resumes cleanly after interruption.
- **Rate limiting:** Polite delays between requests. Full 2023-2025 pull is ~7,200 games.
- **Weather enrichment:** Optional `--enrich-weather` flag pulls historical atmospheric data (pressure, humidity, dew point) from Open-Meteo's free API using venue coordinates and game datetime. Stored as `data/raw/{season}/{gamePk}_weather.json`.

### Stage 2: `bts data build`

Transforms raw game feeds into PA-level feature tables stored as Parquet.

Each row is one plate appearance:

| Column | Source | Type | Example |
|--------|--------|------|---------|
| `game_pk` | game feed | int | 823651 |
| `date` | game feed | date | 2026-03-28 |
| `season` | derived | int | 2026 |
| `batter_id` | matchup | int | 682626 |
| `pitcher_id` | matchup | int | 592662 |
| `lineup_position` | boxscore | int | 3 |
| `is_home` | about | bool | True |
| `hp_umpire_id` | officials | int | 427215 |
| `venue_id` | gameData | int | 3289 |
| `pitch_count` | playEvents | int | 5 |
| `pitch_types` | playEvents | list[str] | [FF,SL,CH,FF,SL] |
| `pitch_calls` | playEvents | list[str] | [B,C,B,F,X] |
| `pitch_px` | pitchData | list[float] | [-0.3,0.6,...] |
| `pitch_pz` | pitchData | list[float] | [2.4,1.8,...] |
| `sz_top` | pitchData | float | 3.32 |
| `sz_bottom` | pitchData | float | 1.68 |
| `final_count_balls` | count | int | 2 |
| `final_count_strikes` | count | int | 1 |
| `launch_speed` | hitData | float | 98.3 |
| `launch_angle` | hitData | float | 22.1 |
| `event_type` | result | str | single |
| `is_hit` | derived | int | 1 |
| `weather_temp` | gameData.weather | int | 46 |
| `weather_wind_speed` | parsed | int | 9 |
| `weather_wind_dir` | parsed | str | In From LF |
| `roof_type` | venue.fieldInfo | str | Open |
| `atm_pressure` | Open-Meteo | float | 1013.2 |
| `humidity` | Open-Meteo | float | 65.0 |

Pitch sequence columns stored as nested arrays in Parquet. Raw pitch data is preserved for EDA to determine optimal aggregations.

Rolling stats, pitcher archetypes, and umpire tendencies are NOT computed at build time — they're computed in the feature engineering step at training time, keeping the build stage deterministic and fast.

## Feature Engineering

All features computed from the PA table at training time. All rolling features use strict `shift(1)` — only data available before the game in question.

### Batter Features (`features/batter.py`)

- **Rolling hit rates:** 7/14/30/60/120 game windows
- **Plate discipline:** K rate, BB rate, chase rate, whiff rate (from pitch sequences)
- **Batted ball quality:** rolling avg launch angle, avg exit velocity, xBA
- **Platoon splits:** H/PA vs RHP, H/PA vs LHP over rolling windows
- **Sprint speed:** seasonal metric. Predicts infield hit probability on weak/topped contact. Source: Baseball Savant sprint speed leaderboard (CSV/API), not available in individual game feeds. Pulled once per season and joined by batter ID.

### Pitcher Features (`features/pitcher.py`)

- **Arsenal profile:** pitch type distribution (% FF, SL, CH, CU, etc.)
- **Zone tendency:** called strike rate, in-zone%, chase-induced rate
- **Rolling effectiveness:** K/9, BB/9, H/9 over recent starts
- **Archetype clustering:** K-Means on arsenal + zone tendency. Maps each pitcher to one of ~8 archetypes. Solves the sparse H2H problem: "how does this batter hit against this *type* of pitcher" rather than this specific pitcher.

### Context Features (`features/context.py`)

- **Lineup position:** 1-9, or pinch hitter flag
- **Home/away**
- **HP umpire tendency:** rolling called strike rate, zone size tendency
- **Park factor:** derived from historical hit rates at venue
- **Weather:** temperature (from feed), wind speed/direction (from feed)
- **Air density:** computed from temperature, atmospheric pressure, humidity, and venue altitude. Single derived feature capturing the physically meaningful variable for ball flight.
- **Dome flag:** zeroes out weather effects for closed-roof venues

### Feature Registry (`features/registry.py`)

Each feature function registered with name and description. `bts eda features` uses the registry to run importance analysis. Experiment configs (TOML) specify which feature subset to use.

## Model Architecture

### Stage 1: PA-Level Model

- **Input:** One row per PA with all engineered features
- **Target:** Binary — did this PA result in a hit? (1/0)
- **Primary model:** LightGBM classifier
- **Secondary model:** MLP (added later for ensemble)
- **Output:** Calibrated `P(hit | this PA)`
- **Training data:** ~1M+ PAs from 2023-2025

### Stage 2: Game-Level Aggregation

Probability math, not a trained model:

```
P(>=1 hit in game) = 1 - PROD_i(1 - P(hit | PA_i))
```

For **prediction** (pre-game):
- Estimate number of PAs from lineup position (leadoff ~ 4.5, 9-hole ~ 3.8)
- Starting pitcher is known. For later PAs, use team bullpen archetype distribution.
- Multiply out estimated PA-level probabilities.

For **backtesting** (post-game):
- Use actual PAs to validate PA-level model directly
- Separately test the pre-game estimation assumptions

### Model Progression

1. LightGBM baseline with validated features
2. Hyperparameter tuning
3. Calibration tuning (critical — bad calibration compounds across PAs)
4. Add MLP, test ensemble
5. Final model selection based on walk-forward P@100

Literature note: the GBT vs neural net debate is overemphasized for tabular data (McElfresh et al., NeurIPS 2023). The BTS-specific evidence (Garnett, Alceo) slightly favors MLPs. An ensemble of both is the safe bet. But the two-stage architecture and features will matter more than model choice.

## Evaluation & Backtesting

### Primary Metric: Precision@K

For each day in the test period, rank all batters by P(>=1 hit).

- **P@1:** Did our single best pick get a hit? (The actual BTS use case)
- **P@10:** Top 10 accuracy (depth of confidence)
- **P@100:** The benchmark metric (SOTA: 85%)
- **P@500:** Depth of signal (Garnett: 77%)

### Walk-Forward Backtesting

```
For each game day D in test season (2025):
    1. Train PA-level model on all data before D
    2. For each batter with a game on day D:
       - Compute features using only pre-D data
       - Predict P(hit | each expected PA)
       - Aggregate to P(>=1 hit in game)
    3. Rank all batters by predicted probability
    4. Record whether top-K picks actually got hits
```

Daily retraining is cheap with LightGBM (seconds per fit). `shift(1)` on rolling features is the inner defense against leakage; walk-forward is the outer defense.

### Secondary Metrics

- **Calibration curve:** Do predictions of 75% result in hits 75% of the time? Critical for PA->game aggregation.
- **Seasonal holdout on 2024:** Train on 2023, test on 2024. Confirms cross-season generalization.
- **Streak simulation:** Given the model's daily P@1, simulate 10,000 seasons. What's the longest streak distribution?

### Experiment Tracking

Each run saves to `data/models/{experiment_name}/`:
- Model artifact
- Feature list
- Config TOML
- Evaluation metrics JSON

Comparison via `bts evaluate compare exp1 exp2`.

## Data Scope

- **Primary:** 2023-2025 (post-shift-ban era, most representative of current game)
- **Backfill:** 2022 tested empirically during Milestone 4. If it helps, include. If it hurts (distribution shift from shift ban), exclude.
- **2026 ABS data:** Not used for training (too few games), but the ABS exploration motivated the umpire features and is part of the writeup narrative.

## EDA Phase Questions

The formal EDA phase answers these before committing to a feature set:

1. Do pitch-sequence-derived features (chase rate, whiff rate, zone contact%) add power over simple rolling averages?
2. Does pitcher archetype clustering outperform individual pitcher stats?
3. Does umpire called-strike tendency add signal beyond noise?
4. What's the optimal rolling window set?
5. Does sprint speed meaningfully predict singles for fast vs slow batters?
6. Does air density pull its weight after park factor is already in the model?
7. Does including 2022 data help or hurt?

EDA produces a report, not a model. The report locks the feature set for training.

## Milestones

### Milestone 1: Data Foundation
- Scaffold project (uv, CLI, tests)
- `bts data pull` — MLB API downloader with incremental resume
- `bts data build` — raw JSON -> PA-level Parquet
- Open-Meteo weather enrichment
- Pull 2023-2025 data
- **Gate:** Correct PA Parquet with hit labels, pitch sequences, lineup positions, umpire IDs, weather

### Milestone 2: Formal EDA
- Implement feature engineering pipeline
- Run EDA analysis on all candidate features
- Answer the 7 EDA questions
- **Gate:** Written EDA report. Feature set locked.

### Milestone 3: Baseline Model
- Train PA-level LightGBM with validated features
- Build game-level aggregation
- Walk-forward backtesting on 2025
- **Gate:** P@100 number to compare against 85% SOTA

### Milestone 4: Optimization
- Feature selection, hyperparameter tuning
- Add MLP, test ensemble
- Calibration tuning
- Backfill 2022 data, measure impact
- Seasonal holdout on 2024
- **Gate:** Final P@100. Streak simulation.

### Milestone 5: Writeup
- Data scientist evolution narrative: v1 (2021) -> field progress -> v2 (2026)
- Methodology, results, what worked/didn't
- ABS exploration as a side story

Each milestone is independently valuable.

## Hardware

Mac only. LightGBM on ~1M rows of tabular data trains in seconds on Apple Silicon. Walk-forward over a full season takes ~30 minutes total. The bottleneck is API data pulls (network I/O), not compute. No GPU needed.

## Key References

- Garnett (2026), "Chasing $5.6 Million with Machine Learning" — current SOTA, 85% P@100
- Alceo & Henriques (2020), "Beat the Streak: Prediction of MLB Base Hits Using ML" — academic benchmark
- Grinsztajn et al. (NeurIPS 2022), "Why do tree-based models still outperform deep learning on tabular data?"
- McElfresh et al. (NeurIPS 2023), "When Do Neural Nets Outperform Boosted Trees on Tabular Data?"
- Nathan, "Physics of Baseball" — ball flight physics, air density effects
- MLB Stats API v1.1 — primary data source, includes ABS challenge data in 2026
- Open-Meteo Historical Weather API — free atmospheric data (pressure, humidity, dew point)
- Baseball Savant Sprint Speed Leaderboard — seasonal sprint speed by player

## Notes

- `data/` directory should be gitignored (raw JSON + Parquet are large). Only code, configs, and the spec are committed.
- The game feed's `hitData` object (launch speed, launch angle) is available on batted ball events within `playEvents`. Not all PAs have batted ball data (strikeouts, walks don't).
- Pitcher archetype count (~8) is approximate. Optimal K determined during EDA via silhouette score or similar.
