# BTS Architecture

## Overview

Two-stage hit prediction model for MLB's Beat the Streak contest.
Stage 1 predicts P(hit) per plate appearance. Stage 2 aggregates to P(>=1 hit) per game.

## Pipeline

1. `bts data pull` — Downloads raw game feeds from MLB Stats API v1.1 to `data/raw/{season}/{gamePk}.json`
2. `bts data build` — Parses game feeds into PA-level Parquet at `data/processed/pa_{season}.parquet`
3. Feature engineering (Milestone 2+) — Computes rolling stats, pitcher archetypes, context features at training time
4. Training and evaluation (Milestone 3+) — PA-level LightGBM, game-level aggregation, walk-forward backtesting

## Key Design Decisions

- PA-level modeling (not game-level) — more training data, natural lineup position handling
- Raw pitch sequences preserved in Parquet nested arrays — EDA determines aggregations
- Features computed at training time, not baked into build — enables fast iteration
- Walk-forward validation with shift(1) — double defense against temporal leakage

## Data Flow

```
MLB Stats API → data/raw/{season}/{gamePk}.json
                         ↓ bts data build
              data/processed/pa_{season}.parquet
                         ↓ feature engineering
              Training DataFrame with all features
                         ↓ model training
              PA-level P(hit) predictions
                         ↓ aggregation
              Game-level P(>=1 hit) rankings
```

## Spec

See `docs/superpowers/specs/2026-03-29-bts-v2-design.md` for full design.
