# BTS Project Instructions

## Quick Start
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model   # Mac/Alienware (full)
UV_CACHE_DIR=/tmp/uv-cache uv sync                  # Pi5 (no LightGBM)
UV_CACHE_DIR=/tmp/uv-cache uv run bts run --date 2026-04-01 --dry-run
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v
```

## Required Prefixes
- All `uv` commands: `UV_CACHE_DIR=/tmp/uv-cache`
- LightGBM needs ARM libomp: `arch -arm64 /opt/homebrew/bin/brew install libomp`
- LightGBM is an optional dep: `uv sync --extra model` to install it

## Safety Rules
- **Never use features computed on full dataset** without shift(1) temporal guard
- **Never trust feature importance or ablation results** if there's any chance of leakage — fix leakage first, then re-evaluate
- **K-Means clustering is NOT safe** for features — cluster assignments are 90.8% unstable across train/test splits
- Run `scripts/leakage_audit.py` and the nuclear test after any feature changes

## Data
- Raw JSON: `data/raw/{season}/{gamePk}.json` (gitignored, ~15GB)
- MiLB raw: `data/raw/milb/{season}/{gamePk}.json` (6,643 games, 2023-2025. Pitch types only available 2023+)
- Processed Parquet: `data/processed/pa_{season}.parquet` (gitignored)
- Regular season only. 7-inning COVID doubleheaders dropped.
- MLB API requires v1.1 (`/api/v1.1/game/{pk}/feed/live`), not v1.

## Strategy Simulation
```bash
# Run blend backtest (5 seasons, ~2-3 hours, needs --extra model)
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate backtest --seasons 2021,2022,2023,2024,2025

# Monte Carlo strategy comparison
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate run --trials 10000

# MDP solver — generates optimal policy file
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate solve --save-policy data/models/mdp_policy.npz

# Exact P(57) for any strategy (no Monte Carlo noise)
UV_CACHE_DIR=/tmp/uv-cache uv run bts simulate exact --strategy combined
```

## Architecture
See `ARCHITECTURE.md` for full details. Key points:
- PA-level LightGBM → game-level probability aggregation
- 13 baseline features + 9 Statcast features, all provably leak-free (date-level shift(1))
- 12-model blend: baseline + single-Statcast variants. `--no-blend` for single model.
- **MDP-optimal strategy**: auto-loads `data/models/mdp_policy.npz` for skip/single/double decisions. Falls back to heuristic if absent.
- **Phase-aware bins**: early season (Mar-Jul) vs late (Aug-Sep) — model degrades late
- **Streak saver tracked**: `saver_available` in `streak.json`, consumed on first miss at streak 10-15
- Projected lineup fallback for morning predictions
- Train on 2019+ data (2017-18 hurts)
- Starter/reliever PA split in aggregation
- `notna().any()` not `.all()` — LightGBM handles NaN natively for Statcast features
