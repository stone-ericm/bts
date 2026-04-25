# BTS Project Instructions

## Quick Start
```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra model   # Mac/Alienware (full)
UV_CACHE_DIR=/tmp/uv-cache uv sync                  # Pi5 (no LightGBM)  [Pi5 BTS DECOMMISSIONED 2026-04-14]
UV_CACHE_DIR=/tmp/uv-cache uv run bts run --date 2026-04-01 --dry-run
UV_CACHE_DIR=/tmp/uv-cache uv run pytest -v

# Scheduler (Hetzner production — systemd --user unit)
UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config ~/.bts-orchestrator.toml
UV_CACHE_DIR=/tmp/uv-cache uv run bts schedule --config ~/.bts-orchestrator.toml --dry-run

# Hetzner cron setup (reproducible install of cron jobs)
bash scripts/cron-setup-hetzner.sh show      # dry-run
bash scripts/cron-setup-hetzner.sh install   # install to bts user crontab
```

## Deployment
- **Deploys trigger on push to `deploy` branch** (NOT main, since 2026-04-21). Workflow: commit/push to main freely; when ready to ship, `git push origin main:deploy`. Gives explicit control over when scheduler restart fires (avoids disrupting live-game scorecard polling).
- **Canary + auto-rollback**: after deploy, workflow waits 30s then checks `systemctl is-active bts-scheduler bts-dashboard` + dashboard HTTP. On failure, auto-reverts to pre-deploy SHA + restarts services.
- **Emergency deploy**: `workflow_dispatch` trigger in the GitHub Actions UI — runs deploy without pushing anything.
- Every push to `deploy` triggers the workflow (no paths filter — `deploy` is a dedicated deploy ref, so any push to it is an intentional deploy). Workflow SSHes as root, runs `git pull --ff-only` + restarts `bts-scheduler` + `bts-dashboard` as user `bts`. **Don't manually `systemctl restart` on bts-mlb after pushing** — workflow does it.
- See `/Users/stone/.claude/projects/-Users-stone/memory/reference_bts_deploy_workflow.md` for full details.

## Feature computation env vars (set in production via `.env` or systemd unit; defaults in code)
- `BTS_ROOKIE_GATE_K` (default `20`): rookie shrinkage strength. Rookies (career PAs < 100) get PA-weighted rolling + pseudocount shrinkage toward 0.2195 league prior on `batter_hr_{30,60,120}g`. Veterans untouched. Set to `0` to revert.
- `BTS_PITCHER_HR_30G_MIN_PERIODS` (default `7`): rolling-window min_periods for `pitcher_hr_30g`. Shipped 2026-04-14 after +0.81pp avg P@1 + +0.46pp MDP P(57) on walk-forward (seed=42 measurement — pending multi-seed re-validation). Set to `10` to revert.
- `BTS_LGBM_RANDOM_STATE` (default `42`): LightGBM `random_state` for all 4 training paths (classifier, ranker, regressor, V-REx). Defaults to 42 for backward-compat with shipped policy. Used by multi-seed audit workflows to measure cross-seed variance without code edits. **IMPORTANT: seed=42 was discovered to be a statistical outlier on 2026-04-14 — all historical single-seed experiment deltas are suspect until multi-seed audit completes.**
- `BTS_LGBM_DETERMINISTIC` (default `0`, OFF): when `=1`, sets LightGBM `deterministic=True` + `force_row_wise=True` in `LGB_PARAMS` for bit-exact-reproducible training across providers. Required for cross-provider seed pooling (OCI E5.Flex showed -1.62pp 2024 P@1 drift vs Hetzner CPX51 on identical seed=42 without this flag — see `project_bts_oci_provider_add.md`). Code-prepped on main 2026-04-24 (`498c5a8`); flip in `.env` on bts-hetzner only after a deliberate re-baseline cycle since it slightly shifts P(57) by changing parallel-reduction order in gradient calculation. **Same flag also unblocks the AX102-U pivot** — Zen-4-to-Zen-4 doesn't save you from non-determinism on dedicated cores.

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

# Multi-metric scorecard (baseline comparison)
UV_CACHE_DIR=/tmp/uv-cache uv run bts validate scorecard --diff data/validation/scorecard_baseline.json
```

## Architecture
See `ARCHITECTURE.md` for full details. Key points:
- PA-level LightGBM → game-level probability aggregation
- 15 baseline features (FEATURE_COLS) + 4 shadow context features (CONTEXT_COLS) + 9 Statcast features, all provably leak-free (date-level shift(1))
- 12-model blend: baseline + single-Statcast variants. `--no-blend` for single model.
- **Shadow model**: `CONTEXT_COLS` (ump_hr, wind, hardness, indoor) run alongside production via `feature_cols_override`. Picks saved to `{date}.shadow.json`. Report: `bts shadow-report`
- **MDP-optimal strategy**: auto-loads `data/models/mdp_policy.npz` for skip/single/double decisions. Falls back to heuristic if absent.
- **Phase-aware bins**: early season (Mar-Aug) vs late (Sep only, `late_phase_days=30`)
- **Streak saver tracked**: `saver_available` in `streak.json`, consumed on first miss at streak 10-15
- **Scheduler daemon** (`scheduler.py`): replaces fixed 11am/4pm/7:30pm cron with dynamic game_time-45min lineup checks; confirmation-based posting via `early_lock_gap`; 1am cron kept as safety-net fallback
- **`private_mode`** config flag (renamed from `shadow_mode` 2026-04-12) — when `true`, picks save but never post to Bluesky. Don't confuse with `shadow_model` (runs the context stack model alongside production for eval).
- **Fallback deadline** uses `_earliest_pick_game_et(daily)` — earliest of primary + double-down game times, not primary alone. Fixes the case where double-down is in an earlier game than primary.
- Projected lineup fallback for morning predictions
- Train on 2019+ data (2017-18 hurts)
- Starter/reliever PA split in aggregation
- `notna().any()` not `.all()` — LightGBM handles NaN natively for Statcast features
