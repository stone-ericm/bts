# BTS Architecture

## Overview

Two-stage hit prediction model for MLB's Beat the Streak contest.
Stage 1 predicts P(hit) per plate appearance using LightGBM.
Stage 2 aggregates to P(>=1 hit) per game via probability math.

Validated results (walk-forward, provably leak-free):
- Single model: P@1=87% (2025), 83% (2024)
- 12-model blend: P@1=87.5% (2025), 84.9% (2024), avg 86.2%
- Tested across 6 seasons (2020-2025): P@1 82-91%, P@500 beats SOTA every year
- SOTA benchmark: Garnett (2026), P@100=85%, P@500=77%

## Data

- **Source**: MLB Stats API v1.1 (`/api/v1.1/game/{gamePk}/feed/live`)
- **Scope**: 9 seasons (2017-2025), 1.5M plate appearances
- **Training window**: 2019 onward (2017-18 hurts — game changed too much)
- **Filters**: Regular season only (no spring training, postseason, exhibitions, 7-inning COVID doubleheaders)
- **Storage**: Raw JSON (`data/raw/{season}/{gamePk}.json`) → PA Parquet (`data/processed/pa_{season}.parquet`)

## Features (15, provably leak-free)

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
| pitcher_catcher_framing | Expanding | Catcher framing proxy (expanding strike rate) |
| opp_bullpen_hr_30g | Rolling | Opposing team's reliever hit rate (30-day, via probable pitcher ID) |
| days_rest | Context | Days since batter's last game |

### Context features (4, shadow model — CONTEXT_COLS)

Always computed by `compute_all_features()` but only used by the shadow model (via `feature_cols_override` param). After 30-day evaluation, may graduate to FEATURE_COLS.

| Feature | Type | Description |
|---------|------|-------------|
| ump_hr_30g | Rolling | Hit rate per HP umpire, 30-day window |
| wind_out_cf | Context | Signed wind vector (direction score × speed) |
| batter_hard_contact_30g | Rolling | Hard-contact rate from categorical hardness column |
| is_indoor | Context | Binary: dome/closed/retractable roof |

Shadow picks saved to `{date}.shadow.json`. Report: `bts shadow-report`.

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
- **12-model blend**: Each model uses baseline 15 features + one Statcast feature variant. Predictions averaged across models for ranking. Improves P@1 by better tie-breaking between near-equivalent top picks.
- **Blend validated**: Window size (7-60d) doesn't matter. 12 models is the sweet spot — fewer loses diversity, more dilutes signal. Different architectures (DT, LR) hurt.
- **MLP ensemble**: Tested, no improvement — trees handle our interaction features better
- **Calibration**: Underconfident at top (predicts 82%, actual 90%), but calibration methods hurt P@K

### Statcast features (9, used by blend variants)

Extracted from game feed pitchData and hitData. Each appears in one blend model variant alongside the baseline 13.

| Feature | Type | Description |
|---------|------|-------------|
| batter_barrel_rate_30g | Rolling | Barrel rate (EV≥98 + sweet LA) — stabilizes at ~50 BIP |
| batter_hard_hit_rate_30g | Rolling | Hard hit rate (EV≥95) |
| batter_sweet_spot_rate_30g | Rolling | Sweet spot LA (8-32°) |
| batter_avg_ev_30g | Rolling | Average exit velocity |
| pitcher_avg_velo_30g | Rolling | Average pitch velocity |
| pitcher_avg_spin_30g | Rolling | Average spin rate |
| pitcher_avg_extension_30g | Rolling | Average release extension |
| pitcher_break_total_30g | Rolling | Mean total break magnitude |
| batter_avg_velo_faced_30g | Rolling | Average pitch velocity faced |

### Rejected features and approaches (2026-03-29)

Tested and rejected after empirical validation:
- **MiLB debut pitcher entropy**: No P@1 improvement (LightGBM handles missing values well)
- **Team defense (BABIP)**: 30-day window too noisy (r=0.19). Prior-season signal was park effects (road-only r=0.12).
- **Granular defense (GB/FB splits, error rate, hard-hit conversion)**: All noise or park effects.
- **Hyperparameter tuning, recency weighting, ranking objective**: No consistent improvement.
- **Adaptive feature selection**: Worse than fixed blend.
- **15+ model blend**: Dilutes signal — 12 is optimal.

## Evaluation

- **Primary**: Walk-forward backtesting (retrain every 14 days)
- **Metrics**: Precision@K at game level, streak simulation
- **Nuclear leakage test**: Manual from-scratch feature computation for random test PAs
- **Multi-season**: Validated across 6 test seasons (2020-2025)

## Orchestration

Hetzner VPS (CPX42, Helsinki) runs scheduler, dashboard, and cron via systemd.

```
┌──────────────────────────────────────────────────┐
│  Hetzner CPX42 (8 vCPU, 16 GB, Helsinki)         │
│  systemd services:                               │
│  - bts-scheduler.service (Restart=always)        │
│  - bts-dashboard.service (port 3003, Tailscale)  │
│  - crontab (check-results, reconcile, data       │
│    refresh, preview pick, lineup collection,     │
│    healthchecks ping)                            │
│                                                  │
│  Tailscale: bts-hetzner (stable identity)        │
│  Deploy: GHA SSH → git pull + systemctl restart  │
│  Backup: R2 bucket bts-backup-data               │
└──────────────────────────────────────────────────┘
```

**Daily lifecycle (scheduler daemon, `Restart=always` + `RestartSec=300`):**
- The scheduler is a single-day daemon: handles one day's games then exits cleanly. Systemd restarts it after 5 min, and it picks up the next day's schedule via `datetime.now(UTC)`.
- Morning init: loads game schedule for the day, plans lineup-check windows
- `game_time - 45min`: runs full prediction cascade at each check (no skip optimization — pipeline determines projected vs confirmed per-batter)
- Short-circuit: if existing pick is already locked (game started or posted to Bluesky), skips the expensive SSH cascade entirely
- `early_lock_gap`: once confirmed lineups are available, posts picks to Bluesky (confirmation-based, not time-based). Gap check excludes batters from non-Preview games (started, finished, postponed).
- Logging: each check logs the selected pick, probability, should_lock decision, and gap vs best projected pick. Pick name/probability recorded in `scheduler_state.json` for audit trail.
- Result polling: starts `game_start + 10min`, checks boxscore every 15min. Posts reply (✅/❌ + streak) as soon as all picks have hits (mid-game early exit) or game goes Final.
- `bts reconcile`: 8-day lookback for scoring changes (hit overturned to error). Recalculates streak from scratch if corrections found. Cron at 2am ET.
- 1am cron remains as a safety-net `bts check-results` fallback. Skips when scheduler has already set result ("hit"/"miss") to avoid double-counting streak. Does NOT post to Bluesky; scheduler owns all posting.

**Key modules:**
- `strategy.py` — MDP-optimal pick logic with heuristic fallback. Auto-loads `data/models/mdp_policy.npz` for provably optimal skip/single/double decisions based on (streak, days_remaining, saver, quality_bin). Double-down must be from a different game. Falls back to heuristic thresholds if policy file absent. Shared by `bts run` and orchestrator.
- `orchestrator.py` — Local prediction (`predict_local`) + shadow prediction (`predict_local_shadow`). TOML config, calls strategy + posting. Shadow uses `feature_cols_override` with separate model cache (`blend_{date}_shadow.pkl`).
- `scheduler.py` — Long-running daemon. Dynamic lineup checks at `game_time - 45min`. Short-circuits when pick is locked. Shadow model runs after production lock (`_run_shadow_prediction`). `bts schedule` CLI command.
- `dm.py` — Bluesky DM notifications on total cascade failure. Uses `api.bsky.chat` directly (not PDS proxy).
- `predict-json` — worker command: runs pipeline, outputs JSON to stdout, logs to stderr.

**Config:** `~/.bts-orchestrator.toml` on Hetzner. `shadow_mode = false` (Bluesky posting live), `shadow_model = true` (context stack runs alongside production for the 30-day eval). Tiers: local only.

**LightGBM is optional:** `uv sync` (Pi5, pick logic only) vs `uv sync --extra model` (workers, full prediction).

## Strategy Simulation

Monte Carlo simulator and MDP solver for evaluating and optimizing play strategies.

```
src/bts/simulate/
    strategies.py       — Strategy dataclass, 7 named profiles, streak-aware thresholds
    monte_carlo.py      — simulate_season(), run_monte_carlo(), load/save profiles
    backtest_blend.py   — 12-model blend walk-forward, saves daily profiles
    quality_bins.py     — equal-frequency quintiles with empirical P(hit), P(both)
    exact.py            — absorbing Markov chain for exact P(57) (no Monte Carlo noise)
    mdp.py              — reachability MDP solver, 103K states, backward induction
    cli.py              — bts simulate {backtest, run, solve, exact}
```

**MDP-optimal strategy (P(57) = 8.91%, phase-aware, different-game doubles):**
- Phase-aware bins: early season (Mar-Aug) vs late (Sep only, `late_phase_days=30`)
- **Different-game doubles**: double-down must be from a different game_pk than primary pick (avoids correlated outcomes — 39.7% of days had same-game doubles). +59% P(57) vs same-game.
- **No densest bucket**: pure blend ranking, no time-window filtering (removed 2026-04-08, was hurting P(57) by 8%)
- At low streaks with many days left: play aggressively, double everything (even Q1)
- At high streaks: skip all but Q4-Q5, stop doubling at streak 46+
- Streak saver tracked: consumed on first miss at streak 10-15

**Backtest profiles:** `data/simulation/backtest_{season}.parquet` (2021-2025, 912 daily profiles). Generated by `bts simulate backtest --seasons 2021,...`.

**Policy file:** `data/models/mdp_policy.npz` (1.8KB). Generated by `bts simulate solve --save-policy`.

## Validation

Multi-metric scorecard for benchmarking model and strategy changes.

```
src/bts/validate/
    scorecard.py         — P@K, miss analysis, calibration, streak metrics, full scorecard assembly
```

CLI: `bts validate scorecard [--save path] [--diff baseline.json]`

Baseline scorecard at `data/validation/scorecard_baseline.json` (2026-04-02).
Investigation scripts in `scripts/validation/`, verdict docs in `docs/validation/`.

## Dashboard

LAN-only web dashboard at `http://pi5:3003`. Single-file Python server using `http.server` (no framework). Serves MLB-themed HTML with inline CSS.

**Key modules:**
- `web.py` — HTTP handler, page rendering, live scorecard HTML, `/api/live` and `/api/live-html` endpoints
- `scorecard.py` — Data extraction from MLB game feed for live scorecard (44 tests)

**Live scorecard (during games):**
- Caught-looking style: pitch grids, SVG diamond with baserunning, trajectory lines
- Shows only picked batters' plate appearances
- `/api/live-html` returns server-rendered HTML fragment; JS polls every 30s and swaps `outerHTML` (no page flash)
- Handles different-game double-downs via `merge_scorecards`
- In-progress PA: pulsing amber border with current pitch count
- Green tint only on hits (single/double/triple/HR), not walks/HBP
- Sticky batter columns (#/name/POS) on horizontal scroll for 7+ PA games

**Lifecycle:** Scorecard appears when game is Live, stays through Final, hidden pre-game.

## Pipeline

```
bts data pull --start 2019-03-20 --end 2025-10-01    # Raw JSON from MLB API
bts data build --seasons 2019,...,2025                 # PA-level Parquet
compute_all_features(df)                               # 15 temporal features
walk_forward_evaluate(df, test_season=2025)            # Walk-forward P@K
```

## Key Learnings

1. **PA-level >> game-level**: Game-level modeling collapses to ~75% P@1 (tested 2026-04-07). PA-level works because the aggregation `1-prod(1-p)` is a better probability model than LightGBM learning P(game_hit) directly, even though all features are date-level.
2. **Leakage is invisible**: Three separate leakage bugs found and fixed (static features, K-Means clusters, doubleheader shift). Each looked correct until tested.
3. **Feature selection is fragile**: Results flip when leakage is present vs absent. Always validate on held-out season.
4. **More data helps, to a point**: 2019+ is optimal. 2017-18 hurts. Expanding features need volume but the model needs relevance.
5. **YAGNI applies to ML**: 13 features beat 18. Simpler models with clean features beat complex models with noisy ones.
6. **Blend diversity > model complexity**: 12 LightGBM variants with different feature subsets beat any single model, hyperparameter tuning, different architectures, or adaptive selection. The power is in tie-breaking via diverse votes, not in individual model quality.
7. **Year-to-year instability is fundamental**: Features that help one season hurt the next. Only the blend consistently improves both test seasons.
8. **Strategy >> model improvements for P(57)**: MDP-optimal play strategy improved P(57) from 0.90% to 8.91% (9.9x) with minimal model changes. Skip bad days, double selectively (different-game only), adapt to days remaining. The exponential nature of streaks (p^57) means small accuracy gains from strategy compound massively.
9. **Same-game doubles hurt P(57)**: rank-1 and rank-2 in the same game have correlated outcomes (same pitcher). Forcing different-game doubles improved P(57) by +59%. 39.7% of days had same-game doubles before the fix.
10. **Quick eval overstates improvements**: Static train-test (train once, predict all days) consistently shows larger gains than walk-forward (retrain every 7 days). Min_periods sweep: +2.7% quick → +0.8% walk-forward. Always validate with walk-forward before shipping.
11. **Bullpen composite polarizes quality bins**: Adding opp_bullpen_hr_30g slightly hurt average P@1 (-0.2%) but improved P(57) by +18% because Q5 (best days) jumped from 89.6% → 92.3%. The MDP exploits stronger peaks through more confident doubling.
12. **Model degrades in September specifically**: Sept P@1 drops to 83.1% vs Aug 85.2%. Phase-aware bins (Sept-only late phase, `late_phase_days=30`) capture this, adding +1.8% P(57).
13. **Competitive validation (2026-04-02)**: 14 items tested against r/beatthestreak community. PA aggregation makes lineup position redundant. Vegas implied run totals add no signal. Miss days are random. Our streak distribution beats community's best model by 14-21%.
14. **Any function receiving the full prediction DataFrame must filter by game status**: Predictions include batters from all scheduled games (started, postponed, etc). Functions like `should_lock` that compare against projected picks must exclude non-Preview games, or postponed/finished games will pollute the comparison.
