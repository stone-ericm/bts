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

### Suspended-game scoring (`is_resumed_portion`, added 2026-06-30)

A suspended-then-resumed MLB game keeps its original `officialDate`, so ALL its PA (pre-suspension AND resumed) file under the earlier date. Per BTS rules the resumed/rescheduled portion is **never evaluated**. `parse_game_feed` flags each PA with `is_resumed_portion` = (per-play `about.startTime` >= the feed's `gameData.datetime.resumeDateTime`; missing startTime → resumed; always False for normal games — the whole branch is gated on `resumeDateTime`, so normal-game rows are byte-identical).

BTS/contest scoring **excludes** the resumed portion via `build.filter_out_resumed_portion` / `build.read_pa_for_bts_scoring` (reads the column only when present, so pre-enrichment parquets no-op → backward-compatible). Model training, feature computation, and the skill-pool `prior_pa`/`prior_hit_rate` lookup **keep** resumed PA (they are real baseball events). Consumers that grade pre-suspension only: the live-forward resolver (`experiment/artifacts.py`), `model/calibrate`, health `realized_calibration` + `slate_auc`, `scripts/canonicalize_realized_picks.build_day_hit_lookup`, `skip_policy_shadow`, and the production scorer (`picks.grade_pick_in_feed` used by `check_hit`; `scheduler._check_hits_midgame` mid-game). Grading: pre-suspension hit → hit; pre-suspension PA, no hit → miss; **zero** pre-suspension PA → void (`void_no_pa` in the resolver via `outcome_game_keys` derived from *unfiltered* PA; `"void"` in the streak via `check_hit(return_status=True)`).

Separately: the daily slate skips resume-date games entirely (`picks.is_resume_date_game`, `officialDate < date`), and the resolver terminal-voids a stranded resume-day candidate as `void_suspended_resume`. Adds `is_resumed_portion` to `PA_COLUMNS` (SCHEMA_VERSION bump — only `data/sync.py` checks it; the resolver reads the parquet directly). The active season self-heals nightly (`bts data build --seasons 2026` full re-parse); the 3am cron gates `sync-to-r2` on a successful build. See `docs/audit/2026-06-29-skip-threshold-and-discrimination.md`.

## Features (16, provably leak-free)

All features use date-level shift(1) — only data from dates strictly before the prediction date.
Verified by nuclear test: 260/260 manual spot checks passed.

| Feature | Type | Description |
|---------|------|-------------|
| batter_hr_7g | Rolling | Hit rate, last 7 game-dates. Never shrunk — captures recent form. |
| batter_hr_30g | Rolling + rookie shrinkage | Hit rate, last 30 game-dates. Rookies (career PAs < 100) get PA-weighted rolling + pseudocount shrinkage toward 0.2195 league prior, K=20 (env `BTS_ROOKIE_GATE_K`). Veterans unchanged. |
| batter_hr_60g | Rolling + rookie shrinkage | Same as 30g, 60-day window. |
| batter_hr_120g | Rolling + rookie shrinkage | Same as 30g, 120-day window. |
| batter_whiff_60g | Rolling | Whiff rate (swinging strikes / swings) |
| batter_count_tendency_30g | Rolling | Avg (balls - strikes) at PA end |
| batter_gb_hit_rate | Expanding | Ground ball hit rate (speed proxy) |
| platoon_hr | Expanding | Hit rate by batter × pitcher handedness |
| pitcher_hr_30g | Rolling | Pitcher's hit rate allowed, 30-day window, `min_periods=7` (env `BTS_PITCHER_HR_30G_MIN_PERIODS`, shipped 2026-04-14 from historical 10). Lets feature activate ~3 starts earlier in a pitcher's career. |
| pitcher_entropy_30g | Rolling | Shannon entropy of pitch type distribution |
| weather_temp | Context | Game temperature from feed |
| park_factor | Expanding | Venue hit rate / league avg (expanding normalization) |
| pitcher_catcher_framing | Expanding | Catcher framing proxy (expanding strike rate) |
| opp_bullpen_hr_30g | Rolling | Opposing team's reliever hit rate (30-day, via probable pitcher ID) |
| days_rest | Context | Days since batter's last game |
| batter_pitcher_shrunk_hr | Expanding × shrinkage | Bayesian-shrunk historical (batter, pitcher) hit rate. Promoted 2026-04-29 after Phase 1 t=+3.35, Phase 2 set-1 +2.77pp, set-2 +3.49pp. Aggregated per (batter, pitcher, date) for no within-day leakage; falls back to league prior 0.2195 (K=10). Inference path explicitly populates this from `lookups["batter_pitcher_hr"]` (predict.py); a missing-bpm bug shipped 2026-04-29 → fixed 2026-04-30 commit `ee4190f`. |

### Context features (5, shadow model — CONTEXT_COLS)

Always computed by `compute_all_features()` but only used by the shadow model (via `feature_cols_override` param). After 30-day evaluation, may graduate to FEATURE_COLS.

| Feature | Type | Description |
|---------|------|-------------|
| ump_hr_30g | Rolling | Hit rate per HP umpire, 30-day window |
| wind_out_cf | Context | Signed wind vector (direction score × speed) |
| batter_hard_contact_30g | Rolling | Hard-contact rate from categorical hardness column |
| is_indoor | Context | Binary: dome/closed/retractable roof |
| park_drag_delta | External as-of table | Park ball-drag regime state (venue rolling-15 Cd minus frozen early-season anchor, shrunk). Read from `data/external/park_drag/park_drag_export.csv` — one row per venue_id × calendar date computed from strictly-prior games, so training merge == serving lookup (`features/park_drag.py`: never-raise loader, mtime-invalidated cache, per-cycle `pinned()` snapshot, staleness→None). Produced daily by `bts park-drag-refresh` (cron 07:45 ET; `features/park_drag_producer.py` — Savant FF fetch w/ browser UA + 403/429 kill-switch); `park_drag_freshness` health source. Added 2026-07-08 (shadow v2 — SHADOW_MODEL_NAME bump, feature-hashed cache `blend_{date}_shadow_{hash}.pkl`, v1 history excluded). **2026 backtest screen = NULL** (`docs/audit/2026-07-08-park-drag-2026-screen.md`): do not promote on alpha grounds; value = regime observability (the expanding park_factor is frozen against mid-season ball changes). Spec: `docs/superpowers/specs/2026-07-07-park-drag-delta-context-feature.md`. |

Shadow picks saved to `{date}.shadow.json` (stamped `shadow_model_version` at creation; unstamped = legacy v1, excluded from v2 status/report/backfill). Report: `bts shadow-report`.

### Skip-policy shadow (counterfactual "pick-the-band" monitor — `skip_policy_shadow.py`)

A second, distinct shadow that is a shadow **policy**, not a shadow model (same predictions,
different action rule). It tests whether the deployed MDP's **skip at streak≥8 / top candidate
< 0.796** rule actually costs streaks on the production (estimated-PA) scale — a question
backtest cannot settle, because the calibrated breakeven (~0.744) sits inside the skipped band's
realized hit-rate range (see `docs/audit/2026-06-20-skip-policy-shadow.md`). **Ground truth via
`decision.json`:** the scheduler writes `data/picks/<date>/decision.json` (schema
`bts_daily_decision_v2` since 2026-08-09; v1 accepted for legacy files) at each true finalization point — pick commit (`_deliver_and_lock_pick`,
delivery branches → `delivered`/`private_locked`/`locked_unconfirmed`), classification-lock (only
when the recovered pick was genuinely delivered), crash-guard, and end-of-day MDP skip. The
scheduler tracks `committed_pick_written` + `final_skip_candidate` across the day; all writes are
best-effort and never affect the pick. The skip-policy shadow reads `decision.json` files: only
entries with `action=="skip" && source=="mdp"` produce a shadow record `{date}.policy_shadow.json`
(`bts_skip_policy_shadow_v1`) for the executable declined candidate; reconciliation fills the
realized outcome from the MLB API. A shadow record is pruned when the corresponding `decision.json`
is later overwritten to a delivered pick (`prune_superseded`). The status artifact
`data/validation/skip_policy_shadow_status.json` (schema v2, 2026-07-10) reports the skipped-band
realized hit rate + a running Wilson CI (MONITORING display only) and a verdict vs the 0.744
breakeven (`below_breakeven` = skip validated, `above_breakeven` = skip costs streaks, else
`straddles`/`insufficient_n`). **Verdict statistics (audit F10, 2026-07-10):** verdicts come ONLY
from pre-registered looks at n∈{30,60,90} resolved divergent days, each a Wilson test at
Bonferroni z=2.394 (0.05/3 two-sided) over the FIRST-c records in date order — deterministic, so
the stateless nightly rebuild replays identical looks; a decisive look is terminal
(`verdict_basis` in the status records which look fired). The old nightly-95%-CI re-test was not
time-uniform (peeking). Records carry a `(policy_npz_sha256, feature_env_hash)` regime
fingerprint for future stratification. Aged records whose decision.json no longer says
mdp-skip are EXCLUDED from the sample as void-equivalent reclassifications and reported
via `counts.aged_superseded_records` (nightly update prints the warning; round-3 F3);
same-day flips are pruned by `prune_superseded` (age-fenced to the eligibility window). The 0.744 derivation is versioned:
`scripts/audit/skip_breakeven_derivation.py` re-derives it from the estimated-PA profiles
(reach57 median p* 0.7418 / E[max] 0.7485 — `docs/audit/2026-07-10-skip-breakeven-derivation.json`).
CLI: `bts skip-policy-shadow-update` + `bts skip-policy-shadow-status`. Surfaced on the dashboard
("Skip-policy shadow" panel). Why `decision.json` (vs reconstructing read-only): see the design
doc — 4 review rounds showed the action + executable candidate aren't otherwise recoverable.

### `decision.json` — authoritative daily action record (`bts/daily_decision.py`)

`data/picks/<date>/decision.json` (schema `bts_daily_decision_v2` since 2026-08-09; readers accept
{v1, v2}) is the single source of truth for "what did production finally do on a date." Written
ONLY by the scheduler at true finalization points; never by `bts run`, preview, or the shadow
model. **v2 adds state provenance on EVERY record** — streak, saver_available, state_source,
state_status, allow_double, contest_source_date, stamped in `orchestrator.run_and_pick` from the
exact DecisionStreakState that fed `select_pick` — plus `second_candidate` on skips (the
executable different-game runner-up). Motivation: the 2026-08-09 boundary-shadow census found
31/44 v1 records state-null (only MDP skips persisted state), making retrospective state
recovery depend on a ledger-as-of join; v2 records are exact by construction. Always best-effort; never raises into
the pick path. **Who reads it:** `bts check-results` gates scoring on `decision.scoreable`
(fallback to `picks.pick_was_delivered` when no decision file exists) — the GH #144 fix that stops
a stale preview `<date>.json` from corrupting the streak on skip days; the skip-policy shadow reads
it to identify genuine MDP skips (`action=="skip" && source=="mdp"`). **When it is written:**
(1) pick commit at `_deliver_and_lock_pick` (`scoreable=True`; `delivery_status ∈ {delivered,
private_locked, locked_unconfirmed}`); (2) classification-lock — only when the existing pick was
genuinely delivered (non-delivered stale-preview locks write nothing); (3) crash-guard at abnormal
exit; (4) end-of-day MDP skip via `_write_endofday_skip` (`scoreable=False`,
`delivery_status="not_applicable"`), fired only when `committed_pick_written` is still False.

**Daemon-path completion (the full #144 fix, shipped 2026-06-22 — PR #145).** The check-results gate
alone left the streak corruptible via two pre-existing daemon paths on a skip day (the projected→real
flip: 3am preview writes a projected `<date>.json`, real lineups flip to a skip, leaving a stale file).
A shared `daily_decision.is_scoreable_commit(date, picks_dir, daily)` (= `decision.scoreable` if a
record exists, else `pick_was_delivered`) is now the ONE "is this a committed pick?" predicate, used by
check-results, result-polling, and the health checks. Closed paths: **(C1)** the fallback
(`_refresh_pick_at_fallback_decision` + its two `run_day` delivery sites) does NOT deliver the cached
preview on a genuine MDP skip — a "standing skip" (`state.final_skip_candidate` set, not committed)
survives a flaky later refresh, but a genuine late pick clears it and delivers, and a cascade *error*
(selection None) still delivers cached (safety net); **(C2)** `run_result_polling` is gated on
`is_scoreable_commit`, not just `state.pick_locked`, so a non-delivered classification-lock is never
scored; **(persistence)** `final_skip_candidate`/`committed_pick_written` live on `SchedulerState`
(carried across a same-day restart by `carry_forward_skip_state`), and `_write_endofday_skip` has an
overwrite-guard (won't clobber an on-disk scoreable record); **(alerts)** `_maybe_alert_missed_pick` /
`health/post_failure` / `health/analytics_artifacts_missing` gate on the skip/commit signal so a
deliberate skip isn't mis-read as a missed/failed pick. `load_decision` rejects malformed/partial
records (missing `action`/`scoreable`/`date`, or wrong `schema_version`) as missing. Two accepted
prod-safe edges remain (private-mode only, prod is public/DM): a private pick whose best-effort
decision-write fails goes unscored; a private pick isn't re-locked on a restart-before-first-pitch.

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

Hetzner VPS (CPX42, Helsinki) runs scheduler, dashboard, and cron via systemd. (Audit fleet uses CPX62 in fsn1 since CPX51 deprecation 2026-04-26.)

```
┌──────────────────────────────────────────────────┐
│  Hetzner CPX42 (8 vCPU, 16 GB, Helsinki)         │
│  systemd services:                               │
│  - bts-scheduler.service (Type=notify,           │
│      WatchdogSec=1800, NotifyAccess=all,         │
│      Restart=always, RestartSec=30)              │
│  - bts-dashboard.service (port 3003, Tailscale)  │
│  - crontab (check-results, reconcile, data       │
│    refresh, preview pick, lineup collection,     │
│    healthchecks ping)                            │
│                                                  │
│  Tailscale: bts-hetzner (stable identity)        │
│  Deploy: GHA SSH → git pull + systemctl restart  │
│  Backup: R2 bucket bts-backup-data               │
│    - artifact sync (parquets/models, manifest)   │
│    - restic repo (operational state, encrypted)  │
└──────────────────────────────────────────────────┘
```

**Operational-state backup (audit F5, 2026-07-10):** `data/picks` (decisions, contest ledger,
delivery IDs, skip-shadow records, the manual saver flag) and `data/health_state` exist ONLY on
the box — gitignored and excluded from the artifact sync — so box loss was irrecoverable
operational state. `bts backup run --set ops` (cron `20 */3`) and `--set archive`
(leaderboard/hetzner_results/external research data, cron `50 4`) push restic snapshots
(encrypted, versioned, deduped) to the R2 bucket under `restic/`; `35 5 Sun` prunes. Secrets:
`RESTIC_PASSWORD` in box `.env` + Eric's Mac Keychain (`r2-bts-restic-password`); R2 creds reuse
the sync's env vars (mapped to `AWS_*` for restic's S3 backend, subprocess env only — never
argv). Each run writes `data/health_state/backup_status.json` (per-set, preserves
`last_success_at` across failures) → `backup_freshness` health source. Restore:
`bts backup restore-drill --target <dir>` restores the latest ops snapshot and verifies the
saver flag, contest ledger, and decision provenance parse — run it after any restore and
periodically (INCIDENT.md). Binary: `scripts/install-restic-hetzner.sh` (pinned + SHA256SUMS-verified,
`~/.local/bin`, no root). The R2 *artifact* sync (`bts data sync-to-r2`) is a separate system:
content-addressed since 2026-07-10 (audit F8) — uploads go to `objects/<sha>/<name>` so the
manifest flip is the only commit point; restores verify into a `.part` temp then `os.replace`;
`verify_manifest` HEADs every referenced object; unreferenced objects >7d old are pruned after
each sync (`--no-prune` to skip).

**Daily lifecycle (scheduler daemon, `Type=notify` + `Restart=always` + `RestartSec=30` + `WatchdogSec=1800` + `SuccessExitStatus=143`):**
- `SuccessExitStatus=143` (both units, added 2026-07-01 box-side; unit files are repo-tracked since 2026-07-10 at `scripts/systemd/` with a `unit_drift` health check — audit F12): the deploy workflow stops units with SIGTERM, which Python exits as 143; without this every deploy logged `Failed with result 'exit-code'`, drowning real crash signal in the journal. A genuine crash (non-zero exit, signal ≠ TERM) still logs as failure and still auto-restarts.
- The scheduler stays alive across days: after IDLE_END_OF_DAY it sleeps until tomorrow's wake (via `_idle_until_next_wakeup`). When the sleep ends and run_day returns, systemd auto-restarts within 30s; new run_day starts with `datetime.now(UTC)`'s new date. Process exits and clean restarts only happen at day boundaries.
- **Heartbeat-watchdog discipline (added 2026-04-22/23)**: any `time.sleep(>60s)` in scheduler.py must be wrapped by one of: `heartbeat_watchdog` (RUNNING-state work like predictions), `_poll_interval_sleep` (result_polling between-iteration sleep), `_watchdog_ping_sleep` (SLEEPING-state waits where pre-sleep heartbeat metadata is authoritative), or `_idle_until_next_wakeup` (end-of-day overnight sleep). Each variant feeds systemd watchdog (notify_watchdog every 60s) AND cooperates with the external check_heartbeat.py monitor's freshness rules. Five bugs found in this class shipped Apr 22-23 — see git log + memory `project_bts_2026_04_23_phase_b_heartbeat.md`.
- **NotifyAccess=all (NOT main)** is required because `uv run` wraps Python in a subprocess; systemd's "main PID" is uv (the launcher), but sd_notify pings come from the Python child. `NotifyAccess=main` rejects the child's pings and TimeoutStartSec kills the service.
- Morning init: loads game schedule for the day, plans lineup-check windows — computed once; a later MLB game-time change does not re-anchor the plan (bit on the 1-game 2026-07-16 slate; backlogged in `docs/optimization-ideas.md`)
- `game_time - lineup_check_offset_min` (code default 45; production TOML **60**): runs full prediction cascade at each check (no skip optimization — pipeline determines projected vs confirmed per-batter)
- Short-circuit: if existing pick is already locked (game started or posted to Bluesky), skips the expensive SSH cascade entirely
- `early_lock_gap`: once confirmed lineups are available, posts picks to Bluesky (confirmation-based, not time-based). Gap check excludes batters from non-Preview games (started, finished, postponed). **Both-slots gate (audit F2, 2026-07-10):** on double days `should_lock` also requires the SELECTED double-down lineup-confirmed — the gap rule alone let a projected DD ride through on the primary's confirmation (11 production days, 4 locked 51–133min early). Applies to normal early locks and the in-loop fallback deferral (same helper); the T−35 final fallback deliberately bypasses `should_lock`, so the gate can only delay delivery within a day, never lose it.
- Logging: each check logs the selected pick, probability, should_lock decision, and gap vs best projected pick. Pick name/probability recorded in `scheduler_state.json` for audit trail.
- Result polling: starts `game_start + 10min`, checks boxscore every 15min. Posts reply (✅/❌ + streak) as soon as all picks have hits (mid-game early exit) or game goes Final.
- `bts reconcile`: 8-day lookback for scoring changes (hit overturned to error). Recalculates streak from scratch if corrections found. Cron at 2am ET. Resolves boxscores unlocked, then applies corrections + the season replay under `picks.scoring_lock` against reloaded state (2026-07-10 review).

**Scoring serialization (`picks.scoring_lock`, 2026-07-09/10):** every writer of pick-result/streak state — `check-results`, the daemon's mid-game and all-final scoring paths, its cap/unresolved/suspended markers (via `scheduler.save_nonterminal_result`, a locked reload that refuses to overwrite a terminal `hit/miss/void`), and `reconcile_results` — holds a shared flock on `data/picks/.scoring.lock` around the read-modify-write, re-loads the pick INSIDE the lock (adopting fresh metadata; failing closed if the file vanished), and skips if a peer already scored. Resolution (network) stays outside the lock. Known deferral: streak.json + pick-file remain a two-file non-atomic pair under crash (healed by the 2am replay + contest anchoring).

**Scheduler state integrity (audit F3, 2026-07-09):** `scheduler_state.json` writes are atomic (`util.atomic_write_text`); a corrupt/torn file is QUARANTINED to `scheduler_state.json.corrupt-<ts>` at load (evidence preserved, fresh day-state, `scheduler_state_integrity` WARNs) instead of crash-looping; state loads BEFORE `notify_ready()`/heartbeat so a failing init can't advertise liveness (`--dry-run` skips the load — it must not mutate). Externally, `scripts/check_heartbeat.py` samples the unit's NRestarts each cron tick: +3 in 20 min, +3 in 60 min, or +4 in 180 min is treated exactly like a stale heartbeat (fail ping) — a crash-loop refreshing its heartbeat every cycle can no longer look healthy.
- 1am cron remains as a safety-net `bts check-results` fallback. Skips when scheduler has already set result ("hit"/"miss") to avoid double-counting streak. Does NOT post to Bluesky; scheduler owns all posting. Since 2026-08-09 the cron passes `--wait-deadline-et 06:00` (in-process 15-min retry until the grader itself reports production+shadow settled; hard deadline; `flock -n` singleton) and `check-results` attempts shadow reconciliation on every exit path; a stale-scoring guard refuses streak-bearing scoring for dates >2 days old without `--allow-stale-scoring` (out-of-order `update_streak` corrupts the streak). Residual stranded results surface via the `result_resolution` health source.

**Key modules:**
- `strategy.py` — MDP-optimal pick logic with heuristic fallback. Auto-loads `data/models/mdp_policy.npz` for provably optimal skip/single/double decisions based on (streak, days_remaining, saver, quality_bin). Double-down must be from a different game. Falls back to heuristic thresholds if policy file absent. Shared by `bts run` and orchestrator. The action choice is a PURE `decide_action(ctx, streak, saver)` over a `DecisionContext` (resolved MDP policy + selected candidates + allow_double); `select_pick` builds the context (impure prep) then calls it — the seam a future uncertainty layer evaluates over a state set.
- `orchestrator.py` — Local prediction (`predict_local`) + shadow prediction (`predict_local_shadow`). TOML config, calls strategy + posting. Shadow uses `feature_cols_override` with separate model cache (`blend_{date}_shadow.pkl`). Both production and shadow paths attach pick provenance v1 (see below) to the DailyPick before save.
- `scheduler.py` — Long-running daemon. Dynamic lineup checks at `game_time - lineup_check_offset_min` (45 default / 60 in production). Short-circuits when pick is locked. Shadow model runs after production lock (`_run_shadow_prediction`). Helpers: `_compute_result_poll_start` (uses `_earliest_pick_game_et` so double-down's earlier game isn't skipped), `_poll_interval_sleep` (result_polling sleep), `_watchdog_ping_sleep` (SLEEPING-state waits — keeps systemd watchdog fed without overwriting heartbeat file), `_idle_until_next_wakeup` (end-of-day overnight sleep). All cooperate with `bts/heartbeat.py:heartbeat_watchdog` (RUNNING-state context manager). `bts schedule` CLI command. `_run_shadow_prediction` reads `data_dir`/`models_dir` from config and threads them into both the prediction call and provenance attachment so the recorded blend hash matches the artifact actually loaded. **Skip-day visibility** (shipped 2026-06-18): when the MDP declines to pick (best candidate below the ~80% pick bar, `strategy.SKIP_THRESHOLD`) the daemon no longer returns silently — `build_skip_summary` feeds a `SKIP —` log line, a one-time *tentative* DM (`maybe_notify_skip`, idempotent via `SchedulerState.skip_notified_at`, dm-mode only, carried same-day across restart by `carry_forward_skip_state`), and the dashboard banner. `skip_summary`/`skip_notified_at` persist on `SchedulerState`.
- `picks.py` — DailyPick + Pick dataclasses, save_pick / save_shadow_pick / load_pick / load_shadow_pick. **Pick provenance v1 (PR #18, deployed 2026-05-04 at `a3bc4d3`)**: every saved pick JSON now carries three optional fields populated at save time: `model_git_sha` (HEAD of the producing checkout), `model_pickle_sha256` (sha of the blend artifact `blend_<date>.pkl` actually used), `policy_npz_sha256` (sha of `mdp_policy.npz` if loaded). All hashes are best-effort/null when the artifact is unavailable; pick saves never fail on provenance errors. Old picks load with `None` via `data.get(...)` backcompat. Future calibration analyses can filter by `model_git_sha == current_deploy_sha` instead of doing deploy-branch archaeology.
- `dm.py` — Bluesky DM notifications for pick delivery and health/cascade failures. Uses `api.bsky.chat` directly (not PDS proxy).
- `predict-json` — worker command: runs pipeline, outputs JSON to stdout, logs to stderr.

**Config:** `~/.bts-orchestrator.toml` on Hetzner. `pick_delivery = "public"` posts picks to the Bluesky feed, `pick_delivery = "dm"` sends picks privately to `bluesky.dm_recipient`, and `pick_delivery = "private"` saves locally only. Legacy `private_mode = true` still maps to local-only delivery when `pick_delivery` is unset. `shadow_model = true` runs the context stack alongside production for the 30-day eval. Tiers: local only.

**LightGBM is optional:** `uv sync` (Pi5, pick logic only) vs `uv sync --extra model` (workers, full prediction).

## Health Monitoring

End-of-day health checks dispatched by `bts.health.runner.run_all_checks()`. Each check module returns 0+ `Alert` objects (level: INFO/WARN/CRITICAL); CRITICAL alerts DM Bluesky via `bts.dm`. 26 sources as of 2026-08-09 (`result_resolution` added — stranded shadow/production results, version-blind, always-attention). Source modules must NOT blanket-catch their own crashes (audit F4, 2026-07-09): an unexpected exception propagates to `_safe_run`, which surfaces it as a CRITICAL `health_runner` alert — expected data-absence stays quiet, per-file content corruption is skipped, but filesystem errors (OSError) propagate.

| Source | Tier | Detects |
|---|---|---|
| `blend_training` | 1 | tomorrow's `blend_<N+1>.pkl` missing at end-of-day → fallback to stale model |
| `pooled_training` | 1 | `<TOMORROW>_status.json` shows under-filled pool (added 2026-04-29; no-op until daily pooled training runs) |
| `post_failure` | 1 | locked pick lacks either public post (`bluesky_posted` + URI) or private notification (`notification_sent` + ID) |
| `restart_spike` | 1 | `NRestarts` delta vs day-anchored checkpoint > threshold (+1/day planned-restart budget across gaps) |
| `calibration` | 2 | top-1 P drift on 7d vs 14d rolling mean |
| `predicted_vs_realized` | 2 | acute drift in mean(predicted) - mean(realized), per-slot (primary + DD legs vs their own p), 14d window vs 28d baseline |
| `realized_calibration` | 2 | absolute LEVEL of overconfidence per bucket — [0.75,0.80) both slots + [0.70,0.75) DD-leg-only (2026-07-12: the DD band had no absolute monitor while its legs ran 0.545 realized vs 0.731 stated; slot-aware because calibrated primaries dilute a pooled bucket). **Pools by production-REGIME fingerprint `(policy_npz_sha256, feature_env_hash)` (audit F6, 2026-07-09)** — the pool is all in-window picks matching the newest stamped pick's fingerprint, so docs/shadow/ops deploys no longer erase the sample while genuine policy/env changes still reset it (`model_pickle_sha256` deliberately excluded: the blend retrains daily). An unstamped/partially-stamped newest pick falls back to the wall-clock `since_deploy_iso` filter (pre-deploy excluded, skip count logged); known residual: predictor-code-only changes don't flip either stamp. Since-deploy timestamp prefers a wall-clock stamp `data/.last_deploy_iso` (written by `deploy.yml` on canary-pass AND rollback — monotonic, so it fixes commit-time≠deploy-time and rollback-moves-HEAD-backward); falls back to `git log %cI HEAD` when absent. Pick run_time vs cutoff is instant-compared, not lexicographic (2026-06-11). Thresholds 8/15/25 pp INFO/WARN/CRITICAL; CRITICAL additionally requires bucket n≥20 OR an exact Poisson-binomial tail ≤1e-3 (2σ-ish small-n readings cap at WARN; effectively-impossible ones — 0-for-8 at 0.73 — still escalate). Slot grading priority: production `slot_results` → PA-frame join (`data_dir`) → day-result proxy (primary on non-DD days only). Per-bucket `incident_key` for same-day DM dedup; WARN-attention listed (persistent WARN reaches the digest from the 2nd consecutive day). |
| `same_team_corr` | 2 | DD pair-realization vs naive independence |
| `projected_lineup` | 2 | % rolling 14d projected_lineup over threshold |
| `pitcher_sparsity` | 2 | % rolling 14d picks with `LIMITED pitcher data` flag (added 2026-04-30 — diagnostic for MiLB-transfer ROI; also catches min_periods regression) |
| `leaderboard_freshness` | 2 | last successful BTS-leaderboard scrape > 12h (WARN) / 36h (CRITICAL); silent if `data/leaderboard/` doesn't exist (pre-deploy state) |
| `park_drag_freshness` | 2 | park_drag external table: failed refresh run (WARN), source data ≥3d behind (WARN) / ≥6d (CRITICAL), manifest generated_at >30h (WARN, cron liveness); silent if `data/external/park_drag/` absent or off-season (Oct-Feb) |
| `disk_fill` | 3 | `shutil.disk_usage` thresholds |
| `pick_entry` | 1 | (added 2026-07-09, audit F1) EOD backstop for check-pick-entered: marker still `alerted` past the submission cutoff → WARN (entry never confirmed); `dm_failed` past cutoff → CRITICAL (unentered AND the alert never reached the operator). Quiet while a late game's window is open; always-attention. |
| `scheduler_state_integrity` | 3 | (added 2026-07-09, audit F3) WARNs while a `scheduler_state.json.corrupt-*` quarantine file is recent (7-day lookback, break-proof): a torn/corrupt day-state was recovered at startup — pick_locked/skip context reset; evidence file preserved. Always-attention. |
| `memory_growth` | 3 | scheduler RSS thresholds (1024/3072/6144 MB tuned 2026-04-28) + Tuesday-EOD weekly digest INFO with median/trend (added 2026-04-29) |
| `streak_validation` | 3 | `streak.json` schema sanity |
| `slate_auc` | 2 | **M3 revisit trigger** (added 2026-06-11): rolling realized slate AUC over persisted daily slates (`data/picks/slates/<date>.json`, written by `bts.slate.save_slate` from `run_and_pick` — full ranked candidate slate, last-write-wins per date, never raises into the pick path). Joins `p_game_hit` to PA-frame any-hit outcomes on (game_pk, batter_id); tie-aware rank AUC, 60d window, min 20 days/200 rows; recomputes ≤1×/7d (cached in `data/health_state/slate_auc_status.json`, written every recompute for dashboards). WARN at AUC ≥ 0.61: the serving-staleness HOLD (`docs/audit/2026-06-11-m3-serving-staleness.md`) assumed ~0.59 discrimination — re-run `scripts/replay_m3_serving_parity.py` before trusting the HOLD further. |
| `contest_state` | 1 | contest-account state (the real MLB BTS streak driving live picks) missing/invalid/corrupt when expected → CRITICAL (no `source_date`, future `source_date`, malformed file). **Coverage-gap semantics (Phase-1 snapshot/coverage split, `56f9726`)**: the gap is counted in **settled picks** (never calendar days), and `source_date` trails the live `activeStreak` counter by ~one settled pick BY DESIGN → gap == 1 is the normal coverage lag, INFO at any time of day; gap ≥ 2 → WARN (logged, not DM'd — can be a transient ledger lag; a genuinely broken fetch DMs separately via the auth-failure path). Also WARNs on a legacy/expired manual override file. History: check added #138; stale + legacy/expired WARN #142; level-aware overnight INFO 2026-06-07; noon-ET escalation 2026-06-12 (retired); aligned to the coverage split `56f9726` (post-Phase-1). |
| `postponed_pick` | 1 | undelivered pick committed to a postponed/missing game (the 2026-05-05 incident class), caught before the stale pick can silently survive to the delivery window |
| `fallback_defer` | 1 | fallback-defer events: makes the archive-unsafe-fallback-candidate path visible; pages only if a defer breaks the never-miss guarantee |
| `analytics_artifacts_missing` | 2 | end-of-day visibility for missing shadow / live-forward validation artifacts after a locked pick |
| `live_forward_resolution` | 2 | canonical live-forward resolution stalled — pending outcomes aging past grace (3d, WARN) / critical (7d) thresholds; the 2026-06-17 suspended-game stall class |
| `mdp_policy_alignment` | 2 | recent production picks collapsing into one MDP quality bin (probability-scale drift vs saved policy boundaries — Gate-B diagnostic; no effect on selection) |
| `backup_freshness` | 3 | (added 2026-07-10, audit F5) restic backup staleness from `data/health_state/backup_status.json`: ops set (picks+health, 3h cron) WARN ≥7h / CRITICAL ≥26h since last success; archive set (leaderboard/results/external, daily) WARN ≥30h / CRITICAL ≥78h; failed-last-run WARN even when fresh. Silent if the status file is absent (backups not armed — local dev). Always-attention. |
| `unit_drift` | 3 | (added 2026-07-10, audit F12) installed `~/.config/systemd/user/bts-{scheduler,dashboard}.service` differs from the canonical templates in `scripts/systemd/` (sha256), or an installed tracked unit has no template. Read-only; install stays an explicit operator action (`scripts/install-systemd-hetzner.sh`). Repeated-attention (2+ days). |

**Tier 1**: silent failures with damage. **Tier 2**: quality decay. **Tier 3**: process integrity.

State files: `data/health_state/memory_growth_history.jsonl` (daily-appended RSS log); `data/picks/lineup_evolution_<date>.jsonl` (one append per `save_pick` call — captures pick trajectory across day's lineup-confirm checks; supports gap #6 analysis of projected-vs-confirmed underperformance, shipped 2026-05-01).

**Truthful heartbeat (H5b, 2026-06-11):** during cascades, `heartbeat_watchdog`
ticks read an in-process progress beacon (`bts.progress`, stage-entry marks
through `run_and_pick`/`run_pipeline`). Progress fresh → `state=running` with
`{stage, stage_age_s, run_id}`; no progress for `heartbeat_stall_after_sec`
(default 900, `[scheduler]` toml) → `state=stalled` → `check_heartbeat.py`
cron POSTs healthchecks /fail within ≤5 min (dashboard `is_heartbeat_fresh`
also fails closed). **sd_notify pings continue during a stall** — the unit has
`WatchdogSec=1800`, and Phase 1 is alert-only (no auto-kill; Codex-reviewed).
Stage durations append to `data/health_state/cascade_stage_durations.jsonl`
(`status ∈ {ok, ok_after_stall, stalled_incomplete}`) — the dataset for any
Phase-2 data-derived thresholds. Spec:
`docs/superpowers/specs/2026-06-11-h5b-truthful-heartbeat-design.md`.

## Contest-account streak (drives live picks)

Live recommendations are driven by Eric's REAL MLB BTS account streak ("contest state"), kept separate from the model/replay `streak.json`. Since real-streak-anchoring Phase 1 (PR #143, 2026-06-17): `bts.contest_state.load_decision_streak_state` returns the real contest streak as the decision streak — **the model replay can never raise it**. Settlement freshness is surfaced as a `status` (fresh/lagged/stale) instead of freezing values, and **doubles are NOT frozen on staleness** (the old freeze-at-`max(model, contest)` + DD-freeze behavior is retired); the pick path still fails closed via `require_contest_state` (prod=true) when contest state is missing entirely, so `model_only` is dashboard-display-only. Automation shipped PR #142 (2026-06-06, Claude+Codex).

- **Auto fetch** — `bts fetch-contest-streak` (cron 4×/day: 01:10/02:10/10:30/13:30 ET) reuses `bts.leaderboard.auth` + the user-profile endpoint (`user_id` from auth/login `success.user.id`) → `bts.contest_fetch` (parse `activeStreak`/`seasonBestStreak`; derive `source_date` from the latest *settled* prediction via `rounds.json`; sanity-gate) → atomic write `data/picks/account_state/contest_streak.json` (`bts_contest_streak_auto_v1`). **Fails safe**: never overwrites on auth/HTTP/shape/identity/staleness failure; identity guard (`--expected-username`) + prior-account guard; throttled DM via `data/health_state/contest_streak_fetch_status.json`. **Auth-failure classification (2026-08-11 incident, 2 Codex rounds):** `fetch_login_session` retries in-process (2s/4s, numeric Retry-After ≤30s) ONLY un-rejection-shaped failures — transport errors, 5xx, zero-length-200 bodies (the observed flap shape); non-empty non-JSON / non-object-JSON 200s are challenge-page-shaped → `TransientAuthError` with NO retry (kill-switch), and 429 → `RateLimitedLoginError` (back off; cookie re-capture would ADD auth traffic). CLI advice matches the class (transient/rate-limited failures never say "re-capture cookies" — the incident's misdiagnosis), malformed bodies can no longer escape as `AttributeError` past `_fail` (that hole silenced DMs exactly on garbage responses), and the DM throttle is **per-category** (`transient`/`rate_limited`/`actionable`; stamps survive successful runs; legacy single-stamp records migrate to all categories) so an outage DM can't suppress a later real cookie-death DM. Profile-stage 429/5xx/transport failures categorize the same way.
- **Precedence** (`load_contest_streak_state`): unexpired manual override (`contest_streak.manual.json` w/ `override_expires_at`) > auto (`contest_streak.json`) > legacy/expired manual fallback (+ health WARN).
- **Manual override** — `bts set-contest-streak` writes an EXPIRING override (`bts_contest_streak_manual_v2`, `--ttl-hours` default 24) for emergencies (e.g. cookie expiry). Can't permanently freeze picks.
- **Streak Saver** = a MANUAL 3-state flag `account_state/saver_state.json` (`not_earned | active | used`; loader-derived `uninitialized` = fail-closed) — the SOLE live saver authority (`bts.saver_state`, read by `load_decision_streak_state`). It is **NOT inferred** from the ledger: the consuming "save" is a streak-preserving transition (e.g. 14→14) that vanishes from the windowed `predictions` snapshot, so available-vs-used is unobservable. `not_earned→active` auto-earns on `best_streak>=10` from a *tracked* `not_earned` (a cold file at >=10 is fail-closed → `bts saver-state --init active`); `active↔used` is MANUAL (dashboard mark-used/undo button or `bts saver-state --use/--undo`); guarded atomic (fcntl-serialized) transitions. The dashboard nudges on a likely save (a stable held No-Hit at 10-15 in the ledger) and warns if active past streak 15; health WARNs if uninitialized in-zone. `contest.saver_available` + `set-contest-streak --saver-available` are retired from the saver path. Shipped 2026-06-18.
- Auth: cookies live in `~/.bts-leaderboard-cookies.json` on hetzner; interactive re-capture via `scripts/capture_bts_cookies.py` when they expire (the one human-in-the-loop dependency — failure alerts, never silently freezes).
- **`bts check-pick-entered` — v2 RE-ENABLED 2026-07-03 (cron `*/15 10-23` ET), hardened 2026-07-04 (Codex review).** Pre-first-pitch DM if today's delivered pick wasn't entered in the MLB app — or was entered as the WRONG pick. v1 (2026-06-12) was disabled same-day because its sole source, the profile predictions array, is **SETTLED-only** → false-alarmed every pre-pitch day. v2 sources entry from the UNION of the profile and `GET api/predictions` (`contest_fetch.fetch_pending_predictions`; found in the app JS bundle), which exposes the pending same-day row. **Identity-verified** (`contest_fetch.pick_entry_status`, added 2026-07-04): entry must contain the DELIVERED player(s) — BOTH double-down slots — mapped BTS-playerId→MLB-feedId via players.json (`cli._fetch_bts_to_mlb`); a wrong player or a missing DD slot alerts as a **mismatch**, an unresolved crosswalk falls back to present-unverified (OUR gap never false-alarms). Rationale: Eric always intends the entered pick to equal the recommendation. **Marker discipline** (`data/health_state/pick_entry_check.json`): the once-per-day marker is consumed only on a confirmed entry or a **successfully-sent** alert; a DM failure writes a retryable `dm_failed` state (which the top-of-loop dedup ignores) so the next run re-alerts — the one alert this feature exists for is never silently lost. `fetch_pending_predictions` raises `ContestFetchError` on a drifted 200 (not `[]`) so schema drift → quiet skip, not a false alarm. Any fetch failure skips quietly WITHOUT consuming the marker. Original incident: 2026-06-11, a delivered pick never entered — this closes that class. **Hardened again 2026-07-06 (Codex-reviewed; `docs/audit/2026-07-06-deferred-dd-and-premature-entry-alert.md`):** the check now gates on `daily_decision.is_scoreable_commit` (the same commit predicate scoring/dashboard use) BEFORE the window/fetch, so a deferred/undelivered *preview* `{date}.json` never triggers a "not entered" DM — the 2026-07-06 incident was a premature DM on a deferred double-down whose early leg (2:10 PM confirmed) locked before the projected primary (9:45 PM) could confirm. The firing window now also excludes the un-submittable final 5 min (deadline = first pitch − 5; the DM reports minutes-to-cutoff). A `state.pick_locked` "union" gate was considered and REJECTED (Codex): `classify_pick_lock_state` sets `pick_locked` on an undelivered preview via status-lookup-failure / game-started, so OR-ing it would reintroduce false alarms. Residual known-limitation: a `private_locked`/`locked_unconfirmed` commit whose best-effort `decision.json` write ALSO failed is not alerted (narrow; private mode not in prod). **v3 (2026-07-09, audit F1 + Codex review): `alerted` is NON-terminal** — after the initial DM, every `*/15` run RE-VERIFIES the account until the entry is confirmed or the cutoff passes, with throttled escalation DMs at T−30 and T−15 minutes-to-cutoff (each tier fires once; an alert sent at/below a threshold consumes it; the marker carries an `escalations` ledger, legacy markers = initial-fired) and a one-time ✅ all-clear DM on the alerted→confirmed transition. **Only an exact identity `match` is terminal**: `present_unverified` (crosswalk gap) writes a nonterminal marker and keeps re-verifying, and fewer entered rows than delivered slots is a `mismatch` regardless of crosswalk coverage. The window's lower bound is strict (exactly first_pitch−5 = already locked, no "0 min" DM). Failed DMs stay retryable at every tier. EOD backstop: the `pick_entry` health source. Known deferral: no postponed-game awareness — a pre-pitch postponement keeps nagging for a selection that will Pass (product call pending). **Hardened 2026-08-11 (auth-flap incident):** auth uses `fetch_login_session(attempts=2)` (bounded — the `*/15` cron is the outer retry loop), and minutes-to-cutoff is computed net of monotonic elapsed fetch time, so a slow auth/profile stage can never produce a "Fix it now!" DM for an entry that locked mid-fetch; that no-DM path still writes the `alerted` marker (detected-and-unresolved, same semantics as the tier-exhausted branch) so the EOD `pick_entry` audit fires. Transient auth outages keep the quiet-skip (v3 design).

## Statcast swing campaign (experimental — NOT in production)

An exploratory campaign testing per-pitch Statcast swing data (miss distance, swing timing, mechanics) as candidate features. No swing feature is in production `FEATURE_COLS`. Spec + plans in `docs/superpowers/specs|plans/2026-06-12-statcast-swing-*`. Modules: `bts.features.swing` (bronze `data/processed/swing_{season}.parquet` → denominator-preserving daily aggregates → `shift(1)` rolling, leak-free), `bts.validate.slate_rank` (paired daily NDCG@10 + season-stratified block bootstrap), `bts.experiment.swing_screen` (declarative arm registry + two screen runners: the original train-once/score runner, and the **residual-stacking** runner `build_prod_prior_oof`/`run_residual_arm` — amendment #3, Codex-co-designed). Stage 0 (bronze + QA) complete; Stage 1 residual screen in flight (2026-06-13). **Residual design** (removes the covered-only-baseline confound): a full-history production prior (production features only) is cross-fit OOF by week-group onto covered training rows + predicted onto eval rows; every arm then trains at the GAME-BATTER SLATE level with `[prod_prior, prior_daily_rank, availability flags, swing_coverage_60g]` + its candidate column, so the paired daily rank-AUC delta isolates swing value *on top of* the full-strength model. Gates (in order): oracle gross-canary must explode (proven, auc→1.0); soft-oracle = power gate (~+0.005); within-date-permuted + mask-only nulls unremarkable. **Key methodology lesson: the controls gate caught FOUR distinct false-conclusion classes before any verdict** — training coverage starvation, negative controls that weren't clean nulls (within-batter permutation leaked batter skill; baseline missing the common scaffolding), and a weak sentinel mistaken for a broken harness (resolved by the oracle canary). Detail in `docs/audit/2026-06-13-swing-screen-codex-r4-codesign.md` + the spec; do NOT promote any swing feature without the full screen→confirm→FDR pipeline.

## Strategy Simulation

Monte Carlo simulator and MDP solver for evaluating and optimizing play strategies.

> **⚠ iid-solver caveat (2026-07-13):** every DP value below (incl. the P(57) headline)
> assumes iid day-types. Realized rank-1 sequences suppress long all-hit windows
> ×0.10-0.14 vs order nulls, making iid milestone values policy-dependent-wrong
> (always-single inflated ×3.6, doubling understated). Use realized-sequence replay
> for policy comparisons — CLAUDE.md "RUN STRUCTURE" gotcha +
> `docs/audit/2026-07-13-dd-p-policy-value-sensitivity.md` finding 5.

```
src/bts/simulate/
    strategies.py       — Strategy dataclass, 7 named profiles, streak-aware thresholds
    monte_carlo.py      — simulate_season(), run_monte_carlo(), load/save profiles
    backtest_blend.py   — 12-model blend walk-forward, saves daily profiles
    quality_bins.py     — equal-frequency quintiles with empirical P(hit), P(both)
    exact.py            — absorbing Markov chain for exact P(57) (no Monte Carlo noise)
    mdp.py              — reachability MDP solver, 103K states, backward induction
    pooled_policy.py    — pooled-seed MDP policy builder (Option 7): compute_pooled_bins,
                          evaluate_mdp_policy, build_pooled_policy. Merges rank-1/rank-2
                          within (seed, date) pairs to prevent cross-seed cartesian pairing.
    cli.py              — bts simulate {backtest, run, solve, exact}
```

**MDP-optimal strategy (P(57) = 8.17% pooled 24-seed, phase-aware, different-game doubles):**
- Phase-aware bins: early season (Mar-Aug) vs late (Sep only, `late_phase_days=30`)
- **Different-game doubles**: double-down must be from a different game_pk than primary pick (avoids correlated outcomes — 39.7% of days had same-game doubles). +59% P(57) vs same-game.
- **No densest bucket**: pure blend ranking, no time-window filtering (removed 2026-04-08, was hurting P(57) by 8%)
- At low streaks with many days left: play aggressively, double everything (even Q1)
- At high streaks: skip all but Q4-Q5, stop doubling at streak 46+
- Streak Saver: a once-per-season mulligan that HOLDS the streak on the first No-Hit at streak 10-15 (it does not increase the streak); now a manual `saver_state.json` flag (see Key modules), and it only changes the suggested action at streak 10-15

**Backtest profiles:** `data/simulation/backtest_{season}.parquet` (2021-2025, 912 daily profiles). Generated by `bts simulate backtest --seasons 2021,...`.

**Policy file:** `data/models/mdp_policy.npz` (1.8KB). Generated by `bts simulate solve --save-policy`.

## Validation

Multi-metric scorecard plus a five-piece SOTA validation methodology stack (PRs #9 → #18, May 2026) for benchmarking model and strategy changes against a held-out lockbox.

```
src/bts/validate/
    scorecard.py         — P@K, miss analysis, calibration, streak metrics, full scorecard assembly
                            (also: compute_scorecard_over_manifest for per-fold #5-manifest evaluation)
    splits.py            — #5 P0/P1: LockboxSpec + FoldSpec + make_purged_blocked_cv (rolling-origin
                            forward-chaining with purge/embargo) + manifest save/load. Lockbox is a stored
                            explicit date range; default lockbox is last 30 game-days of latest tracked
                            complete season.
    proper_scoring.py    — #12 phase 1: log loss, Brier, Murphy reliability/resolution/uncertainty,
                            top-bin calibration. Game/profile-level. Integrated via scorecard output —
                            no standalone CLI.
    conformal_gate.py    — #11 P0/P1: bucket-Wilson lower-bound validity gate + median-bound-width
                            tightness gate over #5 manifest folds. Per-cell PASS/FAIL/INSUFFICIENT_DATA;
                            top-level PRODUCTION_DEPLOY_READY / NO_PRODUCTION_DEPLOY verdict from ship_set.
    ope.py               — DR-OPE, paired hierarchical block bootstrap (Politis–Romano stationary;
                            supports profile-level block-bootstrap CI via n_block_bootstrap, default
                            0; v2.6 used n=500, expected_block_length=7), fitted-Q-evaluation,
                            corrected_audit_pipeline (LOSO with configurable params/rho_pair/policy modes).
    ope_eval.py          — #13 P0/P1: per-fold target-policy V_pi solve + terminal-MC replay
                            cross-check over #5 manifest. Phase-aware late-bin guard. Strict JSON
                            artifact (allow_nan=False).
    rare_event_mc_eval.py — #14 P0/P1: black-box CE-IS rare-event MC over #5 manifest. Train-theta
                            on fold-train, holdout estimate with theta_train + n_rounds=0. Fixed-window
                            estimand (P(max consecutive rank-1 hits ≥ streak_threshold) over
                            n_holdout_dates) — explicitly NOT comparable to #13 V_pi.
    dependence.py        — pearson_residual (+ vectorized pearson_residual_vec, 16× speedup), within-
                            batter-game PA correlation, logistic-normal random-intercept (MoM via
                            brentq inversion), cross-game pair residual permutation test,
                            build_corrected_transition_table, pair_residual_correlation_per_cell
                            (5×5 diagnostic heatmap).
```

**CLI** (`bts validate <subcommand>`):
- `scorecard [--save path] [--diff baseline.json] [--manifest path] [--mc-trials N]` — per-fold scorecard when `--manifest` is provided
- `split-manifest --profiles-dir data/simulation [--lockbox-season N] [--output path]` — build + save a `#5` manifest with explicit lockbox
- `conformal-gate --manifest path [--output path]` — per-(method, alpha) PASS/FAIL + ship_set
- `policy-value-eval --profiles-dir ... --manifest path [--target-policy mdp_optimal] [--output path]` — V_pi + replay
- `rare-event-ce-is --profiles-dir ... --manifest path [--n-rounds-train 8] [--n-final-holdout 20000] [--output path]` — fixed-window event probability per fold
- `falsification-harness ...` — v1/v2/v2.5/v2.6 6-component audit (separate path; see below)

All `#5`-manifest-bound validators (`scorecard`, `conformal-gate`, `policy-value-eval`, `rare-event-ce-is`) carry `lockbox_held_out=true` + `manifest_metadata`. The CV-eval outputs (`scorecard --manifest`, `policy-value-eval`, `rare-event-ce-is`) additionally carry `aggregate_deferred=true` — cross-fold uncertainty is a deferred P1.5+ cycle.

Baseline scorecard at `data/validation/scorecard_baseline.json` (2026-04-02).
Investigation scripts in `scripts/validation/`, verdict docs in `docs/validation/`, audit memos in `docs/sota_audit/`.

**Realized-picks audit artifact (PR #17, 2026-05-04)**: `data/validation/realized_picks_canonical_2026-05-04.parquet` — canonical view of production picks with PA-frame ground-truth attribution and explicit regime cutoffs. Reproducible via `scripts/canonicalize_realized_picks.py --summary`. Memo at `docs/sota_audit/2026-05-04-realized-picks-calibration.md`. Strict-current model verdict was inconclusive (n=5); the post-pooled-MDP-pre-bpm stratum (n=30) showed exploratory overconfidence concentrated in the double-down slot — not a production-deploy claim.

**Synthesis memo (PR #16, 2026-05-04)**: `docs/sota_audit/2026-05-04-falsification-harness-synthesis.md` connects the five validation pieces and the v2.5/v2.6 attribution into a single picture; explicit guardrail that none of #11/#13/#14 has authorized a production deploy.

### Falsification harness (v1/v2/v2.5/v2.6, 2026-05-02 → 2026-05-04)

`scripts/run_falsification_harness.py` — 6-component audit of production's claimed `P(57) ≈ 8.17%`. Uses DR-OPE through `corrected_audit_pipeline`, CE-IS rare-event MC, PA + cross-game dependence diagnostics. v2.5 added 3 mode flags for nested factorial ablation: `--params-mode {pooled,fold-local}`, `--rho-pair-mode {scalar,per-bin}`, `--policy-mode {global,per-fold}`. v2.6 added `--n-block-bootstrap N` for profile-level paired hierarchical block-bootstrap CI on `corrected_pipeline_p57`.

**v1 verdict (2026-05-02)**: `HEADLINE_BROKEN` at `corrected_pipeline_p57 = 0.0083 [0, 0.0375]` (5-fold percentile CI, ≈min/max).
**v2 verdict (2026-05-02)**: `HEADLINE_REDUCED` at `0.0333 [0, 0.1167]` (still 5-fold percentile CI; v2.6 superseded).
**v2.5 attribution (2026-05-03)**: 6-cell nested ablation shows Change A (fold-local params) has **zero observable effect at this metric's resolution** within per-fold branch; Changes B (per-bin rho_pair) and C (per-fold MDP solve) each independently produce a +1.67pp lift on `corrected_pipeline_p57`. v2's "fold-local fix" framing is refuted by attribution. See `docs/sota_audit/2026-05-03-harness-v2.5-attribution.md`.
**v2.6 reconciliation (2026-05-03 → 2026-05-04)**: under profile-level paired hierarchical block-bootstrap (Politis–Romano stationary, expected_block_length=7, n=500), all 6 v2.5 ablation cells gate `HEADLINE_REDUCED` at half-headline=0.04085. The v1 BROKEN classification was a percentile-CI artifact (ci_upper 0.0375 was narrowly below the threshold — about 0.34pp gap on a CI whose grid resolution is 1/120 = 0.83pp, i.e. less than half a metric tick). **The v1→v2 gate-class transition does not survive at this threshold.** Point-estimate attribution (B alone +1.67pp, C alone +1.67pp, B+C +2.50pp via one extra 2023 fold success) survives. See the v2.5 memo's "Addendum: v2.6 block-bootstrap CI" section. Synthesis: `docs/sota_audit/2026-05-04-falsification-harness-synthesis.md` (PR #16).

Cells 100/110 (params=fold-local + policy=global) are operationally undefined per the nested factorial design. v2.5 verdict JSONs (4 ablation cells; cell 000 was the v1 baseline, cell 111 was the v2 production verdict that came in via prior runs): `data/validation/falsification_harness_v2.5_cell{001,010,011,101}.json` plus matching `_heatmap.json` files and the attribution decomposition `data/validation/v2_5_attribution_2026-05-03.json`. v2.6 verdict JSONs (all 6 cells, n_block_bootstrap=500): `data/validation/falsification_harness_v2.6_n500_cell{000,001,010,011,101,111}.json` plus `data/validation/v2_6_n500_summary.json`. Cloud orchestration via `scripts/v2_5_*.{sh,py}` (Vultr provision/run/retrieve/teardown); v2.6 sweep ran locally on Mac (~55 min, vectorized `pearson_residual_vec`).

**Honest production claim status**: the corrected estimate is about 41% of the 8.17% headline (point estimate ~0.0333 in v2/v2.5; v2.6 widens uncertainty without moving the point) — less than half the headline value. No formal retraction has been issued because no replacement number has been certified under lockbox + aggregate-CI methodology — the validation stack at PRs #9-#15 ships the foundation for that future certification but has not yet been used to license a deploy decision.

## Dashboard

LAN-only web dashboard at `http://bts-hetzner:3003` (tailnet). Single-file Python server using `http.server` (no framework). Serves MLB-themed HTML with inline CSS.

**Key modules:**
- `web.py` — HTTP handler, page rendering, live scorecard HTML, `/api/live`, `/api/live-html`, `/api/audit-progress`, `/api/leaderboard` (added 2026-05-01: returns today's BTS-leaderboard consensus pick + our percentile rank as JSON), `/health` endpoints. On a skip day renders a "SKIP TODAY" banner (`render_skip_banner`) from `scheduler_state.json:skip_summary`, naming the streak-dependent pick bar (`strategy.effective_pick_bar`). Suppression keys on the committed-pick predicate (`daily_decision.is_scoreable_commit`), NOT pick-file existence — a stale provisional file (the projected→real flip day, 2026-07-01) must not hide a standing skip. A standing skip owns the page: no hero pick card, no "Waiting for lineups" placeholder, no live-scorecard section (all restored on the next refresh if a late genuine pick clears the banner). History rows whose `decision.json` says `skip` render a muted SKIP marker instead of an eternally-pending dash (`_decision_action`); rows without a decision record keep the pending dash. Player names (batters + opposing pitchers; hero card, pick-history table, scorecard lineup) link to MLB.com game logs via `_player_link` (shipped 2026-06-23): **ID-only** URLs `/player/{mlbam}?stats=gamelogs-r-{hitting|pitching}-mlb&year=` — MLB.com redirects a bare mlbam to the canonical name-slug page *and preserves the query string*, so we never generate a slug or need a name→id crosswalk (a wrong slug silently bounces to `/players`). mlbam (`batter_id`/`pitcher_id`) already travels with every pick and scorecard batter; missing id → plain text fallback
- `scorecard.py` — Data extraction from MLB game feed for live scorecard. Per-batter payload carries `lineup_status` (one of `at_bat / on_deck / in_hole / upcoming / out_of_game / not_in_lineup / pre_game / final`) and `batters_away` (0-8) computed via `_compute_lineup_status` + `_slot_from_bo` helpers. When picked batter's team is currently fielding, `_next_leadoff_id_for_team` derives the right reference batter (their team's next leadoff slot) so distance still computes correctly. ~80 tests.
- `audit_progress.py` — Live in-flight audit monitor. SSHes each box in `boxes.json`, parses `/root/audit.log` completion markers, aggregates per-box + overall progress. Also reports `ps -u bts` audit_attach process status. CLI entry for pre-deploy smoke testing: `python -m bts.audit_progress --provider vultr --dir <name> --seeds-file <path>`. 25 tests.

**Live scorecard (during games):**
- Caught-looking style: pitch grids, SVG diamond with baserunning, trajectory lines
- Shows only picked batters' plate appearances
- `/api/live-html` returns server-rendered HTML fragment; JS polls every 30s and swaps `outerHTML` (no page flash)
- Handles different-game double-downs via `merge_scorecards`
- In-progress PA: pulsing amber border with current pitch count
- Green tint only on hits (single/double/triple/HR), not walks/HBP
- Sticky batter columns (#/name/POS) on horizontal scroll for 7+ PA games
- Upcoming-PA placeholder cells render lineup-distance copy with state-tinted backgrounds (Direction A, shipped 2026-04-24): amber for `on_deck` / `in_hole` (imminent), gray for `N batters away`, red for `OUT`. The first upcoming PA cell is the only one that shows a label; subsequent placeholder cells stay blank. See `docs/superpowers/specs/2026-04-24-batters-away-display-design.md` and `docs/superpowers/specs/2026-04-24-upcoming-cell-polish-design.md`.

**Lifecycle:** Scorecard appears when game is Live, stays through Final, hidden pre-game.

**Audit progress endpoint (added 2026-04-24):**
- `GET /api/audit-progress?provider=vultr|hetzner|oci&dir=<audit_dirname>&seeds_file=<path>` — returns JSON with per-box live progress + audit_attach proc status.
- Why HTTP instead of direct SSH: during a run, `data/<provider>_results/<dir>/` is EMPTY on bts-hetzner — `retrieve_one` only rsyncs at final teardown. Live progress lives in `/root/audit.log` on each box, reachable via the `bts`-user SSH key distributed during provisioning. Exposing as an endpoint means any tailnet caller (laptop, phone) can poll without its own SSH plumbing.
- Shell helper: `scripts/check_audit_progress.sh` — curl + jq + column, defaults to the current Vultr n=100 run. Env-var overridable for other audits.
- Response time ~15–20s for a 26-box fleet (parallel SSH via size-8 thread pool).

## Leaderboard Watcher (added 2026-05-01)

Standalone scraper that captures the public MLB.com BTS leaderboard plus per-user picks logs into parquet. Decoupled from the picks pipeline — failure cannot break daily picks.

```
src/bts/leaderboard/
  ├── auth.py           — load cookies from macOS Keychain (Mac) or file (Linux); xSid mint via POST /auth/login
  ├── endpoints.py      — discovered MLB.com BTS API URL templates
  ├── models.py         — pydantic schemas (LeaderboardRow, PickRow, SeasonStats)
  ├── scraper.py        — HTTP wrappers + parsers; deep active-streak pagination; browser identity; 403/429 kill-switch
  ├── ratelimit.py      — per-function min-gap decorator with optional jitter (next_gap)
  ├── storage.py        — parquet I/O (append-only user_picks; dedupe-on-read)
  ├── analysis.py       — consensus_pick (mtime-prefiltered), percentile_rank
  ├── static_capture.py — content-deduped archival of MLB's PUBLIC static JSONs (stdlib-only, standalone-runnable)
  └── cli.py            — bts leaderboard {scrape, capture-static, status, backfill}
```

**Scheduling: the authenticated deep scrape is DISABLED as of 2026-07-04 (MLB ToS §1(xi) prohibits automated collection; running it on the contest account risked the streak for analytical payoffs that deflated — see `bts_index.md`).** `bts-leaderboard.timer` was `disable --now`'d on the box; **do NOT re-enable it** (incl. via `install-leaderboard-systemd.sh`, which does `enable --now`) — the off state is intentional. Deploys don't touch it. What STILL runs: the `*/30` cron `bts leaderboard capture-static` (public unauthenticated static JSONs — negligible ToS exposure, holds the useful `probabilityStarter` archive) plus the own-account low-volume crons (`fetch-contest-streak`, `check-pick-entered`). **The deep-scrape CODE is retained** for a planned one-time end-of-season archive grab (~2026-09-27; re-enable for one run or `scrape --profile-top-n <big>` once, then disable). When it ran (2026-07-03/04): twice daily at 01:00/10:00 ET, ~25 min per run at the jittered gap.

**Human-fidelity + throttle safety (2026-07-03, `endpoints.browser_headers` / `RateLimitedError`):** the scrape runs at higher authenticated volume (deep board + ~300 profiles) on Eric's REAL contest account, so it presents a normal Chrome-on-macOS identity with the BTS app's own XHR headers (NOT a self-identifying UA), draws jittered 2.0–4.5s gaps (not a fixed metronome), and shuffles profile fetch order. An HTTP **403/429 raises `RateLimitedError` which ABORTS the whole scrape** (back off, don't hammer the account into a hard block), records `rate_limited` in `scrape_status.json`, and DMs Eric (throttled, cooldown only on a real send). This is request hygiene for one authorized personal account on a public leaderboard — no IP/proxy tricks. **Login-time classification joined 2026-08-11:** a 429 at auth/login raises `RateLimitedLoginError` (never retried, never advises the cookie-recapture flow — which would add auth traffic), and non-empty non-JSON 200s are treated as challenge-page-shaped (no retry); only transport/5xx/zero-length-200 shapes retry, so the kill-switch philosophy now covers the login POST too.

**Storage:**
- `data/leaderboard/leaderboard_snapshots/{YYYY-MM-DD}.parquet` — 3 tabs × top-100 rows + the DEEP active_streak board (paginated to streak ≥ 3, ~10-30k rows, deduped by userId; `user_id` column added 2026-07-03, older files lack it). Deep depth means users remain visible after a streak reset — the censoring fix for field-level analyses (calibration frames, streak transitions).
- `data/leaderboard/user_picks/{username}.parquet` — append-only per-user picks log. Pick-log PROFILES are capped at `profile_top_n` TOTAL (default 300, prioritizing the deep active board = the useful pick-logs, then other tabs) to bound the authenticated footprint — the full deep board still lands in the snapshot, only profiles are capped. (Still keyed by sanitized username — a known re-key-by-user_id migration is deferred.)
- `data/leaderboard/season_stats/{YYYY-MM-DD}.parquet` — best/active streak + accuracy per user
- `data/leaderboard/static_snapshots/{feed}/{UTC-stamp}.json` — content-deduped captures of the six public static JSONs; `capture_status.json` records last run/store per feed. ⚠️ Fastly gzips these even with no `Accept-Encoding` and urllib (unlike httpx) won't auto-decompress → `static_capture._maybe_gunzip` magic-byte sniff (without it every feed parses as "invalid"). Snapshots stored before 2026-07-09 are plain `.json`; newer ones are gzipped `.json.gz` (audit F14) — readers must accept both.
- `data/leaderboard/scrape_status.json` — per-run completeness/throttle metadata (`rate_limited`, `active_streak_complete`, `n_leaderboard_rows`, `n_profiles`) so a truncated/aborted snapshot isn't mistaken for the whole field.

**Public static JSONs (discovered 2026-07-03 via the app JS bundle — the app is a web app; www.mlb.com 403s datacenter IPs but `mlb-play.mlbstatic.com` serves everywhere):** `most_selected_players.json` = per-round most-picked players with `numberSelections` + MLB's own `probabilityStarter` model, populated for today AND tomorrow — i.e. PREGAME field consensus as an unauthenticated static file. `suggested_players.json` = MLB's own recommended picks up to 2 rounds ahead. `rounds/units/players/checksums.json` are the lookup tables that make old rows interpretable later (units.json only carries current games — without archival, past unitId→game mappings are lost). ⚠️ `numberSelections` semantics unverified (counts looked identical across adjacent rounds at first capture — possibly a rolling popularity stat, not per-round counts); resolve from a few days of captures before building on it.

**Other-user pick visibility (probed 2026-07-03):** another user's pending pick appears in their profile/round rows pre-settlement but with `playerId` server-side REDACTED (null); `unitId` (the game) and the live hits/atBats line are visible once the row exists. Own rows are never redacted, and `GET api/predictions` exposes the own pending same-day entry (the check-pick-entered v2 source). `api/prediction/leaders` returned 500 (params unknown); `allParticipantsCount` ≈ 57k.

**Auth:**
- Mac: `security` keychain (account `claude-cli`, service `mlb-bts-session-cookies`)
- Linux: `~/.bts-leaderboard-cookies.json` (chmod 600); falls back to `pass` if installed
- One-time interactive capture via `scripts/capture_bts_cookies.py` (Playwright)

**First production scrape (2026-05-01):** captured 309 unique users across 4 tabs; all picks resolved via static lookups for batter_name + batter_team. opponent_team / home_or_away are NaN for historical picks (units.json only carries current/upcoming games). See `docs/superpowers/specs/2026-05-01-bts-leaderboard-watcher-design.md` + `docs/superpowers/plans/2026-05-01-bts-leaderboard-watcher.md`.

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
14. **Single-seed benchmarking is dangerous (2026-04-14/15)**: LightGBM's `random_state=42` produced a +1.29σ outlier on MDP P(57). The 16-seed audit showed true MDP P(57) was 3.50% ± 2.11pp under the OLD single-seed policy, not the claimed 8.91%. Fixed with `BTS_LGBM_RANDOM_STATE` env var + multi-seed audits. Also found and fixed a hardcoded `random_state=42` in `scripts/arch_eval.py:177` that made `rebuild_policy.py` ignore the env var entirely.
15. **Pooled-seed MDP policy (Option 7, SHIPPED 2026-04-15)**: Computing quality bins from profiles pooled across 24 seeds (instead of a single seed) drops per-bin SE by √24 and produces a policy robust to any single seed's luck. A/B validated with four independent signals: 24/24 LOO wins on walk_forward_backtest (+1.93pp), 8/8 on blend_walk_forward cross-path (+1.59pp), MC bootstrap (+2.59pp, 60/80 seed-seasons), chronological replay (+2.31 mean max_streak). Production's in-sample P(57) dropped from 8.91% (inflated) to 8.17% (honest). The new in-sample estimate closely matches LOO holdout (8.38%), confirming it's not overfitting. **Two-metric reporting standard adopted**: avg P@1 for screening, mean MDP P(57) across seeds for shipping. Never ship on single-seed MDP again. Full 48-seed × 32-experiment audit on 4 × Hetzner CPX51 **completed 2026-04-23 19:30 ET** (audit_attach retrieve + auto-teardown clean) + 52-seed Vultr extension on 26 × vhp-8c-16gb-amd in progress (ETA Sat 2026-04-25 morning) = **combined n=100** ready Sat afternoon. Run `scripts/analyze_audit_results.py` against merged data. OCI E5.Flex was attempted as a third provider 2026-04-21 evening but abandoned — new-account 90-day quota moratorium + concurrent-launch accounting cap made it unviable until ~2026-07-20. OCIProvider code preserved in `audit_driver.py` for post-moratorium use.
15. **Any function receiving the full prediction DataFrame must filter by game status**: Predictions include batters from all scheduled games (started, postponed, etc). Functions like `should_lock` that compare against projected picks must exclude non-Preview games, or postponed/finished games will pollute the comparison.
16. **Analytical forward evaluator vs MC bootstrap can disagree in sign for bin-structure changes (2026-04-16)**: Isotonic calibration of `p_game_hit` before binning showed +1.14pp P(57) under the analytical `evaluate_mdp_policy` (t=+3.14, 18/24 seeds) but −1.12pp under MC bootstrap (t=−3.43, 45/120 wins). Root cause: the analytical evaluator computes each policy's value against its OWN bin partition. When a change shifts days across bin boundaries (42% reassigned under isotonic), A and B are evaluated on different day-partitions — not a true A/B. **Rule**: any BTS MDP policy change that alters bin boundaries (calibration, different n_bins, alternative discretization) must be validated with `scripts/mc_replay_ab.py`-style MC bootstrap, not just the analytical. The shipped pooled-policy used both; that's the discipline to keep. Full rejection record at `memory/project_bts_2026_04_16_calibration_rejected.md`.
17. **Gate-class verdicts can be CI-method artifacts (2026-05-04)**: v1's `HEADLINE_BROKEN` classification (5-fold percentile CI ci_upper=0.0375 below half-headline=0.04085) collapsed under v2.6's profile-level paired hierarchical block-bootstrap (Politis–Romano stationary, expected_block_length=7, n=500). At the same threshold, all 6 v2.5 ablation cells gate REDUCED. Point-estimate attribution (B alone +1.67pp, C alone +1.67pp via per-bin rho_pair / per-fold MDP solve) survives the methodology change; the gate-class transition does not. **Rule**: when a methodology change produces a verdict-gate transition, default to two robustness checks before announcing — (1) is the point-estimate attribution actually driven by the change being announced, and (2) is the gate-class transition robust to alternative CI methods at the same threshold. Skipping either check leads to overclaim that has to be retracted under review. Synthesis: `docs/sota_audit/2026-05-04-falsification-harness-synthesis.md`.
18. **PA-frame attribution corrects DD-bias in realized-picks calibration (2026-05-04)**: The 2026-04-25 chronic ~7pp overconfidence finding (n=48) used streak-result attribution, which is biased on double-down days (`result=miss` could mean primary missed or DD missed or both — a hit DD gets attributed as miss whenever the primary missed). The 2026-05-01 health-fix `b08769d` corrected this in alert code; PR #17 was the first retrospective analysis to apply the fix. With PA-frame `(batter_id, date) → day_had_any_hit` attribution on a regime-bounded n=30 stratum (post-pooled-MDP, pre-bpm), primary picks are well-calibrated (gap +1.8pp); the overconfidence concentrates in the **double-down slot** (gap +19.2pp). Strict-current-model n=5 was inconclusive. The DD-slot finding is exploratory — needs more resolved post-bpm picks to track. Memo: `docs/sota_audit/2026-05-04-realized-picks-calibration.md`.
