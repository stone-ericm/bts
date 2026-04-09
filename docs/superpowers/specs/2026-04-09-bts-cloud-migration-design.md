# BTS Cloud Migration — Design Spec

**Date:** 2026-04-09
**Status:** Design spec (pending user review)
**Supersedes:** `2026-03-31-model-orchestration-design.md` (SSH cascade approach retired)
**Related:** `2026-04-08-scheduler-timing-simulation-design.md` (will consume real lineup timing data produced by this work)

## Goal

Move BTS production off personal hardware (Mac, Alienware, Pi5) and onto a single always-on Fly.io machine. Mac becomes a dev-only environment for experimentation; all production services (data pulls, model training, scheduler daemon, dashboard, Bluesky posting, reconciliation) run on the cloud VM.

**Why:** production should not depend on personal workstations being on. Mac sleeps, Alienware has had charmap bugs, and Pi5 — while reliable — ties production to a physical device in the house. A professional architecture has production compute that is:

- Always available (no dependency on "is Mac on?")
- Versioned and reproducible (config + code + data in git + object storage)
- Debuggable from anywhere (no SSH into specific physical machines)
- Deployable via git push (no manual ssh + systemctl dance)

## Non-Goals / Out of Scope

- **Changing the model, features, or MDP strategy.** This is a migration, not a re-architecture of the prediction logic. All model code, feature engineering, and policy files move unchanged.
- **Adding new prediction capabilities.** No new data sources, no new features, no new backtests beyond what's needed to validate scheduler timing changes.
- **Dashboard UI redesign.** The dashboard moves as-is, just to a new host. Tailscale-only access; no auth layer.
- **Replacing Bluesky.** Bluesky remains the event log and public face. Single bot account, single app password.
- **Migrating Pi5's other services.** Pi5 keeps claude-shared, investigation-kb, etc. Only BTS services move.
- **Adding a CI/CD staging environment.** Single production environment on Fly; Mac is the staging/experiment environment via local execution.
- **Replacing LightGBM or any model dependency.** Same stack, just running elsewhere.

## Current State (2026-04-09)

```
┌─────────┐       ┌──────────────────┐
│   Mac   │◄──SSH─│       Pi5        │
│ worker  │       │  orchestrator    │
└─────────┘       │  - bts schedule  │
                  │  - bts dashboard │
┌──────────┐      │  - cron jobs     │
│Alienware │◄─SSH─┤  - Bluesky post  │
│ worker   │      └──────────────────┘
└──────────┘              ▲
                          │
                    ┌─────┴──────┐
                    │  Bluesky   │
                    │  (output)  │
                    └────────────┘
```

**Daily happy path:**
1. Mac's nightly cron runs `bts data pull` + `bts data build` (raw JSON → parquets)
2. Pi5's `bts-scheduler.service` fires lineup checks at `game_time - 45 min`
3. Scheduler SSHes into Mac (fallback Alienware), runs `bts predict-json`, receives predictions
4. Scheduler applies strategy, selects pick, posts to Bluesky, monitors for results
5. `bts reconcile` (Pi5 cron) re-checks last 8 days for scoring corrections

**Problems with current state:**
- Production fails if Mac is off (Alienware fallback has had its own issues)
- Data refresh on Mac only; if Mac's data gets corrupted or stale, workers can diverge silently
- Dashboard is LAN-bound (or Tailscale exit-node workaround when Eric is away)
- Deployment involves 3 machines with different OSes; each requires separate setup and maintenance
- "How is BTS deployed?" is a multi-paragraph answer instead of "read `fly.toml`"

## Target State

```
┌──────────────────────────────────────────┐
│         Fly.io Machine (IAD region)        │
│  ┌─────────────────────────────────────┐  │
│  │  Docker container (uv + Python)     │  │
│  │                                     │  │
│  │  bts-scheduler  (main daemon)       │  │
│  │  bts-dashboard  (HTTP on tailnet)   │  │
│  │  cron: reconcile, check-results,    │  │
│  │        daily data pull+build        │  │
│  │  Tailscale client (sidecar)         │  │
│  │                                     │  │
│  │  /data (50 GB Fly volume)           │  │
│  │    ├── raw/2026/*.json              │  │
│  │    ├── processed/pa_*.parquet       │  │
│  │    ├── models/blend_*.pkl           │  │
│  │    ├── picks/*.json                 │  │
│  │    └── picks/streak.json            │  │
│  └─────────────────────────────────────┘  │
└──────────┬─────────────────┬───────────────┘
           │                 │
    ┌──────▼──────┐   ┌──────▼──────┐
    │ Cloudflare  │   │   Bluesky   │
    │    R2       │   │  (output +  │
    │             │   │  event log) │
    │ - manifest  │   └─────────────┘
    │ - parquets  │
    │ - raw tar   │
    └─────────────┘
           ▲
           │ (pull only, for experiments)
           │
     ┌─────┴─────┐
     │    Mac    │
     │  (dev)    │
     └───────────┘
```

**New daily happy path:**
1. Fly container boots; scheduler daemon starts; Tailscale sidecar joins tailnet
2. Cron inside container runs `bts data pull` + `bts data build` nightly (raw → parquets, incremental)
3. Scheduler fires lineup checks according to config; runs `bts predict-json` locally (no SSH, same process)
4. Scheduler applies strategy, selects pick, posts to Bluesky, monitors for results
5. Reconciliation cron re-checks last 8 days for scoring corrections
6. State lives on Fly volume; snapshots provide Tier 1 recovery; `bts state regenerate` provides Tier 2 disaster recovery

**Pi5:** services disabled and removed. Pi5 hardware retained for its other roles (claude-shared, investigation-kb). Mac: `bts data pull` + `bts data build` nightly cron removed. Mac cron for BTS is empty post-migration. Alienware: BTS repo and config removed; machine retired from BTS pipeline entirely.

## Architecture

### Fly.io deployment

- **App name:** `bts`
- **Region:** `iad` (Ashburn, VA — close to MLB Stats API edge)
- **Machine size:** `shared-cpu-2x` with 4 GB RAM (~$7/mo)
- **Volume:** 50 GB at `/data` (~$7.50/mo)
- **IPv4:** none (Tailscale-only access, saves $2/mo)
- **Estimated total monthly cost:** ~$15/mo (compute + volume + negligible bandwidth)

### Container

- Single Docker image built from `Dockerfile` in repo root
- Base: `python:3.12-slim-bookworm`
- Installed: `uv`, `tailscale`, `caddy` (unused in tailnet mode but reserved), project deps via `uv sync --extra model`
- Runs a small supervisor process (shell script or `s6-overlay`) that starts: Tailscale client, bts-scheduler, bts-dashboard, and a cron loop for periodic jobs
- Single process model is simplest; if supervision feels fragile we can graduate to `s6-overlay` later

### Dashboard access

- Dashboard listens on `0.0.0.0:3003` inside the container
- Tailscale sidecar makes the container a tailnet node under `tag:bts-prod`
- Tailscale ACL rule restricts `tag:bts-prod` to serving traffic only within Eric's tailnet
- Eric accesses via MagicDNS name (e.g., `http://bts:3003`) from any Tailscale-equipped device (Mac, phone, iPad)
- No public domain, no TLS cert management, no auth layer on the dashboard itself

### R2 canonical data layer

- **Bucket:** `bts-backup-data` in Eric's existing Cloudflare account
- **Contents:**
  - `manifest.json` — index with per-file SHA-256, size, upload timestamp, git SHA of producer, `SCHEMA_VERSION`
  - `parquets/pa_{2017..2026}.parquet` (~200 MB total)
  - `models/probable_pitcher_lookup.json` (~1.8 MB, incremental cache)
  - `raw-archive-2017-2025.tar.gz` (~15 GB cold archive of historical raw JSON, uploaded once, read on demand)
- **Writer:** Fly machine (primary producer) via nightly `bts data sync-to-r2` after `bts data build`
- **Readers:** Mac (for pulling parquets when running local experiments); new Fly machines during bootstrap
- **Not in R2:** daily `blend_{date}.pkl` files (regenerable), `streak.json` and pick files (ephemeral state, Fly volume only)

### State persistence

**State lives only on the Fly volume during normal operation.** No auto-commits to git, no R2 backup of state, no external replication during steady-state. The volume is the working cache; the sources of truth for recovery are elsewhere.

**Recovery tiers:**

1. **Tier 1 — Fly volume snapshots** (automatic, default retention). Handles everyday corruption, bad deploys, accidental deletions. Recovery time: ~2 min to attach a snapshot to a new machine.
2. **Tier 2 — `bts state regenerate`** (new command). Handles disaster recovery: Fly region outage, volume loss, migration between providers. Reconstructs state from:
   - The BTS code (git, authoritative for logic)
   - The parquets (R2, authoritative for historical PA data)
   - The MLB Stats API (authoritative for schedules, lineups, results)
   - The Bluesky post history (authoritative for which picks were made and what results were recorded)
   - The committed initial-state snapshot (git, authoritative for pre-migration history)
3. **Tier 3 — Manual reconstruction.** If all of the above fail simultaneously, the system is in an extraordinary failure state that manual intervention must address.

### Initial state snapshot (committed to git once)

At the moment of migration cutover, a one-time snapshot of Pi5's current state is committed to the repo at `data/state/initial-state.json`. This file contains:

- `cutoff_date` — the latest date with fully-resolved results at export time
- `streak_at_cutoff` — current streak value
- `saver_available` — saver flag
- `historical_picks` — array of all pick records with Bluesky URIs, results, dates

The export command (`bts state export`) enforces that **no pick may be in an unresolved state** at export time (refuses to run if any `data/picks/{date}.json` has `result is None`). This makes "exported too early" operationally impossible.

The file is committed once, at migration. It is **not** updated by any automated process post-migration. Post-migration state is derived from Bluesky and MLB API, with the initial snapshot providing the pre-cutoff baseline.

### Schema versioning

- `src/bts/data/schema.py` gets a new `SCHEMA_VERSION` constant derived at import time as `sha256("\n".join(PA_COLUMNS))[:12]`
- `bts data build` asserts the produced DataFrame's columns exactly match `PA_COLUMNS`; fails loudly with the specific diff if not
- `manifest.json` in R2 carries `schema_version` as part of its metadata
- Workers (the Fly machine on bootstrap, Mac on experiment runs) verify manifest's `schema_version` matches their expected value before loading parquets; refuse to proceed on mismatch with an actionable error message
- Any schema change requires: edit `PA_COLUMNS` → run `bts data build` (assert validates) → run `bts data sync-to-r2` (pushes new parquets + manifest) → workers pick up new version on next sync

### Bluesky password consolidation

- `posting.py` and `dm.py` currently read `BTS_BLUESKY_PASSWORD` and `BTS_BLUESKY_DM_PASSWORD` respectively; both map to the same value per memory
- Migration PR refactors `dm.py` to use the shared `get_bluesky_password()` helper from `posting.py`
- A new canonical env var `BTS_BLUESKY_APP_PASSWORD` is read first; legacy names fall back for backward compatibility during cutover
- Fly gets a single `BTS_BLUESKY_APP_PASSWORD` secret; Pi5 keeps working with its two existing env vars until it is retired

### Secrets inventory

| Secret | Location | Purpose |
|---|---|---|
| `BTS_BLUESKY_APP_PASSWORD` | Fly secrets | Bluesky posting + DMs (unified) |
| `R2_ACCOUNT_ID` | Fly secrets | R2 sync |
| `R2_ACCESS_KEY_ID` | Fly secrets | R2 sync |
| `R2_SECRET_ACCESS_KEY` | Fly secrets | R2 sync |
| `TS_AUTHKEY` | Fly secrets | Tailscale pre-auth key (ephemeral, tagged `tag:bts-prod`) |
| `FLY_API_TOKEN` | GitHub repo secrets | Deploy workflow (`superfly/flyctl-actions`) |

**Scope of `FLY_API_TOKEN`:** scoped deploy token for the `bts` app only (`fly tokens create deploy -a bts`), not an org-wide admin token. Blast radius on compromise: unauthorized BTS deploys only.

**Workflow guardrails:** deploy workflow triggers only on `push` to `main` (never on `pull_request`, which avoids secret exposure via fork PRs). Workflow declares `permissions: contents: read` to prevent accidental repo writes.

### Monitoring and alerting

**Heartbeat (scheduler → file):**
- Scheduler writes `/data/.heartbeat` every 30 seconds with JSON: `{timestamp, state, sleeping_until}`
- `state` is one of: `running | waiting_for_games | sleeping_until_X | idle_end_of_day`
- During scheduler sleep periods (between games), heartbeat explicitly declares "sleeping until X" so stale-heartbeat logic knows it's intentional

**Health endpoint (dashboard service):**
- `GET /health` reads `/data/.heartbeat`, returns `200 {"status": "ok", ...}` if fresh or sleeping-intentionally, `503` otherwise
- Response includes `last_heartbeat_age_sec` and `scheduler_state` for easy log debugging

**Fly HTTP health check:**
- Configured in `fly.toml` to hit `/health` every 60 seconds
- Tolerates 2 consecutive failures before action (≈3 minutes of real trouble)
- Restarts the machine on 3rd consecutive failure
- Self-heals from hung scheduler processes without manual intervention

**Healthchecks.io (external watchdog):**
- Free-tier check configured to expect pings every 5 minutes
- A tiny cron inside the container curls `https://hc-ping.com/<uuid>` on a 5-minute schedule
- Alerts Eric via email if no ping received for 10 minutes (i.e., both Fly health checks AND external monitoring see the machine as down)
- Cross-provider check against "Fly lying about its own status"

**Bluesky DM alerts:**
- Existing `dm.py` DM path kept; extended to fire on:
  - Scheduler exception (currently fires only on total cascade failure, which doesn't exist post-migration)
  - Missed pick: no pick posted by `earliest_game_start - 30 min` on a day with games
  - Scheduler heartbeat gone stale outside of declared sleep window (belt-and-suspenders vs Fly health checks)

### Scheduler timing tuning

The current scheduler timing parameters are hardcoded in `scheduler.py`:
- Lineup check offset: 45 min before first pitch
- Fallback post deadline: 15 min before first pitch
- This gives ~10 minutes of alert reaction time, which is insufficient

**Changes:**

1. **Extract hardcoded constants to config.** `scheduler.py:626` and `:680` get their `15` replaced with `sched_config["fallback_deadline_min"]` reads. New TOML keys:

```toml
[scheduler]
lineup_check_offset_min = 60   # was hardcoded 45
fallback_deadline_min = 35     # was hardcoded 15
missed_pick_alert_min = 30     # new — alert threshold
```

2. **Default values are tentative, to be validated by real lineup data.** The numbers above give 30 minutes of reaction time on alert, but are placeholder until the data collection (below) produces an empirical distribution. The validated values replace these before Phase 3 cutover.

3. **Validation gate:** before Phase 3 cutover goes live, `bts simulate backtest` (reusing the existing `2026-04-08-scheduler-timing-simulation-design.md` infrastructure, but with real lineup timing data as input instead of synthetic) must show that the new timing parameters don't drop P@1 by more than 1 percentage point compared to the current -45/-15 timing. If the drop exceeds 1pp, tune the parameters and re-run.

### Lineup posting time data collection

A gap in current knowledge is that no definitive historical data exists for when MLB lineups are posted relative to first pitch. The existing `2026-04-08-scheduler-timing-simulation-design.md` simulates this with `N(60, 15)` synthetic distribution; we want real data.

**New CLI command: `bts data collect-lineup-times`**

- Runs as a cron inside the Fly container, every 5 minutes during an "active window" from 3 hours before earliest game to first pitch of the latest game
- For each game scheduled on the current date, polls `/api/v1.1/game/{pk}/feed/live` and checks `liveData.boxscore.teams.{away,home}.players[*].battingOrder`
- Records the first poll where any batter has `battingOrder` populated as the "lineup confirmation time" for that side
- Writes per-day logs to `/data/lineup_posting_times/{date}.jsonl` with records:
  ```json
  {"game_pk": 775123, "game_time_et": "2026-04-10T19:05:00-04:00",
   "first_away_confirmed_et": "2026-04-10T17:38:00-04:00",
   "first_home_confirmed_et": "2026-04-10T17:42:00-04:00",
   "poll_count": 17}
  ```
- Stops polling a game once both sides have confirmed lineups
- Independent of the scheduler; no interference with production decisions

**New CLI command: `bts data analyze-lineup-times --from-date X --to-date Y`**

- Reads the JSONL logs, computes distributions of `game_time - first_confirmed_time` per side
- Emits a markdown report with percentiles (p10, p25, p50, p75, p90, p95, p99) and histogram
- Produces a recommendation: "with check_offset=X and fallback_deadline=Y, N% of lineups would be confirmed at lock time"

**Bootstrap from existing Pi5 state:**

- `bts data backfill-lineup-times --from-pi5-state` reads the `scheduler_state.json` files already on Pi5 (~6-7 days since 2026-04-03) and extracts coarse-grained data: for each `runs_completed` entry, record the `time` and `new_lineups` count, giving us a rough baseline of "at time T before first pitch, N additional lineups had confirmed."
- This is noisier than the new 5-minute polling but provides data before the collection script has been running for a week.

**Usage in Phase 2:** the collection script starts running on Pi5 at the beginning of Phase 0, in parallel with shadow running during Phase 2, producing a full real-world distribution by the time Phase 3 is ready. The simulation infrastructure from `2026-04-08-scheduler-timing-simulation-design.md` consumes this real distribution instead of its synthetic `N(60, 15)` assumption.

### Deployment flow

**Git-driven auto-deploy:**
- `.github/workflows/deploy.yml` triggers on push to `main` with `paths` filter limiting to code files (`src/**`, `Dockerfile`, `fly.toml`, `pyproject.toml`, `uv.lock`)
- Workflow steps: checkout → install flyctl → `flyctl deploy --remote-only`
- Uses `FLY_API_TOKEN` from GitHub Secrets
- On success, Fly builds the container image, starts a new machine, drains traffic from the old machine, swaps
- Zero-downtime deploys under normal operation; scheduler may lose a few heartbeats during swap but resumes cleanly

**Manual deploy escape hatch:**
- `flyctl deploy` from any machine with the Fly CLI installed and `FLY_API_TOKEN` in environment
- Used if GitHub Actions is itself broken

## Service Inventory (What Moves Where)

| Service | Current | Target | Notes |
|---|---|---|---|
| `bts data pull` | Mac cron | Fly cron | Incremental pull of new game feeds |
| `bts data build` | Mac cron | Fly cron | Incremental rebuild of current-season parquet |
| `bts data sync-to-r2` | (new) | Fly cron | Nightly after data build; atomic manifest update |
| `bts predict-json` | Mac (SSH target) | Fly (in-process) | Called directly by scheduler, no SSH |
| `bts schedule` (daemon) | Pi5 systemd | Fly supervisor | Main production daemon |
| `bts-dashboard` (HTTP) | Pi5 systemd | Fly supervisor | Tailscale-only access |
| `bts check-results` | Pi5 cron (1am) | Fly cron | Result reconciliation safety net |
| `bts reconcile` | Pi5 cron (2am) | Fly cron | 8-day rescan for scoring changes |
| Bluesky posting | Pi5 (in scheduler) | Fly (in scheduler) | Same code path |
| Bluesky DM alerts | Pi5 (via `dm.py`) | Fly (via `dm.py`) | Same code path |
| `scheduler_state.json` | Pi5 volume | Fly volume | Per-day ephemeral state, not migrated |
| `data/picks/*.json` | Pi5 volume | Fly volume | Initial state via committed snapshot |
| `streak.json` | Pi5 volume | Fly volume | Initial value via committed snapshot |
| `data/raw/**` (historical) | Mac (23 GB) | R2 cold archive (tarball) | Mac retains a local copy until confident |
| `data/raw/2026/*.json` | Mac | Fly volume | Incremental, re-pullable from MLB API |
| `data/processed/pa_*.parquet` | Mac | R2 canonical + Fly volume working copy | R2 is source of truth |
| `data/models/mdp_policy*.npz` | Git | Git (unchanged) | Already in git, already works |
| `data/models/blend_*.pkl` | Mac/Pi5/workers | Fly volume only | Regenerated on-demand, not backed up |
| `data/models/probable_pitcher_lookup.json` | Mac | R2 | Incremental cache, synced with parquets |
| Alienware worker | Alienware | **Retired** | Removed entirely from BTS pipeline |

## New CLI Commands

| Command | Module | Purpose |
|---|---|---|
| `bts data sync-to-r2` | `src/bts/data/sync.py` | Upload local parquets + lookup cache to R2, write manifest atomically |
| `bts data sync-from-r2` | `src/bts/data/sync.py` | Download parquets + lookup cache from R2 (respects manifest checksums) |
| `bts data verify-manifest` | `src/bts/data/sync.py` | Read-only check: manifest fresh, schema version matches, checksums OK |
| `bts data archive-historical-raw` | `src/bts/data/sync.py` | One-shot: tar historical raw JSON, upload to R2 as `raw-archive-*.tar.gz` |
| `bts data collect-lineup-times` | `src/bts/data/lineup_collect.py` | Poll MLB API during active window, log lineup confirmation times |
| `bts data analyze-lineup-times` | `src/bts/data/lineup_analyze.py` | Compute distribution + percentiles from collected data |
| `bts data backfill-lineup-times` | `src/bts/data/lineup_analyze.py` | Extract coarse data from existing Pi5 `scheduler_state.json` files |
| `bts state export` | `src/bts/state/export.py` | Export current state to `initial-state.json` (refuses if any pick is unresolved) |
| `bts state regenerate` | `src/bts/state/regenerate.py` | Rebuild pick history + streak from Bluesky + MLB API + initial-state.json |
| `bts state verify` | `src/bts/state/verify.py` | Drift check: regenerate to temp dir, diff against current state, report mismatches |

## Files and Code Changes

### New files

- `Dockerfile` (container image definition)
- `fly.toml` (Fly app configuration)
- `.github/workflows/deploy.yml` (auto-deploy on merge to main)
- `scripts/fly-entrypoint.sh` (supervisor: starts Tailscale, scheduler, dashboard, cron loop)
- `src/bts/data/sync.py` (R2 sync module, ~300 lines)
- `src/bts/data/lineup_collect.py` (lineup polling, ~100 lines)
- `src/bts/data/lineup_analyze.py` (distribution analysis + backfill, ~200 lines)
- `src/bts/state/export.py` (initial state snapshot export, ~100 lines)
- `src/bts/state/regenerate.py` (disaster-recovery state reconstruction, ~500 lines)
- `src/bts/state/verify.py` (drift checker, ~150 lines)
- `tests/test_sync.py`, `tests/test_state_export.py`, `tests/test_state_regenerate.py`, `tests/test_lineup_analyze.py` (new test modules)
- `data/state/initial-state.json` (committed at migration cutover, never auto-updated)

### Modified files

- `src/bts/data/schema.py` — add `SCHEMA_VERSION` auto-derived from `PA_COLUMNS`
- `src/bts/data/build.py` — add runtime assert that output columns exactly match `PA_COLUMNS`
- `src/bts/scheduler.py` — add heartbeat write loop; extract hardcoded `15` and `45` timing constants into config reads (`fallback_deadline_min`, `lineup_check_offset_min`); replace the `run_and_pick` import with direct calls to `run_pipeline` from `bts.model.predict` and `select_pick` from `bts.strategy`
- `src/bts/web.py` — add `/health` endpoint reading `/data/.heartbeat`
- `src/bts/dm.py` — refactor to use `posting.get_bluesky_password()`; accept new env var `BTS_BLUESKY_APP_PASSWORD` as primary
- `src/bts/posting.py` — extend `get_bluesky_password()` to prefer new env var
- `src/bts/cli.py` — register all new CLI commands
- `config/orchestrator.example.toml` — removed `[[tiers]]` sections (no more cascade); add `fallback_deadline_min` and `missed_pick_alert_min` to `[scheduler]`
- `CLAUDE.md` — update deployment section to reflect Fly-only architecture
- `ARCHITECTURE.md` — update to describe cloud-native layout

### Deleted files

- `src/bts/orchestrator.py` — SSH cascade logic is dead code post-migration
- Nothing else; the existing prediction, strategy, posting, data-build code all moves unchanged

## Cutover Plan

### Phase 0 — Prerequisites

External resources and prep work that can happen in parallel with code changes:

1. Create Fly app: `fly apps create bts`
2. Create Fly volume: `fly volumes create data --size 50 --region iad -a bts`
3. Create Cloudflare R2 bucket `bts-backup-data` + scoped API token
4. Create Tailscale OAuth pre-auth key with tag `tag:bts-prod` + update tailnet ACL to allow `tag:bts-prod` → reachable-by-user
5. Create Healthchecks.io project + check
6. Create Fly API deploy token: `fly tokens create deploy -a bts`
7. Add runtime secrets to Fly via `fly secrets set -a bts KEY=VALUE ...`: `BTS_BLUESKY_APP_PASSWORD`, `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `TS_AUTHKEY`
8. Add build/deploy secret to GitHub repo secrets: `FLY_API_TOKEN` (used by the deploy workflow only)
9. Upload historical raw JSON archive to R2: on Mac, `tar czf raw-archive-2017-2025.tar.gz data/raw/2017..2025 && bts data archive-historical-raw`
10. Initial R2 parquet upload: on Mac, `bts data sync-to-r2` (~200 MB one-time push)

**Phase 0 is reversible in minutes** by deleting the Fly app, R2 bucket, Healthchecks.io project, and GitHub secrets. No production impact.

### Phase 1 — Infrastructure code

Write and merge the code changes without activating production impact:

1. Merge PR adding `Dockerfile`, `fly.toml`, GitHub deploy workflow, `scripts/fly-entrypoint.sh`
2. Merge PR adding `src/bts/data/sync.py`, `src/bts/data/schema.py` SCHEMA_VERSION, `build.py` assert
3. Merge PR adding `src/bts/state/export.py` and `src/bts/state/regenerate.py`
4. Merge PR adding lineup data collection (`lineup_collect.py`, `lineup_analyze.py`, backfill command)
5. Merge PR with scheduler heartbeat, `/health` endpoint, config extraction for timing constants, Bluesky password consolidation
6. Deploy workflow auto-pushes the image to Fly on each merge; Fly machine starts but **runs in shadow mode** (no authoritative picks)

At the end of Phase 1, the Fly machine is running the BTS codebase end-to-end, but its posting path is disabled via `shadow_mode = true` in config.

Also at the end of Phase 1, the lineup data collection script (`bts data collect-lineup-times`) is running on Pi5 as a systemd timer, gathering real distribution data for the scheduler timing tuning.

### Phase 2 — Parallel shadow running

Pi5 remains authoritative. Fly runs the full pipeline in shadow mode (no Bluesky posts, writes to a separate results directory). Compare daily:

1. **Fly shadow daemon** runs the normal scheduler flow but:
   - Skips Bluesky posting entirely
   - Writes predictions and pick decisions to `data/shadow/{date}/fly.json`
   - Uploads daily shadow output to R2 at `shadow/{date}/fly.json` for Mac-side diffing
2. **Daily comparison script** (runs on Mac as a cron or manual check):
   - Downloads both `shadow/{date}/fly.json` from R2 and the real `data/picks/{date}.json` from Pi5
   - Compares strictly: exact match on `batter_id`, `pitcher_id`, `game_pk`, `double_down` presence and match, skip-vs-pick classification, and floating-point fields within `|Δ| < 1e-6`
   - Any strict mismatch is investigated; resets the shadow-matching counter
3. **Lineup data collection** runs in parallel on Pi5 via the already-deployed `bts data collect-lineup-times` timer
4. **Phase 2 exit gate:** all of:
   - 7 consecutive days where shadow output matches Pi5 output strictly
   - `bts data analyze-lineup-times` run on the collected data produces a distribution report Eric has reviewed
   - Scheduler timing parameters finalized (either confirmed at the tentative 60/35 defaults or re-tuned based on the data)
   - Parameter choice validated via backtest showing P@1 drop ≤ 1pp vs current -45/-15 timing

During Phase 2, Pi5 continues to be the real production. Fly is provably working but not in charge.

### Phase 3 — State freeze and cutover

Executed on an evening when Phase 2 exit criteria are met. Sequenced carefully:

1. **Wait for today's games to finish.** All picks for the current date must have `result != None` on Pi5.
2. **SSH to Pi5** and run: `cd ~/projects/bts && bts state export --to data/state/initial-state.json`. The export command's resolved-state guard fails loudly if any pick is unresolved; abort and wait if so.
3. **Back on Mac**, pull the snapshot from Pi5 to Mac: `scp pi5:~/projects/bts/data/state/initial-state.json ~/projects/bts/data/state/`.
4. **Commit on Mac**: `git add data/state/initial-state.json && git commit -m "migration: freeze state at cutoff YYYY-MM-DD"` and push.
5. **Merge the cutover PR** that flips `shadow_mode = true` to `shadow_mode = false` in the Fly config. GitHub Actions deploys the new image; Fly machine swap takes ~30 seconds.
6. **Stop Pi5 services** (stopped but not disabled — rollback path):
   ```
   ssh pi5 'systemctl --user stop bts-scheduler bts-dashboard'
   ```
7. **Verify Fly is alive**:
   - `curl http://bts:3003/health` via Tailscale should return `200 {"status":"ok",...}`
   - `fly logs -a bts` shows the scheduler running
   - Healthchecks.io shows ping within expected window
8. **48-hour observation window.** Pi5 services are stopped but not disabled. If anything goes wrong within 48 hours, execute rollback.

### Phase 3-rollback (if anything wobbles in 48 hours)

Rollback is cheap because Pi5 state is intact:

1. `ssh pi5 'systemctl --user start bts-scheduler bts-dashboard'` (Pi5 resumes authority within ~30 seconds)
2. `fly machine stop $(fly machines list -a bts --json | jq -r '.[0].id')` (pause Fly)
3. Investigate Fly failure via `fly logs` and Healthchecks.io history
4. Fix, redeploy to Fly shadow mode, re-run Phase 2 (reset 7-day counter), retry Phase 3 when ready

**Crucial rollback property:** Pi5 state hasn't been touched since Phase 3 step 6; the scheduler resumes exactly where it left off. The only thing that changed during the cutover attempt is that Fly made some Bluesky posts; those are reconciled into Pi5's view on the next `bts reconcile` run since the Bluesky account is the same.

### Phase 4 — Decommission

After 48 hours without rollback, make the cutover permanent:

1. **Disable Pi5 services**:
   ```
   ssh pi5 'systemctl --user disable bts-scheduler bts-dashboard'
   ssh pi5 'rm ~/.config/systemd/user/bts-scheduler.service ~/.config/systemd/user/bts-dashboard.service'
   ssh pi5 'systemctl --user daemon-reload'
   ```
2. **Remove Mac crons** for `bts data pull` + `bts data build`
3. **Retire Alienware**: delete `C:\Users\stone\projects\bts`, remove deps, remove from any config files
4. **Delete dead code**: remove `src/bts/orchestrator.py` (SSH cascade), remove `[[tiers]]` section from `orchestrator.example.toml`, remove SSH worker references from `scheduler.py`. Merge as a cleanup PR.
5. **Archive superseded spec**: move `docs/superpowers/specs/2026-03-31-model-orchestration-design.md` to `docs/superpowers/specs/archived/` with a prefix note that it is superseded by this migration spec
6. **Update documentation**:
   - `CLAUDE.md`: rewrite the deployment section to describe Fly-only
   - `ARCHITECTURE.md`: update the architecture diagram
   - Memory files in `~/.claude/projects/-Users-stone/memory/`: refresh `project_bts.md` to reflect the new cloud-only layout
7. **Turn down the lineup data collection on Pi5** (optional; it's been running in parallel and is no longer strictly needed — can be left running indefinitely or turned off)

## Testing Strategy

### Unit tests (Phase 1 PRs)

- `test_sync.py`: R2 client mock + fixture bucket. Tests `sync-to-r2` writes manifest with correct checksums, `sync-from-r2` downloads only changed files, `verify-manifest` catches schema version mismatch.
- `test_schema.py`: `SCHEMA_VERSION` changes when `PA_COLUMNS` changes, is stable otherwise. `build_season` assert fires on column drift.
- `test_state_export.py`: export refuses when unresolved picks present; produces valid JSON structure.
- `test_state_regenerate.py`: with a mocked Bluesky post history + MLB API responses, regeneration produces expected streak + pick records.
- `test_lineup_analyze.py`: distribution computation produces correct percentiles on fixture data; backfill extracts sensible values from sample `scheduler_state.json`.

### Integration tests (Phase 1)

- Docker image builds and boots; container supervisor starts scheduler + dashboard; `/health` endpoint responds 200 after startup
- Tailscale sidecar joins tailnet with test key; dashboard reachable via tailnet hostname
- End-to-end on a local fixture date: parquets from R2, run `bts predict-json`, verify JSON output shape matches schema

### Production validation (Phase 2)

- 7 consecutive days of strict shadow matching (the primary validation)
- Backtest delta for scheduler timing parameters within tolerance (≤ 1pp P@1 drop)
- Lineup distribution report reviewed and parameters confirmed

### Post-cutover (Phase 3)

- 48-hour observation window with daily manual verification: picks posted, results reconciled, dashboard responsive, health checks green, Healthchecks.io seeing pings
- One manual `bts state verify` run to confirm regeneration matches current state (catches any silent corruption from the cutover process)

## Risks and Mitigations

### Risk: Fly regional outage during a streak-critical day

**Mitigation:** volume snapshots are cross-region by default; a new Fly machine can be created in any region and attached to a restored snapshot. For a truly catastrophic Fly failure, `bts state regenerate` runs from any provider (local Mac, a new cloud) using only Bluesky + MLB API + R2, giving Eric a cross-provider escape hatch. The 48-hour rollback window during cutover is the belt-and-suspenders for "Fly turns out to be fundamentally bad for this workload."

### Risk: `bts state regenerate` bit-rots because it's rarely run

**Mitigation:** `bts state verify` runs weekly as a Fly cron, regenerating to a temp dir and diffing against current state. Mismatches alert via Bluesky DM. This catches parser drift from Bluesky post format changes, MLB API changes, or library version shifts before they matter for a real recovery.

### Risk: Scheduler timing changes degrade accuracy

**Mitigation:** Phase 2 exit gate requires backtest validation showing ≤ 1pp P@1 drop. If the real lineup distribution collected during Phase 2 shows that the tentative 60/35 values are too aggressive, parameters are tuned before Phase 3. Phase 3 is blocked until validation passes.

### Risk: Lineup data collection runs for 7 days and finds that lineups don't post early enough to give Eric a meaningful reaction window

**Mitigation:** the timing tuning is parameterized. If the data shows, e.g., "p95 of lineups confirm at game_time - 40 min," the fallback deadline can't be earlier than 40 min or we lose too many confirmed-lineup picks. The realistic answer might be a 20-25 minute alert window instead of 30. This is a *real* trade-off that the data exposes, not a bug.

### Risk: Bluesky post format inconsistency from past hand-posts breaks regeneration

**Mitigation:** the committed `initial-state.json` captures all pre-migration history, including hand-posted days, as a frozen record. Regeneration uses this file for dates before the cutoff, and only relies on Bluesky parsing for post-cutoff dates. Going forward, the bot is the only poster (policy, not code).

### Risk: GitHub public repo + committed state file accidentally exposes something sensitive

**Mitigation:** `initial-state.json` contains only public information: pick records, batter names, Bluesky URIs, streak counts, result outcomes. Nothing in state is sensitive — BTS picks are already public on Bluesky. Explicit allowlist of fields in the export command (rather than dumping arbitrary state) prevents accidental leakage if future code ever adds sensitive fields.

### Risk: Migration PR breaks local `bts run` / `bts predict` on Mac

**Mitigation:** the legacy code paths stay functional during migration. `bts run` and `bts predict` on Mac read from local parquets unchanged. Only the `orchestrator.py` SSH cascade is deleted in Phase 4; everything prediction-related stays intact. Mac experiments are unaffected.

### Risk: Deploy workflow fails mid-deploy, leaving Fly in a half-updated state

**Mitigation:** Fly deploys are atomic at the machine level; a new machine starts with the new image, drains traffic from the old machine, then the old machine stops. A failed deploy leaves the old machine still running. Manual recovery is `fly machine status -a bts` + restart the stuck machine.

## Cost Summary

| Line item | Monthly cost | Notes |
|---|---|---|
| Fly shared-cpu-2x (4 GB) | ~$7 | Always-on |
| Fly volume (50 GB) | ~$7.50 | $0.15/GB |
| Fly IPv4 | $0 | None (Tailscale-only) |
| Fly bandwidth | ~$0 | Egress < 1 GB/mo |
| Cloudflare R2 storage (20 GB tarball + 200 MB parquets) | ~$0.30 | $0.015/GB/mo |
| Cloudflare R2 egress | $0 | Free per R2 |
| Tailscale | $0 | Personal tier |
| Healthchecks.io | $0 | Free tier |
| GitHub Actions | $0 | Public repo |
| **Total** | **~$15/mo** | |

Compared to current cost of $0/mo running on owned hardware; the $15/mo buys independence from personal workstations, zero-maintenance host, and production-grade deployment.

## Explicit Dependencies on Other Specs

- **`2026-04-08-scheduler-timing-simulation-design.md`**: provides the backtest infrastructure for validating scheduler timing parameter changes. This migration adds a real-data input layer (via `bts data collect-lineup-times`) that the simulation can consume instead of its synthetic `N(60, 15)` distribution.
- **`2026-03-31-model-orchestration-design.md`**: superseded by this spec. The SSH cascade approach it describes is retired; the code module (`src/bts/orchestrator.py`) is deleted in Phase 4.
- **`2026-04-03-dynamic-lineup-scheduler-design.md`**: the scheduler module being migrated. Timing constants are extracted into config per this spec; the scheduler's behavior is otherwise unchanged.

## Open Questions (to be resolved before Phase 3)

1. **Exact scheduler timing values** — the tentative `lineup_check_offset_min=60, fallback_deadline_min=35, missed_pick_alert_min=30` are placeholders until real data from `bts data analyze-lineup-times` is available. Will be finalized during Phase 2.
2. **Healthchecks.io ping cadence** — 5 minutes is a reasonable default but may need adjustment if it produces false alarms during scheduler sleep windows.
3. **Initial state snapshot format version** — current spec defines v1; if future work requires adding fields to `initial-state.json`, the export command will need to handle schema migration.
4. **Whether to delete `src/bts/orchestrator.py` immediately or keep it for a release cycle** — current plan is to delete in Phase 4 after 2 weeks of cloud operation. If Eric prefers a longer retention, the file can stay with a `# DEPRECATED` header.

## User Review Gate

Once this spec is written, Eric reviews the document for:
- Missing decisions that should be captured
- Incorrect assumptions about his intent
- Scope creep or scope shortfalls
- Architectural misunderstandings

After Eric approves, the next step is invoking the `writing-plans` skill to turn this spec into a phased implementation plan with concrete tasks per phase.
