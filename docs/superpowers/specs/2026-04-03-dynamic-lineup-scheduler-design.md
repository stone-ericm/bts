# Dynamic Lineup Scheduler Design

**Date**: 2026-04-03
**Goal**: Replace fixed cron runs with a game-time-aware scheduler that makes predictions on confirmed lineups and commits picks only when data quality is sufficient.

## Problem

The current system runs predictions at 3 fixed times (11am, 4pm, 7:30pm ET). Most lineups aren't posted at 11am, so predictions rely on projected lineups (prior game's lineup). The densest bucket decision is made blind. By 4pm, early games may have already started — locking a pick that was made on stale data.

**Core issue**: The pick commitment point is decoupled from lineup data availability.

## Design

### Scheduler Daemon

A long-running process on Pi5 (`bts schedule`) that manages the full daily lifecycle: lineup monitoring, prediction runs, pick updates, Bluesky posting, and result checking.

**Daily lifecycle:**

1. **Morning init (~10am ET)**: Fetch today's MLB schedule. Extract all game start times. Group games into time buckets (Early < 4pm, Prime 4-8pm, West >= 8pm).

2. **Compute run times**: For each game, schedule a lineup check at `game_time - 45 minutes`. Cluster checks that fall within 10 minutes of each other into a single run (e.g., 7:05pm and 7:10pm games both trigger one check at ~6:20pm).

3. **At each scheduled run**:
   a. Fetch lineup status for all today's games (one schedule API call).
   b. If no new confirmed lineups since last run → skip, no SSH to workers.
   c. If new confirmed lineups → SSH to Mac (Alienware fallback) → `bts predict-json --date X` → full 12-model blend on all batter-game slots.
   d. Re-evaluate strategy: densest bucket, MDP policy, pick selection.
   e. Save candidate pick to JSON (overwriting previous candidate).

4. **Posting decision**: After each prediction run, post to Bluesky (lock the pick) if BOTH conditions are met:
   a. The picked player's game has a confirmed lineup.
   b. The gap between the top pick and the next-best pick with a projected lineup exceeds `early_lock_gap` threshold. (If all remaining games also have confirmed lineups, this condition is trivially met.)

5. **After Bluesky post**: Pick is locked. Scheduler stops all remaining lineup checks for the day. Transitions to result-checking mode.

6. **Fallback**: If it's 15 minutes before the picked game's first pitch and the lineup still hasn't confirmed → post on projected data. Don't miss the game.

### Early Lock Gap Threshold

**`early_lock_gap`**: Configurable parameter (in the TOML config). Default TBD — derived empirically via backtesting during implementation.

**Purpose**: When the top pick has a confirmed lineup but later games don't yet, only lock early if the top pick is meaningfully better than any projected-lineup alternative. If the gap is small, wait — confirmed lineups for later games could change the ranking.

**Backtesting approach**: Simulate the scheduler against historical seasons. For each day, compare the pick made with full confirmed lineups vs. the pick made at each earlier checkpoint. Find the gap threshold that minimizes cases where waiting would have changed the pick to a better outcome.

### Densest Bucket

Bucket density counts all scheduled games regardless of lineup status (confirmed or projected). The number of games in a time slot is known from the morning schedule — lineup confirmation doesn't change how many games are in a bucket.

The 78% override threshold (`OVERRIDE_THRESHOLD`) remains: if the top overall pick exceeds 78% P(hit), it bypasses the bucket rule.

### Doubleheader Game 2

Doubleheader game 2 has a fluid start time that shifts based on game 1's length. The initial `game_time - 45min` from the morning schedule may be inaccurate.

**Handling**: After the initial scheduled check, if game 2's lineup is not yet confirmed, re-check every 15 minutes until a confirmed lineup appears or the game starts. The scheduler detects doubleheader game 2 by checking for two games with the same team on the same date.

### Check-Results Polling

After the pick is locked, the scheduler transitions to result-checking mode:

- **1am ET**: First check of the picked game's status.
- **If Final** → record hit/miss in pick JSON, update streak, stop.
- **If Live** → re-check every 15 minutes (extras, resumed rain delay).
- **If Suspended or other non-terminal status** → stop polling. Flag pick as `result: "suspended"`. Don't record hit/miss — game resumes another day.
- **Cap at 5am ET** → if still Live, flag as `result: "unresolved"`, log warning.

The dashboard and streak logic treat `suspended` and `unresolved` as neither hit nor miss — streak does not advance or reset until resolution.

### Worker Failures

Same as current behavior: if all SSH tiers fail, send a Bluesky DM notification. The last successful candidate pick remains saved. If the posting deadline approaches, post the last candidate.

## What Changes

| Component | Change |
|---|---|
| **New: `scheduler.py`** | Long-running daemon. Fetches schedule, computes run times, triggers prediction runs, manages daily state. CLI command: `bts schedule`. |
| **`strategy.py`** | Add `early_lock_gap` parameter to posting decision. |
| **`orchestrator.py`** | Refactored into a callable function that the scheduler invokes, rather than the top-level entry point. Existing `bts orchestrate` CLI continues to work for manual runs. |
| **`posting.py`** | `should_post_now()` time-based logic replaced by scheduler's confirmation-based logic. Bluesky posting mechanics (API calls, formatting) unchanged. |
| **`picks.py`** | Add `suspended` and `unresolved` result types. |
| **Cron (Pi5)** | Replace 3 fixed prediction runs with one: start scheduler at ~10am ET (or run as a systemd service). Keep 1am check-results as a backup (scheduler handles it, but cron is a safety net). |
| **Pi5 systemd** | New `bts-scheduler.service` unit. |
| **Config** | Add to `~/.bts-orchestrator.toml`: `early_lock_gap` (float), `lineup_check_offset_min` (default 45), `doubleheader_recheck_interval_min` (default 15), `results_poll_interval_min` (default 15), `results_cap_hour_et` (default 5). |

## What Stays the Same

- Workers (`predict-json`), model, features, 12-model blend, MDP policy
- Pick JSON file format (adding new result types only)
- Bluesky posting mechanics, DM notifications
- Streak tracking logic
- Dashboard (reads same pick files)
- Mac/Alienware worker setup
- `bts run` for manual local runs

## State File

`data/picks/YYYY-MM-DD/scheduler_state.json`:

```json
{
  "schedule_fetched_at": "2026-04-03T10:00:00-04:00",
  "games": [
    {
      "game_pk": 123456,
      "game_time_et": "2026-04-03T19:05:00-04:00",
      "lineup_check_at": "2026-04-03T18:20:00-04:00",
      "lineup_confirmed": true,
      "is_doubleheader_game2": false
    }
  ],
  "runs_completed": [
    {"time": "2026-04-03T18:20:00-04:00", "new_lineups": 8, "skipped": false}
  ],
  "pick_locked": true,
  "pick_locked_at": "2026-04-03T18:25:00-04:00",
  "result_status": "final"
}
```

## Testing Strategy

- **Unit tests**: Scheduler run-time computation, clustering logic, early lock gap comparison, doubleheader detection, result status handling.
- **Integration tests**: Mock MLB API responses at different times of day, verify correct sequence of runs, pick updates, and posting decisions.
- **Backtest**: Simulate scheduler against 2021-2025 historical data to derive `early_lock_gap` and validate that confirmed-lineup picks outperform projected-lineup picks.
- **Manual dry-run**: `bts schedule --dry-run` shows what the scheduler would do today without executing predictions or posting.
