# H5(b): Truthful heartbeat — split process-aliveness from progress

**Status: approved 2026-06-11 (Phase 1, alert-only). Codex (gpt-5.5) adversarial
review incorporated.**

## Problem

During the prediction cascade, `heartbeat_watchdog` (bts/heartbeat.py) spawns a
pulse thread that every 60s unconditionally writes a `state=RUNNING` heartbeat
and sends an sd_notify WATCHDOG=1 ping. The pulse certifies "alive and running"
based on nothing but its own thread being alive. If `run_and_pick` wedges in a
non-I/O hang (pandas spin, deadlock — network hangs are already bounded by
15–30s timeouts everywhere, audit H5 part (a)), the pulse pings forever and no
alert ever fires: **wedged-but-pinging**.

The unbounded surface is the in-process local tier
(`run_and_pick` → `run_pipeline`: parquet load → feature compute → 1+12
LightGBM fits → lookups → predict → `select_pick`). SSH fallback tiers are
subprocess-bounded by `timeout_min`.

## Constraints

- **Alert-only (Eric, 2026-06-11).** No auto-kill. The systemd unit has
  `Type=notify WatchdogSec=1800`, so the pulse must KEEP sending sd_notify
  pings during a detected stall — silence for 30 min = systemd restart, which
  is auto-kill by another name. (Prior incident: missing pings during long
  sleeps killed the daemon, 2026-04-23.)
- 5 prior bugs in this code class — stage carefully, keep the pulse path
  trivially simple.
- Existing liveness chain must keep working: pulse writes `data/.heartbeat` →
  cron (5 min) `scripts/check_heartbeat.py` maps state → staleness → POST
  healthchecks.io `/fail` → phone alert. Dashboard `/health` reads the same
  file via `is_heartbeat_fresh`.

## Design

### 1. Progress beacon — `bts/progress.py` (new)

Module-level, thread-safe, **pure in-memory** (never touches files):

- `begin_run(kind: str) -> str` — resets state, returns a fresh `run_id`, and
  seeds the current stage as `cascade_starting` (so a wedge before the first
  pipeline mark still has stage attribution).
- `mark(stage: str)` — **stage-ENTRY semantics**: declares the stage now
  beginning; the previous stage is thereby completed and appended to a bounded
  in-memory transition history (bound **256**; overflow drops oldest + logs).
  Stage names are gerunds of what is about to run (`loading_parquets`,
  `computing_features`, `training_blend_3`, `selecting_pick`, …) — NOT
  completion milestones. The watched age is "how long has the current stage
  been running." `mark()` with no active run is a no-op.
- `end_run(run_id)` — retires the run: snapshot returns None afterwards, and
  the final in-progress stage is completed into the history.
- `snapshot(run_id)` — returns (current stage, stage age, stage generation)
  **only if the beacon's run_id matches and is active**; else None.
- `drain_transitions(run_id)` — returns and clears completed-stage records;
  `[]` on run_id mismatch (no clearing).

Ownership contract: overlapping in-process runs are unsupported (markers are
single-threaded by construction — primary and shadow cascades run
sequentially); `begin_run` supersedes any prior run, and marks always attach
to the current run. A leaked pulse from a failed `join` self-retires because
`snapshot(old_run_id)` returns None. The `bts run` CLI calls `mark()` via the
shared pipeline code harmlessly — without `heartbeat_watchdog` nothing reads
the beacon and no file is touched.

### 2. Marks (~10 one-liners, no signature changes)

- `orchestrator.run_and_pick`: cascade start, predictions returned, decision
  state loaded, pick selected.
- `model/predict.run_pipeline`: data refreshed, parquets loaded, features
  computed, single model trained, **one mark per blend-model fit** (12),
  lookups built, predict done.

Granularity guarantee: no legitimate inter-mark gap approaches 5–6 min
(observed: feature compute ~5 min total, single LGBM fit minutes, parquet
load <1 min) → the 900s threshold has ~3× margin. `compute.py` internals are
deliberately NOT marked (keeps the feature lib clean; its total stays well
under threshold).

### 3. Progress-aware pulse — `heartbeat_watchdog` (modified)

New signature (defaults keep existing callers/tests working):
`heartbeat_watchdog(path, *, kind="primary", date=None, stall_after_sec=900,
interval_sec=60)` — `stall_after_sec` overridable via
`[scheduler] heartbeat_stall_after_sec` in the orchestrator toml; the two
scheduler callsites pass `kind` ("primary" / "shadow") and `date`.

At entry: `run_id = begin_run(kind)`. Each 60s tick, in order:

1. **Always `notify_watchdog()`** — systemd never kills (Phase-1 invariant).
2. `snapshot(run_id)`: None (run retired / superseded) → write nothing, thread
   exits. Stage age < `stall_after_sec` → write `state=RUNNING` heartbeat with
   `extra={stage, stage_age_s, run_id}`. Else → **re-snapshot immediately
   before writing** (mitigates the stall-vs-recovery-mark race; a one-tick
   residual window is accepted and documented) → write `state=stalled` with
   `extra={stage, stalled_for_s, run_id}`.
3. Drain transitions to `data/health_state/cascade_stage_durations.jsonl`
   (append; failures swallowed+logged). Row schema: `{run_id, pid, kind, date,
   stage, started_at, ended_at, duration_s, status, threshold_used_s}` with
   `status ∈ {ok, ok_after_stall, stalled_incomplete}`:
   - On FIRST stall detection **per stage instance** (latch keyed by stage
     generation, reset on stage transition and on new run), append one
     `status=stalled_incomplete` row. The latch flips ONLY on successful
     append (a failed write retries next tick — "exactly once" must not
     become zero times).
   - A stage that completes after crossing the threshold gets
     `status=ok_after_stall` (Phase-2 analysis sees both the stall event and
     the true duration; `stalled_incomplete` is not terminal).
   (Without the incomplete rows, Phase-2 threshold data contains only
   successes — biased against the failures that matter.)

At exit (`finally`): `stop.set()`, `join(timeout=2)`, then `end_run(run_id)`
and a final synchronous drain from the main thread — a late pulse cannot
overwrite the scheduler's post-cascade state heartbeat (SLEEPING etc.)
because its snapshot returns None after retirement.

Pulse-path safety (unchanged properties): atomic heartbeat write (tmp+rename),
broad exception catch, non-blocking exit. Beacon lock guards only tiny dict
ops — the pulse can never block on a lock held by the cascade.

jsonl growth: tens of rows per cascade ≈ single-digit MB/year — explicitly
fine for Phase 1; revisit with logrotate only if Phase 2 extends collection.

### 4. Consumers

- `scripts/check_heartbeat.py`: explicit `stalled` → stale (reason includes
  stage + stalled_for) → `/fail`. Already fails closed on unknown states, so
  deploy order is safe regardless; repeated `/fail` while stalled matches the
  existing behavior for any staleness (healthchecks alerts on transition).
- `bts/heartbeat.py is_heartbeat_fresh`: explicit `stalled` → not fresh
  (today it would fall through to the age check and call a fresh-timestamped
  stalled heartbeat healthy — the one consumer gap found in review).

Worst-case wedge→phone latency: 15 min stall + ≤5 min cron ≈ **20 min**, vs
never today.

## Testing (TDD)

- Beacon: run-token reset; `mark()` with no active run is a no-op; snapshot
  before first mark returns `cascade_starting`; snapshot after `end_run` /
  supersession returns None; drain on mismatched run_id returns `[]` without
  clearing; transition-history bound (burst of >256 marks: oldest dropped,
  logged); stage-entry semantics (durations attribute to the named stage).
- Pulse: stall transition with a fake clock; sd_notify continues during stall;
  heartbeat flips back to RUNNING on recovery; `stalled_incomplete` row once
  per stage instance (second stall in a later stage gets its own row); latch
  not flipped when the jsonl append fails (retries next tick);
  `ok_after_stall` on recovery; RUNNING extras carry stage.
- Exit path: after context exit, a delayed pulse tick writes nothing (cannot
  overwrite a post-cascade SLEEPING heartbeat); final drain captures the last
  stage.
- `check_heartbeat.is_stale`: stalled → stale with stage + duration in reason.
- `is_heartbeat_fresh`: stalled → False.
- Integration: `run_and_pick` under `heartbeat_watchdog` produces marks and
  jsonl rows (mocked cascade).

## Out of scope (Phase 2, gated on the jsonl dataset)

- Any auto-restart (e.g., letting sd_notify go quiet on confirmed stall).
- Per-stage thresholds (Codex suggested static budgets now; rejected as
  YAGNI for alert-only Phase 1 — the data this ships will justify them or not).

## Rejected alternatives

- **Total-cascade budget, no marks**: no stage attribution, slow detection,
  no Phase-2 dataset.
- **Subprocess hard-timeout**: is auto-kill; restructures model caching/memory.
- **Freezing the RUNNING timestamp on stall** (instead of a new state): loses
  attribution; ambiguous between process-dead and progress-stalled (Codex
  concurred).
