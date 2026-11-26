# 2026-07-12 — Eve-of-break restart loop + health-DM spam

## Incident

Last slate before the All-Star break (15 games, all afternoon; DD pair Arraez/Otto
Lopez, both games final by 18:36 ET — result **miss**, streak stays 0). At 18:36:16
the scheduler ran its normal EOD sequence, fired a `predicted_vs_realized` CRITICAL,
and the process exited cleanly (status 0). `Restart=always` relaunched it every ~48s;
each relaunch re-walked the completed day (startup lineup check → "pick already
locked" → EOD health suite → **new CRITICAL DM** → exit). NRestarts went 3 → 50
(~47 duplicate DMs) before the operator session stopped the unit at 19:15 ET.
`systemctl --user stop bts-scheduler`; scheduler left DOWN pending this fix.

## Root cause chain (each necessary, none sufficient alone)

1. **Step 7 wakeup (F-A, root).** `run_day` step 7 called
   `compute_wakeup_time(fetch_schedule(tomorrow))` directly. Tomorrow (7/13) has
   **zero games**, and `compute_wakeup_time([])` returns **today** at the default
   hour — 10:00 ET, 8.6h in the past at EOD. The E1 helper `_next_day_wakeup`
   (built for the no-games-TODAY path, comment literally citing the All-Star
   break) bumps past wakeups to tomorrow morning — step 7 just never used it.
   The failure needs a games-day followed by an empty day: first occurrence
   this season (system went live in April).
2. **Silent idle guard (F-B).** `_idle_until_next_wakeup` treats a past wakeup as
   a silent no-op return — recreating exactly the Restart=always thrash its own
   docstring says it exists to prevent (2026-04-23, NRestarts=21). Kept as a
   return (substituting a guessed wake time could sleep through a REAL game day)
   but now logs loudly.
3. **No DM dedup (F-C).** `dispatch_dm_for_health_alerts` is designed to run once
   per day and sends unconditionally. 47 EOD re-walks → 47 identical DMs.
4. **Day selection until midnight.** `_today_et()` targets the same date until
   00:00 ET (correct per audit O2), so the loop would have run ~5.5 more hours
   (~400 DMs) — and then resumed against empty break days.

## Why the CRITICAL fired at all (F-D)

`predicted_vs_realized` compared the **primary's** `p_game_hit` against the
**day-level** result — but a DD day's result requires BOTH legs to hit.
Decomposition over the live pick files (2026-07-12):

| realized basis                  | 14d gap | 28d gap | drift  |
|---------------------------------|---------|---------|--------|
| day-level result (as shipped)   | +0.2307 | +0.0570 | +0.1737 → CRITICAL |
| primary slot outcome            | −0.0550 | −0.0968 | +0.0419 → below INFO |
| no-DD days only                 | −0.0414 | −0.0414 | +0.0000 |

The primary model is over-delivering (12/14 in the window). The real anomaly is
the **DD leg: 1-for-6 across the six DD days since 7/07** (the five misses:
Alvarez, Turner, Harris, Otto Lopez ×2; the one hit: DeLauter 7/08) at stated
p≈0.74 — P(≤1 hit in 6) ≈ 0.56%, one-sided ≈2.5σ, if calibrated and independent
(round-2 corrected; originally overstated as "~2.9σ"). So the alert was a
mis-specified metric amplifying a real-but-different signal. Fix: per-slot
grading (primary + DD leg each against its own p, via `slot_results`) — the same
attribution `realized_calibration` adopted 2026-05-01 for this exact bias.
Re-scored per-slot (new code run against the live pick data): 14d gap +0.1011
over 22 slots vs 28d baseline −0.0265 over 40 → drift **+0.1276 → WARN**,
honestly attributable to the DD-leg cold streak. The DD-leg run remains worth watching
(`dd_pair_realized_shortfall` stayed quiet — its drift-vs-baseline design dilutes
a two-window-spanning slump; not changed in this batch).

## Monitoring verdicts during the loop

- `check_heartbeat` (cron, external): **caught it in 4 min** — "restart churn:
  NRestarts +5 within 20 min — daemon is crash-looping behind a fresh heartbeat"
  — and correctly cut the healthchecks.io ping (hc.io "down" notification is the
  expected side effect until redeploy).
- `restart_spike` (in-scheduler, F-F): **blinded by its own checkpoint** — it
  advances the baseline unconditionally every run, so 48s re-walks moved it +1
  at a time and a 47-restart storm never summed past the +3 threshold. Fixed:
  the checkpoint anchors to the PROCESSED slate date and only advances across
  a date boundary; multi-day gaps budget one planned exit-restart per elapsed
  day (round-2 #4 — the daily idle→return→Restart=always cycle is a planned
  restart, and no-games days never run health, so a 4-day break would have
  read as a +4 false spike on 7/16).

## Also shipped

- **F-E `fetch_schedule` gameType filter.** No filter existed; the 7/14 All-Star
  Game (statsapi `gameType A`, sportId=1) would have looked like a real 1-game
  slate and run the full lineup-check/pick pipeline. Regular season (`R`) only;
  lenient on a missing key (fixtures). 7/16 Mets@PHI is `R` and unaffected.
- `compute_wakeup_time` gained an optional `now_et` param (threads the caller's
  clock; makes the empty-schedule branch testable and consistent with
  `_next_day_wakeup`'s bump comparison).
- "Tomorrow's wake-up" log line now includes the date (`%a %m-%d`) — the
  time-only format hid the wrong-date wakeup all through the incident journal.

## Fixes (TDD; suite 1808 → 1830 green)

| id  | file | change |
|-----|------|--------|
| F-A | scheduler.py step 7 | use `_next_day_wakeup` (never past, never raises) |
| F-B | scheduler.py `_idle_until_next_wakeup` | loud no-op logging, behavior kept |
| F-C | health/alert.py | per-day (level, incident-identity)-set dedup keyed to the PROCESSED day; union persisted in `health_dm_delivery_status.json`; failed sends never suppressed (H6 intact) |
| F-D | health/predicted_vs_realized.py | per-slot grading; legacy fallbacks (single-pick = day result; legacy DD days excluded symmetrically) |
| F-E | scheduler.py `fetch_schedule` | `gameType == "R"` filter |
| F-F | health/restart_spike.py | day-anchored checkpoint |

## Round-2 adversarial review (gpt-5.6-sol, xhigh) — 10 findings, verdict "not safe as-is"

Blocking findings, all fixed same session:

1. **_next_day_wakeup could skip an early slate** (CRITICAL): post-midnight EOD
   (polling caps 05:00) before an early-start day (London 06:10 ET, wake 05:10)
   → day-bump slept to D+2. Now: a nonempty tomorrow whose wake has passed
   hands off in ~1 min (exit→restart starts the new day). The old thrash-exit
   was accidentally correct here; the bump is reserved for EMPTY tomorrows.
2. **Fetch-failure fallback oversleeps** (MAJOR): "assume 10:00 is safe" without
   a schedule could miss a 09:05 start. Now: 15-min retry handoff.
3. **gameType filter missed the prediction path** (MAJOR): `_fetch_game_slots`
   (preview/run/orchestrate) fetches the schedule independently — the 7/13
   03:00 cron would have previewed the All-Star Game. Now: shared
   `bts.util.is_regular_season_game` used by both fetches. `data.pull
   .discover_games` stays unfiltered (build.py already excludes non-R feeds;
   wasted downloads only).
4. **Planned daily restarts read as spikes across breaks** (MAJOR): budgeted
   one per elapsed day (see monitoring section).
5. **Dedup stamped wall date, not processed date** (MAJOR): post-midnight EOD
   for D consumed D+1's "already sent" budget. Runner now threads
   `now_et_date=today`.
6. **(level, source) too coarse** (MAJOR): two different crashed checks both
   carry `source=health_runner`; the second was suppressible. `Alert` gained
   optional `incident_key`; `_safe_run` sets `health_runner:{check}`.
7. **Legacy-DD fallback was outcome-dependent censoring** (MAJOR): counting
   attributable hit-days while skipping unattributable miss-days inflates
   realized. Legacy DD days now excluded symmetrically (zero such files in the
   live 35d window; correctness for replays/backfills).
8. **Status-file concurrency** (MINOR, deferred): single writer holds in-tree
   (scheduler EOD only; check-results does not dispatch health DMs). Accepted
   risk, noted here; flock if a second dispatcher ever appears.
9. **CRITICAL threshold ~1.3σ under correct overlapping-window SE (≈0.094)**:
   `drift_critical` 0.12 → 0.25 (catastrophic/pipeline tier); WARN 0.08 kept
   at the time — superseded same night to 0.15 by the production-clock
   simulation (see 2026-07-12-dd-leg-calibration.md); it originally stayed
   the attention signal. Day-block bootstrap recalibration queued. ⚠️ Policy
   change — tonight's episode would now be a WARN-attention item, not a
   CRITICAL DM.
10. **Doc math**: 2.9σ → ≈2.5σ (exact tail 0.56%); anchor wording corrected.

## Round-3 (scoped re-review of the fix-of-fixes delta)

- **Fixed (was MAJOR)**: round-2's handoff refactor left `compute_wakeup_time`
  OUTSIDE the try — a malformed game in a successful fetch would raise through
  step 7 into a bare 30s churn loop. Both fetch and computation now share the
  15-min-handoff boundary (test: no-gameDate entry).
- **Fixed (MINOR)**: `attention._with_streak` dropped `incident_key` on
  reconstruction (latent — only CRITICAL producers set it today).
- **Rebutted (claimed off-by-one in the restart budget)**: `threshold =
  base + (gap − 1)` holds the UNPLANNED bar constant at `base − 1` across all
  gaps, exactly preserving the original consecutive-day semantics (3 total =
  2 unplanned + 1 planned). The proposed `base + gap` would silently
  desensitize normal days from 3 to 4. Message arithmetic clarified instead.
- **Deferred (MINOR)**: ~45 min of persistent schedule-fetch failure
  accumulates 3 handoff restarts and fires one `restart_spike` CRITICAL whose
  wording says "restart loop suspected" — a real outage IS operator-worthy;
  only the attribution is imprecise. Wording refinement queued.
- **Deploy preflight (override risk) — verified on the box**: live
  `~/.bts-orchestrator.toml` `[health_checks]` carries only
  `contest_state_expected = true`; no `thresholds` overrides exist, so the
  recalibrated `drift_critical = 0.25` takes effect on deploy.

## Recovery plan

Scheduler is stopped; nothing requires it until the first post-break slate
(7/16 Mets@PHI is the next `R` game — contest rounds exist through the break
but hold no pickable regular-season games; 7/17 full slate). Deploy of this
batch (rides with the already-approved round-3 delta `733d3d6`) restarts both
units via the workflow; heartbeat + hc.io recover on restart.
`.nrestarts_checkpoint` self-heals (legacy checkpoint compared then rewritten
with `day`). No data migration: `health_dm_delivery_status.json` fields are
additive.

Deploy-time checks:
- If the deploy lands after 7/13 03:00 ET, the OLD code's preview cron may
  have written a stray All-Star preview `data/picks/2026-07-14.json` — delete
  it during verification.
- `memory_growth_history.jsonl` took 49 rows for 2026-07-12 (one per loop
  cycle; ~190MB short-lived processes). INFO-digest-only data; optionally
  dedupe to the day's first row.
- Based on the 7/12 snapshot (windows slide before 7/16, so levels may
  shift): the first post-deploy EOD logs a `predicted_vs_realized` INFO
  (per-slot drift +0.1276, below the recalibrated WARN=0.15 — see
  2026-07-12-dd-leg-calibration.md) plus a `realized_calibration` 70-75%
  DD-leg WARN (absolute-level, the new coverage). Either way, the confounded
  CRITICAL is gone.
