# 2026-08-30 — Pick delivered after the submission cutoff (Kwan, 13:36 DM for a 13:40 first pitch)

## Summary

The scheduler DM'd the day's pick — Steven Kwan (CLE, game 824393, first pitch
13:40 ET, BTS cutoff 13:35) + Jeff McNeil (ATH, 16:05) — at **13:36:14 ET**, one
minute after Kwan could still be entered. Eric could not enter the recommended
pick. Nothing alerted: `fallback_defer` validates "never miss" as *a delivered
pick file exists*, which was true.

Three defects combined, none of them new that day:

1. **Deadline-blind sequential orchestration** (the correctness root cause).
   The T−35 fallback refresh finished at 13:20 with Kwan still enterable, then
   **deferred** him on a `has_pending_future_window` boolean snapshotted at
   12:50 — before a 15-min sleep and a 15-min refresh — that counted the
   already-overdue 13:10 check and three post-first-pitch checks as "future
   windows". The loop then ran the overdue 13:10 check (another full cascade)
   and locked the *same* Kwan at 13:36.
2. **No enterability guard anywhere in the lock/deliver path.** Candidate
   filtering keyed only on MLB game status (Preview until first pitch);
   `_deliver_and_lock_pick` never looked at the clock.
3. **~11 minutes of pure sleep per cascade** (the amplifier, chronic all
   season): `pull_feeds` slept 0.3 s after every discovered game even when the
   feed was already cached — 2,112 cached feeds ≈ 10.6 min — plus ~1 min of
   per-date schedule discovery and a full parquet rebuild, on every intraday
   cascade, re-refreshing data that cannot change until tomorrow.

`FALLBACK DEFERRED` had fired three times in four days (8/27, 8/29, 8/30). The
first two "worked" by re-picking later games; the third re-picked the abandoned
batter after his cutoff.

## Timeline (ET)

| Time | Event |
|---|---|
| 10:00 | Schedule: 14 games; checks at 11:15, 12:35, 13:10, 14:10, 15:05, 18:20 (offset 60). |
| 11:15→11:31 | Check: Kwan 75.7%, gap 1.5% vs a PROJECTED contender (Arraez, PHI 16:07) → `should_lock=False`. |
| 12:35→12:51 | Check: same. Next check (13:10) is after the fallback deadline (13:40−35 = 13:05) → sleep to 13:05. |
| 13:05→**13:20** | Fallback refresh (15.5 min). `should_lock=False`; **FALLBACK DEFERRED** ("4 future check(s) with pending lineup data" = `len(next_checks)`, not a count of pending windows). `2026-08-30.json` deleted. |
| 13:20→**13:36** | Overdue 13:10 check runs. Arraez is not in PHI's lineup at all; the phantom contender evaporates; gap 5.5% → `should_lock=True` → **DM 13:36:14**. Cutoff was 13:35. |

Post-incident partner pick (McNeil + Blaze Alexander) was hand-entered; see the
pair-correlation audit for why same-game pairing is fine.

## What the adversarial review (Codex, gpt-5.6-sol, repo access) corrected

- Root-cause **ranking**: the latency bug is the amplifier; the stale, sequential,
  deadline-blind state machine is the correctness cause; the missing delivery
  guard is the containment failure. (The first draft had latency first.)
- The deferral predicate was **not evaluated at 13:20** — it was snapshotted at
  ~12:50 and reused after the blocking sleep + refresh.
- The log's "4 future check(s)" is `len(next_checks)`; only two games were
  actually unconfirmed at 13:20 (823987, 824636), both after first pitch.
- A selection-time *margin* (the first draft's fix B) would have excluded Kwan
  at 13:20 while he was still deliverable — masking the defect. Correct
  layering: a hard cutoff at the delivery chokepoint + a live-only completion
  budget before starting/deferring another cascade.
- `fallback_deadline_min` is the latest cascade **start**, not a delivery
  deadline; the config semantics were never written down.
- The May 23 deferral tests use a frozen clock and a mocked refresh, so they
  could not see a refresh crossing a scheduled target or the cutoff.
- With the sleep fix alone, cascades are ~5 min (not 4) and the DM lands ~13:15 —
  sufficient for this incident, not a guarantee in general.

Full findings: the session's Codex review (11 findings, all confirmed or
accepted-as-overstated; disagreements were on emphasis, not mechanism).

## Fix (branch `fix/late-pick-delivery-guard`, 2026-08-30)

| Layer | Change | Test |
|---|---|---|
| Cutoff constant | `bts.picks.SUBMISSION_CUTOFF_MIN = 5`, `earliest_pick_game_et`, `submission_cutoff_et`; cli + health import it | `tests/test_submission_cutoff.py` |
| **Hard guard** | `_deliver_and_lock_pick` refuses at/after cutoff for the earliest slot (all modes incl. private): archives `refused_delivery_*.json`, removes `<date>.json`, records `state.delivery_refusals`, DMs a CRITICAL, leaves `pick_locked=False`. Stamps `DailyPick.delivered_at`. | `tests/test_late_delivery_guard.py` |
| Live re-pick | `_games_past_cutoff` → `run_and_pick(unavailable_game_pks=…)` → `select_pick` drops already-unenterable games (no margin; offline callers unchanged) | `tests/test_cutoff_candidate_exclusion.py` |
| Block reason | `LockDecision` (`block_reason`, `contender_game_pk`) replaces the 3-tuple | `tests/test_lock_decision.py` |
| **Planner** | `plan_fallback_action` decides AFTER the refresh, against the live clock and re-synced confirmations: gap block → defer only if the contender's window can finish (`max(start, now) + budget ≤ cutoff − reserve`), else **deliver**; projected primary → legacy defer; gate-only/unknown/true → deliver | `tests/test_fallback_plan.py` |
| Wiring | pre/post-refresh confirmation sync; overrun checks the refresh already covered are coalesced; cascade durations measured (`runs_completed.duration_sec`, `state.fallback_refreshes`); `effective_cascade_budget_min` = max(config, slower of last two + 2); fallback deadline floored at cutoff + budget + reserve | `tests/test_incident_2026_08_30.py` (advancing clock), `tests/test_scheduler.py` |
| Latency | `pull_feeds` no sleep on cache hits; `_refresh_season_data` memoized per (season, yesterday) via `data/processed/.refreshed_*` (`BTS_REFRESH_ALWAYS=1` forces) | `tests/data/test_pull.py`, `tests/test_refresh_memo.py` |
| Health | `late_delivery` source (CRITICAL at/after cutoff or on a refusal; WARN inside the reserve); `fallback_defer.same_pick` compares both slots | `tests/health/test_late_delivery.py` |

### Policy change (approved by Eric 2026-08-30)

A **gap-rule** block (the pick is confirmed but a *projected* contender in another
game is within `early_lock_gap`) no longer abandons the enterable pick when the
contender's confirmation window cannot finish before this pick's deliver-by time.
Under the old rule 8/27 (Perdomo), 8/29 (Arraez's early DD) and 8/30 (Kwan) were
all abandoned; under the new rule they deliver at ~T−20. A **projected primary**
keeps the 2026-07-06 product choice (abandon the early slate). The
`_should_defer_at_fallback` helper remains for its unit tests but is no longer on
the run_day path.

### New config keys (`[scheduler]`, defaults in code)

- `cascade_budget_min` (12): assumed cascade duration; raised intraday to the
  slower of the last two measured cascades + 2.
- `operator_reserve_min` (10): time Eric gets between the DM and the cutoff.
- `fallback_deadline_min` (35, unchanged) is the latest cascade **start**; it is
  floored at `5 + budget + reserve` (27 with defaults, so the box value is
  unchanged; a 25-min morning value is raised to 27 with a log line).

### Counterfactuals with the fix

- 8/30: fallback refresh ends 13:20 → contender window (15:05 run) infeasible →
  deliver Kwan at 13:20 (T−20). With the latency fix the refresh ends ~13:10 →
  deliver at T−30.
- 8/27 12:45: Perdomo delivered at T−20 instead of abandoned.
- 8/29 15:50: Arraez was the *projected primary* → still deferred (unchanged).
- 7/06 (early confirmed DD, late projected primary): still deferred (unchanged).

## Follow-ups

- Deploy in the idle window; confirm on 8/31 that the second cascade logs
  `already refreshed through … skipping` and `runs_completed[*].duration_sec` ≈ 5 min.
- The singleton-slate / moved-up-game gap (`docs/optimization-ideas.md`) is
  unchanged; the planner composes with it.
- `check_confirmed_lineups` swallows HTTP errors per game and returns "nothing
  confirmed" — fine for the planner (a failed sync can only make it *more*
  willing to wait), but worth a log line.
