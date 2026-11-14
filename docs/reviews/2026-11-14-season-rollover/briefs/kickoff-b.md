# Kickoff B: a day with no selected pick gets a last attempt (S6)

You are a fresh author. Work only in this worktree (`~/projects/bts-sr-b`, branch `sr-lastcheck`, created from `main`). Project rules: `CLAUDE.md` (read "2027 season" and the Architecture notes on the scheduler, the cutoff guard and the fallback deadline) and the rulings ledger in `docs/superpowers/plans/2026-11-12-2027-season-roadmap.md` (R2, R13, R14, R16, R18 are settled; do not reopen them).

Files you may change: `src/bts/scheduler.py` and tests. **Do not change `plan_run_times`, `compute_run_times`, `_deliver_and_lock_pick`, or anything under `src/bts/` other than `scheduler.py`** (another author is changing `contest_state.py` and the data path in parallel).

## Rules
- Tier A (pick-path behaviour). RED first: write each failing test, run it, record that it fails for the stated reason, then implement. Report the RED counts.
- Each item is a table of states. A row marked "unchanged" must behave exactly as today; if you cannot make the code match a row, stop and say so in the handover instead of choosing.
- Smallest change that satisfies the tables. No refactors, no renames, no new configuration keys, no new files under `src/` unless a row cannot be met without one.
- Tests must not use the network, must not shell out to git, must not depend on a branch name, and must pin `today=`/`now=` (never `date.today()` or the wall clock).
- Run Python as `.venv/bin/python`; focused tests: `TZ=America/New_York .venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Do not run the whole suite (this machine is throttled; the lead runs the gate elsewhere).
- For each item, break the new behaviour once on purpose (a mutation), show a test fails, and restore the source exactly. Report which test caught which mutation.
- Do not commit, push, or touch any other worktree. Do not read `~/projects/claude-shared` or `~/bts-runs`.
- The reviewers' probes that demonstrated these defects are in `docs/reviews/2026-11-14-season-rollover/probes/` (written against older commits and scratch paths; raw material for your RED tests, not tests themselves).
- This file has a long incident history. Every path that runs today when a pick file exists, or when a skip is recorded, must be byte-for-byte the same decisions at the same times: prove it by running the existing day-level tests listed below unchanged, and say so.
- Existing tests that must stay green without edits: `tests/test_thin_slate_checks.py`, `tests/test_late_delivery_guard.py`, `tests/test_postponed_delivery_guard.py`, `tests/test_pickpath_review_pp1.py`, `tests/test_pickpath_review_pp2.py`, `tests/test_incident_register_2026.py`, `tests/test_scheduler.py`, `tests/test_scheduler_decision_record_integration.py`, `tests/test_cold_start_integration.py` (except the one expected failure below).

## S6. A day with no selected pick gets a last attempt before each remaining cutoff
`src/bts/scheduler.py::run_day`, the block after the lineup checks ("5. Fallback"), and the missed-pick alert near the top of the file. Found by the final g2 review (`docs/audit/2026-11-14-g2-fixes-codex-final.md`, cold start, item 1) and confirmed by reading the code: the end-of-checks fallback only acts on an existing pick file, so when every check returned no predictions the day ends with no pick, no decision and no alert until the next morning's `bts day-outcome`. On a team's first game of a season there is no prior lineup to project, so this happens whenever the lineups post after the last check: the reviewer's run of the real `run_day` for a 19:10 one-game Opening Night with offset 60 and lineups first available at 18:25 ran checks at 16:10, 17:10 and 18:10 and saved nothing.

| State when the planned checks are exhausted and the pick is not locked | Today | Required |
|---|---|---|
| a pick file exists, undelivered, its fallback deadline ahead, no standing skip | wait for the deadline, then the fallback | unchanged |
| a standing skip (`final_skip_candidate` or `skip_summary`) | no wait | unchanged |
| no pick file, no standing skip, and at least one of today's regular-season games still has its fallback deadline ahead | the day ends silently | for each remaining distinct game start, in order: sleep until that start's fallback deadline (the same `resolve_fallback_deadline_min` and floor the existing wait uses, applied to that game's start), then run one more check. If the check returns a pick, everything from there is the existing handling of a check that returns a pick, unchanged: the confirmation gate, then that pick's own fallback deadline (for a pick in the game whose deadline was just reached, that means delivery now). Stop making extra attempts at the first check that returns a pick or records a skip |
| no pick file and no game with its deadline ahead | the day ends silently | the day ends, as today |

Alert: when the day's attempts are over (the planned checks and any extra attempts) with no pick file and no recorded skip, although today's schedule had at least one regular-season game, send ONE alert through the existing missed-pick alert transport saying that no pick was selected today, and send it then, not at the next morning's check. Today `_maybe_alert_missed_pick` returns without alerting when there is no pick file. A day with a recorded skip, a delivered pick, or no regular-season game sends nothing new.

RED test to start from: `tests/test_cold_start_integration.py::test_opening_day_late_lineup_gets_a_last_attempt` (an expected failure adopted from the reviewer's probe).


## Tests
1. `tests/test_cold_start_integration.py::test_opening_day_late_lineup_gets_a_last_attempt` is a strict expected failure today. It must pass with the `xfail` mark removed, and for the stated reason: a pick delivered after the lineups appear at 18:25 for a 19:10 start. `test_opening_day_late_lineup_checks_today` pins the checks at 16:10, 17:10 and 18:10 "and no later feed request": that last clause is the old behaviour; change only that clause, to the new expected attempt, and say so.
2. New day-level tests, each driving the real `run_day` with only the network, the clock and the DM transport faked:
   - lineups never appear before the cutoff: the extra attempt runs, selects nothing, the day ends with no pick, and exactly one alert says no pick was selected;
   - a recorded skip: no extra attempt, no alert;
   - a multi-game day where every planned check fails (prediction outage) and predictions recover before a later game's deadline: a pick for a still-enterable game is delivered;
   - a day where a pick file exists and is undelivered: the existing fallback behaves exactly as before (same delivery time).
3. State what the extra attempts cost in the worst case: how many additional prediction cascades a day can now run, and when.

## Deliverable
`.codex-review/sr/handover-b.md`, under 80 lines: what changed (functions and the control flow in words), the RED count and reason, the mutations and the tests that caught them, the focused test command and counts for the listed files, `git diff --stat`, the worst-case cost, and any table row you could not match. When the file is complete, print `SRB-<first 8 hex characters of the sha256 of the handover file>` and then `DONE`.
