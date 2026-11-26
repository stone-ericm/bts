# Kickoff 2: remove the leg reduction (the lead's contract was wrong)

Same session, same worktree (`/Users/eric/projects/bts-pp-c1`), same rules as `.codex-review/pp-c1/kickoff.md` (test first, focused tests only, no commit/push/stash, handover + marker).

## What happened
A fresh reviewer blocked your rewrite: `.codex-review/pp-c1-review/report-r1.md` (read it in full; its probes are at `/tmp/test_ppc1_review_r1.py`). You implemented the contract you were given faithfully; the contract was the defect. Leaving the double on disk until the delivery save lets `run_single_check`'s stale-cutoff archival delete the whole cached pick first, and a caught pre-send error then loses a day that the previous code kept. Every design tried so far for "reduce a double to its playable leg" has had a window like this.

## The new contract (smaller again): there is no reduction
The delivery guard refuses a pick only when NONE of its games is playable. A double with at least one playable leg is delivered exactly as it was before 2026-11-13: unchanged, as a double.

1. In `_deliver_and_lock_pick`: compute the void games as now. Refuse (`_refuse_void_game_delivery`, unchanged) only if every committed game of the pick is void: a single on a void game, or a double with both legs void. Otherwise deliver the pick unchanged. Print one stderr line when a double has one void leg (which game, which state); nothing else.
2. Delete `_playable_leg`, `_keep_playable_leg`, the `selection = None` reset that existed only for the reduction, and the `leg_dropped_game_not_playable` branch in `src/bts/health/late_delivery.py`. No reduction record on state, no mutation of the DailyPick, no extra save.
3. Everything else stays: the guard still runs before the cutoff check; an unreadable schedule delivers; the refusal's archive, state record, and CRITICAL DM that names the game and not the batter.

Why this is acceptable: in the contest a leg on a postponed game is a Pass, so a delivered double with one postponed leg scores exactly like a single of the other leg (the grader already returns hit + void = +1). Known costs, which the lead is ledgering and you should NOT try to fix: the pick DM and the decision record still name the postponed leg; the entry checker can report that leg as not entered; the cutoff guard still uses the earliest leg's first pitch even if that leg is the void one (as it did before 2026-11-13).

## Tests
- Replace the reduction tests with tests of the new contract: a double with one void leg (either leg) is delivered as the same double in DM, public and private modes, the caller's object and the pick file are unchanged doubles, the decision record says double, no refusal record, no alert DM, one stderr line; a double with both legs void and a single on a void game are refused as before; an unreadable schedule delivers; the health check renders no reduction wording.
- Keep the reviewers' day-level scenarios as regressions of the OUTCOME, adapted to the new contract (the playable leg's delivery must not be lost): round 1's B2 day (`tests/test_pickpath_review_pp1.py::test_postponed_dd_does_not_erase_playable_primary`: now the double is delivered), the death-before-save restart, and from `/tmp/test_ppc1_review_r1.py` the caught-pre-send-error cases and the earlier-void-cutoff restart cases. The reference for a double with one void leg is the scheduler as it was BEFORE the guard existed: `git show 4199360:src/bts/scheduler.py` (the commit just before `b205ae0` added the guard). Run each scenario against that reference first. Where the reference delivered, yours must deliver; where the reference itself lost the day (for example a cached double whose earliest leg is past its cutoff while every later prediction cascade fails), yours may lose it too, and that is a pre-existing limit, not yours to fix. Say in the handover, per scenario, which of the two it is and show the run.
- Delete `tests/test_pickpath_review_pp2.py` tests that only pin the removed reduction; say which.
- Mutation check each new behaviour (refuse-only-when-all-void; no mutation; no record; no alert).
- Before handing over, run the kickoff's focused set plus any file you add or change, and paste its last line.

## Deliverable
Write `/Users/eric/projects/bts-pp-c1/.codex-review/pp-c1/handover-2.md` in the same shape as before, under 120 lines, including the per-scenario reference comparison. When the file is complete, print one line `PPC2-<first 8 hex characters of the sha256 of the handover file>` and then `DONE`.
