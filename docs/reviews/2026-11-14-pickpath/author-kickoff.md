# Kickoff: fresh author for the delivery guard's leg reduction (pick-path review pp1, finding C1)

You are a fresh AUTHOR session, not a reviewer. Work only in this worktree: `/Users/eric/projects/bts-pp-c1` (branch `pp-c1`, at `main`). Another session wrote the code you are about to change; it was blocked three times in review and has been stopped. Do not defer to its design where this brief says otherwise.

## Read first, in this order
1. `CLAUDE.md` sections "Work policies" and "2027 season".
2. `docs/audit/2026-11-14-pickpath-codex-r1.md` and `docs/audit/2026-11-14-pickpath-codex-r2.md` (the two review reports; round 2's C1 and its three notes are your obligations).
3. `src/bts/scheduler.py`: `_deliver_and_lock_pick`, `_delivery_game_statuses`, `_void_committed_games`, `_playable_leg`, `_keep_playable_leg`, `_refuse_void_game_delivery`, `_archive_and_remove_pick`, and how `run_day` calls `_deliver_and_lock_pick` and the shadow trigger afterwards.
4. `src/bts/health/late_delivery.py`.
5. `tests/test_postponed_delivery_guard.py`, `tests/test_pickpath_review_pp1.py`, `tests/conftest.py` (the autouse fixture at the end).
6. `.codex-review/pp-c1/reviewer_probes_pp2.py`: the reviewer's round-2 probes. Two fail today: the crash between archive and save (`[False]`), and the failed send reported as delivered.

## Obligations (from round 2)
- **C1 (blocking).** `_keep_playable_leg` archives and UNLINKS the cached double, then saves the single. A process death between the two leaves no pick file; the restarted day cannot fall back and ends undelivered although the playable leg was still enterable.
- **Note, health wording.** `late_delivery.check` says the kept batter "was delivered alone" from a record written before any send; when the send then fails, that is false.
- **Note, inline shadow name.** `run_day` still holds the original double after the reduction, so when the PRIMARY was the dropped leg, the inline shadow comparison is given the dropped batter's name.
- **Note, WARN never sent.** The reduction's WARN is passed in `dispatch_dm_for_health_alerts`'s `alerts` argument, which keeps only CRITICAL; no DM is sent.

## The contract to implement (decided by the lead; narrower than the code you are replacing)
When exactly one leg of a double is on a postponed or cancelled game:
1. Reduce the pick to a single of the playable leg **in place, on the DailyPick object the caller passed**, so every later reader in `run_day` sees the single.
2. **Do not archive, unlink or separately save the pick file for the reduction.** The live file stays the double until the normal delivery path saves the pick, which replaces it atomically. A death at any point before that save must leave the double on disk, so the restarted day reduces it again and delivers.
3. **No separate alert DM for the reduction.** Print one line to stderr. The pick DM that follows names the single.
4. Record ONE entry on `state.delivery_refusals` (reason `leg_dropped_game_not_playable`, the void game(s) and state, the kept batter). The health check renders it as a WARN that states the reduction and the kept leg and makes **no claim that anything was delivered**.
5. The commit decision record must describe the single: the cycle's `SelectionResult` (which still says double, with both candidates) must not leak into it.
6. Then continue through the existing cutoff guard and delivery path unchanged.

Unchanged and not to be re-opened: a single on a void game, or a double with both legs void, is refused as now (archive, remove, CRITICAL DM naming the game and not the batter); an unreadable schedule delivers (ruling R10); the guard runs before the cutoff check (round 1's N1).

If the narrower contract cannot satisfy an obligation, STOP and say so in the handover. Do not add apparatus (new files, new state machines, new alert paths) to make it fit.

## How to work
- Test first. For each obligation write the test, run it, and keep the failing output (RED) before changing production code; then GREEN. Start from the reviewer's two failing probes: adapt them into `tests/test_pickpath_review_pp1.py` (or a sibling `tests/test_pickpath_review_pp2.py`), with imports from this worktree, not `/Users/eric/projects/bts`.
- Add a test for each numbered contract item above that is not already pinned, including: a death before the delivery save leaves the double on disk and a restart delivers the single; a failed send leaves the health check making no delivery claim; the caller's object is the single after the call (primary dropped and double-down dropped); no alert DM is dispatched for a reduction; both scorers add exactly 1 for the delivered single.
- Update or delete tests in `tests/test_postponed_delivery_guard.py` that pin the behaviour you remove (the archive of the double, the WARN dispatch, the "was delivered alone" wording). Say which and why in the handover.
- Mutation check: for each new behaviour, break it in the source, show the named test that fails, and restore.
- Run focused tests only: `TZ=America/New_York .venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Before handing over, run this set and paste its last line: `tests/test_postponed_delivery_guard.py tests/test_pickpath_review_pp1.py tests/test_late_delivery_guard.py tests/test_thin_slate_checks.py tests/test_incident_register_2026.py tests/test_scheduler.py tests/test_scheduler_decision_record_integration.py tests/test_scheduler_decision_record.py tests/test_streak_applied_dates.py tests/health` plus any file you add. Do not run the whole suite.
- Do not commit, push, stash, or change branches; the lead lands the change. Do not edit files outside `src/bts/scheduler.py`, `src/bts/health/late_delivery.py` and `tests/` without saying why. Do not read `~/projects/claude-shared`, `~/projects/bts-c2-framing-results` or `~/bts-runs`.

## Deliverable
Write `/Users/eric/projects/bts-pp-c1/.codex-review/pp-c1/handover.md`: files and functions changed; for each obligation and contract item, the test that pins it with its RED and GREEN lines; the mutants and which test each kills; the last line of the focused run; anything you did not do or are unsure of. Keep it under 120 lines.

When the file is complete, print one line `PPC1-<first 8 hex characters of the sha256 of the handover file>` and then `DONE`.
