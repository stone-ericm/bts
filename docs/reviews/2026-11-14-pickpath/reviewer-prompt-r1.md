# Adversarial review: the delivery guard's leg reduction, rewritten by a fresh author (tier A)

Round 1 of at most 2. You are a fresh reviewer with no part in the earlier rounds. Work read-only in this worktree: `/Users/eric/projects/bts-pp-c1` (branch `pp-c1`). The change under review is UNCOMMITTED: `git diff` and the new file `tests/test_pickpath_review_pp2.py`. Do not read `~/projects/claude-shared`, `~/projects/bts-c2-framing-results` or `~/bts-runs`.

This code decides whether and what a real contest account's daily pick is delivered in 2027. Assume the author is wrong somewhere and find where.

## Background
- A delivery guard was added on 2026-11-13: a pick on a postponed or cancelled game is not delivered (`scheduler._void_committed_games`, `_refuse_void_game_delivery`), and a double with exactly one such leg is reduced to a single of the other leg (`_playable_leg`, `_keep_playable_leg`).
- Two earlier review rounds are archived: `docs/audit/2026-11-14-pickpath-codex-r1.md` and `-r2.md`. Round 2 blocked on C1 (the reduction unlinked the cached double before saving the single: a crash between the two lost the day) and left three notes (a health message claiming a delivery before any send; the inline shadow comparison given the dropped primary's name; a WARN that was never sent).
- The previous author was stopped. A fresh author rewrote the reduction to a narrower contract (its brief: `.codex-review/pp-c1/kickoff.md`, "The contract to implement"; its handover: `.codex-review/pp-c1/handover.md`). Treat the handover's claims as claims to test, not as evidence.

## What to establish
1. **C1 and the three notes:** closed, by running the code? Reproduce each original failure against the committed code (`git stash` is NOT allowed; use `git show HEAD:<path>` into a scratch copy if you need the old behaviour) or rely on the archived probes at `.codex-review/pp-c1/reviewer_probes_pp2.py`, then show the new behaviour.
2. **The in-place reduction itself.** The DailyPick the caller passed is now mutated before the cutoff check and before any save. Find a path where that mutation produces a wrong outcome: the delivery is then refused (past the cutoff, or the send fails) and the caller or a later cycle keeps using the mutated object or the file on disk (which is still the double); a later check the same day re-reads the double from disk while the postponed game has since been rescheduled or its status has changed back; the reduced pick's `policy_decision`, `runner_up` or other fields that still describe a double; `run_day`'s three call sites and what each does with `daily` after a False return; the fallback's cached-pick path; private, DM and public modes; a restart at each point.
3. **Records:** `state.delivery_refusals` gets one entry per reduction call. Can repeated calls (retries, restarts, each scheduled check on the same day) grow it without bound or produce a misleading health alert? What does the `late_delivery` health check now say on an ordinary reduced-and-delivered day, and is a WARN there right?
4. **Anything the rewrite dropped** that the earlier code did correctly and nothing now does (for example a record that a reduction happened when the pick file is later the single; the archive of what was originally selected).
5. **False greens** in `tests/test_pickpath_review_pp2.py` and the two tests changed in `tests/test_postponed_delivery_guard.py`: a test that would still pass with the behaviour it names broken.

Not open for re-litigation unless it produces a wrong outcome nobody has named: the rulings in the roadmap ledger (`docs/superpowers/plans/2026-11-12-2027-season-roadmap.md`, R10 unreadable schedule delivers, R11 a reduced double is delivered as a single), and the refusal of a single or an all-void double.

You may run focused tests and write probes under `/tmp`: `TZ=America/New_York .venv/bin/python -m pytest <files> -q -p no:cacheprovider`. Do not run the whole suite. Do not edit the worktree.

## Rules for findings
A BLOCK names a concrete input or state, the wrong production outcome, the file and line, and the smallest fix, and says how you demonstrated it. A finding about test-code quality is a NOTE unless the test makes a false claim about behaviour that nothing else covers. A false refusal or a lost delivery matters as much as a wrong delivery.

## Deliverable
Write `/Users/eric/projects/bts-pp-c1/.codex-review/pp-c1-review/report-r1.md` with: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `C1 and the notes`, `Blocking findings`, `Notes`, `False greens`, `What I ran`. Under 150 lines.

When the file is complete, print one line `PPR1-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
