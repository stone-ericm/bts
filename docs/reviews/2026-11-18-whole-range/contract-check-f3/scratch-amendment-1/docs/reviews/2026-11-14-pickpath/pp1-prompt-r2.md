# Adversarial review, round 2 of 2 (final): the fixes for your pp1 BLOCK

Same rules as round 1 (`.codex-review/pickpath-2026-11-13/prompt-r1.md`): read-only in `/Users/eric/projects/bts` at `HEAD`. Your r1 report is archived at `docs/audit/2026-11-14-pickpath-codex-r1.md`. The change is the top commit: `git show HEAD`.

## Scope
1. **B1** closed? `picks._replay_season_streak` returns None when today's pick is unresolved and today is in `applied_dates`. Re-run your interleaving. Then try to break the fix: an interleaving where this refusal now leaves the streak wrong or stuck (for example a marker left behind by a scorer that never retries; midnight between the crash and the retry; the next-morning 02:00 and 07:40 reconciles; `reconcile` and the daemon both running).
2. **B2** closed? One void leg of a double: the other leg is delivered as a single (`scheduler._playable_leg`, `_keep_playable_leg`). Re-run your day-level probe. Then try to break it: the kept leg's pick file, decision record, DM text, scheduler state, later result grading and streak increment (a single must add 1, not 2; `streak_increment_for_resolved_hit`), the shadow and live-forward paths that read the pick file, a restart between the archive and the save of the single, and the case where the kept leg is itself past its cutoff.
3. **N1, N2** as fixed.
4. Anything in this commit that produces a wrong production outcome that was right before it.

Your r1 probes were adopted as `tests/test_pickpath_review_pp1.py`; say if the adoption weakened any of them.

A false refusal or a lost delivery matters as much as a wrong delivery. A finding about test-code quality is a NOTE unless it makes a false claim about behaviour that nothing else covers. Say what you ran.

## Deliverable
Write `/Users/eric/projects/bts/.codex-review/pickpath-2026-11-13/report-r2.md` with: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `B1`, `B2`, `N1 and N2`, `New findings`, `Notes`, `What I ran`. Under 120 lines.

When the file is complete, print one line `PP2-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
