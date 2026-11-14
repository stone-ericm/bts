# Adversarial review, round 2 of 2 (final): the season-rollover fixes after both round-1 reviews and the test gate

Same rules as round 1. This checkout is now at the new commit under review; the change is still everything since the base `12dbaaa`: `git diff 12dbaaa HEAD`. What changed since your round 1: `git diff b3b1e3c HEAD`.

## What happened in round 1
Two reviewers and the test gate, between them:
- **Streak state.** (a) A malformed old pick file made the new history scan raise on every decision: fixed, such files are skipped. (b) The saver's season still followed the latest settled round's date, so last season's saver flag leaked into this season. (c) A separate audit showed the decision passed a saver that is not earned yet to the policy as "used", and the tracked policy chooses differently on that at streaks 7 to 9. (d) Working through these, the lead found both of its earlier definitions of an observation's season wrong in one state each and replaced them with one table.
- **Scheduler.** (e) An extra attempt whose pick the confirmation gate did not lock ran a second cascade, overran the cutoff and lost the pick. (f) The extra-attempt generator read the loop's `result` across `continue`. (g) An all-postponed slate got an extra cascade and a false CRITICAL. (h) The missed-pick alerts keyed their once-a-day dedupe on the wall clock. (i) The day-end alert's wording was false after a refusal, and the 8/30 incident test failed. (j) One fixture depended on `streak.json` alone.

## The contracts now
In `.codex-review/sr-review/contracts/`: `kickoff-a.md` (S3, S4, S5 still as written there), `kickoff-a3.md`, **`kickoff-a4.md` (S1 v3, S7, S2 v2: this replaces the S1 and S2 rows of kickoffs A and A2)**, `kickoff-b.md` with `kickoff-b2.md`, `kickoff-b3.md`, `kickoff-b4.md` (S6 and its corrections). The authors' handovers are in `.codex-review/sr-review/handovers/` for reference; do not rely on them.

## What to establish
1. Your round-1 findings: closed, by running the code.
2. The other reviewer's findings (a) to (j) above that you did not report: closed? Run them.
3. **The new code, which no one has reviewed:**
   - **S1 v3 (`kickoff-a4.md`):** every row of the seven-row table and the manual-override rule; any state the table does not cover or decides wrongly (name the state and what the bot does on that day); year and time-zone edges; the first seven days of a season in order, with and without the contest fetch working; a mid-season day with a broken rounds mapping; can the bot still refuse to pick with no way to recover without a human, and can it now pick with a streak it should not trust?
   - **S7:** the six-row saver table against what `simulate/mdp.transition_outcomes` means by the saver; every caller that reads the saver (decision, DM text, `decision.json`, dashboard, health) and whether any two now disagree; the fetch path's promotion rule; the change in policy actions at streaks 0 to 15 on realistic days.
   - **S2 v2:** false alarms and missed alarms on days one to three and mid-season.
   - **S6 corrections:** the reuse of the attempt's own lock decision (limited to extra attempts: is that limit sound, and does anything else now differ for planned checks?); the playable-games filter (a game that becomes void or playable during an attempt, doubleheaders, a failed status read); the alert's slate date across midnight and across restarts.
4. False greens in the tests added or changed since round 1, and any assertion in an existing test that was weakened. `tests/test_incident_2026_08_30.py` was edited: check that nothing it asserted before is lost.
5. Anything else in `git diff 12dbaaa HEAD` you now consider wrong.

Findings in code this change does not touch are NOTES for a follow-up.

## Deliverable
`.codex-review/sr-review/report-codex-r2.md`, under 150 lines: `Verdict` (SIGN, SIGN WITH EDITS, or BLOCK), `Round-1 findings`, `Blocking findings`, `Notes`, `False greens and changed assertions`, `What I ran`. A BLOCK names a concrete input or state, the wrong production outcome, the file and line, the smallest fix, and how you demonstrated it. When the file is complete, print `SRRC2-<first 8 hex characters of the sha256 of the report file>` and then `DONE`.
