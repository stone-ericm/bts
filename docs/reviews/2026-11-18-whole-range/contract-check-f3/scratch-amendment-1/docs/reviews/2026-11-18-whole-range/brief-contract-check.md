# Check a build contract before anyone builds from it

You are an adversarial reviewer of a CONTRACT, not of code. Assume it is wrong. Change no tracked file; probes go in `.codex-review/wr-check/`. No network. `.venv/bin/python` is synced. Small probe scripts may run directly; any pytest invocation goes through `~/bts-runs/yield_run.sh .codex-review/wr-check/<name>.log <command…>`, one at a time (a `nice` warning is not an exit). Create files with your edit tool, never a shell heredoc; do not edit a script after asking to run it.

The contract is `docs/reviews/2026-11-18-whole-range/CONTRACTFILE` (draft 1). It comes from the whole-range review: `docs/reviews/2026-11-18-whole-range/FINDINGS.md` and the slice reports with their reproduction scripts beside them. `CLAUDE.md` "2027 season" says what each earlier change set does and what is deliberately not fixed.

An author with no other context will build exactly what the table says; two independent reviewers will then attack the result against the table. For EACH row:
1. **Reproduce the defect on this tree** (run the slice's script or your own minimal one). Is the "today" column right?
2. **Is the required behaviour right and complete?** Derive what the code must do at each caller and each state; find the state the row does not cover, the place two competent authors would diverge, and the case where the fix makes something worse (a pick that went out before and now does not; an alert that fired and now does not; a 2026 number that moves; a refusal that now fires every cycle with no way out). For anything that silences or rewords an alarm, write the "must still fire" cases.
3. **Is it achievable and testable as written?** Name the test that would be RED on the base, and any required test that cannot fail.
4. **The standing hostile-input rows** (interrupt mid-step, malformed file, midnight and year boundary, failed or unconfirmed send, restart mid-step): which matter for this row, and what must hold.
5. **Interactions** with the other rows of this contract and with the other fix set (`kickoff-fix-a.md` and `kickoff-fix-b.md` are built in parallel by different authors: name every file both will touch and say how to keep them apart).

Classify each finding **MUST FIX** (an author would build something wrong, two authors would build different things, or a reviewer could block the result against the table) or **NOTE**.

Deliverable: `.codex-review/wr-check/contract-check.md`, under 130 lines. Verdict first: BUILD FROM THIS, BUILD WITH THESE EDITS (each edit written as the replacement text for the row: use this verdict whenever every MUST FIX has one), or NOT YET (only if a row's design has to change; then say the smallest design change, and whether the remaining rows can be built without it). When complete, print `MARKERTAG-<first 8 hex characters of the sha256 of contract-check.md>` on its own line and `DONE`.
