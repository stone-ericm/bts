# Check a build contract before anyone builds from it: change set 4a, version 2 (second and last check)

You are an adversarial reviewer of a CONTRACT, not of code. Assume it is wrong. No tracked file is to be changed; probes go in `.codex-review/cs4a-check2/` inside this copy. No network, box or contest account (install an audit hook that forbids DNS and connect in any probe that calls scheduler or delivery code). `.venv/bin/python` is synced; small probe scripts may run directly; any pytest invocation goes through `~/bts-runs/yield_run.sh .codex-review/cs4a-check2/<name>.log <command…>`.

The contract is `docs/reviews/2026-11-14-season-rollover/build/kickoff-cs4a.md` (version 2). A first reviewer checked version 1 and said NOT YET: its report is `build/cs4a-check/contract-check.md`, with its probes beside it (reuse them). The lead then rewrote the contract and changed the design in two places (ruling R29 in `docs/superpowers/plans/2026-11-12-2027-season-roadmap.md`): the command accepts only today's date, and it no longer writes the saver flag: on its day the observation itself is the saver authority. The table at the end of the contract says how each of the first check's findings was answered. Read also `design/RULING-R24.md` (the owner's decisions the contract must serve).

An author with no other context will build exactly what the table says; two independent reviewers will then attack the result against the table.

Report on:
1. **Each finding of the first check (F1 to F8):** closed by the text as written, or not. Run what can be run against `main`.
2. **What version 2 newly gets wrong.** In particular:
   - Row 9 and row 11 make the observation the saver authority on its day. Find every reader of the saver flag on the path from a decision to a delivered pick and to its record (`load_decision_streak_state`, the scheduler, `strategy`, `decision.json`, the pick file's `policy_decision`, the scorer's saver consumption in `picks.update_streak`, the dashboard, `maybe_auto_earn_saver` in the fetch): after this change, does any of them act on the flag file in a way that contradicts the observation's saver on that day, and is that contradiction a wrong pick, a wrong record, or only a display? The contract claims only the decision and the health check change.
   - `SaverState(state=…, season=…)` constructed from the observation: does that constructor and `policy_saver_available` accept it as the row says (run it)?
   - Row 2's "checked BEFORE anything on disk is read" with click's parameter source: implementable for every listed option, including flags with defaults and the boolean pair?
   - Row 3's two writes: log first, then the file. Is every failure outcome stated, and can the stated tests fail?
   - Row 8: "the manual file is parsed first, as today" against the real function.
   - The ten-step validity definition: is it one implementation? Any property whose check depends on an earlier one in a way the order does not settle?
   - "One clock per operation" for the health check and the command.
3. **Rows that are wrong against the real code,** states no row covers, promises the code cannot keep, places two competent implementers would diverge.
4. **Operator traps that remain,** given that this slice is declared not deployable alone and the contract names its limits. Only traps the contract does NOT name count.
5. **Tests:** required behaviour with no test asked for; tests asked for that cannot fail.

Classify each finding: **MUST FIX** (an author would build something wrong, or two authors would build different things, or a reviewer could block the result against the table) or **NOTE**. Do not report as MUST FIX a limit the contract already names.

Deliverable: `.codex-review/cs4a-check2/contract-check.md`. Verdict first: BUILD FROM THIS, BUILD WITH THESE EDITS (each edit written as the replacement text for the row or definition; use this verdict whenever every MUST FIX has such a replacement text), or NOT YET (only if the design itself, not its wording, has to change). Under 110 lines. When complete, print `CS4AD-<first 8 hex characters of the sha256 of contract-check.md>` on its own line and `DONE`.
