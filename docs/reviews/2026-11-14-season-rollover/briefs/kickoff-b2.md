# Kickoff B2: two corrections after the first review round (S6)

The fresh Codex reviewer found no blocking defect in your scheduler change: its fifteen real-day scenarios passed. Two things need correcting.

## 1. A fixture that fails once the other change set is merged
`tests/test_scheduler_last_attempt.py`, the test that expects a skip recorded during an extra attempt (around lines 164 to 169; its helper around 97 to 99 plants only `streak.json` = 57). The other change set, merged beside yours, makes the decision's model streak 0 until a root-level pick dated in the current year has settled, and ignores a contest observation recorded in an earlier year. With that in place your fixture's 57 is no longer the decision streak, the day delivers instead of skipping, and the test fails at line 167.

Make the fixture state its streak the way production will: a contest observation file `account_state/contest_streak.json` recorded in the fixture's year (`{"active_streak": 57, "recorded_at": "<fixture date>T12:00:00Z", ...}` with whatever other fields the loader requires; read `src/bts/contest_state.py::_parse_state_file`). Freeze the contest-state clock to the fixture date as well as the scheduler clock (the loader takes `now=`; find how `run_day` reaches it and pin it the same way your other clocks are pinned). The test must pass on this branch AND on a tree that contains the other change set: do not depend on `streak.json` alone anywhere in this file. Check every other test in the file for the same dependence.

## 2. The generator reads a variable it does not own
`checks_with_last_attempts` reads `result` after each `yield`. `result` is assigned in the loop body by `run_single_check`, but the body can `continue` before assigning it (the `if now < target: continue` after the watchdog sleep, and the coalesced branch). After such an iteration the generator sees the PREVIOUS check's result, or no `result` at all if none ran. So an extra attempt that did not actually run can end the extra attempts because an earlier planned check had returned a pick that was later refused or archived; with no earlier check it raises `NameError`.

Required: the generator decides from what happened in the attempt it just yielded, never from an earlier check. If the attempt did not run a check, it is not a "check that returned a pick": move on to the next distinct start. Smallest change that does this (for example a per-iteration value the loop body sets, reset at the top of every iteration). Do not change the behaviour of any existing path.

Tests (RED first): an extra attempt whose sleep returns early so the body takes the `continue`, (a) after a planned check whose pick was refused and archived: the next distinct start still gets its attempt; (b) with no check having run before it: no exception.

## Deliverable
`.codex-review/sr/handover-b2.md`, under 30 lines: what changed, RED counts, the focused command from your first handover with counts, `git diff --stat` against your commit `75994d3`. Do not commit. Print `SRB2-<first 8 hex characters of the sha256 of the handover file>` and then `DONE`.
