# Kickoff: change set F3, the morning report names a skip decided on a state that later changed (2026-11-25, version 1; base `main` at `5e0167b5`; F2, which is in review, touches only the health check and runner, so F3 is independent of it)

Finding C3-B1 of the closing test, VISIBILITY ONLY (`CLOSING-TEST.md`; `closing-test/carbon-slice3/report.md` B1; ruling R60). Tier A (it changes what the operator is told about a day's end state; it is read-only). Session `bts-wr-fresh13` (drafted by `bts-wr-fresh12`). Branch `fix/wr-f3`. **The scheduler's skip is NOT changed** (R40's class: Eric accepted that a skip is not re-decided).

## What is wrong on the base
An MDP skip decided at a lineup check is kept for the day (`scheduler.final_skip_candidate`) and written at end of day as `decision.json` with `action: skip`, `source: mdp`, and the state it was decided on (`streak`, `saver_available`, `state_source`, `state_status`, `contest_source_date`). If the contest state for that date then changes (the operator files a corrected dated observation, or revokes the season switch) and the later checks return nothing, the skip stands, and at 08:30 `bts day-outcome` reports `deliberate_skip`: exit 0, no DM. The operator is never told that the day was skipped on a state that was later corrected.

## The design (read-only; `src/bts/day_outcome.py` only)
1. **Evidence, read once, read-only, in `gather`:** (a) `account_state/contest_manual_requests.jsonl` (the append-only log every `bts set-contest-streak --date` write appends to before it writes the observation; change set 4a): the LAST line, in file order, that is a JSON object whose `effective_date` is the processed date D; (b) `account_state/season_contract.json` (the switch): its `status`, `revoked_at` and `switched_at`. Any file that is absent, not a regular file, unreadable, undecodable, or not of that shape is "no evidence" for that item; a malformed LINE of the request log is skipped.
2. **A new problem state `skip_state_changed`** (not in `OK_STATES`, so exit 1 and the DM), returned by `classify_day` in place of `deliberate_skip` for a decision with `action == "skip"` and `source == "mdp"` when at least one of:
   - **(i) a later dated observation differs:** the last request line for D exists and its `active_streak` differs from the skip record's `streak`, or the saver availability it implies (the live rule, `saver_state.policy_saver_available`, applied to its `saver_state` and `active_streak`) differs from the record's `saver_available`;
   - **(ii) the switch changed on D:** the switch's `revoked_at` or `switched_at` falls on D in ET and the skip record's `state_source` is the automatic kind.
   Otherwise the base's `deliberate_skip`. Every other state is computed exactly as on the base.
3. **The text (R64).** `detail`: `The policy skipped the day on streak S (saver available: A, from the K state), and the contest state for D changed afterwards: <what>. The scheduler does not re-decide a skip, so no pick was made on the changed state. Nothing needs doing for D now; check that the current state is right before the next slate with \`bts season-contract status --picks-dir DIR\`.` where `<what>` is `a dated observation for D gives streak S2 (saver available: A2)` for (i) and `the season switch was revoked at T ET` / `the season switch was enabled at T ET` for (ii), joined with `; ` when both. The DM is the base's `message()` (`BTS day check D: SKIP STATE CHANGED. …`). The status command is read-only.
4. **Not covered, written as limits:** the skip decision time is not recorded, so a dated observation filed BEFORE the skip check that the skip did not use (because it was invalid or for another date) is not distinguished from a later one; an automatic fetch that changed the reading after the skip (no dated observation, no switch change) is not seen (the automatic file is overwritten; its history is in `contest_ledger.jsonl`, not read here); a correction filed after 23:59 ET on D carries the next date and is not D's evidence.

## Must still catch (identical to the base)
| # | State | Must still happen |
|---|---|---|
| K1 | an MDP skip with no request line for D and no switch change on D | `deliberate_skip`, exit 0, no DM |
| K2 | an MDP skip whose last request line for D equals the skip's streak and saver availability (the skip was decided on that observation) | `deliberate_skip` |
| K3 | a skip not decided by the MDP | `skip_not_deliberate`, as the base |
| K4 | every non-skip state (`entered`, `entered_game_unverified`, `not_entered`, `no_decision`, `no_games`, `commit_not_scoreable`, `research_commit`, `delivered_under_research`, `contradiction`, `intent_unknown`, `schedule_unverifiable`, `entry_unverifiable`) | byte for byte the base's, whatever the request log and switch hold |
| K5 | the request log or switch absent, a directory, a named pipe, unreadable, undecodable, a JSON list, lines with other dates, malformed lines | no exception; that item is "no evidence"; a named pipe never blocks (read with the project's non-blocking regular-file reader) |
| K6 | the command's exit codes 0, 1, 2 and the "could not run" DM | unchanged |

## Must change (RED on the base)
| # | State | Base | F3 |
|---|---|---|---|
| M1 | Carbon's dated case: skip on streak 30 at 16:10; a dated observation for D with streak 1 filed at 16:30; later checks return nothing | `deliberate_skip`, exit 0, no DM | `skip_state_changed`, exit 1, the DM with (i) |
| M2 | Carbon's revoke case: skip on an automatic reading; `bts season-contract revoke` on D | `deliberate_skip` | `skip_state_changed` with (ii) |
| M3 | a correction that changes only the saver (streak equal, availability different) | `deliberate_skip` | `skip_state_changed` |
| M4 | both (i) and (ii) | — | one state, both reasons in the detail |

## Standing rows
- **An interrupt mid-step:** read-only; nothing is written.
- **A malformed or non-regular file:** K5.
- **Midnight ET and the year boundary:** D is the processed date (`--date`, the cron's `$YESTERDAY`); the switch times are converted to ET before the date comparison; a revoke at 00:05 on D+1 is not D's evidence (limit); December 31 to January 1 needs no special case (the request log has no season filter beyond `effective_date`).
- **A failed or unconfirmed send:** the base's DM handling (a failed DM leaves exit 1).
- **A restart mid-step:** stateless.

## The author's work (tests first; ONE commit)
`src/bts/day_outcome.py`, a new `tests/test_wr_f3_skip_state_changed.py`. The existing files that pin `day_outcome` (found by searching the tests for `day_outcome`, `day-outcome`, `classify_day`, `gather`, `deliberate_skip`: the contract check lists them by running) pass unchanged; if one must change, STOP and ask.

## Review
Gate; a fresh Codex xhigh review and a Carbon review; at most two rounds; merge on a Codex SIGN.

## Known and not fixed
1. Design 4's three cases.
2. The skip itself is not re-decided (R40); the report comes the next morning.
3. Only the last request line for D is compared.
