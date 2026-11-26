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

## Amendment 1 (2026-11-26 01:11, `bts-wr-fresh13`, after the contract check `WRCHKF3-275302f2`, BUILD WITH THESE EDITS; `contract-check-f3/`). It overrides the text above wherever they differ.

The check reproduced both faults (a 16:10 skip on streak 30; a dated streak 1 filed at 16:30, or a revoke; 08:30 `deliberate_skip`, exit 0, no DM). It found version 1 unbuildable as written: (a) the request log records attempts, not installed observations (a failed install leaves a valid line; a request line can precede the skip); (b) every skip record holds `state_source: "contest"` (dated, automatic and legacy alike), so rule (ii)'s "automatic kind" never matches; (c) the files cannot order a correction against the skip, because the skip's decision time is not recorded; (d) the text claimed an order, said "Nothing needs doing for D now", and used a literal `DIR`. Its remedy records skip provenance (selection time, observation kind, live requirement) in the scheduler and its saved state.

**My ruling (design authority under the lead; cost if wrong: F3 stays silent in some sequences it could have caught, and it can alert in research mode after a revoke; it can never change a pick, a decision record or a state file):** F3 stays READ-ONLY in `src/bts/day_outcome.py`, with NO scheduler or decision-record change. Recording provenance in the scheduler's saved state is a pick-path change with the known rollback hazard (`scheduler.load_state` quarantines a state file with an unknown field) and is out of scope for a visibility piece; it is ledgered. Instead F3 compares the skip record with the contest state ON FILE for D at report time and makes NO claim about order.

### The design, replacing design 1 to 3
1. **Base first.** `classify_day` computes the base outcome exactly as today. Only when it is `deliberate_skip` (an MDP skip on a playable day with valid intent) is the evidence read; every other outcome is byte for byte the base's (K4).
2. **Evidence, read once and read-only, each item independently "no evidence" when absent, not a regular file, unreadable, undecodable, too deep or too large, or invalid** (`read_regular_file_text`; the existing validators: `contest_state.validate_dated_observation` for a dated observation, the season-switch validator for the switch):
   - (a) **The dated observations for D on file:** `account_state/contest_streak.manual.json` and every `account_state/contest_streak.manual.revoked-*.json` (the files `bts season-contract revoke` sets aside), each counted only if it validates as a dated observation whose effective date is D. Of those, the one with the latest offset-aware `recorded_at`. The request log `contest_manual_requests.jsonl` is NOT read.
   - (b) **The switch** `account_state/season_contract.json`: its validated `revoked_at`, for the season of D.
3. **Reasons:**
   - (i) **Dated difference:** (a) exists and its `active_streak` differs from the skip record's `streak`, or the policy saver availability it implies (`saver_state.policy_saver_available(SaverState(state=<its saver_state>, season=D.year), <its active_streak>)`) differs from the record's `saver_available`. The record's `streak` must be an exact non-negative `int` and `saver_available` an exact `bool`; otherwise no reason (a malformed record keeps the base result).
   - (ii) **Revoke on D:** `revoked_at` converts to D in ET, the entry intent is `enter`, and NO dated observation for D on file has exactly the skip record's streak and policy saver availability (if one has, the skip could have been decided on it, and a dated observation in effect is used whatever the switch says).
   - With at least one reason the state is `skip_state_changed` (not in `OK_STATES`: exit 1 and the base's DM handling); otherwise the base's `deliberate_skip`.
4. **The text (R64), adopted from the check with the order claims removed.** `detail`: `The policy recorded a skip for D using streak S and policy saver availability A.` + one sentence per reason + `Check MLB for unsettled entries and verify the current streak, season best and saver. Follow checklist row B8a for the current ET date and year before changing any state.` Reason (i): `A dated observation now on file for D gives streak S2 and policy saver availability A2.` Reason (ii): `The season switch records a revocation on D at T ET.` A and A2 are `available` or `unavailable`; T is `HH:MM`. The DM is the base `message()` (`BTS day check D: SKIP STATE CHANGED. …`). No path, no `DIR`, no executable command.

### Concrete examples (R70 item 2); each a test
| # | Sequence | Expected |
|---|---|---|
| M1 | 16:10 MDP skip on streak 30 (automatic); 16:30 the real `bts set-contest-streak --date D --streak 1 …` installs; later checks return nothing; 08:30 D+1 | `skip_state_changed`, exit 1, reason (i) with S2 = 1 |
| M2 | automatic skip on 30; `bts season-contract revoke` on D after it (the manual file absent) | `skip_state_changed`, reason (ii) |
| M3 | skip on a dated 30/`used`; a later dated observation for D with 30/`active` installed | `skip_state_changed`, reason (i) (availability differs) |
| M4 | automatic skip on 30; a dated 1 installed; then revoke on D (the dated file set aside as `revoked-*`) | one state, both reasons, one DM |
| K1 | MDP skip, no dated observation for D on file, no revoke on D | `deliberate_skip`, exit 0, no DM |
| K2a | a dated 30 installed at 09:00, the 16:10 skip decided on it | `deliberate_skip` |
| K2b | a dated request that failed to install (request line appended, observation not written) | `deliberate_skip` (the request log is not read) |
| K2c | a pre-check `enable` and fetch on D, the skip on the new automatic reading | `deliberate_skip` (enable is not a reason) |
| K2d | the B8a restore: revoke on D at 09:00, a dated 30 filed after it, the skip on that dated 30 | `deliberate_skip` (a dated observation on file equals the skip) |
| K3 | a non-MDP skip on a playable day | `skip_not_deliberate`, as the base |
| K4 | every outcome other than `deliberate_skip` (including `entered_late`, and MDP skips masked by `intent_unknown`, `schedule_unverifiable`, `no_games`, all-void games), with any evidence on disk | JSON, message, exit and send byte for byte the base's |
| K5 | each evidence file absent, a directory, a named pipe, unreadable, invalid UTF-8, malformed, deep JSON, a long number, the wrong date or season; a malformed skip record | no exception, never blocks; that item is "no evidence"; valid evidence elsewhere still counts |
| K6 | exit 0 clean, 1 for problems (including a failed DM), 2 with the existing could-not-run DM | unchanged |
| K7 | December 31 skip, a revoke at 00:30 on January 1 ET | not D's evidence: `deliberate_skip` |

### Known and not fixed (replacing the list above)
1. No order is known: a dated observation for D filed BEFORE the skip but not used by it (it failed validation at the time, then was corrected) reads as a difference (it is one).
2. A correction that returns to the skip's values, a best-only change, a legacy override, an automatic-only change and a change written after D (the next day's dated observation replaces the file) are not seen.
3. In research mode reason (ii) is not used; in enter mode a revoke on D before a skip decided on a private path is reported.
4. The skip is not re-decided (R40); the report comes the next morning.
5. Recording skip provenance in the scheduler (the check's remedy) is ledgered, not built.

### Files
`src/bts/day_outcome.py`, a new `tests/test_wr_f3_skip_state_changed.py`, and the `bts day-outcome` description in `CLAUDE.md` (the new state). No existing test changes (the check ran the 12 pinning files against a scratch version: none failed); a change is a STOP.
