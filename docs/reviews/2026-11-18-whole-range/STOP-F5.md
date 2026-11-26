# STOP: change set F5 (the entry watchdog every 15 minutes all day, with an early gate), after round 2

Written 2026-11-26 00:30 ET by `bts-wr-fresh13`. **Not merged. `main` is unchanged** (the watchdog still runs `*/15 10-23`, so a game before about 11:15 ET gets fewer checks and one before 10:05 gets none: finding BL-O4 stays open). Branch `fix/wr-f5` (pushed): round 1 `5a80f0cc`, round 2 `84e2a0fc`, records to this file. Under the kickoff (R57) F5 goes to the lead; no follow-up of the session's own.

## What was built
Round 1 (`bts-wr-fresh12`): the line `*/15 * * * *` with F1b's lock and limit and a new `--cron` flag; with `--cron` a start before 10:00 ET defers (`base_slot_available`) when a 10:00 to 23:45 quarter-hour is inside the committed pick's window. Round 2 (Amendment 2 and its two errata): an `early_cron_run: true` key (written only when true) on markers and receipts of an early run that went past the gate; at an original-hours `--cron` run inside the window such a marker is not terminal and its tiers do not count, while F1's wrong-game observation still counts; the morning report ignores early receipts as evidence when the final pick has an original-schedule slot; a `--cron` run that displaces another date's `alerted` or `dm_failed` marker first copies it to `pick_entry_check.<date>.json`, which the health check reads when the live marker is for another date.

## Round 2 verdicts
- **Codex xhigh: BLOCK** `WRF5REV2-67ca3192` (`review-fix-f5-codex-r2/report.md`). All six round-1 rows (R1 to R6) hold. Six new findings, all in the round-2 state rules:
  1. a MANUAL check between an early run and 10:00 carries the early tiers or the early `confirmed` into a marker without the early key, so the original-hours run again suppresses the only DM or says `already confirmed` (the round-2 key is lost when a manual run rewrites the marker);
  2. after an early wrong-game alert and an early verified correction, a pitch correction that adds a base slot sends a second identical all-clear (stated in the handover; the contract did not state it);
  3. a pitch corrected to gain a base slot and then back again before the morning: the base pages `NOT ENTERED`, the candidate reports `entered` (the final-slot rule as written);
  4. a named pipe at the final pick path blocks the morning report (the new pick read is a blocking open);
  5. a live marker with invalid UTF-8: the base's health run reports a CRITICAL (`UnicodeDecodeError`), the candidate reports nothing;
  6. a date displaced twice (a failed send, then a later manual confirmation for that date, then another early run): the first copy outlives the recovery and that date's health sends a false `never confirmed` CRITICAL.
  Two text edits (cron header and `ARCHITECTURE.md`; the `--cron` help). Held: crontab parity, 124 manual invocations equal to the base, traffic (no request on any no-attempt run), F1b's lock and limit, the clock-change projection, 1,079 tests, 8 of 10 scratch changes (resetting on manual runs too, and preferring the copy when the live marker is for D, are not noticed by any test).
- **Carbon High:** still running at the time of writing, for the record (`review-fix-f5-carbon-r2/` when it ends).
- **Gate:** the gate on `6eed95a4` was cancelled at about 15% (00:09 to 00:26) to free the slot; no gate has run on `84e2a0fc`. Round 1's gate on `0f14c380` was green.

## The session's own mistakes on F5
1. Amendment 2 item 3 first wrote the slot interval with its ends reversed (the author stopped before any edit; erratum).
2. Amendment 2 first required `early_cron_run: false` on every other receipt, which would have changed F1's pinned parity digests for manual runs (the author stopped; erratum 2).
3. **The class of round 2's findings 1, 3 and 6 was in view and not written into the contract:** Amendment 2 marked early state but did not say what happens when a run WITHOUT the key (a manual run, or a later write) inherits that state, nor that the morning slot rule and the dated copy are judged on files that can change again before they are read. The must-still-catch table had round 1's sequences only, not the sequences round 2's own rules create. Finding 4 (a blocking open in the new pick read) and finding 5 (the live read's decode error) are the standing "malformed or non-regular file" row, which the contract named and the review of the author's diff did not check line by line.

## Ways forward for the lead (R57: the lead rules)
1. **Smaller F5 (recommended by the session):** keep round 1's gate and drop round 2's state machinery; instead, make the extra early runs WRITE NOTHING that a later run or reader trusts: an early run that passes the gate verifies and alerts as now, but records its marker under a separate name (`pick_entry_check.early.json`) that no original-hours run, manual run, health check or morning report reads as terminal; the morning report counts an early receipt only when the final pick still has no base slot. This removes findings 1, 2, 6 by construction; 3 remains as a stated limit (or the reviewer's rule: exclude early evidence when a bound non-early receipt observed the account at or after 10:00); 4 and 5 are two-line fixes (non-blocking pick read; the base's decode error kept on the live read). It needs its own contract check first.
2. **Accept BL-O4 as a written limit** for 2027: an opener before about 11:15 ET gets fewer watchdog checks (none before 10:05); the morning report pages `ENTRY UNVERIFIABLE` after such a day; the operator confirms the entry in the MLB app by hand on those days (a checklist line, January with Eb).
3. **Round 1 alone** (the gate without round 2) is not an option: round 1's BLOCK (B1, B2) stands for it.
4. Leave unmerged.
