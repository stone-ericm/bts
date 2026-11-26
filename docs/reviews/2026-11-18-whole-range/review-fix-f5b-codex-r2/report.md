# Verdict

**BLOCK** on `88b0427f` for the F5b change set. The named round-one cases pass. The change can suppress an early health alert and can hang the newly eligible scheduled check.

## Blocking findings

1. **A slow live all-clear can hide the last early warning failure.** On March 24, batter 1010/game 101 had a 10:05 pitch and a 09:30 manual warning. The 09:44:50 manual recovery wrote `confirmed` before sending its all-clear. Replace the pick with batter 1020/game 102 at 09:44:55, also for 10:05. At 09:45 the installed check saw the old entry and its warning for the replacement failed. The old all-clear took 30 seconds, so its receipt says 09:45:20. The 10:00 run is already at cutoff. A historical March 23 manual check then displaces the live marker. `src/bts/health/pick_entry.py:70–72,84–85` drops the early CRITICAL because the older write's receipt says 09:45:20. This delay exceeds the milliseconds described in Amendment 2. `last-slot-r1.log` shows the CRITICAL on `e66cc74c` and `last-slot-r2.log` shows no alert on `88b0427f`. Preserve the 4 MiB receipt read and compare the actual live marker write-completion time, recorded in the existing seal's metadata so the live marker, receipt and seal bytes can stay identical.
2. **An eligible early cron check can hang on a writerless units capture.** Commit a 06:10 pick at 04:55 and put a FIFO with no writer at its season's `*.json.gz` units capture path. Supply a batter match at 05:00. The candidate's installed schedule reaches `src/bts/entry_receipt.py:211`, an ordinary blocking open from both qualification (`:280`) and receipt construction (`:398`). The paired base's first scheduled run at 10:00 exits 0 outside the window and seals its receipt. A regular capture confirms normally on the candidate. With the pipe, the candidate reaches the outer timeout, and the stack shows the second blocked open after the probe interrupted the first. F1 already documents this for an eligible live check. F5b newly exposes it to scheduled checks on slates with no eligible original slot. Use a nonblocking open and reject non-regular descriptors in both early loads, applying the existing unavailable-capture handling and preserving the live callers. See `nonregular-capture-{base,candidate,stack}.log`.

## Edits

- Update the published-time caveat in `docs/ops/pick-entry-receipt-v1.md:9` with the ordering guarantee delivered by the fix. The adopted text otherwise matches. The rendered help and deferral text give no independent R64 block.

## Notes

- Current-format ledgers sent no all-clear after an unsuccessful warning, a cutoff reached during the fetch, or a check without a recipient. Another pitch correction after a completed all-clear and no new warning also stayed quiet. A pre-release `e66cc74c` marker created without a recipient can still yield a false all-clear through the legacy tier fallback at `src/bts/cli.py:2038–2039`. The base has no early ledger to migrate. See `b1.log` and `b1-round{1,2}-old-ledger.log`.
- The 10 ms delayed-publication control reproduced Amendment 2's stated limit. With publication before the early observation, health paged on both commits. See `published-limit-r{1,2}.log`.
- The sandbox rejected the `nice` priority adjustment. The bounded commands continued serially. The tracked tree and index stayed clean.

## 1. Round-one findings

- **B1.** Both missing-entry and wrong-game warnings survived a pitch-only correction and produced one all-clear on confirmation. Repeats made zero account calls. UTC `Z` and naive UTC spellings of the same instant stayed terminal. Manual comparison cases also sent one clear. See `b1.log`.
- **B2.** A 74,621-byte sealed live recovery with 326 captures kept health quiet after a historical marker displacement. The 04:55 manual response paused before replacement, the 05:00 early warning failed, and the 05:01 live replacement and publication suppressed historical fallback as specified. The same-date live marker also took precedence without a seal. See `b2.log`.
- **B3.** An unparseable `observed_at`, a missing offset, a prior ET date and a list-valued `receipt` each caused the next check to read the account and warn. A cutoff underflow returned exit 1 and sent `ENTRY UNVERIFIABLE` in morning. Early health reported the failed warning with current selection unknown. The next early check skipped the invalid pick. See `b3.log`.
- All eleven targeted cases failed on archived `e66cc74c` and passed on the candidate. The reviewed architecture, docstring, CLI help and receipt-guide replacements matched Amendment 2. The initial help assertion hit a terminal-width wrap and passed at width 105. Actual console help, deferral stdout and the subsequent live warning were captured. See `round-one-red.log`, `rendered-help.log` and `texts-wide.log`.

## 2. Introduced defects and base comparison

- Archived `bf78b642` source and candidate runs printed their respective `bts.__file__`. With identical fixture paths, pinned ET clocks and receipt IDs, 14 manual and 14 original-hours cases matched exactly in exit, stdout, stderr, captured DMs, live marker bytes, receipt and seal bytes, and morning results with no early evidence. Sixteen live health inputs matched at three times, including the base's existing list and UTF-8 exceptions and writerless live-marker wait. See `parity-{base,candidate,compare}.log`.
- Installer show/install/remove kept all other lines and retained the lock and timeout wrapper. Nominal normal and DST day counts were 96, 92 and 100. The early-reader matrix stayed bounded for bad files, including writerless early-marker and final-pick pipes. The newly reachable capture timeout and the health alert suppression beyond the stated publication interval are detailed above. See `schedule.log` and `health-inputs.log`.

## 3. Tests

- The 77 watchdog tests passed. All 29 search-matched existing test files ran, with 1,255 passed and eight skipped. The contest ledger parser added nine passes. The installer syntax and shim checks, backup inclusion and offline bootstrap ran. The only existing test differences are the two approved hour assertions in `test_entry_intent.py:169` and `test_wr_f1b_lock.py:59`.
- The candidate audit-driver run reproduced the archived base's 33 failures and 16 errors with the same test IDs. GNU rsync 3.x is absent on this machine.
- Five scratch changes each failed their targeted tests. These covered same-date live precedence, the early-pick cap, the receipt flag after the gate, and leg binding in the morning and health readers. See `guard-*.log`.

ran: archived-base byte comparisons, round-one replays, publication timing and capture checks, installer and operator text, watchdog and matched tests and scripts, audit driver, five scratch test checks; did not run: full fast suite, native Linux wrapper or live cron, live account or DM transports.
