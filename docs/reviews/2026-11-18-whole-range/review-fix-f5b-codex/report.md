# Verdict

**BLOCK** on `e66cc74c` with Amendment 1. The paired live-path checks matched `bf78b642`. Early-state handling loses all-clears, can revive a resolved health alert, and mishandles malformed evidence.

## Blocking findings

1. **A pitch correction loses the all-clear for the same legs.** Commit batter 1010/game 101 at 04:55 for 06:10. The installed 05:00 run sends a missing-entry warning or a wrong-game warning for game 102. Correct the pitch to 06:20 and enter game 101. The 05:15 run proves the right game, writes `confirmed`, and sends no all-clear. At 05:30 it is terminal. The paired manual sequence sends the all-clear in both cases. `src/bts/cli.py:2037–2041` drops the warning history needed at `:2165–2166`. Carry a pending all-clear bit for the same legs across pitch corrections, independently of the reset timing tiers, until confirmation. The same comparison uses raw time strings. Rewriting the same 06:10 ET instant as `10:10:00Z` also opens a terminal marker and makes another account read. Compare parsed pitch instants. `early.log` records both warning sequences and the UTC spelling control.
2. **A sealed live recovery can still produce a false end-of-day CRITICAL.** On July 15 a 05:00 early warning fails. A 05:15 manual check verifies game 101 and seals a `confirmed` write. A historical July 14 check then replaces the global marker. With two unit captures July 15 health stays quiet. With 326 distinct captures from the same season, the genuine later sealed receipt is 74,621 bytes and health revives the old CRITICAL. `src/bts/health/pick_entry.py:61` applies the 64 KiB marker limit to receipts whose inventories grow over the season. In a separate replay with small receipts, pause a 04:55 manual confirmation after the account response, let the 05:00 early failure write, and then complete and seal the manual write. The same historical displacement revives the CRITICAL because `:73–75` compares response time, even though the manual write actually followed the early observation. Use a compact, durable time for the successful live marker replacement, bound to its seal, and keep lookup bounded without rejecting valid full-season receipts. Preserve the existing live marker and receipt bytes. See `health-season-inventory.log` and `health.log`.
3. **Malformed early inputs can short-circuit checking or abort reporting.** After a genuine 05:00 early confirmation for a 06:10 pick, corrupt `observed_at` to `"not a datetime"` or change `receipt` to a list. A missing entry at 05:15 gets `already_confirmed`, zero account calls and no warning. The list is even sealed as `references.confirmed_by`. Validate these fields before trusting the marker in `src/bts/entry_receipt.py:103–117`. Separately, after a genuine early observation set the final pick's `game_time` to `0001-01-01T05:00:00Z`. It parses, but the cutoff subtraction raises `OverflowError` in morning (`src/bts/day_outcome.py:186`) and in early health (`src/bts/health/pick_entry.py:99`). The morning command exits without its report or DM. Calculate and validate the cutoff in the bounded pick reader so both consumers apply the existing unavailable-pick behavior. `early.log` and `contract-edges.log` show the failures.

## Edits

- `ARCHITECTURE.md:355` can say “Early confirmation is terminal while the saved leg identities and first-pitch times match the early ledger. The live ledger retains its existing confirmation and retry rules.” This describes the reopening rule.
- `src/bts/cli.py:1907–1912` can say “The installed watchdog launches every 15 minutes ET. It prints the outcome when it skips a check. An eligible early check uses the dated early ledger. Manual and original-hours checks use the live ledger. While eligible, an unresolved entry is checked again, with throttled reminders at 30 and 15 minutes to cutoff and an all-clear when a warned entry is confirmed.” At `:1896` use “Each watchdog CLI invocation attempts to publish one entry receipt and its seal.”
- `docs/ops/pick-entry-receipt-v1.md:7–8` can say “A pre-10:00 ET cron check that passes the early gate writes any resulting marker to `data/health_state/pick_entry_check.early/<ET date>.json`. The morning report uses qualifying early observations only with a readable final pick that has the same legs and no nominal original-hours slot.” These are wording edits under R64.

## Notes

- The amended limits reproduced. A manual fix after the final early slot can leave the early warning without an all-clear. A warning and a clear can each be sent by both ledgers. An interrupt after the confirming replacement can leave the all-clear unconfirmed. A later bound live absence wins in the morning after the final pick loses its original slot.
- The archived checks were run with fresh installer output and candidate imports. Some original assertions encode the failures they found, the retired dated-copy layout, or v1 eligibility. The original wrappers also assume old checkout paths and, in the R11 repeat, hit same-day dispatch dedup. The adapted replay logs and the F5b tests supply the amended assertions. The first installer comparison fixture tried to recreate its shim directory. `schedule_probe.py` reuses one shim HOME and produces a byte comparison.
- Tracked code and tests were read-only. The historical source came from `git archive bf78b642 src`. The sandbox logged denied priority adjustment and the bounded runs continued. Native Linux wrapper tests skipped on macOS.

## R1–R13

| Row | Result on candidate |
|---|---|
| R1 | A 10:05→10:50 correction preserves three original-slot wrong-game warnings and `ENTRY UNVERIFIABLE`. |
| R2 | A replacement commit preserves the same three warnings and morning result. |
| R3 | A 10:06 correction leaves the initial 10:00 warning available after an early warning or failed send. |
| R4 | March 23 live bytes and the previous day's CRITICAL survive the March 24 early run. |
| R5 | The 10:15-start scratch change failed the 10:06 pitch test. The original 10:06, 10:15 and 10:20 cases passed. |
| R6 | 09:45 defers for 10:20. The sole original slot binds game 102 and warns. |
| R7 | The 09:55 manual crosswalk gap leaves the 10:00 one-minute warning. |
| R8 | Manual check reads and warns after an early batter-only confirmation. |
| R9 | One early wrong-game warning, one early all-clear, then no 09:45 or 10:00 DM. |
| R10 | Writerless final-pick FIFO returns exit 1 and `ENTRY UNVERIFIABLE` with early evidence alone, or `NOT ENTERED` with a bound manual absence. |
| R11 | Invalid live UTF-8 raises and the real health runner sends CRITICAL. |
| R12 | The Dec 31 live recovery survives both Jan 1 early runs. No false CRITICAL in this sequence. |
| R13 | Runtime manual tiers and same-date live precedence hold. Resetting manual tiers failed the suite. The precedence assertion missed a scratch change when the recovery also had a valid later seal. The no-seal control is in `health-live-precedence-no-seal.log`. |

### Amendment 1 additions

| Case | Result |
|---|---|
| Same-day replacement | A replacement is checked again after confirmation and gets its own unspent warning tiers. |
| Manual absence and manual fix | The later absence gets a morning DM. A manual fix after the final early slot can leave the early warning open, as stated. |
| Two ledgers | Each ledger sends its own warning and all-clear when both participate. |
| Corrected cutoff and late confirmation | Both genuine post-cutoff confirmation and a revised earlier cutoff give `ENTERED LATE`. A pitch-only correction loses the warning's all-clear in B1. |
| Adjacent dates and stale fallback | Early failure survives rollover. The normal later sealed live recovery suppresses fallback, with the two counterexamples in B2. |
| Invalid marker destination | The missing write is reported on stderr and in the sealed receipt, with exit 1 retained. |
| Non-regular final pick | Early health returns the failed-send CRITICAL with its observation time and unknown selection. |
| Delayed lock holder | The shims reproduce the documented loss of the sole 10:00 launch. |

## 1. Scope and base parity

`schedule-replay.log` compares real `show` and `install` for both intents with the same shim paths. All other lines are byte-identical. The watchdog differs only in the hour field and `--cron`. `parity-*.log` prints both import paths and compares 28 manual/original-hours cases with equal stdout, stderr, exits, DMs, changed paths and file contents after UUID and fixture-path normalization. Health and morning results matched with no early evidence.

## 2. Early ledger

`early.log`, `timing-morning.log` and the 60 watchdog tests exercise 06:10 and 10:05 slots, replacements, wrong-game binding, sends, correction, terminal confirmation, rollover and an invalid destination. The bad destination reports on stderr and in a sealed receipt with preserved exit 1. B1 and B3 show failures on correction and malformed state.

## 3. Health and morning

`health.log` and `live-health-*.log` cover the live/early marker matrix and a writerless FIFO under deadlines. Sixteen live inputs match the base at three times, including JSON lists, invalid UTF-8 and the FIFO. B2 covers failed sealed-receipt suppression. `timing-morning.log` and `contract-edges.log` cover identity binding, slot gain and loss, final cutoffs, `entered_late`, non-regular/oversized/invalid final picks, and F3/F4 together with a genuine early receipt. B3 covers the cutoff overflow.

## 4. Gate and timing

Real installed arguments pass exact 10:00, strict cutoff, inclusive 75 minutes, the earlier double-down leg, UTC/offset times and the 23:45 last slot. `schedule-replay.log` projects 92/96/100 clock-change/ordinary starts. `archive-contract-lock_boundary-main.log` and the wrapper tests preserve `flock -n`, 840 seconds plus 30 seconds, and reproduce the documented delayed-holder skip of the sole 10:00 slot.

## 5. Traffic and receipts

No-pick, preview, out-of-window, deferred and terminal gates sent no DM, made no account request and wrote neither ledger. The closing replay confirmed at 05:15 and produced 96 receipt pairs. `timing-morning.log` verifies early flags on eligible observations and their absence on local gates. `readers.log` verifies the backup `ops` roots include the dated ledger and runs the matched audit bootstrap on 21 synthetic days. The contest-ledger parser passed. The audit source inventory found no direct consumer of the new key or ledger path.

## 6. Operator text

`archive-operator-rerun.log` captures the cron header, architecture, receipt guide, help and stdout. At 09:45 a 10:05:01 pitch defers to the nominal 10:00 slot. At 10:00:02 it is already outside the window. The flag help and deferral stdout both say nominal. The edits above cover stale or imprecise wording. No R64 text block.

## 7. Tests and scratch changes

The 60 new cases passed. The 29 matched existing files gave 1,255 passes and eight skips. The auxiliary contest parser added nine passes and the archived Carbon probe added six. The separate wrong-game run passed all 70 cases. `test_audit_driver.py` returned the same 33 failures and 16 errors on candidate and base with GNU rsync 3.x unavailable. Only the two permitted existing assertions differ.

The new watchdog suite caught writing the early marker to live, date-only state, removing writer leg binding, dropping seal suppression, admitting early morning evidence despite an original slot, dropping the installed flag, moving the gate to 10:15, and resetting manual live tiers. It missed preferring early to a same-date live marker, lifting the early-pick cap, moving the receipt flag before the gate, and the two extra reader-only leg binding changes. Each surviving copy passed all 60 tests. The no-seal precedence, padded-pick, pre-gate flag and unchanged-decision/changed-leg checks exercise these gaps directly.

## 8. Standing cases

A send failure stays retryable after an interrupted retry and process restart. Atomic replacement leaves old or new ledger bytes according to the interruption point. Early FIFO, bad JSON/UTF-8, oversized markers and invalid tiers are handled locally. The midnight run opens only the new date and old Dec 31 state stays available. `early.log`, `health.log`, `timing-morning.log`, `contract-edges.log` and `watchdog.log` contain the cases.

## 9. Remaining diff

The code under review matches `e66cc74c`. `bash -n` and `git diff --check` passed. Health evidence ordering and the early-state input checks are covered by the findings above.

ran: paired cron generation, base/candidate CLI and health grids, archived F5 and contract probes with adapters, direct replays, matched pytest files and script checks, and one-at-a-time scratch changes; did not run: native Linux wrapper or live cron, live transports or backups, or the full fast suite.
