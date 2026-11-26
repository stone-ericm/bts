## Verdict

**BLOCK** on `84e2a0fc` against `a4e8278d`. All six round-one rows hold. The new marker and receipt paths still change alert delivery in the cases below.

## Blocking findings

1. **A manual check can make early tiers suppress the only original-hours DM.** On March 24 at 09:45 the 10:05 pick is absent and the early alert spends all three tiers. Correct first pitch to 10:06. A 09:55 manual run sees a row with an unavailable crosswalk and writes `present_unverified` with the inherited tiers, dropping their early origin. At 10:00 the row is gone. Both versions read the account, but only the base sends the one-minute warning. A manual check can also skip verification outright: with an early batter-only confirmation, a 10:00 manual check after correction and wrong-game binding says `already confirmed` while the base warns. `src/bts/cli.py:2030,2071,2127`. Keep early confirmation and tier provenance separate from manual marker state so manual checks use base state and original-hours cron can reset inherited early tiers while retaining unresolved wrong-game evidence. Replayed with `state_replay.py` and `surviving_mutations.py` against both trees.
2. **One wrong-game alert can produce two identical all-clears.** The 09:15 run on a 10:05 pick sees game 102 for game 101. The operator corrects the entry and the 09:30 run sees the new unit in game 101 and sends an all-clear. Only the saved pitch changes to 10:50. The 10:00 run sends the same all-clear for the same verified entry. The base sends none in this replay. `src/bts/cli.py:2045,2158,2161`. When bypassing an early `confirmed` marker, leave the account recheck enabled and withhold an all-clear for its already-resolved alert. The handover notes this behavior. `state_replay.py` captured both DMs.
3. **A restored early-only pitch can turn a known absence into a clean morning.** With the same decision on disk, 09:45 confirms the right entry for 10:05. A metadata correction to 10:20 adds the 10:00 slot. At 10:00 both versions see the entry removed and warn. Before the morning report the saved pitch is corrected back to 10:05. The base pages `NOT ENTERED`. The candidate reports `entered` and sends nothing. `src/bts/day_outcome.py:74–78,189`. The final-slot rule does this as written. Exclude early evidence when a bound non-early receipt observed the account at or after 10:00, even if the final pitch no longer affords a slot. `state_replay.py` captured both morning outputs.
4. **A named pipe at the final pick path blocks the morning DM.** Seal a genuine early `no_pick` receipt for 10:05, keep `decision.json`, and replace `<date>.json` with a FIFO without a writer. Given the same files the base emits `NOT ENTERED` and exits 1. The candidate blocks before output or a DM. `src/bts/day_outcome.py:155–156,189`. Read the final pick through a nonblocking open with a regular-file check on the opened descriptor and treat an unavailable pick as unavailable slot information. `morning_fifo.py` used the same real sealed receipt in both trees and stopped the candidate after 0.2 seconds.
5. **Invalid marker encoding silently suppresses the base's health CRITICAL.** Put `b'{"reason":"\xff"}'` in the only live marker, with no early field or dated copy. The health runner on the base sends a CRITICAL for `UnicodeDecodeError`. The candidate produces no alert. `src/bts/health/pick_entry.py:31–40,83–85`. Preserve the base's decode-error behavior on the live read and tolerate decode errors on the dated-copy read. `fallback_recovery.py` ran the real health dispatch with captured DMs in both trees.
6. **Displacing a recovered date can resurrect its resolved CRITICAL.** December 31's 18:00 send fails. January 1's 05:00 check archives the failure. A historical manual check for December 31 at pinned 18:15 confirms the entry and health sees it as resolved. January 1's 05:15 check replaces that live confirmation. December 31 health falls back to the first copy and sends `never confirmed` on the candidate. The base has no 05:15 launch and sends nothing. `src/bts/cli.py:2021,2049–2070` and `src/bts/health/pick_entry.py:86–95`. Retire the dated fallback when a live confirmation for its date is displaced, before publishing the new live marker. `fallback_recovery.py` captured the false CRITICAL. The first-copy rule does not describe this recovery consequence.

## Edits

- `scripts/cron-setup-hetzner.sh:18` and `ARCHITECTURE.md:355` describe the deferral in terms of a saved selection. Replace the gate sentence with “Before 10:00 ET, a check inside a committed pick’s entry window defers if a nominal 10:00–23:45 ET quarter-hour on the same date is also inside that window.” A preview or outside-window invocation returns at an earlier gate.
- `src/bts/cli.py:1890` help. Use “Before 10:00 ET, defer an in-window check of a committed pick if a nominal 10:00–23:45 ET quarter-hour is also inside that window.” The deferral stdout and the receipt outcome row already describe the nominal slot accurately.

## Notes

- The audit-driver failure and error node IDs match the archived base exactly. Both lack GNU rsync 3.x. The first combined test invocation timed out and was superseded by completed shorter groups. `nice` reported that the sandbox denied the priority adjustment. All recorded completed groups used bounded runs with `TZ=America/New_York`.
- The initial replay reused a unit ID across two games and reproduced an ambiguous binding. The final all-clear replay used the corrected entry’s distinct unit ID. The parity rerun pinned the fixture writer’s separate lineup-capture timestamp. The round-one replay pointed the original functions at their relocated tracked fixture. The interrupt check reads the sealed receipt’s `KeyboardInterrupt` detail because Click translates the exception at its command boundary.
- ET wall-time arithmetic across clock changes, nominal slots versus delayed starts, the morning count including no-attempt receipts, and January's checklist and `INCIDENT.md` deferral are contract limits or decisions.

## Must-still-catch rows

| Row | Candidate versus base |
|---|---|
| R1 | Holds. After the same-pick correction, 10:00, 10:15 and 10:30 read and warn for game 102. The morning pages `ENTRY UNVERIFIABLE`. |
| R2 | Holds. Replacement with game 102 warns about entry in game 101 at all three slots. The morning pages `ENTRY UNVERIFIABLE`. |
| R3 | Holds. With no intervening manual check, the revised 10:06 window sends the base’s 10:00 alert after the early tiers were spent. |
| R4 | Holds. The 05:00 run displaces the March 23 marker and March 23 health still sends the base’s CRITICAL. |
| R5 | Holds. The scratch 10:15 search start fails the new tests and Carbon’s original sole-slot tests. |
| R6 | Holds. The original mutation witness on the unmodified candidate defers at 09:45 and warns at 10:00 for 10:20 first pitch. |

## 1. Scope and manual parity

`installer.py` called `show`, `install` and `remove` with crontab shims for both intents. All 19 other emitted lines per intent and foreign bytes matched. Only the watchdog's hours and `--cron` flag differed. Both versions removed the old and commented marked lines. Refusal and missing-crontab cases passed in the installer tests. `parity_pinned.py` printed each `bts.__file__` and matched 124 manual invocations across 122 sequences for output, exits, transports, file names and bytes, including marker, receipt and seal bytes with normalized IDs. Normal health and morning reads on those inputs matched. Finding 5 is an exception for live encoding errors. The reviewed code paths matched `84e2a0fc`. Existing test changes were limited to the two authorized assertions.

## 2. Early state and later original-hours checks

`round1_relocated.py` ran the original `alert_probe.py` functions, checked the changed outcomes where the old assertions expected round-one failures, and ran `mutation_witness.py` in candidate mode. The closing fixture gained its 05:15 confirmation. Late commitment, late game binding, entry removal, the moved double's wrong leg and the out-of-order replay against an original-hours terminal marker all gave the expected outcomes. Findings 1–3 cover the failures after an intervening manual run, a prior all-clear and a slot gained then lost.

## 3. Dated copy and health

`marker_checks.py` verified exact CRLF bytes for December 31 copied January 1, no copy on auth failure, preservation of the existing copy's inode and bytes, and two successive displaced dates still paging. Health uses the valid copy with a different live date, uses live state for D, stays quiet with absent or malformed live state, and still raises on a live list. Malformed, non-object, wrong-date, invalid-encoding, FIFO and directory copies were ignored. A live FIFO did not block cron or health, a malformed live marker was replaced, and a live directory still caused the atomic write to raise. Findings 5–6 cover the missing CRITICAL and the stale fallback.

## 4. Gate and boundaries

`gate_readers.py` used arguments from the emitted line, which leaves `--window-min` at 75. Verified 08:49 versus 08:50 for 10:05, cutoff equality at 10:00, the 10:06 sole slot, exactly 75 minutes at 10:00 for 11:15, and 76 minutes at 10:00 for 11:16. A shortened 19-minute window and a double with a late primary used the earlier leg and the specified window. The morning helper agreed on the default-window slot cases. The evening 19:10 case kept the five base slots. `standing.py` showed the nominal 10:00 slot at first pitch 10:05:01 and a delayed 10:00:02 invocation outside the window.

## 5. Traffic, receipts and readers

`gate_readers.py` drove all 96 no-pick launches and the 40 added slots with preview, evening, terminal and shared-window states. No account reads, DM attempts or marker writes occurred, and each invocation sealed a receipt. Real early observations used the existing captured account calls. The captured restic `ops` command included `data/health_state` and preserved the dated copy and seals. `audit_readers.py` obtained the same sealed audit bundle and bootstrap CSV result with and without a genuine early receipt and dated marker. Findings 3–6 cover reader edge cases.

## 6. Lock, limit and clock changes

The emitted watchdog still runs `flock -n` before `timeout --verbose -k 30 840`. The shim took contention, timeout, kill and interruption paths in `test_wr_f1b_lock.py`. Real Linux binary cases were skipped on this Mac, as was the opt-in overlap diagnostic. Projection from the real cron fields yielded 92, 96 and 100 slots for spring, ordinary and fall dates, all spaced 900 seconds in UTC. The skipped and repeated hours made no account request in the pinned no-pick runs. Fetch sharing stayed at 10:30 and 13:30.

## 7. Operator text

The cron header, architecture paragraph and receipt guide reflect the new cadence and lock. The guide describes publication attempts, seals, nominal slot counts and the true-only field. Captured `--cron` help and stdout matched the round-one wording. The edits above clarify gate order. No text met the R64 blocking threshold.

## 8. Standing rows

`standing.py` interrupted both real installers before shim publication. Both retained the old table and published the full table on retry. The previous day's after-midnight pitch did not affect January 1. An interrupted send left no marker, an unconfirmed send wrote `dm_failed`, and restart retried. `marker_checks.py` interrupted the watchdog after copy publication and before marker replacement, then verified restart kept the original copy and replaced the live marker. `morning_fifo.py` found the new non-regular-file regression.

## 9. Tests and deliberate changes

Every test file from the brief's search ran in separate groups. The new F5 suite and existing groups gave **1,079 passed, 8 skipped**. Carbon's original probe added **6 passed**. The audit-driver gave the same **33 failures and 16 setup errors** on both trees.

`mutations.py` used a fresh scratch copy per case and verified imports. Tests caught writing the early key on every run, dropping the reset, resetting the wrong-game observation, dropping the morning slot rule, including its cutoff, dropping the copy, overwriting an existing copy, and dropping `O_NONBLOCK`. The full new suite stayed green with reset enabled on manual invocations and with the dated copy preferred even when live state was for D. These are test gaps. `surviving_mutations.py` supplies the missing inputs for both. The extra 10:15 slot-start change also failed tests.

## 10. Other changes

The first-copy and final-slot rules need the recovery cases above covered by the contract and tests. No other code finding came from the diff.

ran: generated-cron comparisons, archived-base replays, original review probes, reader and recovery checks, matched pytest groups, Carbon probe, audit-driver baseline, and scratch changes; did not run: real Linux flock/timeout or Debian cron, the opt-in overlap diagnostic, the whole fast suite, or real network and transport calls.
