# Verdict

**BLOCK** for `1d30de86` on `1e69d0f0`. F1 can forget an unresolved wrong-game observation after the warning fails, issue a false all-clear and stop checking the pick.

## Blocking findings

**B1. A failed warning loses its observation on the next different result.** At 18:25 ET on March 24, the watchdog sees no entry and sends `initial`. At 18:35, batter 1010 matches, pending row 77 is in round 1 and the capture binds it to game 102 while the saved game is integer 101. The first wrong-game DM fails. The marker is `alerted/wrong_game` with only `initial` consumed. At 18:45, a crosswalk gap writes `present_unverified` and erases the only evidence of the observation. No entry and a batter mismatch are two other ways to erase it.

At 19:00, the crosswalk again resolves the batter. A second real capture now also binds unit 77 to game 101, making the inventory conflicting and qualification `unit_unverified`. F1 writes terminal `confirmed/match`, exits 0 and sends “✅ BTS pick entry confirmed for 2027-03-24 (match).” A1(b) requires an unresolved result and A2 requires the still-due first warning. The same gap after a first failed warning without any prior DM silently confirms.

The observation is inferred from the replaceable reason or a consumed tier at `src/bts/cli.py:2023`. Marker updates at lines 2072 and 2161 overwrite the reason. The smallest fix is to persist an unresolved observation bit on same-date nonterminal markers when overwriting the only evidence, read it in A1(b), and leave `wrong_game` in `escalations` reserved for a consumed warning. The regression failed on all three intermediate results and passed with this fix in a scratch copy. Its 68 non-FIFO F1 tests stayed green. [Regression](failed_observation.py) · [F1 output](failed-observation-candidate.log) · [Fixed scratch output](failed-observation-fixed.log) · [First warning without prior DM](sequence-run.log)

## Edits

- **A3 missing fields.** The locked-delivery double says “The scheduler tried to send None + DD Batter 1020” even though the leg detail correctly names `batter 1010`. Amend the locked-line name rule to use the A3 fallback for either missing name while retaining ` + DD ` between names. The fixture sentence becomes “The scheduler tried to send batter 1010 + DD Batter 1020.” When a missing saved game ID is listed in case (b), replace “recommended game None” with “recommended game ID unavailable.” The affected legs and verification directions remain clear under R64. [Rendered text](sequence-run.log)
- **A3 cutoff.** The inherited “cutoff passed during fetch” line also fires when the responses finished before cutoff and the local capture read crossed it. Use “check-pick-entered: submission cutoff passed during this check; skipping the nag (entry can no longer change)” on the wrong-game path and update the A3 cutoff sentence accordingly. The observed skip itself was correct. [Pinned cutoff](cutoff-candidate.log)
- **A6.2 health.** Replace the last clause with “After the cutoff, pick-entry health reports WARN for a final `alerted` marker and CRITICAL for a final `dm_failed` marker. A final `present_unverified` or `confirmed` marker produces no pick-entry health alert.” Both the failed warning and a subsequent crosswalk gap refute the unconditional WARN claim. [Health states](outcome-candidate.log)
- **A6.4 rollback.** Replace the first sentence with “After rollback, an in-window batter match overwrites an `alerted`/`wrong_game` marker with terminal `confirmed` and attempts an all-clear DM if a recipient is configured.” The second sentence about treating it as batter-only still applies. [Rollback](rollback-base.log)
- **A6.5 morning priority.** Replace the sentence about an earlier clean receipt with “The morning report never names a wrong game. Timely game proof or a timely batter match with only confirmed or unit-unverified slots can produce a clean morning result while the watchdog still reports unresolved game verification.” The *later* conflicting capture produced a clean morning result with an active watchdog WARN. [Morning states](outcome-candidate.log)

## Notes

- The review used Amendments 1–3, including M10. The separate capture reads, shared reminder tiers, terminal confirmation, and accepted capture limits behave as specified where checked.
- Unreadable saved times raise before fetching on both revisions. The optional first-pitch omission in A3 is unreachable for an unreadable saved time through the current command. [Time case](sequence-run.log)
- An archived-source path test fails identically on changed and unmodified scratch copies. Its normal-checkout run passes. The GNU rsync 3.x prerequisite failed identically on base and F1. No tracked file was edited. [Archived test](tests-archived-scripts2.log) · [Base audit](tests-audit-base.log)
- The sandbox declined the priority change requested by `nice`. Runs remained bounded and serial. Victoria 3 was absent at the process checks before the FIFO groups.

## 1. Revision and comparison

`git diff 1e69d0f0 1d30de86 -- src docs/ops` shows the watchdog, receipt helper and one operations sentence. Only the new F1 file changed under tests. Both archives printed `bts.__file__` and loaded from their requested revisions.

Compared 5,510 paired runs across the slot states and verifier results, prior marker statuses and all tier subsets, delivery states, four fetch failures and identity mismatch, send success/failure/no recipient and window/tier/cutoff times. The grid crosses slot states with each other and runs the marker/timing and fetch combinations separately. It compared stdout, stderr, attempts and sent DMs, marker bytes with receipt ID/time normalized, exit and receipt fields. 4,690 were identical, 708 took the A1 warning path, 58 resolved a prior observation, and 54 showed A2 changes to the initial alert or tier preservation on a prior wrong-game history. B1 requires successive runs. [Comparison](grid_run.py) · [Example shard](grid-cross-0-candidate.log) · [Tier shard](grid-ladder-0-candidate.log)

## 2. Contract rows

Ran N1–N13 and M1–M10 through the real command and file-backed fixtures. F1's tests passed all rows as written. Independently checked the window boundaries at 18:24/18:25 and 19:34/19:35, including a delivered legacy pick without a decision. The additional sequence across N5 and M5/M9 produced B1. [F1 tests](tests-f1.log) · [FIFO rows](tests-fifo.log) · [Window boundaries](rows-candidate.log)

## 3. False alarms and missed observations

36 double cases across the pending and settled profile source shapes, including rows in both, matched the expected game outcome for each leg. Single-slot checks at saved game IDs `null`, `"101"`, `101.0`, `true` and `101` alerted only for the integer. Shared-batter picks without a prior observation confirmed. No additional A1(a) miss or false alarm was found in these checks. B1 misses the required A1(b) follow-up. [Rows](rows-candidate.log) · [Game IDs](sequence-run.log)

## 4. Escalations

Checked both first-alert orders, entry removal, full correction and capture change after a terminal marker, a crosswalk gap, each tier's failed send and retry, no recipient, cutoff during the check and failed all-clear. A successful first wrong-game DM leaves `initial` due. The successful warning survives a crosswalk gap, reminders retry then fire once, and a double with one game proved remains unresolved until both are proved. A failed first wrong-game DM survives an immediate retry and fails to survive B1's intermediate results. The first driver incorrectly expected an identical countdown a minute later. The second checked the elapsed minute and passed. [Sequences](sequences.log) · [Corrected countdown and remaining tier checks](sequence-run.log)

## 5. Operator text

Compared complete first, reminder, locked-delivery priority across all tiers, unresolved stdout, and batter-only text in M10. Checked one affected leg of a double, both affected legs, missing and non-string names and case (b) without a `wrong_game` leg. The valid-input strings matched the contract. The Edits section covers the observed text gaps. None of those text gaps met R64's blocking rule. [Render checks](sequence-run.log) · [M10 tests](tests-f1.log)

## 6. Standing rows

Checked invalid gzip, truncated gzip, invalid JSON and UTF-8, top-level nonobjects, noniterable units, missing/null/false/zero units, mapping/string units, invalid unit rows, directory and denied capture reads. Raised reads gave terminal batter-only confirmation and two diagnostics on F1. Silent shapes had the base outcome. Checked KeyboardInterrupt and SystemExit, publication success and failure after an interrupt, midnight start anchoring, UTC capture year versus ET date, invalid and unreadable markers, instance of failure after send and the FIFO cases. Base and F1 reproduced the inherited marker failures. [Standing runs](standing-candidate.log) · [Base](standing-base.log) · [FIFO](tests-fifo.log)

## 7. Receipt and morning report

On the 708 A1 warning comparisons, the receipt differed only in `marker_status` and `exit_code` after normalization. The verifier still recorded the batter match and publication qualified it independently. `day-outcome` exercised timely and late proof, timely batter-only proof, absence with and without a wrong-game observation, wrong game alone, failed alert, identity gap and unavailable receipts. Capture changes between the reads reproduced both accepted disagreements between watchdog and morning. Real rollback confirmed the same wrong-game entry, sent the base all-clear, then skipped further checks. See the A6 edits for the exact limits. [Morning](outcome-candidate.log) · [Separate reads](outcome-capture-changes.log) · [Rollback](rollback-base.log)

## 8. Tests and scratch changes

The search found 62 test files, including F1. All were attempted in separate bounded groups. 3,526 passed, four skipped and one xfailed. The audit driver's 33 failures and 16 setup errors occurred on both revisions because GNU rsync 3.x is absent. [Test list](tests-list.log) · [Audit output](tests-audit.log)

The tests noticed dropping the integer check, dropping the shared-batter check, dropping case (b), consuming `initial` with the first wrong-game alert and swapping the texts. **No runnable test noticed confirmation when any leg is confirmed.** The independent double check caught it when the primary was proved and the second leg became unbound after a warning. The full matched set was also attempted on that variant, with only the same audit prerequisite and archived-path failures. This is a test coverage note. [Scratch changes](mutation.py) · [F1 false green](mutation-any-leg.log) · [Independent failure](mutation-any-leg-independent.log)

## 9. Other diff review

The receipt note in `docs/ops/pick-entry-receipt-v1.md` accurately describes the two reads and batter-only verifier. The matching source and test trees between `1e69d0f0` and the author's `006fafef` base explain why the authored comparison hashes also pass here. No further code finding.

ran: both revision archives, 5,510 comparisons, file-backed row/sequence/text/standing/receipt/morning/rollback checks, all 62 matched test files, six scratch changes and a scratch fix; did not run: whole fast suite, network or live transports, audit assertions blocked by absent GNU rsync 3.x.
