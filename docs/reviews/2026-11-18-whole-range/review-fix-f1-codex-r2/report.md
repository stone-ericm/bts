# Verdict

**BLOCK** for `1d30de86` and `15b27474` on `1e69d0f0`. Round two fixes the reported serial sequence. An older overlapping invocation can still erase the observation and cause a terminal all-clear without game proof.

## Blocking findings

**B1. A slow invocation overwrites a newer wrong-game observation with its older view of the marker.** On March 24, 2027 ET, the saved pick is batter 1010 in integer game 101, first pitch 19:40. At 18:25 a missing entry sends `initial`. Invocation A starts at 18:30, reads that marker and waits on the empty pending response. At 18:45 invocation B sees the batter in target-round unit 77, which the local capture binds to game 102. B's warning DM fails and B writes `alerted/wrong_game` with `game_unresolved: true` and `escalations: ["initial"]`. A resumes with the empty response and writes `alerted/no_pick` without `game_unresolved`.

Add a second gzip capture binding the same unit to game 101. At 19:00 the real check sees `unit_unverified` because the captures conflict. It writes terminal `confirmed/match`, exits 0 and sends `✅ BTS pick entry confirmed for 2027-03-24 (match).` A1(b) and R1 require the unresolved marker and the still-due first warning. A5.4 states that overlapping runs can both send. This overlap loses the observation and makes the later all-clear false.

The prior marker is read before the fetch at [cli.py:2004](../../src/bts/cli.py#L2004). `mark` uses that snapshot at [cli.py:2024](../../src/bts/cli.py#L2024), including the tier-exhausted write at [cli.py:2170](../../src/bts/cli.py#L2170). The smallest fix is to serialize the marker decision for a picks directory. After fetching, load the latest marker under the lock, use it for A1(b) and the tier decision, and hold the lock through the marker update and all-clear decision. Atomic replacement of the JSON alone allows a stale run to replace a complete newer marker. The probe nests B in A's captured pending fetch to control the order. Both invocations use the real command, capture reader and marker writer. [Probe](interleaving.py) · [Failing output](interleaving.log)

## Edits

None. The three R2 text changes render as specified and the three corrected limits held. B1 requires a code change.

## Notes

- R1 explicitly makes an unreadable marker follow the base. Corrupting the marker after a warning and then producing conflicting captures yielded a batter-only match with no DM. Separate capture reads, shared cutoff tiers, no-recipient tier consumption and the terminal confirmation limit also behaved as stated. [Sequences](sequences.log) · [Separate reads](morning-capture-change.log)
- An unreadable saved second-leg pitch raised before any fetch on both revisions, as stated in R2. The text log line labeled `only affected leg first warning` shows the 19:05 reminder. The script checks the actual 18:30 first-warning string separately before printing that line. [Candidate](texts-candidate.log) · [Base](texts-base.log) · [Assertions](texts.py)
- The machine rejected the priority adjustment requested by `nice`. Runs stayed serial and bounded. Victoria 3 was absent at each process check. No tracked file changed.

## 1. Scope and base comparison

Reviewed both `git show` outputs and both base-to-commit diffs under `src` and `docs/ops`. The combined diff changes the watchdog, the receipt's shared slot extraction and one receipt documentation sentence. `bts.__file__` in the logs identifies the base archive under this directory and the checkout's `src` for the candidate.

The stratified grid covered every verifier result and slot qualification, singles and doubles, each prior marker status and all tier subsets, another date, unreadable markers, each delivery state, each transport failure and identity mismatch, three DM outcomes, and ET window, tier and cutoff boundaries. Compared stdout, stderr, DM attempts and deliveries, raw marker bytes with receipt ID and clock normalized, exit status and receipt fields with IDs and clocks normalized. Of 10,368 pairs, 7,449 were identical. The remainder took the expected A1(a) path (816), A1(b) path (480), full-proof resolution (227), or R1 nonterminal update (1,396). B1 needs successive invocations with interleaved completion. [Grid](grid.py) · [Qualification counts](grid-candidate-qualification.log) · [Ladder counts](grid-candidate-tiers.log) · [Fetch counts](grid-candidate-fetch.log)

## 2. Contract rows

N1–N13 and M1–M13 passed in the F1 tests through the real command with synthetic pick files and real gzip captures. This includes the added raising-read row M10, the two failed-observation rows and M13's double requiring proof of both legs. [F1 run](tests-f1.log)

## 3. False alarms and missed matches

Checked 36 doubles on each revision with pending-only, settled-profile-only and duplicate-across-source rows per leg, covering both legs right, each leg wrong alone and both wrong. F1 named only the proved wrong leg or legs. With game IDs `null`, `"101"`, `101.0`, `true` and `101`, only the exact integer triggered A1(a). Same-batter doubles with identical or different games confirmed with no prior observation. No extra serial A1 alarm or miss appeared. B1 reproduces a missed same-date follow-up. [Candidate](rows-candidate.log) · [Base](rows-base.log)

## 4. Tiers and event order

Ran 54 orders crossing earlier absence, wrong-game send success/failure/no recipient, intervening removal/mismatch/crosswalk gap and their DM outcomes. The failed first warning retried, a successful one left `initial` due on removal, and `t30` and `t15` each retried on failure and fired once on success. Full game proof cleared the marker and attempted the all-clear as due. A failed all-clear stayed terminal, and a later capture change fetched nothing after confirmation. A cutoff reached during the check preserved the unresolved observation and skipped the DM. [Sequence output](sequences.log) · [F1 run](tests-f1.log)

## 5. Texts under R64

Compared whole first-warning, reminder and `locked_unconfirmed` strings, unresolved stdout and Amendment 3's batter-only wording. Used a double with one affected leg and two different pitch times, missing names, a quoted picks directory and case (b) with no leg currently classified `wrong_game`. Cutoff crossed during the capture read with a timely contest response. The rendered verification messages and cutoff line matched the amended contract. [Rendered text](texts-candidate.log) · [Batter-only cases](tests-f1.log)

## 6. Standing rows

Reran corrupt gzip, truncated gzip, invalid JSON and UTF-8, nonobject top level, noniterable units, directory, denied read and silently accepted malformed shapes. Raising reads and silent shapes produced their respective Amendment 2 and 3 outcomes. Checked KeyboardInterrupt and SystemExit during the new read, a second-read failure during publication, midnight and the UTC/ET year boundary, invalid and unreadable markers, interruption after send and FIFO behavior. Base checks reproduced the inherited marker exceptions. [Candidate standing rows](standing-candidate.log) · [Base standing rows](standing-base.log) · [FIFO and interrupted write](tests-f1.log)

## 7. Receipt and morning outcome

Across the 1,296 A1 warning pairs, the normalized receipts differed only in `marker_status` and `exit_code`. The receipt kept the batter verifier and its own game qualification. Real `day-outcome` reads checked timely proof, timely batter-only evidence, late proof, absence, wrong game alone, failed warning, a crosswalk gap and missing evidence. The morning outcome could be clean (`entered` or `entered_game_unverified`) alongside an unresolved watchdog marker, and the morning text never named the wrong game. Actual rollback to the archive confirmed and attempted the base all-clear when configured, with the failed send and no-recipient cases also checked. [Morning](morning-candidate.log) · [Independent capture reads](morning-capture-change.log) · [Rollback](rollback-base.log)

## 8. Matched tests and scratch changes

The specified search identified 63 test files. Ran all in bounded groups. The two F1 files passed all 83 tests. The other runnable files produced 3,456 passes, four skips and one expected failure. The audit driver had the same 33 failures and 16 setup errors on both revisions because this machine lacks GNU rsync 3.x. Both F1 test files are additions against the base. [Test list](tests-list.log) · [Candidate audit](tests-audit-candidate.log) · [Base audit](tests-audit-base.log)

All six isolated scratch changes were caught by tests. Removing the integer check failed 1 test, removing the shared-batter check failed 2, dropping case (b) failed 10, consuming `initial` failed 13, confirming with any proved leg failed 2, and swapping the texts failed 18. B1 passed through the existing green suite and failed the new interleaving check. [Scratch runner](mutations.py) · [Integer](mutation-integer.log) · [Shared batter](mutation-shared.log) · [Case (b)](mutation-prior.log) · [Initial](mutation-initial.log) · [Any leg](mutation-any-leg.log) · [Texts](mutation-texts.log)

## 9. Other diff review

The receipt operations sentence correctly describes the independent capture reads and batter-only verifier. The two read timings reproduced both directions of disagreement the contract permits. [Capture change output](morning-capture-change.log)

## 10. Round-one finding and its class

Ran the archived `failed_observation.py` on both code revisions. The crosswalk, removal and mismatch branches all failed on round one and passed on round two. The new F1 suite also covers the no-prior-DM and partial-double cases. Ran independent sequences across sends, cutoff and re-entry, including real round-one markers with only a reason or only a consumed tier and a real ordinary base marker. Those serial sequences agreed with R1. The interleaving in B1 reaches a terminal confirmation before any full-proof check, with neither a raising capture read nor a damaged marker. [Round one](replay-round1.log) · [Round two](replay-candidate.log) · [Legacy handoff](legacy-candidate.log) · [Interleaving](interleaving.log)

## 11. Unaffected runs and marker readers

The grid found no extra difference on a clean-history day. After the cutoff, health produced WARN for `alerted`, CRITICAL for `dm_failed`, and no pick-entry alert for `present_unverified` or `confirmed`. Rollback read actual F1 markers containing `game_unresolved`, followed the base batter check and wrote a terminal marker. The midnight/year check kept previous-date state out of the new day's decision. [Grid](grid-candidate-tiers.log) · [Health](morning-candidate.log) · [Rollback](rollback-base.log) · [Midnight](standing-candidate.log)

## 12. R2 edits

Verified the fallback locked-delivery names with ` + DD `, `recommended game ID unavailable`, and `cutoff passed during this check` character for character. The corrected A6.2 health status rule, A6.4 rollback all-clear attempt and A6.5 clean-morning coexistence were reproduced in sections 7 and 11. [Text run](texts-candidate.log) · [Morning and health](morning-candidate.log) · [Rollback](rollback-base.log)

ran: both source archives, 10,368 paired grid cases, N1–N13 and M1–M13, sequence and interleaving probes, text and standing rows, receipt and morning checks, rollback, all 63 matched test files, six scratch changes; did not run: the whole fast suite, live or network transports, audit assertions that require the absent GNU rsync 3.x prerequisite.
