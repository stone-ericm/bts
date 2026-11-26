# F2 round 1 review

## Verdict

**BLOCK** on code commit `2e20fdfc` against `a4e8278d`, reviewed at `f9d6cf9d`.

## Blocking findings

1. **An unreadable saver flag can discard earlier contest alerts.** At July 15, 2027 22:00 ET, use an in-effect `used` observation at streak 12 and make `account_state/saver_state.json` a symlink into a directory the scheduler cannot traverse. `Path.exists()` raises `PermissionError` before the saver reader's exception handler. A symlink pointing to an overlong name similarly raises `OSError`. The base preserves the coverage INFO. F2 raises in both modes, and the runner substitutes a `health_runner:contest_state` CRITICAL. In `src/bts/health/contest_state.py:215`, catch `OSError` for this comparison and return the collected alerts. [Base](comparison-base.log) · [F2](comparison.log)

2. **The new receipt can overwrite the live saver flag.** For a dated `used` observation and an `active` flag, make `health_dm_delivery_status.json.dated_used_mismatch.json` a symlink or hard link to `account_state/saver_state.json`. The base leaves the flag byte-for-byte intact. F2 sends the new DM and overwrites the flag with receipt JSON. The real accepted automatic read at streak 9 the next day changes from `saver_available=True` on base to `False` on F2. At `src/bts/health/runner.py:347`, use a receipt path that refuses aliases or writes by atomic replacement, with an alias-safe read for dedup. Preserve the existing digest's receipt handling. [Base](runner-edges-base.log) · [F2](runner-edges.log)

3. **A bad receipt-path override can stop the mismatch send.** Set `health_checks.thresholds.health_dm_delivery_status` to `/`, keep an existing CRITICAL, and present the dated `used`/`active` mismatch. The base sends the digest and returns even though the receipt path is non-regular. F2 sends that same digest, then `Path.with_name` raises `ValueError` before the mismatch transport is called. At `src/bts/health/runner.py:351`, handle a path with no basename and attempt the WARN with the default separate receipt or with receipt persistence disabled. [Base](runner-edges-base.log) · [F2](runner-edges.log)

4. **B8a's saver procedure can spend a saver the operator just restored.** Operator A follows the live DM to B8a, verifies MLB shows used, and reads an active flag. Operator B marks it used, then sees a corrected MLB result, presses Undo, and corrects the dated observation to `active`. A continues the documented `--use` step. With synthetic MLB readings and the real dashboard/CLI, the flag goes from the restored `active` back to the wrong `used`. B8a (`docs/ops/2027-season-start.md:31`), reached from `src/bts/health/contest_state.py:229`, checks MLB before reading the flag and gives no recheck between that read and the write. Add this instruction to B8a. “Coordinate saver corrections with anyone using the dashboard. Immediately before writing, refresh MLB and the scheduler's flag. If either changed since your verification, stop and reconcile the current state before retrying.” [Operator run](operator-text.log)

## Edits

- The displayed command alone reads the default picks directory. In an isolated shell where that directory and the scheduler's directory both existed, it reported `used` from the former and `active` from the latter. Replace the final live sentence with “Use the `bts season-contract status` command with the scheduler's picks directory as shown in checklist B8a. Follow B8a's saver reconciliation steps for any correction.” Label the B8a subsection for this warning and define D as the ET date printed in the warning. [Operator run](operator-text.log)

## Notes

- The current F2 tests give false greens on this Python build. The installed JSON decoder accepts 3,000 levels, and the test's top-level list is then rejected by the shape check. At 20,000 levels the decoder actually raises. F2 preserves the collected alerts and the mutant raises in both modes. Use a fixture that proves decoding reached its limit. [Mutation](mutation-no-recursion-catch.log) · [Decoder check](decoder-false-green.log)
- The old saver reader swallows the FIFO test's `TimeoutError` as an `OSError`. The first read can return normally after its alarm. The second then hangs because the test sets only one alarm for both modes. A bounded run of that mutant hit the outer timeout. A separate alarm per mode confirmed both signals were swallowed, and the 28 authorised CS4a cases independently rejected the mutant. Give each read its own alarm and use an exception the reader cannot swallow. [Outer timeout](fifo-false-green.log) · [Alarm and CS4a check](live-reader-timeout.log)
- In the unchanged attention code (`src/bts/health/attention.py:118`), a persisted mismatch row with `streak: "bad"` raises before either dispatch. With that file the base sends the existing CRITICAL. F2 exercises this inherited parser issue when the mismatch is present. [Base](runner-edges-base.log) · [F2](runner-edges.log)

## 1. Scope and paired parity

The code commit changes two production files and the two authorised test files. The other changes between the specified base and code commit are review records. Archived `a4e8278d` with `git archive a4e8278d src` and printed `bts.__file__` for each side. All 288 pairs matched in raw alert order and content, captured transport calls, and status file bytes. The pairs covered 14 contest scenarios, five flag inputs, live and historical clocks, recipient present or absent, and eight real scheduler runs with health enabled or disabled. [Probe](probe_parity.py) · [Output](parity.log)

## 2. Comparison

Ran 714 checks per source across 17 flag inputs, seven streaks, all three dated states, and both modes. The ordinary current-season `active` and `not_earned` flags each added one WARN for dated `used`, after coverage, with the specified source and key. `used`, uninitialized, absent, stale/boolean season, unreadable regular file, invalid UTF-8/JSON, list, directory, FIFO, and a 5,100-digit integer preserved the prior alerts. Linked-file read failures are finding 1. The parity grid covered expired, future, legacy and invalid-v3 observations. A second paired pass checked dated `not_earned` at every streak from 0 to 400 with a FIFO flag. Both modes retained the base warning exactly at 10–15. The real decoder-limit follow-up exercised the `RecursionError` catch. December 31 in January UTC, January 1 ET, and a December 31 rerun after midnight used the correct ET seasons. Replacing either file before the historical reread removed the mismatch. [Base](comparison-base.log) · [F2](comparison.log) · [Streak sweep](dated-streaks.log) · [Decoder](decoder-false-green.log)

## 3. Delivery

Ran the real EOD runner and dispatcher with captured transport. An existing 1,000-grapheme digest and its receipt were byte-identical to the control. The 446-grapheme mismatch was a separate WARN-only DM, sent second. Both same-date incident orders sent and deduplicated independently in live and historical mode. The label incremented across consecutive processed days and preceded the single historical pointer. Ordinary thrown failures in either transport allowed the other DM to proceed and retried on rerun. An empty ID counted as sent. A pre-receipt interrupt in the first dispatch stopped the second. An interrupt in the second preserved the first receipt. Findings 2–3 cover receipt failures outside that normal sequence. [Probe](probe_delivery.py) · [Output](delivery.log)

## 4. Operator text and B8a

Rendered the July live DM, December live DM and after-midnight historical DM. With the dated observation on the mistaken side, real status showed the two values and correcting the same-day observation left the active flag intact. The dashboard changes were visible on a fresh status check. The gap after that check is finding 4. A verified `--use` the next morning changed the next decision read and left the real, already saved automatic pick and expired observation byte-for-byte intact. Late December displayed the correct ET year. In the new ET year, B8a's D-year guard and the historical pointer led to read-only inspection. [Probe](probe_operator_text.py) · [Rendered messages and CLI results](operator-text.log)

## 5. Tests and scratch changes

Ran every file from the specified search plus the amended contract index, 79 distinct files. Final group totals were 5,519 passed, 12 skipped, one expected failure, and one sandbox failure at the unrelated `test_web_saver` loopback bind. An initial group timed out in OPE. The rerun of the other files passed, and the isolated OPE run passed all 16 cases. The new F2 file also passed all 50 cases by itself. [File list](test-files.txt) · [OPE rerun](pytest-ope.log) · [Loopback result](pytest-05.log)

The existing-test diff adds a bounded-reader spy and its call assertion inside `saver == "used"` only. The two flag inputs, seven streaks and two expected settings make 28 cases. The other assertions and the live loader trap stay intact.

One scratch change at a time, the focused suite detected dropping the key, using the live reader, choosing the wall-clock year, warning on a `used` flag, folding into the digest, dispatching first, and sharing the receipt. Removing the recursion catch survived the focused suite but failed the deeper check. Moving the new block above the old dated saver warning also survived. Their `used` and `not_earned` predicates are mutually exclusive, so that move has no visible effect for a valid observation. [Scratch driver](probe_mutations.py) · [False greens](probe_test_false_greens.py)

## 6. Standing rows

Direct runs covered the interrupt before each receipt write, retry after restart, then dedup, malformed and non-regular flag and receipt files, December 31/January 1 both live and historical, and a failed or unconfirmed send. FIFO and directory receipts did not stall or displace the existing digest. Receipt aliases and unreadable target paths failed as above. No real transport was used. The sandbox rejected the requested priority change and the commands continued as the brief specifies.

## 7. Other review

The naming of the extra incident, the separate DM and receipt, the historical pointer placement, and both message templates matched Amendment 1 in the ordinary cases. No tracked files or checkout Git state were changed.

ran: archived-base and F2 probes, real EOD and scheduler paths, rendered text and CLI/dashboard checks, all 79 selected pytest files, nine scratch changes; did not run: full fast suite, live MLB or Bluesky, socket assertions beyond the denied loopback bind.
