# Verdict

**SIGN WITH EDITS** for `b3d5170b` on post-F3 base `ddbc531a`.

## Blocking findings

None. The checks preserved existing day states and delivered alerts, and found no F4 write to pick, decision, streak, or health state.

## Edits

1. **Distinguish an absent target from a failed read of an open descriptor.** `src/bts/day_outcome.py:89–99` treats every `FileNotFoundError` from `src/bts/util.py:21–44` as an absent target. On a clean `no_games` day with a regular file containing `{}`, `probe_file_matrix.py` opened the real file and injected `ENOENT` from `os.read`. The CLI returned exit 0, a null reason and zero DMs. The contract calls for exit 1 and `cannot be read (FileNotFoundError)`. Mark `FileNotFoundError` as an absent target only during the bounded helper's `stat` or `open`, and let later failures reach the generic read-error handler. The failure was injected, and the base sends no alert in this case.
2. **Make the morning advisory itself a B8b(2) trigger.** The first morning read can flag a file after a successful reconcile. B8b(2) at `docs/ops/2027-season-start.md:32` gives the specific lead handoff for a replay warning or failed reconcile. Add after its first sentence: “For a morning report with a model streak file advisory, give `bts-lead` the report, its host and ET run time, and any grading or reconcile errors, receipt paths, and health alerts. If the lead session is unavailable, give the same information to Eric. Verify which jobs completed, including when reconcile reported success.” The normal DM already gives the handoff. The compact DM relies on B8b(2).
3. **Name the source of the compact report on manual runs.** At `src/bts/day_outcome.py:79–80` replace “See day-outcome JSON in cron.log and B8b item 2 in docs/ops/2027-season-start.md before the next affected job.” with “Find the day-outcome JSON in cron.log for a scheduled run or on stdout for a manual run. Follow B8b item 2 in docs/ops/2027-season-start.md before the next affected job.” The installed cron redirects there. The direct command emits the JSON on stdout and sends the same compact DM.
4. **Qualify the recovery procedure for a non-regular target.** The amended known limit and B8b(2) should say: “Complete reconcile replay can rebuild a replaceable regular file from verified final history after the blockers are resolved. Preserve a non-regular target during triage and have the BTS lead or Eric prepare a safe target before running reconcile. The grader and reconcile can block on a FIFO. The 08:00 ET correction cutoff still applies.” `probe_operator.py` bounded a real reconcile read of a FIFO at 120 ms on both revisions, then verified that an atomic rename could replace that FIFO. This is an existing scorer behavior. `day-outcome` diagnosed the FIFO without opening it.
5. **Keep the code commit within the named scope.** Revert the additional `CLAUDE.md:66` sentence in `b3d5170b`. It describes F4 and has no runtime effect. The amended author-work list allows the CLI help, two implementation files, the new test file and two existing assertion changes.

## Notes

- The checked-in tests stay green when the streak read moves before classification. `probe_order.py` passes on F4 and fails on that scratch change. It writes `{` immediately after the real F3 skip review and checks that the command reads it exactly once, after that review, and reports it.
- The standalone scorer and reconcile still have different acceptance rules. A readable prior-season object with a string count can trigger this advisory while the new-season scorer starts at zero. An interrupted scorer can also leave a valid streak snapshot beside an unresolved pick. These are stated limits of the file check.
- The sandbox printed the expected `nice` warning. The first matrix attempt stopped at a sandbox-denied local Unix socket bind. The completed rerun used a temporary local socket with approval. The parity comparison used the repository on `PYTHONPATH` after its first invocation failed to import the test fixtures.

## 1. Changes and base parity

Ran `git show b3d5170b`, `git diff ddbc531a b3d5170b` and `git diff --check`. The contract-file delta is Amendment 2 in the preceding review-record commit. The code commit adds the three expected source deltas and the new F4 tests, makes the two permitted existing assertion changes, and also touches `CLAUDE.md` (Edit 5). F1 asserts and removes only the null field from each parsed morning result, including the second result in N11. All existing digests stand. F3 adds the one field to the shared expectation for `as_dict()` and CLI stdout.

`probe_parity.py` printed distinct `bts.__file__` paths from `git archive ddbc531a src` and the checkout. All 122 CLI comparisons agreed byte for byte after removing just the appended null field, including all 16 states, the all-void schedule, missing config, and transport timeout and missing-ID controls. `skip_state_review` used both a mismatched observation and a same-day revocation. Eleven default-reader comparisons agreed, including Unicode, newline translation, over 1 MiB, permissions, FIFO, directory and symlinks. The default path keeps the original descriptor check, UTF-8 text read and close. Existing caller tests passed.

## 2. File rule

`probe_file_matrix.py` ran the contract-check input matrix through real `bts day-outcome` on a clean day and a `NO DECISION` day. Syntax, UTF-8, non-objects, invalid counts, mode 000, directory, FIFO, socket, symlink loop, over-limit digits and deep nesting all kept the day state and produced the designated reason and DM. Omitted count, accepted counts, ignored metadata and dangling links remained quiet on clean days. Exactly 1 MiB passed. A file grown after the first read stopped at 1,048,577 total bytes. A FIFO renamed over the file before open was opened with `O_NONBLOCK`, rejected by `fstat` and never read. The injected `EIO` was diagnosed. The injected post-open `ENOENT` exposed Edit 1. No matrix case changed a base outcome into exit 2 or a traceback.

## 3. DM delivery and content

`probe_dm.py` used the real `sendMessage` path and an offline transport counting graphemes with ICU. The 1,000-code-point combined DM sent in full and the 1,001-code-point DM sent compact. The 833-character base alert, the long-detail alert, the accented and combining-character names, and an existing 3,601-code-point alert with exactly 1,000 graphemes all reached the transport. Compact DMs retained the base problem head and the full stdout detail and pick. The clean long-pick case used the review head. A base alert of 1,001 graphemes was already rejected when the file was good. With a broken file the compact alert sent. No accepted base alert was lost to the length limit.

## 4. R64 and B8b(2)

`probe_operator.py` rebuilt the two BL-O3 receipt variants. The scheduler and 01:00 grader failed on `[]`. The 02:00 and 07:40 replays warned and then failed at the display read. The independent CRITICAL still sent. Base 08:30 reports exited 0 without a morning DM. F4 retained `entered_game_unverified` or `entered`, exited 1 and sent the review DM. Its wording leaves both alerts available for the lead, preserves evidence and says grading or replay may be affected.

At 07:40 a correction saved before reconcile's final display failed. After the older blocker was resolved, the 08:20 replay rebuilt the regular file and left the 08:00-frozen correction unchanged. On a separate day, a file with `{"streak":false}` installed after a successful 07:40 reconcile still sent an advisory. These runs support the checklist edits for a stand-alone advisory and a FIFO. The first-day case in item 6 supports treating the report as an advisory before supervised recovery. Following the rendered instructions did not demonstrate an R64 outcome requiring a block.

## 5. Tests and false greens

Ran the exact `grep -rlE` search from the brief. All 14 matched test files ran, including 56 F4 cases. `pytest-f1.log` has 70 passing in its own 43-second run. `pytest-day.log` has 294 passing. `pytest-related.log` has 271 passing and four conditional skips. The three script matches passed shell/Python syntax checks. One is the cron caller and two match unrelated names.

Seven single scratch changes ran in isolation. The F4 tests caught missing-as-problem (`test_k1_k2_every_day_state`), accepting `bool` (`test_m2_bad_contents[true]`), dropping 1 MiB (`test_m2_bad_contents[read-limit]`), dropping compact DM (`test_k3_compact_dm`), dropping `RecursionError` handling (`test_k4_decoder_limits_preserve_no_decision[nesting]`) and restoring the clean state name (`test_k3_m3_bad_file_preserves_day_state`). The early read survived all 635 checked-in cases that ran. The independent ordering check failed on the scratch copy with `['read', 'classified']` and showed a false clean result after the file changed during classification.

## 6. Standing rows

`probe_lifecycle.py` pinned 23:59 and 00:01 ET across New Year and ran the explicit previous-day 08:30 report on New Year's Day and a simulated opening morning. Valid prior-year state remained quiet. A string count in that prior-year file triggered the advisory while the new-season scorer loaded zero. An interrupt before atomic replacement preserved the old bytes. An interrupt after replacement and before the scorer saved the pick left a valid count and applied-date marker. The report made no writes and the scorer restart settled the pick without applying the streak twice.

An interrupt during the real bounded read closed the descriptor. With a broken file, a captured send timeout or missing response ID left exit 1 and one attempt, with the expected stderr. A new invocation sent the same report. With no recipient the JSON and exit were unchanged. Snapshots around the report showed no state writes. The malformed and non-regular cases are in item 2.

## 7. Other diff checks

Reviewed the CLI help, `DayOutcome` defaults, `classify_day` call sites, and default reader callers. The new help names the added exit-1 reason. Independent missing-config behavior and the F3 review path matched the base. The report's byte and DM bounds matched the adopted contract in the completed checks. `git status --porcelain` showed no tracked changes from the review.

ran: commit and scope checks, archived base parity and default reader, file matrix and read faults, ICU grapheme transport, BL-O3 and operator replays, ET boundaries and lifecycle, all 14 matched test files, three script syntax checks, seven scratch changes, and the independent ordering check; did not run: four conditionally skipped cases, the full suite, production cron or transport, or a real filesystem returning ENOENT from an open descriptor.
