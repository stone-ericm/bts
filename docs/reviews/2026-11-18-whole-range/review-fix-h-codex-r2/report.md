# Review of change set H, round 2

## Verdict

**BLOCK** on the whole change `0a2bec10..84805852`, reviewed from `ac2818eb`. The round-1 scalar failure is closed. A different malformed base still sends a double where H4/A3 require a forced single.

## Blocking findings

**B1. A matched base with nonnumeric boundaries sends two picks.** Keep the shipped integer action table `(58, 181, 2, 5)` and horizon `180`, replace just the NPZ's `boundaries` with `np.array(['low', '.70', '.80', '.90'])`, and save a validated one-bin tail bound to the new file's SHA-256. The real loader decodes four strings into a list, so every current Amendment 2 check passes. A real reach-57 lookup with streak 56, one day, saver used and probability 0.84 raises `TypeError` comparing the probability with the first boundary. The amendment's stated coverage of bases that cannot answer the lookup is incomplete. [strategy.py:367](/Users/eric/projects/bts-work/wr-rev-h2-cx/src/bts/strategy.py:367) · [mdp.py:188](/Users/eric/projects/bts-work/wr-rev-h2-cx/src/bts/simulate/mdp.py:188)

At noon ET on 2027-09-26, with dated `57/57/used`, doubles allowed and two eligible games, the orchestrator and scheduler chose Batter 0 (0.84) and Batter 1 (0.83). The scheduler saved both, sent both in the captured DM and committed a scoreable `double` with tail SHA `0f81ee9fa398248bed3e2d2df0efcc773186a83fcdb4c04d182dbfcb0a8918a7` and no degradation. H4 requires Batter 0 only, a degradation reason and a null tail SHA. Nested-list and complex boundaries reproduced the same lookup error and wrong resolver action. [Probe](/Users/eric/projects/bts-work/wr-rev-h2-cx/.codex-review/wr-h2/probe_artifacts.py) · [Reproduction log](/Users/eric/projects/bts-work/wr-rev-h2-cx/.codex-review/wr-h2/paired-counterexample.log)

## Edits

- Require each decoded boundary to be a real number inside `_validate_longest_base` and make that requirement explicit in Amendment 2. A scratch check for decoded `int` or `float` values rejected all three bad lists, sent only Batter 0 and recorded the degradation with a null tail SHA. The scalar and healthy short controls still behaved as required. Keep the paired-string scheduler case as a regression check. [Scratch rerun](/Users/eric/projects/bts-work/wr-rev-h2-cx/.codex-review/wr-h2/scratch-boundary-content-fix.log)

## Notes

- The 58/58 fetch refused the reading with an error and retained the accepted 57/57. R33 reused the saved choice in both directions across the boundary. The fresh single's recorded runner-up matched the accepted A4 limit.
- Boolean streak inputs can raise in the legacy NumPy lookup on both trees. The display results matched. The literal 57 in the new shape check is the number of rows needed by the unchanged reach-57 lookup. It is independent of the longest-streak threshold.
- Review runs were sequential and bounded. `nice` reported a permission warning. Initial probe failures involved two-bin fixtures, scratch path resolution and checks in the grading, rollback and round-1 replay scripts. Follow-up runs supplied the results below. The expected failing paired-artifact check is retained.

## 1. Rule and regime

- `probe_rule.py` checked 36,400 independent day/rate cases over days 1–400, including zeros, equal rates, exact and adjacent-float ties, perfect rates and subnormals. Dispatch used raw days −2, 0, 1–4, 180, 181 and 271 and streaks including 56–58, 100, a huge integer, −1 and booleans. It checked 2,016 resolved contexts and 144 inherited boolean exceptions, with every new-regime decision field, both saver states and both execution clamps checked.
- Scratch constants 56 and 58 moved the objective, resolver, fresh selection, display bar and walk state. The state-reading cap remains the separate H2 limit. See `rule.log` and `boundary-{56,58}.log`.

## 2. H1–H9 and artifact failures

- The original paired-scalar inputs sent a double on archived round 1 and sent a forced single with null tail SHA on round 2. The short-horizon control used four raw/effective days on September 23. Its finale counterpart doubled. The same replay covered 19,512 H1–H3 resolutions across every raw day 1–271.
- Real paired NPZs tested additional ranks, each empty axis, wrong axis sizes, boolean/float/complex/structured/text dtypes, a single negative or huge unsigned action, very large and differently typed horizons, scalar/object boundaries, and B1. Read-only and memory-mapped tables worked in the decoded context. Serialized boolean and numeric text horizons normalized to usable integers. Direct decoded contexts rejected non-`int` horizons and non-list boundaries.
- The real selection call forced single with reason and null tail SHA on the listed missing, corrupt, truncated, directory, mismatched, two-bin, nonpositive-horizon, malformed-manifest and no-artifact cases. Loader, base-check and rule exceptions degraded, and both interrupt types propagated. The positive-day bar was 0.0 without artifact access. B1 is the exception to the corrupt-base requirement. See `paired-counterexample.log`, `artifacts.log` and `artifacts-followup.log`.
- The real orchestrator and scheduler separately selected, saved, reloaded, sent and committed all eight H5 date/input pairs. July 15 was single and September 24–26 were double for dated and switched automatic readings. All had the recorded tail SHA and valid `chosen_on`. H7 wrote the new objective. H8 preserved it in the empty-refresh and classification-restart commits. H9 passed with both zero-day cases intact. See `lifecycle.log` and the test logs.

## 3. A6 parity with the base

- `probe_parity_followup.py` imported the real checkout and `git archive 0a2bec10 src` in separate processes and printed each `bts.__file__`. All 61,435 cases matched on every decision field and the direct display bar. This included all 1,511 G guard cells, raw days 0–271, effective days 0–240, bin edges and adjacent floats, heuristic thresholds, tail stops, both clamps, best status and values, malformed and missing artifacts, alternate horizons, fallback years and zero days at and above the boundary. The author's larger archived comparison also passed in the focused suite.
- The only observed display exception change was the intended A2 conversion of a legacy loader `TypeError` to `None`. The stricter base validator is called only from the longest-streak path. `mdp.py` is the same blob on both commits, `f2a6d1331cdf9e46c83beffff2451815a9d21389`. See `parity-followup.log`.

## 4. Objective readers

- The v3 reader returned `longest_streak`. The reach-57 predicate and shadow excluded even a synthetic MDP skip with that objective. A finalized single yielded zero alignment samples and the census labeled it `excluded_tail_objective`, with empty row diffs and zero primary samples. Fresh, cached and restart writers retained the required metadata.
- The entry CLI matched the game and batter and sealed the exact decision bytes in the receipt. Day-outcome reported `entered`, the dashboard rendered the single and the post-failure check cleared it. The results CLI graded one slot, moved the model streak 57 to 58 once and preserved pick metadata on a second call. See `readers.log`, `results.log` and `lifecycle.log`.

## 5. Standing rows

- Malformed and non-regular files and rule interrupts are in item 2. In a scheduled sequence, a rejected send cleared the attempt and wrote no commit, and a retry delivered. A transport interrupt left the attempt marker. The real `run_day` classification restart wrote `locked_unconfirmed` without calling the transport again. An empty refresh delivered a disk-reloaded pick with its original objective and SHA.
- The finale doubled at one day and the following day skipped at zero. On December 31 at 23:59 ET the old automatic reading was accepted and the slate skipped. On January 1 at 00:01 ET the switch was refused. Installing a current-season automatic reading produced a single at 271 raw/180 effective days and the usual unconfigured-year warning.
- Archived base code read the actual delivered pick and record, normalized the objective to `unknown`, left the single scoreable and left the bytes intact. The base writer refused a new `longest_streak` record. See `lifecycle.log`, `rollback-followup.log` and `parity-followup.log`.

## 6. Season walk

- The full walk exited 1 with the same 4,599 original records, all 612 warnings and the same boundary checks as G. Warning counts at streaks 50, 55 and 56 were 141, 157 and 314. The 219 new rows gave 207 singles (23 preseason), three doubles on September 24–26, and nine post-finale skips. All 210 plays had valid `chosen_on` and the required slots. No new warning. The decompressed H JSON and saved command output matched the fresh run.
- The checker reported a planted July 15 named skip and September 26 single even with a `reach57` objective. It also reported missing effective days, source, artifacts, choice validation and the second slot. **Note:** a planted finale single with all three stored day fields set to four passed because the checker trusts the stored reference day count. The independent date calculation verified the actual run. See `walk-check.log`.

## 7. Tests and scratch changes

- All ten requested suites passed, 109 + 105 + 68 = 282 tests. The two existing test edits match A10, and the walk test file only adds tests.
- The tests caught changes to the tie order and operands, acceptance of two bins, skip on degradation, a missing enum, a null bar, use of the base horizon and loading before dispatch. They also caught dropped checks for a boolean horizon, integer dtype, streak minimum, day/saver/bin axes and both action extrema. Dispatch at zero survived the new H file alone and failed the existing objective test.
- **Note:** Removing each of the integer horizon type, positive base horizon, boundary list type, ndarray type and rank checks still passed 281 tests in each scratch run. The archived parity test was excluded in scratch because its child pins the checkout. Independently, a float horizon, consistent zero-day table, tuple of numeric edges and table-like object each changed a forced single to a double when its corresponding check was removed. Dropping the rank check leaves the exact shape check, which still rejects wrong ranks. See `scratch-group*.log`, `scratch-wider*.log` and `mutation-examples.log`.

## 8. Other diff checks

- Reviewed both code commits together and confirmed the checkout's source, scripts and tests match `84805852`. No tracked file was changed in the review.

ran: all requested focused suites, full season walk, archived comparison and rollback, independent rule, paired artifact, scheduler, reader, calendar and scratch checks; did not run: whole fast suite, known FIFO hangs, live services or the shared wrapper
