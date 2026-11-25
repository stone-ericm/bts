# Review of change set H, round 1

## Verdict

**BLOCK** on `813935bc`, reviewed against `0a2bec10` from checkout `91d96148`. Amendment 1 governs.

## Blocking findings

**B1. A corrupt base table with an exactly matched tail sends a double.** On 2027-09-26 at noon ET, set a dated 57/57/used reading and leave two scheduled games eligible (Batter 0 at 0.84 and Batter 1 at 0.83). Keep the base NPZ keys, shipped boundaries and `season_length=180`, but replace `policy_table` with the scalar `np.array(0, dtype=np.int8)`. Rebind the shipped one-bin tail to the SHA-256 of that file and save it through the validating writer. The tail validates and the artifact health check returns zero alerts. The production rebuild script can bind this pair because it hashes the base bytes directly. [rebuild_tail_policy.py](/Users/eric/projects/bts-work/wr-rev-h-cx/scripts/rebuild_tail_policy.py:131)

The base array cannot supply a table lookup. The longest-streak prerequisite only checks that the array is present and that the horizon is positive. The real scheduler saved and sent **Batter 0 + Batter 1** and wrote a scoreable `double` record with `tail_policy_sha256=3475402f39b3178870a72ae08fffffcf83943d3c7a871dc90a9f6831163714cc` and no degradation. H4/A3 require **Batter 0 only**, a degradation reason and a null decision tail SHA for a corrupt base. Reject the malformed table shape inside `_longest_action` before using the tail rates. A four-dimensional shape check in scratch turned this delivery into the required forced single and preserved the healthy short-table case. [strategy.py](/Users/eric/projects/bts-work/wr-rev-h-cx/src/bts/strategy.py:365)

Reproduction and scratch rerun are in `probe_paired_base.py`, `paired-base.log` and `scratch-followup.log`. The probe's `--fixed` mode checks the required delivery and record.

## Edits

- Add the SHA-matched malformed base case to H4's artifact checks as part of B1. Require the real scheduler to send one slot and commit the degradation and null tail SHA.

## Notes

- A8's cap and retained automatic reading reproduced. R33 reused saved picks in both directions. A4's fresh-single runner-up in the record reproduced. These are accepted limits.
- With a shipped table and long remaining horizons, boolean streaks can reach the base NumPy lookup and cause an `IndexError`. The archived and reviewed sources gave the same exceptions and returned the same display bar, `None`.
- The new walk checker catches both required wrong actions. It also accepts a real finale row with `days_effective`, `source` and `artifacts` cleared. The complete run independently checked the resulting records.
- The requested niceness setting reported a permission warning. Runs continued sequentially with a timeout. The first two-bin fixture failed validation, the first 240-day comparison supplied too few table rows, and the first scoring check used the record's slot name. The follow-up runs used a validated fixture, an extended table for the alternate horizon and the saved pick's `pick` key. Original logs are retained.

## 1. Rule and regime

- `probe_rule.py` checked 36,400 combinations of days 1–400 and 91 valid rate pairs against a separate recurrence. It covered exact ties, adjacent floats, subnormals, zero rates, perfect rates and equal rates. It also checked every decision field on 23,040 longest-streak resolutions.
- Dispatch covered 55–60, 100, a 101-digit streak, −1 and both booleans around raw days −3, 0, 1–4, 28, 180, 181 and 271. The rule probe reached the inherited boolean exception in its final display of sample cases. The archived comparison completed those cases.
- In independent scratch copies at boundaries 56 and 58, the objective, resolver, fresh selection, direct bar and new walk row followed the constant. At 56 the walk delivered the finale double. At 58 the stock dated CLI still refused 58/58 under the documented H2 cap. The pure and direct selection paths doubled at 58.

## 2. Rows H1–H9 and artifact failures

- H1–H3. The paired-base probe checked 19,512 resolutions over every raw day 1–271. It exercised both savers, low and high candidate probabilities, streaks 57/58/60/100, disabled doubles and an unavailable different-game partner. Rule and executed actions matched.
- H4 and H6. The artifact probes used real temporary NPZs for A3's missing, corrupt, truncated, directory, mismatch, two-bin, zero and negative horizon, neither-file and malformed-manifest cases. They added wrong tail schema/rates, oversize files, missing base table fields and injected loader/rule errors. These forced single with a reason and null decision tail SHA. The paired scalar case is B1. The positive-day bar returned 0.0 without calling the loader. Rule and loader interrupts propagated for both `KeyboardInterrupt` and `SystemExit`.
- H5. Called `orchestrator.run_and_pick` and independently ran the scheduler lineup check, disk reload, captured DM and commit on July 15 and September 24–26 for both required state types. All eight choices, selected slots, exact `chosen_on`, best/trust fields and record SHAs agreed.
- H7–H9. The v3 writer and reader accepted `longest_streak`. An empty-refresh cached commit and a real scheduler classification restart recovered the saved objective and SHA. The H9 test kept both legacy zero-day cases and passed for the new positive-day objective.

## 3. A6 parity with the base

- `probe_parity_extended.py` printed imports from an actual `git archive 0a2bec10 src` and from the reviewed source. It matched 683,626 cases including all 1,511 G guard cells. It compared every decision field and the direct bar over raw days 0–271, with effective days 0–180 for shipped artifacts and up to 240 with an alternate positive horizon.
- The grids included every shipped bin edge and adjacent floats, heuristic thresholds and executable partner products, best/trust inputs, the tail stop, missing/invalid/mismatched pairs, two bins, shorter and longer horizons, several years and zero days at and above the boundary. No returned bar differed. An ordinary legacy loader failure raised on the base and produced `None` under A2. Both commits have `mdp.py` blob `f2a6d1331cdf9e46c83beffff2451815a9d21389`.

## 4. Objective readers

- The fresh scheduler and both recovery writers retained `longest_streak`. The objective reader returned it and the reach-57 predicate rejected it. The skip shadow wrote nothing even when given a synthetic MDP skip with that objective. The alignment sample stayed at zero and the census reported `excluded_tail_objective`, no row diffs and an empty primary sample.
- The actual entry CLI matched the batter and game, sealing the decision bytes in its receipt. The day-outcome CLI returned `entered` with Batter 0. The dashboard rendered the single and the post-failure check cleared the delivered pick. The results CLI graded one slot and advanced the model streak from 57 to 58 exactly once, preserving `chosen_on` and the saved policy decision.

## 5. Standing rows

- `probe_delivery.py` reloaded a saved selection after an empty refresh, then delivered it. A known send failure cleared its marker and wrote no record, and a later retry succeeded. An interruption during transport left the durable marker. Both a re-entry into delivery and a real `run_day` classification restart recorded `locked_unconfirmed` without a second send.
- September 26 doubled and September 27 took the zero-day legacy skip. The old automatic switch was accepted at 23:59 ET on December 31 and refused at 00:01 on January 1. Installing accepted 2028 state yielded a single at 271 raw and 180 effective days. The walk also checked the fallback-year warning.
- A fresh 58/58 fetch failed visibly and left 57/57 accepted. Both R33 reuse directions delivered the original pick and objective with the later displayed streak. A real refresh after falling to 0 produced a genuine MDP skip.
- The archived base parsed an H record, normalized its objective to `unknown` and treated it as scoreable. Its writer refused a new record with the H objective. Interrupt and malformed-file standing checks are in item 2.

## 6. Season walk

- Ran the full walk. Exit 1, 4,818 records and 612 warnings. All 4,599 original records, all warnings and the year-boundary checks exactly matched G. Warning counts at 50/55/56 were 141/157/314.
- The 219 new days gave 207 singles including 23 preseason, doubles on September 24–26 and nine legacy post-finale skips. All 210 plays had valid choice records and the expected slots. There were zero added warnings. The complete JSON also matched the author's saved H result.
- A named July 15 skip and a single on September 26 both raised explicit action findings when their recorded objective was set to `reach57`.

## 7. Tests and deliberately wrong scratch changes

- All ten requested suites passed in three bounded runs, 92 + 114 + 55 = 261 tests. The existing objective test in `test_tail_policy.py` and the resolver and bar grid in `test_wr_g_guard.py` had precisely the A10 behavioral edits. The season-walk test file only added tests.
- Tests caught double-first ties, swapped operands, accepting two bins, accepting a non-positive horizon, dispatch at zero days, skipping on degradation, dropping the objective and returning a null longest-streak bar.
- 260 tests passed with a scratch change that used a positive base horizon, and the same 260 passed with artifact loading moved before the fresh dispatch. The parity test was deselected in scratch because its child imports the real checkout. The independent rerun changed a raw-four selection with a valid one-day base from single to double under the first change. It observed `load, dispatch` under the other change. The reviewed source called `dispatch, load, dispatch`.

## 8. Other diff checks

- The fresh source changes are confined to strategy, the tail-policy module and the decision enum. The other code additions are in the walk and tests. The reviewed tree has no tracked edits from this review.

ran: all requested suites, complete walk, archived comparison, independent rule, artifact, delivery, restart, reader and scratch probes; did not run: the whole fast suite, known FIFO hangs or live services
