# H2 round 1 review

## Verdict

**SIGN WITH EDITS** for `b12e144c` against `36fd3e35`. The production change met the amended validation and live-play contract in the offline review. Complete the paired deploy and rollback instructions with the merge.

## Blocking findings

None.

## Edits

- Update `CLAUDE.md` and `docs/ops/2027-season-start.md` A7/B8b with the paired deploy and rollback rule required by Amendment 1. Deploy H with H2. Before reverting past H2, have the lead verify active and best in both observations and pending pick metadata. An expired dated observation can block a valid automatic observation on the older code. Coordinate verified replacement where the rollback can represent the current state and obtain a fresh selection for incompatible pending metadata. The base still recognizes an already committed pick as scoreable. The author handover records this rule and the file scope that left it out of the deploy docs. [`rollback-base.log`](rollback-base.log)
- Clarify K4's “absent active” as absence of both count fields. The automatic reader at `src/bts/contest_state.py:338` also supports the `streak` alias, including for switched v2 files. A covered automatic file using only the alias reads at 57 on the base and 400 on H2. Explicit `active_streak: null` is refused even with the alias present. [`raw-head.log`](raw-head.log) [`raw-base.log`](raw-base.log)

## Notes

- The prior-observation identity read can raise `UnicodeDecodeError` on invalid UTF-8, in addition to the documented `AttributeError` for a JSON array. A directory at the observation path can raise `IsADirectoryError` during replacement. Both revisions reproduced these inherited outcomes in `src/bts/cli.py:2392` and the observation write. [`standing-head-final.log`](standing-head-final.log) [`standing-base.log`](standing-base.log)
- The initial FIFO and typo probes exposed gaps in the review fixtures. The final runs interrupt FIFO reads explicitly, capture both DM call sites, and check the missing-coverage health alert separately from the morning acceptance line. `nice -n 19` reported a priority permission warning. The bounded runs finished at the available priority. [`standing-head-final.log`](standing-head-final.log) [`typo-confirmed.log`](typo-confirmed.log)

## 1. Diff and six bounds

Reviewed `git diff 36fd3e35 b12e144c -- src` and verified that the checkout's production, test and walk sources match `b12e144c`. The diff changes S1–S6, the shared cap and its imports, four diagnostics and the classifier docstring. Both imports printed the expected `bts.__file__`, including the archived base under this review directory.

`probe_bounds.py` compared 4,912 outcomes on each side. The grid included -1, 0, 1, 56, 57, 58, 100, 399, 400, 401, a 101-digit integer, booleans, floats, strings, null, missing fields and compound JSON. After normalizing the four message bounds, differences occurred only when a count entered 58–400. The accepted observations matched the low-count record shape and the dated CLI wrote the same two files. The ordered checks, exception types, non-bound diagnostics and refusal write behavior held. Another 21 direct classifier inputs preserved the same conversions and errors outside the intended trust change. [`source.diff`](source.diff) [`bounds-head.log`](bounds-head.log) [`raw-head.log`](raw-head.log)

## 2. K1–K10

- K1–K4 and K6: live and historical file parsing rejected invalid active and dated best values in order. Automatic non-integer best still reads as missing. Both 400/400 and high best below active had the required outcomes. Dated CLI option, date, order and bound failures returned usage errors before reads or writes. The undated CLI retained its legacy behavior. [`bounds-head.log`](bounds-head.log)
- K5: 216 fetch refusals per revision crossed dry-run, recipient and the six-hour cooldown boundary. All exited 2, logged actionable status, preserved observation/ledger/saver bytes and attempted a DM only with a recipient at or beyond cooldown. Stubbed HTTP responses also drove the real profile parser and confirmed rejection of malformed prediction shapes. [`refusals-head.log`](refusals-head.log) [`profile-head.log`](profile-head.log)
- K7–K9: the regular, byte, shadow and text pick loaders agreed with the required-delivery matcher. Invalid metadata was absent, previews and dated mismatches were refused, and both valid high kinds loaded. Parsed best was trusted at 400 with current provenance and untrusted or missing on the specified controls. The ordered dated checks, switch refusal reasons, authority precedence and saver tests passed. [`refusals-head.log`](refusals-head.log) [`pytest-2.log`](pytest-2.log)
- K10: the switched fetch replaced 57/57 with 58/58, appended one ledger row and the next fresh selection used 58 under H. A 401/401 response preserved the accepted files and used the existing failure and alert path. The separate stale warning stayed clear at age 14 hours and became old after the boundary. The base refused 58 and kept 57. [`refusals-head.log`](refusals-head.log) [`refusals-base.log`](refusals-base.log)

## 3. M1–M5 and live paths

Ran six complete days through dated CLI or switched fetch, morning and historical checks, the scheduler's selection and save, delivery and commit, entry check, day-outcome and grading from stubbed final boxscores. Both modes accepted 58, 100 and 400. At 100 and 400 the restarted scheduler recovered and sent the saved pick after an empty refresh. The scoreable commits retained the supplied best and objective. The cached commits had the documented null live streak field. Entry matched the batters, and day-outcome reported `entered_game_unverified` with the supplied batter-only entry data.

Repeated grading left the observations intact and applied each day's model increment once, reaching 60, 101 and 402. A separate dashboard check displayed each high contest count with a different model count of 6. The M3 tests also ran the dated 60/75/used and switched 12/60 paths. The search for 57, `TARGET_STREAK` and `min(` found no seventh live validation cap. [`day.log`](day.log) [`display.log`](display.log) [`bound-search.log`](bound-search.log) [`pytest-1.log`](pytest-1.log)

## 4. Policy with trusted best above 57

Ran 1,385,100 comparisons per revision for streaks 0–56, raw day counts 0–269, both savers and 15 quality values including each bin edge and its neighbors. There were 8,537 guarded baseline decisions per revision. Actions, objectives, effective best, skip reasons and guard results matched best 57, and the base and H2 signatures matched. At 57, 58, 100, 399 and 400 the zero-day case and H's positive-day choices matched the controls, including double restrictions. [`policy-head.log`](policy-head.log) [`policy-base.log`](policy-base.log)

## 5. Accepted typo costs

On the finale a dated 75/75 in place of 57/57 still delivered a double and recorded 75/75. Best 157 in place of 57 produced the same tail skip at active 12, with supplied best 157 in the record and 57 in the skip text. When the real best was a beatable 12, the same typo changed double to skip. The automatic 5/157 typo changed double to skip and earned the saver, which persisted after correction to 5/5. These accepted typos generated no fetch or morning error alerts.

Two additional cases exercised the limits. A dated 75/75 in place of 0/57 changed a tail skip to a delivered double. An automatic 5/400 with no settled coverage kept its best untrusted, played the double, and left the saver at `not_earned`. The morning journal accepted the count and the independent health check raised its existing missing-coverage alert. [`typo-confirmed.log`](typo-confirmed.log)

## 6. Standing rows

Interrupted the fetch before each of observation replacement, status replacement, ledger append and saver replacement. Writes already completed stayed in place, and the observation was either the old complete record or the new complete record. Compared 40 malformed and non-regular cases across eight data paths with equal outcomes on both versions. Historical reads returned without blocking. Ordinary FIFO reads were bounded by the probe. Both a dated and a switched 58 were accepted at 23:59 ET on December 31 and ineligible at 00:00; new activation and fetch restored automatic eligibility.

A definite send error cleared the attempt and a restarted retry delivered. An interrupted send kept the attempt across restart, locked as `locked_unconfirmed` and sent the incident alert without a second pick send. An automatic correction reused the cached pick even when a fresh check would skip. A best-only dated correction followed by an empty refresh refused the cached pick as expected. On rollback the base rejected 58 in either observation file, refused a best-only dated 58, stripped incompatible `chosen_on`, and still recognized the committed control. [`standing-head-final.log`](standing-head-final.log) [`standing-base.log`](standing-base.log) [`rollback-base.log`](rollback-base.log)

## 7. Full season walk

Regenerated all 5,037 rows with zero findings. The 4,818 old records and year-boundary results equaled `walk-ea-g-h.json.gz`. All 219 `dated/58/60/used/0.96` rows yielded 207 singles, three doubles on September 24–26 and nine skips September 27–October 5. The entire regenerated result equaled the committed H2 result. [`walk.log`](walk.log)

## 8. Tests and scratch changes

3,218 tests passed in four groups. These cover the new H2 file, every changed existing test file, H, the morning check, stale delivery, and the season-walk suite. Reviewed all changed test lines against Amendment 1. Existing edits were the listed refusal fixtures, bound and fixture-dependent message updates, and the added walk test. Independent existing assertions were kept. [`test-changes.diff`](test-changes.diff) [`pytest-1.log`](pytest-1.log) [`pytest-2.log`](pytest-2.log) [`pytest-3.log`](pytest-3.log) [`pytest-4.log`](pytest-4.log)

The focused tests caught restoring 57 at each of S1–S6, a fetch cap of 399 or 401, removal of the automatic active cap or fetch cap, removal of `best >= active` in fetch, and replacing the real tail policy target with 400. Two changes passed all four groups. The dated outer `chosen_on` cap can move to 401 with no observable benefit because the identity still requires `active == streak` and `active <= best <= 400`. Changing the currently unused `contest_state.TARGET_STREAK` to 400 also passed, leaving a gap for an explicit contract assertion. The direct classifier probe checks that value. The stored-walk test reads committed data. The fresh run in item 7 checked its claims against real output. [`mutations.log`](mutations.log) [`mutations-group-1.log`](mutations-group-1.log) [`mutations-group-2.log`](mutations-group-2.log) [`mutations-group-3.log`](mutations-group-3.log) [`mutations-group-4.log`](mutations-group-4.log)

## 9. Other diff findings

No further production defect reproduced. The tracked tree is clean. The scratch scripts, archived base, mutation evidence and regenerated walk are under `.codex-review/wr-h2c/`.

ran: source and test diff checks, independent base and H2 probes, full season walk, 3,218 requested tests, and 14 scratch changes; did not run: full fast suite, deploy gate, live MLB requests, or external DM delivery.
