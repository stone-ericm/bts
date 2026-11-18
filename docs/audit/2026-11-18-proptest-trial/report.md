# Property-testing trial for `compute_all_features`

Base `51adfc7f` on local branch `discovery/proptest-features`. Started 2026-11-18 14:51:49 EST and ended at 15:01:01 EST, for 9 minutes 12 seconds elapsed. Hypothesis 6.151.12 was available in the existing virtual environment. Session token usage was unavailable.

## Harness and contract

`.codex-review/proptest/test_prefix.py` (205 lines, commit `2ace1602`) generates 2–34 game-dates straddling the closing days of 2024 and opening days of 2025. Every frame has a doubleheader, multiple PAs per game, and recurring matchups among one or two batters and two pitchers. Both pitch profiles include pitch measurements and batted-ball observations. Some frames are reverse-ordered. The probable starter lookup and park-drag table getter return synthetic data, so the real bullpen aggregation and park-drag merge run without reading either artifact.

Each example draws one cutoff D. By stable PA identity, it compares all 30 declared model, context, and Statcast columns on D with a prefix through D. It also compares those columns through D after changing hits, pitch sequences and measurements, count, and batted-ball observations on and after D. The function's docstring says *strictly before D*, so the outcome perturbation includes D in both the changed input and the checked output. PA identity and matchup keys (including pitcher handedness), schedule, and as-of game context are held fixed. These checks exercise leakage from observations within a fixed schedule. External artifact provenance and raw or intermediate observational columns in the returned DataFrame are outside the generated comparison.

`days_rest` is checked against the gap between successive distinct game-dates for each batter, including the offseason. These histories feed the rolling and expanding features across both seasons. The existing `scripts/leakage_audit.py` uses season-cut and manual spot checks on real data. This trial instead generates collision-heavy cases and changes outcomes inside the comparison boundary to exercise the shift-by-date guard described in `CLAUDE.md`.

## Three inspected frames

Five frames were printed and inspected using preview seed 5118 before running pytest. These are three complete row projections from that preview. Profile 0 includes a called strike, a foul, and a hard batted ball. Profile 1 includes a ball, a swinging strike, a soft ground ball, and a missing spin measurement. Full inputs can be reconstructed with `make_frame(case)`; the case fields are date count, fall date count, doubleheader index, PAs per game, alternating batter flag, lefty reliever flag, reverse row flag, hit bit mask, pitch-profile bit mask, and cutoff index.

### First preview, cutoff 2024-09-29

`(2, 1, 0, 2, False, False, False, 0, 0, 0)`

```text
PA  Date        Game  Batter  Pitcher  Hand  Venue  Hit  Profile
 0  2024-09-29   100      11       21     R     31    0        0
 1  2024-09-29   100      11       22     R     31    0        0
 2  2024-09-29   101      11       21     R     31    0        0
 3  2024-09-29   101      11       22     R     31    0        0
 4  2025-03-27   102      11       21     R     32    0        0
 5  2025-03-27   102      11       22     R     32    0        0
```

### Third preview, cutoff 2025-03-29

`(4, 1, 0, 2, True, True, True, 184, 66, 2)`

```text
PA  Date        Game  Batter  Pitcher  Hand  Venue  Hit  Profile
 9  2025-03-31   106      12       22     L     31    0        1
 8  2025-03-31   106      11       21     R     31    0        0
 7  2025-03-29   104      11       22     L     31    1        0
 6  2025-03-29   104      11       21     R     31    0        1
 5  2025-03-27   102      11       22     L     32    1        0
 4  2025-03-27   102      12       21     R     32    1        0
 3  2024-09-29   101      12       22     L     31    1        0
 2  2024-09-29   101      11       21     R     31    0        0
 1  2024-09-29   100      12       22     L     31    0        1
 0  2024-09-29   100      11       21     R     31    0        0
```

### Fifth preview, cutoff 2024-09-23

`(5, 4, 4, 2, True, False, False, 198, 152, 0)`

```text
PA  Date        Game  Batter  Pitcher  Hand  Venue  Hit  Profile
 0  2024-09-23   100      11       21     R     31    0        0
 1  2024-09-23   100      12       22     R     31    1        0
 2  2024-09-25   102      12       21     R     32    1        0
 3  2024-09-25   102      11       22     R     32    0        1
 4  2024-09-27   104      11       21     R     31    0        1
 5  2024-09-27   104      11       22     R     31    0        0
 6  2024-09-29   106      11       21     R     31    1        0
 7  2024-09-29   106      12       22     R     31    1        1
 8  2025-03-27   108      12       21     R     31    0        0
 9  2025-03-27   108      11       22     R     31    1        0
10  2025-03-27   109      12       21     R     31    1        0
11  2025-03-27   109      11       22     R     31    0        1
```

## Planted leaks

Each was applied independently to an in-memory scratch copy and discarded with the process. Seed 5118, 400-example generation budget per run. All three failed the same-day outcome comparison on the intended column.

| Feature and change | Shrunk case | Cutoff and observed change | Pytest time |
|---|---|---|---:|
| `batter_hr_7g`, remove the date shift | `(3, 1, 0, 2, False, False, False, 0, 0, 2)` | 2025-03-29, PAs 6–7 went from 0 to 0.3333333333 | 4.47 s |
| `batter_pitcher_shrunk_hr`, stop subtracting the current date's hits from the cumulative numerator | `(2, 1, 0, 2, False, False, False, 0, 0, 0)` | 2024-09-29 doubleheader, PAs 0–3 went from 0.2195 to 0.4195 | 2.34 s |
| `pitcher_catcher_framing`, remove the date shift from the expanding mean | `(5, 1, 0, 2, False, False, False, 0, 0, 4)` | 2025-04-02, PAs 10–11 went from 0.5 to 0.4 | 5.64 s |

The logs and wrapper watch files are under `.codex-review/proptest/`. The sandbox emitted priority warnings from `nice`; pytest ran and the wrapper recorded the exit status.

## Unmodified source

| Seed | Configured examples | Pytest time | Result |
|---:|---:|---:|---|
| 5119 | 1,000 | 143.56 s | Pass |
| 5120 | 1,000 | 143.71 s | Pass |

Both seeded baseline runs passed, for 2,000 configured examples and 287.27 seconds of pytest time. No counterexample occurred in the fixed-matchup generated property. The nested Hypothesis check uses a filter-free generator and an explicit example budget. Pytest reports the pass and elapsed time for the enclosing test. Its statistics footer omits the nested example count.

## Confirmed violation in an additional structural check

`opp_bullpen_hr_30g` for an unchanged starter PA on April 6, 2025 was NaN when another same-game PA was against the starter and 0.5 when that other PA was against a reliever. The fully specified example is `.codex-review/proptest/regressions/test_bullpen_same_day_availability.py` (51 lines, plain pytest). It is minimized by the ten-date gate: one reliever PA on each of ten prior dates, with alternating outs and hits, followed by the fixed target starter PA and the one PA whose pitcher changes. The unmodified source reproduced the failure in 0.67 s.

This is a definite violation of the strictly-prior-date contract. At `compute.py:475–494`, the rolling average uses prior observations and the team/date merge key exists only when that date contains an observed reliever PA. The value's availability thus reveals whether a reliever appeared that day, even on the earlier starter PA. The target PA, prior history, pregame probable starter, opposing team, and all hit outcomes are identical between inputs. The serving lookup in `model/predict.py:289–293` uses the most recent available team value from its history. The reproduction establishes the computation violation. Its effect on served predictions was outside this trial.

## Reference calculations and limits

`.codex-review/proptest/test_feature_references.py` (87 lines) independently selects strictly prior PAs and computes the seven-game-date mean, the rookie pseudocount and career-PA gate for all three windows, both expanding platoon gates, and the prior batter/pitcher shrinkage. Comparisons run for every PA. Seed 5122 produced 100 passing examples, zero failing and zero invalid, plus three explicit boundary cases. These exercise unequal date weights across an offseason, the transition at exactly 100 career PAs, and distinct histories for left- and right-handed matchups. Pytest reported 4 passing tests in 16.16 seconds. The reference checks and bullpen reproduction are in commit `9eb7a91e`.

All five requested steps were completed. History length was capped at 34 game-dates, so the trial did not exercise expiration from the 60- and 120-date windows, and the two pitch profiles and periodic hit masks limit variation within dates. Only the additional bullpen case changes observed matchup structure. Real artifact contents, serving effects, and the other kickoff leads were outside the trial. No production files or dependencies were changed. Both commits are local and this report is uncommitted.

## Was it worth it?

Yes. In roughly nine minutes the trial established that three independent same-day regressions are detectable, checked 2,000 generated schedules with fixed matchup structure, and documented a real leak exposed by varying that structure in a targeted follow-up. The contrast in coverage matters when choosing the inputs to perturb. I would try the tied-quantile lead in `simulate/quality_bins.py` next. A small generator with heavily repeated scores could check bin and policy-index agreement directly, including the smallest cases where ties collapse adjacent bins.
