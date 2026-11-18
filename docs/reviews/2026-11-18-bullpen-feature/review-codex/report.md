# Bullpen official-date review

## Verdict

**SIGN** for `1135c217` against `43fc5628`. The change removes the observed same-day availability defect on supported official-date inputs. The 2025-only replay fills 941 PAs across 30 team-dates and preserves every previously valued bullpen bit.

## Blocking findings

None.

## Notes

- The shrunk test at `tests/test_bullpen_same_day.py:54` accepts two missing values. An always-missing mutant demonstrates this false green. The warmed doubleheader and loop tests require finite values at eligible gaps. The shrunk test could require 0.5 on both arms.
- The single-PA canary at `scripts/leakage_audit.py:13` has limited coverage. A scratch mutant that uses the base logic on multi-PA dates passed the canary and reproduced `NaN` → 0.4 on a two-PA date. The doubleheader test covers that larger target.
- The standalone `scripts/phase5_bullpen.py:165` experiment already includes the feature in its baseline via `BLEND_CONFIGS`, and its treatment appends a duplicate at line 146. This predates the change. Its coverage print uses its own calculation at line 123.

## Identity on real data

Pandas 3.0.1, frozen 20,087-game lookup with verified SHA-256, park drag absent in both arms. The first three runs compared 88 other columns exactly, including column and row order, index types and dtypes. The perturbation runs used a row ID to check alignment to each input after the pipeline's existing date/game sort and index reset.

| Input | PAs | Games covered | Known team / probable PAs | Base → branch missing | Fills |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | 181,694 | 2,425/2,425 | 181,694 / 181,620 | 12,372 → 11,546 | 826 |
| 2025 | 182,083 | 2,425/2,425 | 182,083 / 181,899 | 12,262 → 11,321 | 941 |
| 2024–25 | 363,777 | 4,850/4,850 | 363,777 / 363,519 | 13,343 → 11,546 | 1,797 |
| 2025 shuffled | 182,083 | 2,425/2,425 | 182,083 / 181,899 | 12,262 → 11,321 | 941 |
| 2025 duplicate index labels | 182,083 | 2,425/2,425 | 182,083 / 181,899 | 12,262 → 11,321 | 941 |
| 2025 duplicated PAs | 182,624 | 2,425/2,425 | 182,624 / 182,440 | 12,297 → 11,356 | 941 |
| 2025 strings and timestamps, each | 182,083 | 2,425/2,425 | 182,083 / 181,899 | 12,262 → 11,321 | 941 |

Every run had zero losses, zero valued bit differences and zero maximum valued absolute delta. Within each implementation, the entire timestamp and string outputs matched exactly. [Identity log](real.log)

## The rule and invariance

The independent date-by-date loop matched all 182,083 rows in 2025, with exact missing masks and maximum finite error 5.6e-17. The 92-row synthetic loop matched exactly with both object and float team keys. It covered 9/10/>30 prior buckets, an initial eligible gap, adjacent unequal means, a trailing gap, same-team doubleheaders, cross-year carry, a new team ID, known team/unknown probable, absent or null team, null date and expiry through undefined buckets. Both empty-history controls matched the base schema exactly. [Reference](probe_common.py) · [Edges](edges.log)

On real 2025 PAs, swapping every target-day pitcher to the starter or a reliever, flipping every hit, and adding/removing PAs and another game preserved every target-day bullpen value. Cut dates were July 23 (team 113 gap), May 1 (ordinary), May 4 (doubleheader), March 18 (first) and September 28 (last). Future PA mutations preserved every output column through each cut on 114,932 / 35,001 / 38,294 / 74 / 182,083 PAs respectively. The last cut used a synthetic later date copied from real PAs. The first cut had no warmed team. [Invariance log](invariance.log)

Strict predecessor selection excludes equal and later parsed keys. Off-contract intraday keys can split one calendar day. Flipping the midnight PA moved the 04:00 feature from 0.454545 to 0.363636 with naive, UTC and New York timestamps. On both code versions, a future null team coerced fabricated IDs `2**53` and `2**53 + 1` to the same float before the join, changing two earlier values from 0 and 1 to 0.5. Real MLB IDs fit exactly in that dtype. Both versions rejected the mixed timezone/format string probe during date parsing. [Edge probe](probe_edges.py)

## The join

Unsorted input, ties across teams and reliever PAs, integer lookup IDs with object/float PA-side keys, `None`/`NaN`, a missing lookup entry and a known team with no history all passed. Both as-of inputs were globally date-ordered with nonnull, unique team-date keys and object match dtypes. The returned team dtype matched the base and no temporary key leaked. Daily groupby makes the history unique and target deduplication followed by one result per key makes the final right side unique. An injected duplicate attachment raised `MergeError` at `compute.py:501`. [Join probe](edges.log)

All 1,516,736 PAs from 2017–25 finished in fresh processes. Base took 29.242 s at 7,431.8 MiB peak RSS and branch 29.225 s at 7,452.2 MiB. Missing bullpen PAs fell from 20,834 to 11,388. Each timing measures the feature call and each RSS is the process high-water mark. [Run log](real.log)

## Serving

The production lookup builder received strict prefixes of the 2025-only frame. All 15 other lookup entries were byte-identical at each cut. For the 30 teams, March 19 had zero valued teams in both arms. April 14, July 25, September 21 and December 31 had all 30 valued in both, with 1, 0, 1 and 0 numeric changes respectively. [All teams at each cut](actual-lookups.csv)

Across 4,794 team-dates, old/new training had 331/301 missing and both serving arms had 331 missing. Old training versus old serving differed on 4,362 of 4,433 finite pairs, with 30 values exclusive to each arm. New training versus old/new serving differed on 4,391/4,362 of 4,463 finite pairs, with 30 training-only and 301 jointly missing for each. New-serving mean/max absolute difference from new training was 0.004355/0.037074. [Team-date replay](morning-parity.csv)

Live slots read the opponent team's latest stored finite value for both starter and reliever scoring. The 30 next-date rows reproduced the handover. Of 4,463 finite old/new lookup pairs across the season, 29 changed, all on those next dates. The signed delta ranged from −0.95 to +1.58 percentage points in the bullpen input. A fresh training run would also see the newly filled inputs. Score and pick deltas are unmeasured. [Full-precision next dates](following-games.csv)

| Team | Filled date | Next date | Old lookup | New lookup and new training |
| ---: | --- | --- | ---: | ---: |
| 135 | 2025-04-13 | 2025-04-14 | 0.1225558860 | 0.1310521603 |
| 117 | 2025-04-26 | 2025-04-27 | 0.1545568300 | 0.1528190013 |
| 119 | 2025-04-29 | 2025-04-30 | 0.2100677313 | 0.2089092091 |
| 138 | 2025-05-09 | 2025-05-10 | 0.2138173072 | 0.2151330967 |
| 135 | 2025-05-10 | 2025-05-11 | 0.1784673821 | 0.1942568558 |
| 117 | 2025-05-15 | 2025-05-16 | 0.1632592134 | 0.1596644422 |
| 134 | 2025-05-18 | 2025-05-19 | 0.2022139024 | 0.2046497998 |
| 114 | 2025-05-19 | 2025-05-21 | 0.2474676231 | 0.2454474211 |
| 116 | 2025-05-25 | 2025-05-26 | 0.1937125849 | 0.1960273997 |
| 117 | 2025-05-30 | 2025-05-31 | 0.1484663502 | 0.1588830169 |
| 139 | 2025-05-31 | 2025-06-01 | 0.1957838485 | 0.2068949596 |
| 133 | 2025-06-04 | 2025-06-05 | 0.2407629016 | 0.2468992652 |
| 140 | 2025-06-06 | 2025-06-07 | 0.1973746911 | 0.1937080244 |
| 113 | 2025-06-10 | 2025-06-11 | 0.2000189653 | 0.1969229901 |
| 121 | 2025-06-11 | 2025-06-12 | 0.1836899679 | 0.1745396411 |
| 144 | 2025-06-11 | 2025-06-13 | 0.2013995944 | 0.1990491671 |
| 114 | 2025-06-20 | 2025-06-21 | 0.1934460537 | 0.2045571648 |
| 138 | 2025-06-27 | 2025-06-28 | 0.2229700980 | 0.2214820027 |
| 137 | 2025-07-03 | 2025-07-04 | 0.2072617104 | 0.2120862718 |
| 143 | 2025-07-06 | 2025-07-07 | 0.2124853130 | 0.2060567416 |
| 111 | 2025-07-08 | 2025-07-09 | 0.1719446509 | 0.1717913942 |
| 111 | 2025-07-12 | 2025-07-13 | 0.1818202542 | 0.1792975081 |
| 144 | 2025-07-12 | 2025-07-13 | 0.1773157086 | 0.1728424041 |
| 143 | 2025-07-22 | 2025-07-23 | 0.2143174358 | 0.2088473503 |
| 113 | 2025-07-23 | 2025-07-25 | 0.1670838303 | 0.1670838303 |
| 110 | 2025-08-01 | 2025-08-02 | 0.2083868684 | 0.2099741700 |
| 141 | 2025-09-11 | 2025-09-12 | 0.1899477537 | 0.1889862152 |
| 114 | 2025-09-12 | 2025-09-13 | 0.2085722516 | 0.2113500293 |
| 113 | 2025-09-18 | 2025-09-19 | 0.2100094168 | 0.2151376219 |
| 134 | 2025-09-20 | 2025-09-21 | 0.2063922254 | 0.1968684159 |

## The audit canary

Pristine base exited 0. The actual branch script on base source gave `NaN` and 0.4, failed the starter arm and exited 1. The branch script on branch source gave 0.4 on both arms and exited 0. Existing stdout matched across all three runs. Audit 1 was MINOR (1,384/144,755, 0.96%). Audit 3 printed 10/10 platoon, 10/10 ground ball and 5/5 park. Audit 4 matched 0.1786, 0.1622, 0.2128, 1.0182, 0.1608 for bullpen, and 0.2882 for batter/pitcher. Three branch canary calls were identical. Forced lack of warm-up failed both finite checks. [Audit log](audit.log) · [Base with canary](audit-base-with-canary.txt) · [Branch](audit-branch.txt)

## Tests and false greens

Base failed the shrunk, warmed doubleheader, loop and real-slice tests. Its two passes were `test_bullpen_empty_history[False]` and `[True]`. Neither has a usable reliever-date bucket and the lookup toggle only controls whether the team column is present. [Base pytest](pytest-base.txt)

| Scratch mutation | Tests that failed |
| --- | --- |
| Allow exact as-of matches | Shrunk, doubleheader, loop, real slice |
| Forward-fill shifted rolling values | Shrunk, doubleheader, loop, real slice |
| Remove team dtype alignment | Shrunk, doubleheader, loop with incompatible merge keys |
| Sort by team first | Loop with unsorted date keys |
| Restore shift and exact-date merge | Shrunk, doubleheader, loop, real slice |
| Require nine defined rates | Loop, real slice |
| Use a 29-bucket window | Loop, real slice |
| Drop null-key exclusion | Loop with a null date key |
| Always emit missing | Doubleheader, loop, real slice |

Every requested mutant failed a synthetic test. Hidden files yielded 5 passes and the optional slice skipped. The new file, optional-LightGBM test and existing feature tests gave 87 passes in one process. The new file gave 6 passes in each of 2027 and 2028, with Python datetime, pandas and libc-backed time confirming the process-local wall clock. [Mutation log](tests.log) · [Hidden files](pytest-hidden.txt) · [Combined](pytest-combined.txt) · [2027](pytest-2027.txt) · [2028](pytest-2028.txt)

## Consumers

The direct readers are prediction, the M3 parity replay, the audit and the standalone Phase 5 experiment. The production `dropna().last()` lookup and M3 replay's presence and parity counts can move as designed. Training, PA and game simulations, experiments and calibration checks consume the feature through `FEATURE_COLS`. The standard train and score any-feature masks already have a nonmissing batter/pitcher prior. No consumer break or test pin to an old bullpen value was found.

The daily `blend_{date}.pkl`, shadow blend cache (feature names and park-drag fingerprint), and simulation blend cache (seed/day, feature names and fit settings) omit computation identity. Use fresh or computation-keyed caches for a later score comparison.

## Rows 1 to 5

1. Holds. Whole 2025 and synthetic loop plus real perturbations and valued-bit identity.
2. Holds. Unknown metadata, null date, cold window and empty-lookup schema probes.
3. Holds. Exact 2024, 2025 and combined frame comparisons plus input-order and key variants.
4. Holds. Actual prefix lookups for every team and replay of all next-date values.
5. Holds. Git diff confined to the three authorized tracked files and caches identified above.

## What I ran

Archived base `src scripts` with `git archive`. Serial `yield_run.sh` invocations ran the [historical comparison and serving](probe_real.py), [invariance](probe_invariance.py), [staged audit](probe_audit.py) and [pytest matrix](probe_tests.py). The sandbox denied priority lowering, and each wrapper was allowed to exit before starting its successor. The [small synthetic probe](probe_edges.py) ran directly. Real inputs were limited to 2017–25 and the audit directory contained only 2023–25 PA symlinks and the frozen lookup. Tracked files were untouched. No network, backtests, model fits or full fast suite.
