# Bullpen official-date attachment

Base `43fc56289bf61c517a5bf19a0ac65db17c72c11f`, branch `fix/bullpen-same-day`. Fix commit `1135c217e67e1443bbcadc4299cb13d6cb64eaa8`.

## Implementation

- `src/bts/features/compute.py`, bullpen block at lines 455–507. Compute rolling means on the existing per-team reliever-date buckets and attach the strict predecessor by official date. The predecessor position can carry an undefined mean. Team dtype normalization is confined to temporary as-of keys, and the many-to-one merge uses the original PA-side keys.
- `tests/test_bullpen_same_day.py` lines 16, 125, 142, 173, and 183 cover the shrunk regression, warmed doubleheader perturbation, plain-loop window, empty-history controls, and 2025 slice comparison.
- `scripts/leakage_audit.py` lines 9 and 174 exercise a deterministic warmed side with two variants of the same target PA. The audit exits 1 if either variant differs from the finite prior reference.

## Focused tests and base reproduction

| Test | Base | Fixed |
| --- | --- | --- |
| Shrunk same-day availability regression | RED, untouched PA was missing with no reliever, 0.5 with relief | PASS |
| Warmed doubleheader presence and outcomes | RED, both starter-only PAs missing against 0.3. Both relief variants passed before the starter-only assertion | PASS |
| Hand-built frame against the plain-loop window | RED, four qualifying PAs missing | PASS |
| Empty history with populated lookup | PASS | PASS |
| Empty probable lookup | PASS | PASS |
| 2025 team 113 slice against the plain-loop window | RED, 61 PAs missing out of 3,845 | PASS |

Base targeted run had 4 failures and 2 passes. Fixed targeted run plus existing context, park drag, cold start, prediction feature coverage, determinism, and experiment feature tests had 77 passes and 1 expected failure. The fixed focused and base targeted runs used real 2025 PA data and the frozen lookup for the optional slice. That test skips when those inputs are unavailable.

## Historical feature identity

The comparison ran the actual `43fc5628` source and the fixed source on the main checkout's read-only PA parquets, with pandas 3.0.1 and park drag off for both. Both used the same 20,087-entry frozen mapping at `~/bts-runs/inputs/probable_pitcher_lookup.2017-2025.json` (SHA-256 `4a8989587f97412bcb7498da3a27d8cd1ff0321a5186028fc17f8ce9f4536454`). The PA history for each comparison starts with the first date in the named input.

| Input | PAs | Games in lookup / total | PAs with known team | PAs with known probable | Old missing | New missing | Filled | Lost | Valued bit mismatches | Max absolute difference on old-valued PAs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 181,694 | 2,425 / 2,425 | 181,694 | 181,620 | 12,372 | 11,546 | 826 | 0 | 0 | 0 |
| 2025 | 182,083 | 2,425 / 2,425 | 182,083 | 181,899 | 12,262 | 11,321 | 941 | 0 | 0 | 0 |
| 2024–2025 | 363,777 | 4,850 / 4,850 | 363,777 | 363,519 | 13,343 | 11,546 | 1,797 | 0 | 0 | 0 |

In all three runs, `assert_frame_equal(check_exact=True, check_index_type=True, check_column_type=True)` passed for all 88 other columns in original order, with exact indices, row order and dtypes. In-process pickle bytes for the other-column frames matched in all three runs. The bullpen column matched at the float-bit level on all previously valued rows and on every row missing in both versions. Results are in `.codex-review/bullpen/identity.log`.

## Serving lookup replay

History source for these 4,794 team-dates is the 2025 PA parquet alone, with the frozen 2017–2025 probable mapping. For each target date the replay selects the last nonmissing bullpen value from rows with official dates strictly earlier than the target, as `predict._build_feature_lookups` does. The replay can reuse precomputed PA values because even the old bullpen value on a prior row only depends on that row's relief presence and older buckets. All rows sharing a team and date were checked for value and mask equality before collapsing them. Other lookup columns and their row order passed the exact frame comparison above.

| Left vs right | Both finite | Unequal finite | Left only | Right only | Both missing | Mean absolute difference | Max absolute difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old training vs old lookup | 4,433 | 4,362 | 30 | 30 | 301 | 0.00438417 | 0.03707443 |
| New training vs old lookup | 4,463 | 4,391 | 30 | 0 | 301 | 0.00438635 | 0.03707443 |
| New training vs new lookup | 4,463 | 4,362 | 30 | 0 | 301 | 0.00435470 | 0.03707443 |
| Old lookup vs new lookup | 4,463 | 29 | 0 | 0 | 331 | 0.00003165 | 0.01578947 |

Old training is missing on 331 team-dates and new training on 301. Both serving versions are missing on 331. The tables count numeric inequality among finite pairs and show absolute differences averaged over all those pairs.

At the next played date after each of the 30 filled gaps, the new lookup matches the new training value. Twenty-nine lookups move. Team 113 on July 25 has the same numeric value in both versions. Values below are rounded; `.codex-review/bullpen/following-dates.csv` contains the full-precision values.

| Pitching team ID | Filled date | Next played date | Old lookup | New lookup | New training |
| ---: | --- | --- | ---: | ---: | ---: |
| 135 | 2025-04-13 | 2025-04-14 | 0.1225558860 | 0.1310521603 | 0.1310521603 |
| 117 | 2025-04-26 | 2025-04-27 | 0.1545568300 | 0.1528190013 | 0.1528190013 |
| 119 | 2025-04-29 | 2025-04-30 | 0.2100677313 | 0.2089092091 | 0.2089092091 |
| 138 | 2025-05-09 | 2025-05-10 | 0.2138173072 | 0.2151330967 | 0.2151330967 |
| 135 | 2025-05-10 | 2025-05-11 | 0.1784673821 | 0.1942568558 | 0.1942568558 |
| 117 | 2025-05-15 | 2025-05-16 | 0.1632592134 | 0.1596644422 | 0.1596644422 |
| 134 | 2025-05-18 | 2025-05-19 | 0.2022139024 | 0.2046497998 | 0.2046497998 |
| 114 | 2025-05-19 | 2025-05-21 | 0.2474676231 | 0.2454474211 | 0.2454474211 |
| 116 | 2025-05-25 | 2025-05-26 | 0.1937125849 | 0.1960273997 | 0.1960273997 |
| 117 | 2025-05-30 | 2025-05-31 | 0.1484663502 | 0.1588830169 | 0.1588830169 |
| 139 | 2025-05-31 | 2025-06-01 | 0.1957838485 | 0.2068949596 | 0.2068949596 |
| 133 | 2025-06-04 | 2025-06-05 | 0.2407629016 | 0.2468992652 | 0.2468992652 |
| 140 | 2025-06-06 | 2025-06-07 | 0.1973746911 | 0.1937080244 | 0.1937080244 |
| 113 | 2025-06-10 | 2025-06-11 | 0.2000189653 | 0.1969229901 | 0.1969229901 |
| 121 | 2025-06-11 | 2025-06-12 | 0.1836899679 | 0.1745396411 | 0.1745396411 |
| 144 | 2025-06-11 | 2025-06-13 | 0.2013995944 | 0.1990491671 | 0.1990491671 |
| 114 | 2025-06-20 | 2025-06-21 | 0.1934460537 | 0.2045571648 | 0.2045571648 |
| 138 | 2025-06-27 | 2025-06-28 | 0.2229700980 | 0.2214820027 | 0.2214820027 |
| 137 | 2025-07-03 | 2025-07-04 | 0.2072617104 | 0.2120862718 | 0.2120862718 |
| 143 | 2025-07-06 | 2025-07-07 | 0.2124853130 | 0.2060567416 | 0.2060567416 |
| 111 | 2025-07-08 | 2025-07-09 | 0.1719446509 | 0.1717913942 | 0.1717913942 |
| 144 | 2025-07-12 | 2025-07-13 | 0.1773157086 | 0.1728424041 | 0.1728424041 |
| 111 | 2025-07-12 | 2025-07-13 | 0.1818202542 | 0.1792975081 | 0.1792975081 |
| 143 | 2025-07-22 | 2025-07-23 | 0.2143174358 | 0.2088473503 | 0.2088473503 |
| 113 | 2025-07-23 | 2025-07-25 | 0.1670838303 | 0.1670838303 | 0.1670838303 |
| 110 | 2025-08-01 | 2025-08-02 | 0.2083868684 | 0.2099741700 | 0.2099741700 |
| 141 | 2025-09-11 | 2025-09-12 | 0.1899477537 | 0.1889862152 | 0.1889862152 |
| 114 | 2025-09-12 | 2025-09-13 | 0.2085722516 | 0.2113500293 | 0.2113500293 |
| 113 | 2025-09-18 | 2025-09-19 | 0.2100094168 | 0.2151376219 | 0.2151376219 |
| 134 | 2025-09-20 | 2025-09-21 | 0.2063922254 | 0.1968684159 | 0.1968684159 |

## Leakage audit and nuclear test

Ran the checked-in audit with the worktree's `src` on `PYTHONPATH` and only the read-only 2023–2025 PA files and frozen probable lookup staged in `.codex-review/bullpen/audit-root/`. Park drag was absent in the staged directory.

- Audit 1 classified doubleheaders as MINOR, 1,384 of 144,755 batter-dates (0.96%).
- Audit 3 reported 10/10 platoon checks, 10/10 ground-ball checks, and 5/5 park-factor checks passing.
- Audit 4 used the same PA on the last 2024 date in the 2023–24 and 2023–25 computations. Both the base and fix had all six matches at the audit's 0.001 tolerance. This check holds the target date's PAs fixed.

| Audit 4 feature | 2023–24 | 2023–25 |
| --- | ---: | ---: |
| `batter_hr_7g` | 0.1786 | 0.1786 |
| `batter_hr_30g` | 0.1622 | 0.1622 |
| `platoon_hr` | 0.2128 | 0.2128 |
| `park_factor` | 1.0182 | 1.0182 |
| `batter_pitcher_shrunk_hr` | 0.1688 | 0.1688 |
| `opp_bullpen_hr_30g` | 0.1608 | 0.1608 |

Audit 5 set the target PA to all starter with no hit, then all relief with a hit, with an independent prior reference of 0.4. On the base it reported missing and 0.4 respectively and exited 1. On the fix both variants returned 0.4 and the audit exited 0.

## Runs

Each pytest invocation and long script ran serially through `~/bts-runs/yield_run.sh` with `TZ=America/New_York`, `PYTHONDONTWRITEBYTECODE=1`, and the worktree's `.venv/bin/python`. Logs and the replay CSVs are in `.codex-review/bullpen/`.

| Log | Run | Result |
| --- | --- | --- |
| `red.log` | Focused bullpen pytest on base | 4 expected failures, 2 passes |
| `audit-red.log` | Staged leakage audit on base | Audit 5 caught one of two variants. Audit 4 matched all six |
| `green.log` | Focused bullpen and existing feature pytest | 77 passes, 1 expected failure |
| `identity.log` | Real-frame identity and serving replay | 3 comparisons passed, 4,794 team-dates replayed, 30 next-date results |
| `audit-green.log` | Staged leakage audit on fix | Exit 0, all six nuclear comparisons matched and both canary variants passed |
| `fast.log` | Unsandboxed fast suite on committed tree | 9,213 passed, 37 failed, 16 errors, 24 skipped, 6 deselected, 1 xfailed in 28:59 |

The committed-tree fast suite exited 1. The 53 `FAILED` and `ERROR` IDs matched `~/bts-runs/gate-failed.txt` exactly, with no extra or missing IDs. The bullpen module had five passes and one skip because the optional real inputs were absent in the gate's working directory. That same test passed in the focused run with the explicit 2025 inputs. The worktree has no tracked changes after the gate, and the handover and scratch evidence are uncommitted.

## Known limits

- An opener listed as probable, a probable who did not start, or a missing probable can still skew or suppress the historical PA classification and later windows.
- Resumed PAs retain the suspended game's original official date. A historical bucket can therefore contain events that happened later than its stored date.
- The serving lookup still takes the last nonmissing stored value. After a reliever date that stored value excludes the reliever date itself, as in the M3 audit.
- Training inputs change on about half a percent of 2025 rows in the season-only run. A retrained model can change probabilities and policy actions at fixed bins. Existing backtest baselines, saved profiles, and audit reports describe their historical computation and coverage.
- Daily production `blend_{date}.pkl` uses the date as its key. Shadow blend pickles add feature names and park-drag fingerprint, and simulation blend caches use seed/day, feature names, and fit settings. Those keys omit this computation change. Use fresh caches or caches explicitly tied to the changed computation for later model or scorecard comparisons.

No baseline fast suite was rerun. No backtests, artifact rebuilds, model comparisons, or 2026 outcomes were used. No push.
