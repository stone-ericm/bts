# WR-Ea handover

Commit `7faef25508d7b1adb77614c60f60d006e741455e` is the single commit on top of `d903cfc4` on `fix/wr-ea`. No push was made.

## Items and RED evidence

| Item | Test in `tests/test_wr_e_texts.py` that failed on the baseline |
| --- | --- |
| 1. Morning warning restricts the command to before the first check | `test_stale_warning_maximum_detail[morning]` |
| 2. Dashboard heading reads “LAST RECORDED NO-PICK SUMMARY” | `test_skip_surfaces` (all ten parameter cases) |
| 3. A raising class lookup preserves the base summary with an unverified candidate | `test_candidate_with_raising_class_lookup_preserves_base_summary` |
| 4. Delivery warning points to B8a for a later date | `test_stale_warning_maximum_detail[delivery]` |

The RED run recorded 13 failures and 43 passes in `.codex-review/wr-ea/red.log`. At the longest tested detail of 177 characters, the dispatched bodies with their envelopes measured **895 characters for morning** and **875 for delivery**, both under 900.

All 19 C14 scratch mutations were detected by tests in `test_wr_e_texts.py`. They covered restoring the highest prediction, below at equality, dropping the K2 label, both case-order changes, each of the three old journal endings, each of the three escaping omissions, each missing saver sentence, both old refusal endings, a string-probability exception, the old heading, the old pointer, and dropping the first-check qualification. Logs are in `.codex-review/wr-ea/mutation-*.log`.

## GREEN and preserved behavior

| Required command group | Passed |
| --- | ---: |
| Operator texts, optional LightGBM, web saver | 69 |
| Scheduler skip visibility and tail objective | 48 |
| Stale delivery | 62 |
| Morning state | 61 |
| Season-walk tests | 18 |

The first GREEN attempt passed 68 tests and hit a sandbox permission error at the dashboard HTTP test's loopback bind. The rerun with approval passed all 69. Tests ran directly in the required groups.

- The base-comparison tests against `63669c42` covered summary existence and preserved fields over unusable and valid prediction frames, complete lineup-check returns outside the allowed summary fields, and shipped bars across both regimes and saver states. The source comparison showed an empty strategy diff.
- Skip-visibility and notice lifecycle tests covered summary persistence, the missed-pick gate, and once-per-day notices. The state compatibility test exercised both directions using the actual base loader and current loader.
- The morning and stale-delivery suites exercised the transport, status paths, incident key, date, deduplication, retry and interrupt boundaries. The source diff in these warning functions is limited to the two contract text replacements. The late and postponed refusal tests verified the warning, archive, record and removal of the pick under their respective sources. Ea's source diff leaves the EOD path untouched.
- Malformed display fields, HTML, line breaks and unusable probabilities were covered by the operator text tests. Morning tests also covered unreadable startup records. Failure, unconfirmed send, interrupt and restart cases were covered by the notice lifecycle and warning suites. Midnight ET was covered before and during delivery and at startup, and the walk boundary checks covered December 31 to January 1. The warning date sources are identical to the baseline.

## Season walk

The direct walk passed before and after committing. On the final commit it produced 4,599 runs with zero problems, and the full decoded output matched `scripts/audit/season_walk/results/walk-e.json.gz`. The comparison against the earlier baseline also agreed on every action, bar and day count. The 841 former skip-message findings split into 229 K3 and 612 K4. The checker caught a false “below” comparison planted in a valid in-season K3 record. The saved result files required no update.
