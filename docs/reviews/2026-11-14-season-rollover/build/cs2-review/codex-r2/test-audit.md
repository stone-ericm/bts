# Final test review

`compare.py` proves all 40 assertions in `test_contest_state.py` are identical to round 1. Removing only the ten added `now=_PINNED_2026` keywords, their import and constant reproduces the round-1 module AST. Its other explicitly timed expiry tests retain their original clocks. The pinned year retains the nonzero model fixtures and the model-9/contest-7 counterexample; it does not relax an assertion. Original round-1 fixtures under a 2027 clock produce four failures / thirteen passes, confirming the reported hazard. The model-9 case passes there with a zero model diagnostic, making its anti-max premise vacuous; the pin restores that premise.

All fourteen changed test files ran with a default clock at January 1, 18:00 UTC, in both 2027 and 2028, advancing with real elapsed monotonic time. `future_clock.py` replaces the stdlib datetime/date classes and their bound aliases in bts/test modules after collection; explicit fixture dates/clock monkeypatches win. The production CLI, scheduler and contest-state clock values are asserted; alias lists are saved in `clock-2027-aliases.json` and `clock-2028-aliases.json`. Binary consumers are preloaded, and a captured real monotonic function avoids changing timing-mock call counts. Final runs: 2027 562 passed / 8 skipped; 2028 560 passed / 8 skipped / 2 failed. Both failures are `test_row6_cli_void_display_reads_scored_season`: its March 24, 2027 date becomes 283 days old, so scoring refuses before reaching the mocked void resolver. Pin `bts.cli._now_et` to that pick's night. No other season-sensitive assertion/premise issue was found in the touched fixtures.

Initial diagnostic harness versions produced a frozen-heartbeat failure, consumed a mocked monotonic clock, and (for the isolated original contest file) loaded a binary datetime consumer after replacing datetime, causing a process abort. Those are harness artifacts, not product failures; original logs are retained, and the corrected runs use `*-final.log` names. The two full-file final clock runs already had the binary dependencies loaded by collection.

## All 16 adopted parameterized cases (10 functions)

| Function in test_streak_season_cs2_review.py | Cases | Review |
| --- | --- | --- |
| test_b1_deeply_nested_streak_json_isolated_by_season_readers_and_reconcile | 1 | Rewritten read oracle now expects RecursionError, not fresh values. Dateless equivalence, [] markers, unchanged bytes and actual complete reconcile rebuild are asserted. Fails on round-1 source at rebuild. |
| test_sequence_2026_to_2027_hit_racing_scorers_double_crash_and_reconcile | 1 | Real writes, early reply, model/saver readers, cross-year reset, BaseException gap, pp1 B1 guard, retry and real 02:00 CLI replay all asserted. The peer run is sequential, so "racing" overstates this test alone; retained probe.py supplies an actual barrier-driven CLI/polling race. |
| test_sequence_opening_night_miss_does_not_trigger_2026_saver_zone | 1 | Prior 12/available tests the correct rollover direction (0/available, not 12/spent). |
| test_sequence_opening_night_void_keeps_legacy_file_untouched_then_day2_stamps | 1 | Preserves literal legacy bytes and asserts the next hit stamps 2027 with only its own marker. |
| test_mid_season_saver_exact_boundaries_on_tagged_file | 5 | 9,10,12,15,16 exercise both sides of each zone boundary, with numeric and saver assertions. |
| test_mid_season_second_miss_after_spent_saver_resets_to_zero | 1 | Same-date retry followed by next-date miss distinguishes idempotency from spent-saver reset. |
| test_corrupt_streak_json_stops_the_scorer_and_is_left_intact | 3 | Rewritten to nonzero CLI exit and byte preservation for void/hit, unresolved hit pick and persistent health failure. Generic exit/error checks are supplemented by precise error assertions in the main new file and the independent 54-scorer matrix. |
| test_contest_dates_dec31_to_jan01_boundary | 1 | Real dated writes verify the calendar-year boundary and sole new marker. |
| test_reconcile_summary_end_to_end_at_utc_et_new_year_boundary | 1 | Real replay/summary on either side of ET New Year, with UTC clock and explicit tag assertion. |
| test_row12_backup_verify_and_export_accept_season_tagged_streak | 1 | Real unchanged consumers accept the tagged file. Its clean verification of tagless regenerated values characterizes row 12's known omission, not correctness of restored seasonal diagnostics. |

The adopted module header at lines 16–17 still says deep corruption must return fresh values; that prose contradicts the amended contract and the corrected function assertions. Update the header. No adopted assertion requests the obsolete reset.

## Main new file additions / rewritten assertions

- Old five-case "unreadable means fresh" oracle is replaced by six exact exception-type cases, adding decoder recursion. This is the amendment's intentional contract change, not an assertion loosened for green.
- Eighteen hit/miss/double corruption cases require the exact base exception and byte preservation.
- Six seasonal-save cases require a default-saver error and explicit-saver rebuild with exact tag, counter, saver and marker values.
- Non-string dates now require ValueError. The test only checks final bytes; independent `attack.py` also spies that no reader was reached.
- Unpadded selection fallback requires a true 2027 saver from stale state and exercises actual strategy. This protects strategy only; the new shadow and polling reader parses remain stricter (blocking finding).
- Three bad-season argument cases require TypeError and unchanged bytes. They pass on pre-CS2 base for the wrong reason (unsupported keyword); assert an informative guard message to make them red there. They do fail on round-1 source, and `attack.py` tests all public guards with existing and missing files.
- `test_row5_reconcile_rebuilds_a_deeply_nested_file_as_it_did_before` invokes save_streak directly, despite its name. Actual reconcile is independently covered by the adopted deep test and six complete-history probes.

New files total 118 cases: all pass currently; base source 86 failed / 32 passed, round-1 source 36 failed / 82 passed. Base's 32 greens are the eight original compatibility controls, eighteen legacy corrupt-update failures, three rewritten adopted corruption cases, and the three unsupported-keyword TypeErrors. Round 1's 82 greens are 66 retained original cases, four already-raising deep reader/update cases, and twelve adopted cases unaffected by the fixes. No test-code note is a blocker; independent real-code probes cover the relevant behavior.
