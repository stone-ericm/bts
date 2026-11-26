# PP-C1 kickoff 3 handover

Changed only `tests/test_pickpath_review_pp2.py` during this follow-up, plus this required handover. Production source was not edited; scheduler, late_delivery and picks SHA-256 values match this turn's starting values.

`test_delivered_hit_plus_void_double_adds_exactly_one` now covers both the existing Final/explicit-zero-AB fixture and a Preview/Postponed fixture with no roster or batting line. The new fixture returns Preview/Postponed in the actual mocked schedule JSON, so schedule-based void resolution must supply the Pass. Both leg orientations run through both real scorers.
The daemon cases now call `run_result_polling(daily.pick.game_pk, ...)`, matching real run_day even when the primary is void. Existing hit+void slot, exact +1 streak and reconcile assertions remain.

## RED / GREEN
- Before editing: original four cases passed (`4 passed in 9.65s`, `/private/tmp/ppc3-baseline.log`).
- Confirmed the false green: disabling only `_slot_is_voided` in memory left the original four cases passing (`4 passed in 2.70s`, `/private/tmp/ppc3-original-mutant.log`).
- Mutation command: `TZ=America/New_York .venv/bin/python /tmp/ppc1_r2_disable_void_resolution.py tests/test_pickpath_review_pp2.py::test_delivered_hit_plus_void_double_adds_exactly_one -q -p no:cacheprovider --tb=short`.
- RED after adding the Preview cases: `4 failed, 4 passed in 3.83s` (`/private/tmp/ppc3-red.log`). All four `preview_no_roster` cases fail: CLI leaves `slot_results=None`; daemon returns `unresolved` instead of `final`. The original Final/zero-AB cases still pass.
- GREEN in a fresh process without the mutation: `8 passed in 3.49s` (`/private/tmp/ppc3-green.log`), using `.venv/bin/python -m pytest` on the same test with `TZ=America/New_York -q -p no:cacheprovider`.

## Required focused run
Ran `TZ=America/New_York .venv/bin/python -m pytest tests/test_postponed_delivery_guard.py tests/test_pickpath_review_pp1.py tests/test_late_delivery_guard.py tests/test_thin_slate_checks.py tests/test_incident_register_2026.py tests/test_scheduler.py tests/test_scheduler_decision_record_integration.py tests/test_scheduler_decision_record.py tests/test_streak_applied_dates.py tests/health tests/test_pickpath_review_pp2.py -q -p no:cacheprovider`.
Last line: `======================= 695 passed, 8 skipped in 25.39s ========================`
Full log: `/private/tmp/ppc3-focused.log`. Eight existing skips are private-mode cases without contest-state fetch or transport. `git diff --check` passed.
No production mutation on disk, whole suite, commit, push, stash, branch change, real transport, or account/outcome access. No outstanding follow-up identified.
