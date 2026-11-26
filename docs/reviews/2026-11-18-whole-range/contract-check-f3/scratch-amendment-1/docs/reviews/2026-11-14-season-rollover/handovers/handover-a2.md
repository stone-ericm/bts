# SRA2 handover

A2 supersedes A1's season definition: `src/bts/contest_state.py::_observation_season_year` now prefers parsed recorded_at's New York year, then source_date.year, then unknown. `src/bts/health/contest_state.py::check` says the old observation was recorded in that season and requests the current-season fetch.
Updated `tests/test_season_rollover_state.py` covers every corrected row, conflicting field years, New York year boundaries, invalid/missing timestamps, manual overrides, and unchanged health coverage checks (none/INFO/WARN) for a fresh recording with last season's round.
S3/S4/S5 source hashes are identical before/after A2. No commit or push.

RED: the two required auto-file cases both failed before source changes: optional mode returned model-only zero; required mode raised ContestStateError despite a current recording. All updated state cases before source changes: 19 failed, 55 passed. Baseline focused run: 307 passed.
Mutations rerun separately (all pytest exit 1):
- S1 invert season comparison: `test_s1_prior_season_is_absent`, 24 failures.
- S1 bypass model gate: `test_s1_model_fallback_needs_current_settlement`, 10 failures, 6 passes.
- S2 invert season comparison: `test_s2_prior_season_alert_names_season_and_fetch`, 6 failures.
- S2 disable first-fetch exemption: `test_s2_first_fetch_zero_without_settlement_is_not_critical`, 5 failures.
- Additional A2 mutation restoring source_date precedence: `test_s1_current_recording_with_prior_round_is_used` and `test_s2_current_recording_with_prior_round_keeps_coverage_checks`, all 5 failed.
Each mutation was restored byte-for-byte with SHA-256 equality verified. Final focused run: 330 passed; diff whitespace check clean.

Same focused command:
`TZ=America/New_York .venv/bin/python -m pytest tests/test_season_rollover_state.py tests/test_season_rollover_data.py tests/test_contest_state.py tests/test_contest_state_best_trust.py tests/test_contest_state_corrupt_precedence.py tests/health/test_contest_state.py tests/data/test_pull.py tests/data/test_build.py tests/test_data_build_empty.py tests/test_decision_saver_fallback.py tests/test_saver_state.py tests/test_tail_policy_r3_fixes.py -q -p no:cacheprovider --tb=short`

`git diff --stat` (cumulative tracked A1+A2 changes):
```text
 src/bts/contest_state.py               | 28 +++++++++++++++++++++++++---
 src/bts/data/build.py                  | 13 ++++++++++---
 src/bts/data/pull.py                   | 14 +++++++++++++-
 src/bts/features/compute.py            |  2 ++
 src/bts/health/contest_state.py        | 20 +++++++++++++++++---
 tests/health/test_contest_state.py     | 27 +++++++++++++++------------
 tests/test_contest_state.py            | 25 +++++++++++++------------
 tests/test_contest_state_best_trust.py |  5 +++--
 tests/test_data_build_empty.py         | 10 +++++++---
 9 files changed, 105 insertions(+), 39 deletions(-)
```
Also include both untracked new test files: `tests/test_season_rollover_state.py` (74 cases), `tests/test_season_rollover_data.py` (29 cases). No unmatched table rows. External gate/review and real-artifact check remain with the lead; A1's missing-data limit on the leakage/nuclear audit is unchanged.
