# SRA handover — S1–S5

All table rows matched. Only the five permitted source files and tests changed.
No commit/push; scheduler.py, strategy.py and model/predict.py untouched.

- S1 — `src/bts/contest_state.py::load_decision_streak_state`: reject prior-season selected observations (source_date first, otherwise recorded_at in New York); preserve unexpired manual overrides. Model streak is zero without a current-year settled root pick; no streak.json write. Shared `_observation_season_year` supplies the season definition.
  RED: 26 failures (old observations still supplied 18 or did not raise; model fallback retained 9 without a current settlement).
  Mutations: invert season comparison → `test_s1_prior_season_is_absent` caught all 12 cases; bypass model gate → `test_s1_model_fallback_needs_current_settlement` caught 10 cases (6 unchanged cases passed).
- S2 — `src/bts/health/contest_state.py::check`: CRITICAL names the prior season and current-season fetch; first current-season zero observation without a source_date or current settlement is exempt. Other missing-date cases remain CRITICAL.
  RED: 8 failures (3 missing/wrong old-season messages; 5 false first-fetch CRITICALs).
  Mutations: invert season comparison → `test_s2_prior_season_alert_names_season_and_fetch` caught 3 cases; disable zero-streak exemption → `test_s2_first_fetch_zero_without_settlement_is_not_critical` caught 5.
- S3 — `src/bts/features/compute.py::_build_probable_pitcher_lookup`: skip non-digit stems before int conversion. Numeric lookup/cache bytes match the control with and without sidecars.
  RED: 5 failures (4 conversion/cache errors plus weather-only conversion failure).
  Mutation: remove stem guard → `test_s3_sidecars_do_not_change_numeric_cache` caught 4 cases, including a valid JSON feed under a signed numeric filename.
- S4 — `src/bts/data/pull.py::download_game_feed`: require JSON object, retain the three attempts, write original bytes to a same-directory .tmp file, close, os.replace, clean in finally. Existing targets remain untouched, including invalid cached bodies.
  RED: 11 failures (7 invalid bodies persisted, retry/atomicity failed, write/rename failures bypassed, pull_feeds failed to skip).
  Mutations: remove JSON validation → `test_s4_invalid_body_retries_without_files` caught 7; replace os.replace with direct write → `test_s4_retries_invalid_then_publishes_identical_bytes_atomically` caught 1.
  Write/rename error tests also verify no temporary files remain; pull_feeds test verifies SKIP and continuation.
- S5 — `src/bts/data/build.py::build_season`: enumerate with iterdir, catching only missing-directory FileNotFoundError; no-feed builds raise once the output exists, naming season/directory and preserving output. Prior empty-season CLI expectations updated.
  RED: 8 failures (4 existing-output false successes; 4 swallowed listing errors).
  Mutations: remove existing-output guard → `test_s5_no_requested_feed_noop_only_before_output_exists` caught 4 (4 pre-season cases passed); restore glob enumeration → `test_s5_invalid_season_directory_raises` caught all 4.

Every mutation ran separately, failed with pytest exit 1, and was restored byte-for-byte with SHA-256 equality checked. Season-best classification and saver logic are unchanged; the prior-season best test now checks classification directly because the decision loader discards that observation.

Validation:
- Before changes: 189 existing focused tests passed.
- Initial new-test RED: 58 failed, 20 passed (78 cases), before any source edit. Two more unchanged manual-override cases were added afterward.
- Final focused command (307 passed):
  `TZ=America/New_York .venv/bin/python -m pytest tests/test_season_rollover_state.py tests/test_season_rollover_data.py tests/test_contest_state.py tests/test_contest_state_best_trust.py tests/test_contest_state_corrupt_precedence.py tests/health/test_contest_state.py tests/data/test_pull.py tests/data/test_build.py tests/test_data_build_empty.py tests/test_decision_saver_fallback.py tests/test_saver_state.py tests/test_tail_policy_r3_fixes.py -q -p no:cacheprovider --tb=short`
- New tests use local fixtures/mocked responses, pinned now values, no network/git/branch assumptions. Touched existing decision/health tests also pin now.
- `git diff --check`: clean. Full suite/gate not run, as instructed; lead owns the external gate/review and real-artifact check.
- `TZ=America/New_York .venv/bin/python scripts/leakage_audit.py`: could not start (FileNotFoundError: data/processed/pa_2023.parquet). Its nuclear check was not reached; no leakage-audit pass is claimed.

`git diff --stat` (tracked changes; new tests remain untracked):
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
Additional new tests: `tests/test_season_rollover_state.py` and `tests/test_season_rollover_data.py` (80 cases total). Include both when collecting the change set.
