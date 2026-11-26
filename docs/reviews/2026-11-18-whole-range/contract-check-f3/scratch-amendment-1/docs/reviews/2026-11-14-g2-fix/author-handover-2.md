# G2 follow-up author handover

Worktree `/Users/eric/projects/bts-pp-c1`, branch `g2-fix`, uncommitted changes above `282c438`.
No commit, push, stash, branch change, deployment, agent launch or real-account access.

## Shared RED/GREEN evidence
- Existing focused baseline before source changes (`baseline-2.log`): `109 passed in 50.95s`.
- RED (`red-2.log`): `24 failed, 74 passed in 50.52s` (A: 1, B: 17, C: 4, D cron: 2).
- GREEN (`green-2.log`): `100 passed in 49.40s`, including every changed test file.
- Accepted-limit control (`limit-2.log`): `2 passed in 2.30s`; this documents retained behavior, not a new safety guarantee.
- All named tests below also pass in the final focused run.

## A — remove F2
- Restored only `src/bts/scheduler.py::plan_run_times` from `git show main:src/bts/scheduler.py`; verified the function's bytes, including its docstring, match main exactly after all mutations.
- Deleted all four merge tests from `tests/test_thin_slate_checks.py`.
- Added exactly one planner regression: `test_close_early_and_regular_runs_are_retained_as_separate_retry_opportunities`.
- It pins the 13:10/14:05 slate, offset 60: `10:10, 11:05, 12:05, 12:10, 13:05`, including the separate game-202 12:05 and game-101 12:10 attempts. Its reason records Codex's outage through 12:08.
- RED: `FAILED ...::test_close_early_and_regular_runs_are_retained_as_separate_retry_opportunities` before restoration.
- GREEN: all 14 thin-slate tests pass. No A mutation requested; exact-byte restoration is checked separately. `run_day` unchanged.

## B — distinguish unknown feed metadata from no regular games
- Changed `src/bts/data/build.py::build_season`: require readable, nonblank string `gameData.game.type` on every non-weather feed; missing/bad envelope or unreadable type raises `ValueError` naming the field and file. Validation happens even beside valid regular or spring feeds.
- Count a matching game type before the seven-inning exclusion, so an all-excluded regular slate still reaches the existing schema error instead of returning None. Row-extraction/schema errors and invalid JSON still raise.
- Missing/empty/weather-only directories and positively non-regular feeds retain the no-op; existing parquet preservation and mixed-season continuation controls stay green. No 03:00 cron-chain change.
- `tests/test_data_build_empty.py`: 12 malformed metadata cases; typeless feed before/after a spring or regular feed with an existing parquet; regular seven-inning exclusion; truncated JSON; multiple spring feeds; weather-only directory.
- RED: 12 `test_cli_rejects_missing_or_unreadable_game_type_with_filename` cases, four `test_cli_typeless_feed_fails_even_beside_other_feeds` cases, and `test_cli_seven_inning_regular_game_still_takes_schema_error_path` failed.
- GREEN: all 27 real-CLI data-build tests pass; existing build/schema tests pass too.
- Source mutations, all killed and restored (test names are in `tests/test_data_build_empty.py`):
  - `B-missing-metadata`: skip missing envelope → `test_cli_rejects_missing_or_unreadable_game_type_with_filename[feed0]`.
  - `B-nonstr-type`: allow non-string type → same test `[feed5]` (null).
  - `B-empty-type`: allow empty type → same test `[feed10]`.
  - `B-blank-type`: allow whitespace-only type → same test `[feed11]`.
  - `B-no-filename`: omit file from error → same test `[feed1]`.
  - `B-skip-after-nonregular`: ignore bad feed after spring feed → `test_cli_typeless_feed_fails_even_beside_other_feeds[zzzzzz.json-S]`.
  - `B-count-after-seven-inning-filter`: move counter back after exclusion → `test_cli_seven_inning_regular_game_still_takes_schema_error_path`.
  - `B-no-positive-exclusion`: remove no-op for positively excluded feeds → `test_cli_empty_season_is_success_without_writes[non_regular]`.

## C — handle the builder's None return in prediction refresh
- Changed only `src/bts/model/predict.py::_refresh_season_data`: on None, print one stderr line naming the season and absence of regular games, then return before len(df) or marker publication. This additional source file is explicitly required by kickoff-2 C.
- Added four tests in `tests/test_refresh_memo.py`: real spring-only build; repeated None builds with an existing parquet; historical parquet loading through real features; real preview CLI with downstream model boundaries stubbed and no `Failed:` line.
- RED: all four new refresh/pipeline/preview tests failed with the old len(None) path.
- GREEN: all 9 refresh tests pass, including the five existing memoization controls. The historical tests load real 2025 and 2026 fixture parquets and execute real `compute_all_features`; training, lookups and serving are stubbed.
- Source mutations, all killed and restored (test names are in `tests/test_refresh_memo.py`):
  - `C-no-none-guard`: remove None branch → `test_no_regular_games_yet_returns_without_marker_and_retries`.
  - `C-no-return`: retain log but fall through to len(None) → `test_none_build_return_does_not_memoize_even_with_existing_parquet`.
  - `C-memoize-none`: publish a marker in None branch → the same existing-parquet test (next call incorrectly skips).
  - `C-no-stderr-line`: send new line to stdout → `test_no_regular_games_yet_returns_without_marker_and_retries`.
  - `C-bypass-historical-loading`: return an empty frame after refresh → `test_pipeline_loads_historical_parquets_and_computes_features_without_current_season`.

### C caller trace (source inspection plus synthetic execution)
- `build_season` has two production callers on these paths: CLI `build` already handles None; `_refresh_season_data` now handles it. No other direct build-return consumer found on the prediction path.
- `run_pipeline`: after refresh, reads every existing sorted `pa_*.parquet`, concatenates and computes features. It requires historical parquet(s), not a file for the prediction year; no parquet at all still raises the existing error.
- `_build_feature_lookups`, `train_model`, `train_blend`, `predict`: consume the combined historical frame, not a current-year parquet. Training filters by the historical minimum training year, not the prediction year. Serving's long-rest flag does not require current-year PA.
- `orchestrator.predict_local`: calls that pipeline; absent dated blend cache causes training from historical frames. Its optional calibration block checks `current_pa.exists()` and logs `No pa_2027.parquet; calibration skipped` when absent. Verified with `BTS_USE_CALIBRATION=1`; no change needed.
- `run_cascade`/`run_and_pick`: local tier delegates to predict_local; returned predictions proceed to slate/state/selection without reading a current-year parquet. Synthetic trace reached selection with one prediction and tier `local`; no further fix needed.
- `bts preview`: loads a dated blend if present, otherwise calls the pipeline and then selects from returned predictions. No independent current-year parquet read. Synthetic CLI reached selection, exit 0, no `Failed:` line.
- SSH's `predict-json` worker and CLI `predict` use the same pipeline, with no separate current-year parquet requirement; synthetic predict-json emitted a JSON prediction. Shadow local prediction uses the same historical glob with refresh disabled after production refresh.
- `trace-2.py` / `trace-2.log`: all four executed callers (predict_local, run_and_pick, preview, predict-json) loaded historical frames and ran real features without a 2027 parquet or refresh marker. Model training/serving, selection and network leaves were stubbed; this is not proof of model quality or live entry.
- Independent existing limit: `_fetch_prior_lineup` returns no projected fallback before any current-year Final game; a posted lineup remains available through the live feed. This is not a parquet/read failure; not changed.

## D — one yesterday check, and the receipt limit
- Removed the 01:30 day-outcome line from `scripts/cron-setup-hetzner.sh`; retained the existing 08:30 line and every argument, including `$YESTERDAY`. Updated timing comment and `tests/test_entry_intent.py` to require exactly one check under either intent.
- RED: both `test_the_day_outcome_check_is_installed_under_either_intent` cases failed before line removal. GREEN: all 48 entry-intent tests pass. No D mutation requested.
- Added `tests/test_day_outcome_entry_sequence.py::test_limit_later_wrong_game_binding_is_not_reobserved_after_batter_only_confirmation`. Existing sequence was factored into a helper with a configurable later feed ID.
- It verifies one actual fetch, followed by two already_confirmed receipts; later units bind game 9002 while selection names 9001; fresh qualification would say wrong_game, but the sealed observation remains unit_unverified and day-outcome remains entered_game_unverified. Both sequence tests pass. F3(a), F3(b) and F4 product code unchanged.

## Final verification and remaining work
- Mutation ledger `mutations-2.log`: 13 killed; individual `mutant-2-*.log` files contain named failure output. Script verified source restoration before the final focused run.
- Command (exit 0), `focused-2.log`:
`TZ=America/New_York .venv/bin/python -m pytest tests/test_day_outcome.py tests/test_entry_intent.py tests/test_thin_slate_checks.py tests/test_late_delivery_guard.py tests/test_postponed_delivery_guard.py tests/test_incident_register_2026.py tests/test_scheduler.py tests/health tests/test_refresh_memo.py tests/test_data_build_empty.py tests/test_day_outcome_entry_sequence.py tests/test_build_schema_assert.py tests/data/test_build.py -q -p no:cacheprovider`
- Last line: `============================= 765 passed in 56.50s =============================`
- `git diff --check` passes. No requested item unfinished. Lead review/commit/landing remain outside this author task; no production execution or real model training was performed.
