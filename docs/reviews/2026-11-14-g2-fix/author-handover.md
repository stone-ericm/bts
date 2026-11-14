# G2 four-fix author handover

Worktree `/Users/eric/projects/bts-pp-c1`, branch `g2-fix`. No commit, push, stash, branch change, agent launch, external-account access or deployment.
Only the allowed product/test files changed; supporting logs and this handover are in `.codex-review/g2-fix/`.

## Evidence
- Existing tests before source changes: `107 passed in 44.55s`; data-build baseline (`build-baseline.log`): `25 passed in 1.85s`.
- Initial RED (`red.log`): `22 failed, 118 passed in 46.82s`.
- Initial GREEN (`green.log`), all four fixes and data-build tests: `169 passed in 45.59s`.
- Additional precedence RED (`red-late-qualification.log`): `1 failed, 55 passed in 1.85s`; GREEN (`green-late-qualification.log`): `56 passed in 1.51s`.
- Additional deduplication control (`green-dedup.log`): `1 passed in 1.65s`.
- All tests named below pass in the final focused run. Every mutant was applied to the source, run separately, killed by an assertion failure, and restored in `finally`.
- Exact mutant commands, killing test names and failure summaries: `mutations.log` (29), `mutations-extra.log` (1), and `mutant-*.log`.

## F1 — pre-season build no-op
- Changed `src/bts/data/build.py::build_season`: return `None` before schema validation/output creation only when no eligible feed survives the existing game-type/7-inning filters. Missing directories naturally have no feeds. A regular-season feed with no valid rows still raises the existing schema error.
- Changed `src/bts/cli.py::build`: print one stderr line for the no-op and continue to the next requested season. The 03:00 cron chain is untouched.
- Added `tests/test_data_build_empty.py`: missing/empty/non-regular/excluded-doubleheader directories; existing parquet bytes and mtime; available season alone and before/after an empty season; genuine schema error stays nonzero.
- RED: `FAILED ...::test_cli_empty_season_is_success_without_writes[missing]` (also empty/non_regular/excluded_doubleheader), preservation and both mixed-season orders; 7 failures in the initial RED run.
- GREEN: all 9 CLI tests pass (`green.log`, final `focused.log`); existing data-build/schema tests also pass.
- Mutants and killers (names below are in `tests/test_data_build_empty.py`):
  - `F1-missing`: restore missing-directory exception → `test_cli_empty_season_is_success_without_writes[missing]`.
  - `F1-empty`: remove eligible-feed no-op → `test_cli_empty_season_is_success_without_writes` (4 failures).
  - `F1-overwrite`: overwrite parquet during no-op → `test_cli_empty_season_preserves_existing_parquet`.
  - `F1-swallow-error`: no-op on zero rows despite eligible feeds → `test_cli_regular_season_build_error_still_fails`.
  - `F1-warning`: send warning to stdout → `test_cli_empty_season_is_success_without_writes`.
  - `F1-stop-other-seasons`: return instead of continue → `test_cli_builds_available_seasons_even_beside_an_empty_one[2027,2026]`.

## F2 — cluster the combined thin-slate plan
- Changed only `src/bts/scheduler.py::plan_run_times`: put existing regular runs into the early-check clustering pass; retain the earlier time and union game IDs without duplicates.
- Added four tests in `tests/test_thin_slate_checks.py`; existing expectations were not changed. Grid: two offsets × 49 separations, asserting gaps greater than 10 minutes and unique IDs.
- RED: `FAILED ...::test_regular_and_early_checks_at_the_same_minute_merge`, `test_two_game_grid_has_no_adjacent_runs_inside_cluster_window`, and `test_early_check_near_regular_check_uses_earlier_time_and_union`.
- GREEN: 17 thin-slate tests pass in `focused.log`; original singleton, normal-slate and late-start controls remain green.
- Mutants and killers (same test file):
  - `F2-separate-plans`: restore independent plans → `test_regular_and_early_checks_at_the_same_minute_merge`.
  - `F2-later-time`: move merged run to later time → `test_early_check_near_regular_check_uses_earlier_time_and_union`.
  - `F2-no-union`: discard merging game ID → `test_regular_and_early_checks_at_the_same_minute_merge`.
  - `F2-duplicate`: append duplicate IDs → `test_merged_checks_list_each_game_once`.

## F3 — ordinary day outcomes and yesterday's cron checks
- Changed `src/bts/day_outcome.py::OK_STATES`, `_entry_state`, `classify_day`, `gather`: add clean `entered_game_unverified`, using existing receipt binding plus an in-time legacy match and nonempty slots containing only confirmed/unit_unverified; detail explicitly says the game was not verified. An in-time fully qualified entry still wins. A later game confirmation does not erase an in-time batter match.
- `gather` counts void statuses with `picks._is_void_detailed_state`; a positive schedule consisting entirely of void games is `no_games`, regardless of decision. Zero-game commit contradictions and partly void slates retain their behavior.
- Changed `scripts/cron-setup-hetzner.sh`: both checks use yesterday, at 01:30 and 08:30; timing comments updated. Updated the cron test in `tests/test_entry_intent.py`.
- Added classifier/real-file command tests in `tests/test_day_outcome.py` and `tests/test_day_outcome_entry_sequence.py`. The latter adapts Gemini's actual check-pick-entered sequence: unbound feedId, observed match, refreshed units, two already_confirmed receipts, then the real day-outcome CLI. Auth/DM/network leaves are mocked; receipt production/discovery/sealing are real.
- RED: `FAILED ...::test_batter_match_with_only_unverified_games_is_clean`, `test_all_void_games_are_no_games_even_with_a_decision`, both command cases and the real receipt sequence; both cron intent cases failed. The extra precedence test also failed before its ordering fix (see Evidence).
- GREEN: all 56 classifier/command tests, the receipt-sequence test and 48 entry-intent tests pass in `focused.log`.
- Mutants and killers (`tests/test_day_outcome.py` unless qualified):
  - `F3-clean-state`: remove clean-state membership → `test_batter_match_with_only_unverified_games_is_clean`.
  - `F3-no-batter-match`: disable new entry state → `tests/test_day_outcome_entry_sequence.py::test_batter_only_observation_then_already_confirmed_is_clean`.
  - `F3-full-priority`: bypass fully confirmed entry → `test_fully_confirmed_entry_wins_over_batter_only_match`.
  - `F3-any-slot`: accept any allowed slot instead of every slot → `test_batter_match_with_other_qualification_states_stays_unverifiable` (all 7 excluded states fail).
  - `F3-no-cutoff`, `F3-no-match`, `F3-no-date`, `F3-no-sha`, `F3-no-account`, `F3-degraded`: remove each corresponding guard → `test_batter_only_match_requires_every_entry_binding_and_cutoff` (the corresponding parameter fails).
  - `F3-no-observed`, `F3-no-season`: remove each guard → `test_batter_only_match_requires_same_season_and_observed_outcome`.
  - `F3-null-sha`: permit absent decision hash → `test_batter_only_match_cannot_bind_without_a_decision_hash`.
  - `F3-no-void`: disable all-void classification → `test_all_void_games_are_no_games_even_with_a_decision`.
  - `F3-any-void`: accept partly void slate → `test_partly_void_slate_still_needs_a_decision`.
  - `F3-zero-void`: treat zero games as all-void → `test_a_commit_on_a_day_with_no_games_is_a_contradiction`.
  - `F3-no-status-count`: omit status counting → `test_command_all_postponed_or_cancelled_is_clean`.
  - `F3-cron-timing`: restore 23:50 → `tests/test_entry_intent.py::test_the_day_outcome_check_is_installed_under_either_intent`.
  - `F3-cron-date`: retain 01:30 but check today → the same cron test (both intents fail).

## F4 — no refused batter in void-game health wording
- Changed `src/bts/health/late_delivery.py::check`: game_not_playable message contains label and games, without batter; late-cutoff branch unchanged.
- Added `tests/health/test_late_delivery.py::test_void_game_refusal_names_games_and_label_without_batter`.
- RED: `FAILED ...::test_void_game_refusal_names_games_and_label_without_batter`.
- GREEN: all 12 late-delivery health tests pass in `focused.log`, including the existing late-cutoff batter wording control.
- `F4-batter-name`: restore refused batter text → the new test fails.

## Final focused run
Command (exit 0), log `focused.log`:
`TZ=America/New_York .venv/bin/python -m pytest tests/test_day_outcome.py tests/test_entry_intent.py tests/test_thin_slate_checks.py tests/test_late_delivery_guard.py tests/test_postponed_delivery_guard.py tests/test_incident_register_2026.py tests/test_scheduler.py tests/health tests/test_data_build_empty.py tests/test_day_outcome_entry_sequence.py tests/test_build_schema_assert.py tests/data/test_build.py -q -p no:cacheprovider`
Last line:
`============================= 740 passed in 56.19s =============================`
`git diff --check` passes. All 30 source mutants killed and restored before this run.

## Limits and remaining work
- No requested implementation contract left unfinished; no known failing check. Evidence uses synthetic feeds/receipts and mocked external services, not a live account or restored production box.
- B5 and `run_day` were not changed, as directed. Lead still needs the fresh review and landing; no review agent was launched by this author.
- `CLAUDE.md` still describes 23:50 and the old clean-state list; it is outside the edit allowlist. The allowed source and cron comments are updated. Receipt producer/qualification rules are unchanged: the new clean state explicitly does not claim game verification.
