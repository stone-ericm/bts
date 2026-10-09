# 2026 framing test: round-2 test-first ledger (code review c1's findings)

Written by the lead on 2026-10-09, under the manager's ruling that test-first means evidence:
- **One line per finding.** Each line gives the failing tests, the RED run (its command and the failing assertions) and the GREEN run after the fix.
- **Raw outputs.** They are in `tdd-round2/`. Pytest's temporary paths are shortened to `<pytest-tmp>`.
- **The source.** Code review c1 is `report-c1.md`, sha256 `58a8dc00e16e9d1af9139629cea25dc50887d83bd89a06f3ae75bc064617ebf4`, a BLOCK on `4fa1ba4`.
- **The fixes.** They are commit `bdab58b`. Its tests and the revised mutation check are commit `9507294`.

**How to read a RED.** Several RED runs fail with "no attribute" or "unexpected keyword". That shows the rule's function did not exist yet, which is weaker than a failing assertion. For each such rule, the mutation ledger at `9507294` shows the assertion-level failure: break that one span and the same tests fail. The mutants are named on each line.

| Finding | Failing tests | RED (before the fix) | GREEN (after) | Mutants |
|---|---|---|---|---|
| 1. Strict prediction covers only the day-level blend | `test_f26_hook.py`: `…non_finite_reliever_score`, `…score_series_missing_a_row`, `…reliever_score_of_the_wrong_length` | 3 failed: `DID NOT RAISE ValueError` (`F1-red.txt`) | 3 passed (`F1-green.txt`) | M25–M28 |
| 2. Validation certifies only internal consistency | `test_f26_run.py`: `test_validation_refuses_coherently_wrong_profiles` (5 cases: labels, game ids, batter ids, basis, an 11th row) | part of 14 failed: `no attribute 'Trusted'` (`F2F3-red.txt`) | part of 15 passed (`F2F3-green.txt`) | M29–M33 |
| 3. The A-posted coverage gate trusts its own count | `test_f26_run.py`: `…count_says_otherwise`, `test_validation_rederives_the_catcher_evidence` (6 cases), `…recorded_counts_and_ids…`, `test_the_aggregate_recomputes_every_catcher_value` | part of 14 failed: `no attribute 'Trusted'`; the aggregate case: the evidence file did not exist (`F2F3-red.txt`) | part of 15 passed. The aggregate test was then strengthened to edit all ten runs identically, so only the recomputation can refuse it, and `…catcher_evidence_differs` was added for the single-run case | M34–M40 |
| 4. The 2019 history start is not enforced | `test_f26_rules.py`: `…catcher_id_before_2019_refuses`, `…pre_2019_rows_without_a_catcher_id_are_fine`; `test_f26_run.py`: `…refuses_pre_2019_catcher_history_before_any_walk_forward` | 3 failed: 2 `no attribute 'history_start_problem'`, 1 `DID NOT RAISE InputRefused` (`F4-red.txt`) | 3 passed. After adding a STOPPED record for the refusal, the run test passed again (`F4-green.txt`, second line) | M41 |
| 5. The pre-read gate does not bind the source inventory | `test_f26_run.py`: `…preparation_function_itself_refuses…` (2), `…refuses_any_path_but_the_declared_ones` (3), `…gated_preparation_reads_the_declared_sources`, `…historical_pins_must_be_the_screens`, `…source_manifest_must_be_exactly_the_pinned_games` (8), `…table_must_cover_exactly_the_pinned_games` | 16 failed: `no attribute 'DATA_DIR'` (6), `'sources_problem'` (8), `'historical_pins_problem'`, `'table_games_problem'` (`F5-red.txt`) | 16 passed (`F5-green.txt`) | M42–M46 |
| 6. Accepted runs are not bound to clean launcher completion | `test_f26_run.py`: `…records_its_launcher_unit`, `test_validate_run_binds_the_run_to_its_clean_launcher_receipt` (10 cases), `…waits_for_the_earlier_seeds_clean_receipt`, `…launch_refuses_a_seed_that_already_has_a_run…` | 13 failed: 10 `unexpected keyword argument 'c1_dir'`, 1 `KeyError: 'launcher_unit'`, 2 `DID NOT RAISE SystemExit` (`F6-red.txt`) | 13 passed (`F6-green.txt`) | M47–M50 |
| 7. Malformed records are silently repaired | `test_f26_rules.py`: `…malformed_non_starting_record…` (5), `…well_formed_bench_and_pitcher_records…`, `…well_formed_table_is_accepted`, `…malformed_or_inconsistent_table_refuses` (19), `…pa_game_ids_must_be_exact_positive_integers` (5), `…returns_the_sorted_distinct_ids` | 29 failed, 3 passed: 5 `assert (102 is None)` (the side was still identified), 6 `no attribute 'pa_game_ids'`, 18 `DID NOT RAISE InputRefused`. The 3 that passed are the well-formed guards and the duplicate-side case, which the old duplicate check already refused (`F7-red.txt`) | 32 passed (`F7-green.txt`) | M51–M55 |

**One more check, added for finding 1, found while reading the real prediction path.** `_predict_lgbm_classifier`, used by all twelve blend models, scores NaN on a row whose features are all missing. Strict prediction would refuse that, but only hours into a walk-forward. So the run now stops before any walk-forward (STOPPED `all_features_missing`, code 5) if a 2026 row has every feature of some model missing. A's catcher column is counted as missing for this check.
- `weather_temp`, in every model, has no missing value in the pinned 2019-2025 files (`int64`, 0 nulls). 2026 is unverified until the preparation read.
- The test is `test_a_2026_row_with_every_model_feature_missing_stops_before_any_walk_forward`.
  - RED: `assert 0 == 5`. The run completed instead of stopping (`F1b-red.txt`). An earlier RED attempt failed only on the test's own setup, a broken self-check (`assert 4 == 5`), and was corrected before this RED.
  - GREEN: 1 passed (`F1b-green.txt`).
  - Mutant: M65.

**The nonblocking items (8–11)** are implemented in `bdab58b`:
- identified catcher ids are listed, not counted;
- the 2026 resumed-flag totals are reported;
- the block-resampling constant indicator and the dependence-disagreement flag are reported;
- the streak summary (seeds below zero) is reported;
- the aggregate's own CPU is reported.

These were implemented before their tests; the tests were added afterwards, in the commit that adds this ledger:
- `test_validation_reconciles_the_recorded_counts_and_ids_with_the_evidence` (the identified-id list);
- `test_the_manifest_reports_the_2026_resumed_flag_totals`;
- `test_the_block_resampling_constant_flag`;
- `test_the_aggregate_reports_its_indicators_streaks_and_own_cpu`.

The aggregate's CPU is spent outside the launcher, as is the preparation's. Both are reported for the manager to account by hand; they are not in the ledger.

**Two tests changed while the fixes went in.** Neither weakens a check:
- `test_arm_summary_…` assumed scalar streak values. A real `diff_scorecards` stores `{baseline, variant, delta}`, so the test now uses that shape, and a new test runs `arm_summary` on a real diff.
- Two older damage tests now accept the earlier, stricter refusal message: "label" before "scorecard", and "counts" before "identified".

**Not covered by any test or mutant.**
- The real twelve-model walk-forward and the real launcher are still stubbed in the run tests (review c1, finding 11). Only an admitted real run exercises them.
- The `>=` versus `>` mutant on the +0.3pp threshold differs only at exact float equality, so it is not a meaningful mutant.
