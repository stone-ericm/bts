# 2026 framing test: test-first ledger for code review c1's and c2's findings

Written by the lead on 2026-10-09, under the manager's ruling that test-first means evidence:
- **One line per finding.** Each line gives the failing tests, the RED run (its command and the failing assertions) and the GREEN run after the fix.
- **Raw outputs.** They are in `tdd-round2/`. Pytest's temporary paths are shortened to `<pytest-tmp>`.
- **The source.** Code review c1 is `report-c1.md`, sha256 `58a8dc00e16e9d1af9139629cea25dc50887d83bd89a06f3ae75bc064617ebf4`, a BLOCK on `4fa1ba4`.
- **The fixes.** The fixes for c1's findings and their tests are commit `bdab58b` (corrected after review c2, which found that this line had attributed the tests to `9507294`). `9507294` adds the revised mutation check and two more tests.

**How to read a RED.** Several RED runs fail with "no attribute" or "unexpected keyword". That shows the rule's function did not exist yet, which is weaker than a failing assertion. For each such rule, a mutant breaks that one span and the same tests fail. The mutants are named on each line. The evidence is the final ledgers (`mutants-1af3b95.tsv`, `mutants-bc56731.tsv` and this round's). The first round-2 ledger, `mutants-9507294.tsv`, had ten survivors and is not that evidence for every rule (corrected after review c2). Review c2 also found that some kills fail on a message rather than showing the invalid input accepted; the labels now say so where it applies (M28, M52).

| Finding | Failing tests | RED (before the fix) | GREEN (after) | Mutants |
|---|---|---|---|---|
| 1. Strict prediction covers only the day-level blend | `test_f26_hook.py`: `…non_finite_reliever_score`, `…score_series_missing_a_row`, `…reliever_score_of_the_wrong_length` | 3 failed: `DID NOT RAISE ValueError` (`F1-red.txt`) | 3 passed (`F1-green.txt`) | M25–M28 |
| 2. Validation certifies only internal consistency | `test_f26_run.py`: `test_validation_refuses_coherently_wrong_profiles` (5 cases: labels, game ids, batter ids, basis, an 11th row) | part of 14 failed: `no attribute 'Trusted'` (`F2F3-red.txt`) | part of 15 passed (`F2F3-green.txt`) | M29–M33 |
| 3. The A-posted coverage gate trusts its own count | `test_f26_run.py`: `…count_says_otherwise`, `test_validation_rederives_the_catcher_evidence` (6 cases), `…recorded_counts_and_ids…`, `test_the_aggregate_recomputes_every_catcher_value` | part of 14 failed: `no attribute 'Trusted'`; the aggregate case: the evidence file did not exist (`F2F3-red.txt`) | part of 15 passed. In this round (c2's R2-2) the aggregate test became `…recomputes_every_catcher_value_even_against_a_forged_expectation`, and the single-run case is now refused by validation (`…single_runs_tampered_value…`) | M34–M39 (M40 retired: each run must equal the pinned expectation, M73) |
| 4. The 2019 history start is not enforced | `test_f26_rules.py`: `…catcher_id_before_2019_refuses`, `…pre_2019_rows_without_a_catcher_id_are_fine`; `test_f26_run.py`: `…refuses_pre_2019_catcher_history_before_any_walk_forward` | 3 failed: 2 `no attribute 'history_start_problem'`, 1 `DID NOT RAISE InputRefused` (`F4-red.txt`) | 3 passed. After adding a STOPPED record for the refusal, the run test passed again (`F4-green.txt`, second line) | M41 |
| 5. The pre-read gate does not bind the source inventory | `test_f26_run.py`: `…preparation_function_itself_refuses…` (2), `…refuses_any_path_but_the_declared_ones` (3, replaced in this round by `…but_the_inventorys`), `…gated_preparation_reads_the_declared_sources`, `…historical_pins_must_be_the_screens`, `…source_manifest_must_be_exactly_the_pinned_games` (8), `…table_must_cover_exactly_the_pinned_games` | 16 failed: `no attribute 'DATA_DIR'` (6), `'sources_problem'` (8), `'historical_pins_problem'`, `'table_games_problem'` (`F5-red.txt`) | 16 passed (`F5-green.txt`) | M42–M46 |
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

The aggregate's CPU is spent outside the launcher, as are the preparation's and the expect step's. Review c2 found that preparation did not report its CPU, though an earlier version of this ledger said it did. From this round, `prepare` and `expect` report `cpu_s` (`test_the_preparation_reports_its_own_cpu`, `…expect_step…`), and the aggregate reports `aggregate_cpu_s`. They are charged to the allowance, as design §7 requires. Review f1 found that an earlier version of this line said they were not counted, following a manager ruling of 10/09 that the manager has since withdrawn as an erratum. The manager records these costs by hand into the effective total before any allowance or launch.

**Two tests changed while the fixes went in.** Neither weakens a check:
- `test_arm_summary_…` assumed scalar streak values. A real `diff_scorecards` stores `{baseline, variant, delta}`, so the test now uses that shape, and a new test runs `arm_summary` on a real diff.
- Two older damage tests now accept the earlier, stricter refusal message: "label" before "scorecard", and "counts" before "identified".

**Not covered by any test or mutant.**
- The real twelve-model walk-forward and the real launcher are still stubbed in the run tests (review c1, finding 11). Only an admitted real run exercises them.
- Corrected after review c2: an earlier version said the `>=` versus `>` mutant on the +0.3pp threshold was not meaningful. Review c2 built a legitimate {-1, 0, 1} matrix whose mean is exactly 0.003, so the boundary is attainable. It now has a test (`…a_mean_of_exactly_the_practical_threshold_is_positive`) and a mutant (M66).
- The stale-bytecode account of the first round-2 run's M57 and M59 survivals is a plausible diagnosis, not a measured cause: no cache bytes or mtimes were kept. Review c2 also showed that a stale cache can fake a kill, as well as hide one, so the `1af3b95` commit message's "never fake a kill" is too strong. The runner purges caches before every mutant run and writes no bytecode, which closes the mechanism review c2 demonstrated.

## Code review c2's findings (round 3 of fixes; review c2 sha256 `6613515c2aba5c828d8910a2b4f8491f5456b3456687d529e154d5758dc583d0`, BLOCK on `d13eae8`)

| Finding | Failing tests | RED (before the fix) | GREEN (after) | Mutants |
|---|---|---|---|---|
| R2-1. The pre-read gate does not check X-37's source inventory | `test_f26_run.py`: `…source_inventory_is_read_at_the_exposure_commit`, `…source_inventory_must_be_cited_published_and_exact` (6: wrong sha256, not published, published before X-37, extraction, selection, extra field), `…x37_row_without_a_citation…`, `…refuses_any_path_but_the_inventorys` (3), `…refuses_without_an_inventory`, `…reports_its_own_cpu` | part of 15 failed: `no attribute 'SELECTION'` (10), `'source_inventory'` (2), `KeyError: 'cpu_s'` (`R2-red.txt`) | part of 15 passed (`R2-green.txt`) | M43, M68–M72 |
| R2-2. Coverage is self-certified at the next-seed gate | `test_f26_run.py`: `…a_fabricated_catcher_value_is_refused_by_validation_and_the_next_seed_gate`, `…unavailable_coverage_cannot_be_faked_into_identified` (the review's case: every rate missing, values faked to 0.5 and counts rebuilt in both records) | 2 failed: `DID NOT RAISE RunInvalid`, i.e. the fabrication passed validation (`R2-red.txt`) | 2 passed (`R2-green.txt`) | M73, M74 |

**How the R2 fixes work.**
- **R2-1:** X-37's description must cite `docs/audit/c2-framing-2026-source-inventory.json` and its sha256. `source_inventory` checks that the file was published in the exposure commit itself and has exactly the declared directories, the registered selection rule and this code's extraction fields. The admission gate requires it in every mode. `prepare` refuses any directory that is not the inventory's.
- **R2-2:** a one-time `expect` step, run off-launcher after preparation, computes the expected per-side-game catcher values from the pinned inputs. It writes `expected_catcher_evidence.2026.parquet`, which is pinned as an input and bound by the inputs row. Every validation compares each retained value with it exactly and takes the reason from the expected value, so the next-seed gate no longer trusts a run's own values. The aggregate still recomputes from the admitted inputs, and its recomputation now guards the pinned expectation itself (`…forged_expectation`: a coordinated forgery of the expectation and all ten runs is accepted by validation and refused only by the recomputation).

**Review c2's nonblocking items.**
- **The disagreement flag.** It is now computed in `dispose` and covers both directions. Its test, `…dependence_disagreement_flag_covers_both_directions`, uses the review's 10-by-42 matrix, where the iid bound is below zero and the block bound above, and does not repeat the expression.
  - RED: `KeyError: 'dependence_disagreement'` (`NB-red.txt`).
  - GREEN: passed (`NB-green.txt`).
  - Mutant: M67.
- **Four guard tests passed on their first run,** because they guard checks that already existed. Their failing direction is their mutant:
  - `…a_mean_of_exactly_the_practical_threshold_is_positive` (M66);
  - `…a_side_game_moved_to_another_date_is_refused` (M35; it isolates the side-game set check, which the counts check used to shadow);
  - `…a_table_with_two_away_records_and_no_home_refuses` (M55; with that check removed the table is accepted, not refused downstream);
  - `…the_expect_step_writes_the_evidence_every_run_must_carry` (M74). This test was written after `expect` itself, which was implemented with R2-2's red tests above.
- **Mutant labels.**
  - M28 is relabelled as a defense-in-depth check: with it removed, a short score array still fails downstream in pandas, so its kill is by message.
  - M52's label notes that its bench-record case is genuinely accepted under it.
  - M64 is relabelled: it changes the block generator's seed.
  - M40 is retired (see finding 3 above).
