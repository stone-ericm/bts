# F1 round 2 handover

Commit `15b2747431b378f3e2883ad98a5af5890acd0ea2` on branch `fix/wr-f1`. The handover is outside the commit.

## Tests first

The working tree's source and tests before the test edits matched `1d30de86`. The selected round-two test run reported 14 failures, 69 deselected. The source edits followed that run.

| Item | Test initially failing on `1d30de86` |
| --- | --- |
| R1 and adopted reviewer regression | `test_failed_observation` exercised all three intermediate results. The final ambiguous binding yielded exit 0 and terminal confirmation after each bridge. |
| M11 | `test_failed_observation[<bridge>-M11]` failed for crosswalk, removal and mismatch after a delivered initial alert. |
| M12 | `test_failed_observation[<bridge>-M12]` failed for the same three bridges with no previous DM. |
| M13 | `test_m13_partial_double_proof` saw exit 0 with only one leg proved. |
| R2 locked-delivery names | `test_locked_delivery_names_fallback` saw `None + DD Batter 1020`. |
| R2 unavailable game ID | `test_unavailable_recommended_game_id` saw `recommended game None`. |
| R2 cutoff copy | `test_m7_cutoff_passes_during_fetch` saw `during fetch`. |

`test_legacy_observation_does_not_assume_initial` pins all three prior-marker signs. `test_ordered_pairs_and_triples_after_observation` walks 144 ordered pairs and 1,728 ordered triples across the six verifier results and both DM outcomes. It checks the marker and attempted DMs before the first fully proved result. Both tests failed before the source edits.

The round-one F1 file changed only in `test_m7_cutoff_passes_during_fetch`. It expects the new cutoff text and checks the unresolved marker field. A6.2, A6.4 and A6.5 are contract clarifications per the ruling. They required no code or red tests. The pinned N11 health and morning scenario passed.

## Scratch changes

Each change was applied in isolation to a scratch copy of the package and then restored.

| Scratch change | Detected by |
| --- | --- |
| Omit the field from `alerted` | 7 failures across the reviewer regressions and M7, including cutoff marking. |
| Omit the field from `dm_failed` | All 3 M12 bridge cases failed. |
| Omit the field from `present_unverified` | M11 and M12 crosswalk cases falsely confirmed. |
| Confirm a double with any leg proved | M13 falsely confirmed and failed. |
| First wrong-game warning consumes `initial` | M2 and M8 failed. |

## Matched tests

Both F1 files passed, 83 tests. The other 60 matched files below passed in serial bounded batches, with 3,456 passes, 4 skips and 1 expected failure. The separate `tests/test_audit_driver.py` run reported 33 failures and 16 setup errors due to missing GNU rsync 3.x, consistent with the known base failure. The OS denied the `nice` priority adjustment. Every pytest run was bounded and serial.

### Batch 1 (41 passed)

- `tests/experiment/test_cli_screen_nested_split.py`
- `tests/experiment/test_cli_select_nested_split.py`
- `tests/experiment/test_cli_select_pooled.py`
- `tests/experiment/test_diagnostics.py`
- `tests/experiment/test_integration.py`
- `tests/experiment/test_run_selection_feature_cols.py`
- `tests/experiment/test_run_selection_keep_rule.py`
- `tests/experiment/test_run_selection_multi_seed.py`
- `tests/experiment/test_runner_nested_split.py`
- `tests/health/test_pick_entry_source.py`

### Batch 2 (187 passed)

- `tests/leaderboard/test_scraper_stealth.py`
- `tests/scripts/season_ledger/test_compile.py`
- `tests/scripts/season_ledger/test_matching.py`
- `tests/scripts/season_ledger/test_reconcile.py`
- `tests/scripts/season_ledger/test_rows.py`
- `tests/scripts/test_benchmark_bridge.py`
- `tests/scripts/test_field_products_builders.py`
- `tests/scripts/test_field_products_compare.py`
- `tests/scripts/test_field_products_ledger.py`
- `tests/scripts/test_field_products_run.py`

### Batch 3 (239 passed)

- `tests/scripts/test_mining87_production.py`
- `tests/scripts/test_mining87_run.py`
- `tests/scripts/test_mining87_surfaces.py`
- `tests/scripts/test_mining87_units.py`
- `tests/scripts/test_phase_d_pooled_policy_outer_eval.py`
- `tests/scripts/test_phase_d_pooled_policy_postmortem.py`
- `tests/scripts/test_readme_live_record.py`
- `tests/scripts/test_w13_tests.py`
- `tests/test_automatic_reading_cs4c.py`
- `tests/test_cli_integration.py`

### Batch 4 (542 passed, 1 xfailed)

- `tests/test_cold_start_integration.py`
- `tests/test_contest_fetch.py`
- `tests/test_day_outcome.py`
- `tests/test_day_outcome_entry_sequence.py`
- `tests/test_entry_intent.py`
- `tests/test_entry_receipt.py`
- `tests/test_incident_register_2026.py`
- `tests/test_live_gate_cs6a.py`
- `tests/test_pick_chosen_on_cs5a.py`
- `tests/test_pickpath_review_pp1.py`

### Batch 5 (1,850 passed, 4 skipped)

- `tests/test_postponed_delivery_guard.py`
- `tests/test_reconcile_receipt.py`
- `tests/test_saved_pick_cs5a_review.py`
- `tests/test_saved_pick_paths_cs5a.py`
- `tests/test_scheduler_decision_record.py`
- `tests/test_scheduler_decision_record_integration.py`
- `tests/test_scheduler_tail_objective.py`
- `tests/test_season_contract_cs4b2.py`
- `tests/test_season_contract_cs4b2_edges.py`
- `tests/test_stale_delivery_cs4c_carbon.py`

### Batch 6 (597 passed)

- `tests/test_strategy.py`
- `tests/test_streak_season_cs2.py`
- `tests/test_tail_policy_r3_fixes.py`
- `tests/test_tail_policy_strategy.py`
- `tests/test_wr_a_pick.py`
- `tests/test_wr_b_contract.py`
- `tests/test_wr_b_edges.py`
- `tests/test_wr_c2_carbon_v2.py`
- `tests/test_wr_c2_r2_carbon.py`
- `tests/test_wr_h_longest.py`
