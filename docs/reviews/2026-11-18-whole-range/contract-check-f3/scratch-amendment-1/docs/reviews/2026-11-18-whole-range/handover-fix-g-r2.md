# Change set G, Round 2

Commit `cfc62745` on `fix/wr-g`.

## Tests first

1. `test_real_commit_replaces_prior_decision` failed on all six cases. Confirmed delivery retained the unconfirmed or private status in two cases. Three replacement paths retained the previous batter. The undecodable record left the commit flag unset. `test_restart_after_decision_before_flag_save` still checks byte preservation through classification and end-of-day skip.
2. `test_guarded_delivery_records_selected_batter[warmup]` failed because the DM named batter 100 and the guarded record's primary was null. Fresh DM and public delivery retained their captured candidate fields. A refresh without a decision now records the saved slots for unguarded picks too.
3. Both `test_malformed_donor_cells` cases raised `IndexError` on the stored donor code 3. `test_malformed_queried_cell` verifies the queried-cell exception still agrees with raw lookup.
4. `test_guard_journal_saver_index` failed for the string `"0"`, which logged saver available. The cases cover both Boolean and integer values and both numeric strings.
5. Both `test_invalid_guard_keeps_raw_parity_check` cases failed to halt. They cover the impossible guard with bin 999 and an in-range guard on a raw single.
6. `test_tail_ignores_reach57_donor`, the available-saver case of `test_guard_journal_saver_index`, and `test_guarded_delivery_records_selected_batter[public]` passed on the original head. Each failed on its own scratch change, respectively applying the guard in tail, always printing saver unavailable, and dropping the guard from a public commit. All three changes were undone. The round-trip test now names the skip predicate it exercises.
7. The original `test_reserved_saver_changes_the_real_policy_action` failed because the guard made the no-saver action double. Its Amendment 3 fixture uses streak 9 on 2027-08-20, 38 policy days left, probability .79 in bin 0, and the `not_earned` flag. Both saver states are outside the CSV. The actions are skip without the saver and single with it.

## Verification

- Focused tests and all commit/delivery caller files yielded 1,421 passes, 12 skips, and one expected failure across four runs.
- The unchanged lookup grid checked 314,940 inputs per lookup, with 4,533 guarded changes across exactly 1,511 cells. The resolver grid checked 327,120 inputs per side and the direct-bar grid checked 21,808 states per side.
- The season walk exited 1 as expected. All 4,599 records and boundary results were compared. There were 252 skip-to-single changes, including 229 formerly flagged in-season records and 23 preseason records, 612 in-season all-skip warnings, and zero other differences. The fresh JSON matched the decompressed saved bytes. The transcript matched after accounting for the saved priority warning and appended comparison summary, so both saved artifacts were retained.
- All seven items were built and verified under Amendments 2 and 3.
