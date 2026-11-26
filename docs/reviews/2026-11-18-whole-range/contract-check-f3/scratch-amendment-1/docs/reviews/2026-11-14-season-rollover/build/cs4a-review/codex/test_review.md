# Per-test audit

The 2027 and 2028 frozen runs agree: branch 690 passed; base 680 failed and 10 passed. Each test calls production code. No required behavior is asserted only through a mock replacing its implementation. The existing file does not call the policy resolver or scheduler; independent review probes cover both.

Twenty-two functions have failures against the base. The three wholly GREEN functions comprise five cases; runtime mutations of the actual parser/writer made all five fail (`control-mutations.log`). The other five GREEN cases are ordinary base Click rejections and are explicitly named controls, not new-behavior evidence.

| Test function | Base passed / failed | Assertions / raises contexts | Review |
|---|---:|---:|---|
| `test_write_and_noop_order_output_and_isolation` | 0 / 3 | 12 / 0 | Real CLI and files; wraps append and the utility imported by production; checks exact identity, order, isolation, and a write-free no-op. |
| `test_rejections_before_reads_and_writes` | 4 / 20 | 5 / 0 | Real CLI; wraps Path.read_text, Path.open and builtins.open. Four source/source-date nodes are named GREEN controls. |
| `test_missing_options_controls` | 1 / 2 | 2 / 0 | Real CLI with an actual Path.read_text tripwire. Saver without date is already rejected by the base. |
| `test_unpadded_date_and_reason_only_correction` | 0 / 1 | 6 / 0 | Actual date parsing, three distinct request-log lines and next-day rejection. |
| `test_ordered_invalid_dated_observation` | 0 / 34 | 2 / 1 | Real loader and validator, automatic-file tripwire, numbered errors on two dates. |
| `test_missing_required_properties` | 0 / 16 | 0 / 1 | Real loader must raise the specified numbered ContestStateError; pytest.raises is the assertion. |
| `test_first_failure_and_ignored_keys_and_offset` | 0 / 1 | 3 / 1 | Real parser; paired failure priority, ignored values and equivalent UTC expiry. |
| `test_readability_errors_and_command_correction` | 0 / 5 | 3 / 1 | Real file/command behavior with a read fault for permission simulation; directory failure is real. |
| `test_out_of_date_is_absent_even_with_legacy_expiry` | 0 / 4 | 2 / 0 | Real selection and health; no fallback to other-date v3. |
| `test_decision_observation_is_authority_and_isolated` | 0 / 480 | 3 / 0 | Real loader; cs.load_saver_state and cs.contest_state_is_fresh are the actual call-site aliases, patched as tripwires. |
| `test_midnight_dst_year_boundary_and_clock_rollback` | 0 / 5 | 3 / 0 | Real decision loader at each boundary and after rollback. |
| `test_one_captured_clock_across_midnight` | 0 / 4 | 6 / 0 | Clock replaces datetime module and imported production aliases; real reader/decision/health/CLI operations. |
| `test_health_observation_saver_without_flag_read` | 0 / 84 | 4 / 0 | Real check; flags.load_saver_state is exactly the source of health's function-local import. |
| `test_warning_commands_pasted_into_real_cli` | 0 / 1 | 6 / 0 | Real warning and CLI; two concrete corrections executed. Third is form-only here; independent delivery_probe executes it with N/B supplied. |
| `test_failed_append_or_replacement_preserves_old_authority_and_retry` | 0 / 5 | 10 / 0 | Real CLI with injected I/O failures. util.atomic_write_text is imported locally by the production writer, so its spy intercepts the real call. |
| `test_corrections_replace_existing_kind` | 0 / 4 | 2 / 0 | Real files and CLI replacement, including invalid dated contents. |
| `test_three_day_intermediate_sequence` | 0 / 1 | 4 / 0 | Real CLI/loader; intentionally pins next-day automatic fallback. |
| `test_deterministic_two_writer_interleavings` | 0 / 5 | 6 / 0 | Nested real invocations at read/append/atomic-write/output boundaries; spies wrap the real operations. |
| `test_interruption_retry_boundaries` | 0 / 2 | 5 / 0 | Actual append/replacement with KeyboardInterrupt at the intended boundaries; real retry and log counts. |
| `test_legacy_fixed_clock_bytes_and_output_control` | 1 / 0 | 5 / 0 | Exact bytes/output preservation. Deliberately mutating the production writer makes this GREEN control fail. |
| `test_wrong_schema_remains_lax_legacy_control` | 3 / 0 | 4 / 0 | Exact legacy field assertions. Deliberately changing real parser results makes all three GREEN nodes fail. |
| `test_automatic_v3_schema_still_uses_automatic_parser_control` | 1 / 0 | 4 / 1 | Real automatic parsing and undated refusal. The same parser mutation makes this GREEN node fail. |
| `test_noop_compares_valid_raw_snapshot_identity_only` | 0 / 1 | 2 / 0 | Real CLI/validator; actual selection-loader and atomic-writer aliases are forbidden. Snapshot and unchanged output are asserted. |
| `test_legacy_option_defaults_are_not_explicit_commandline_options` | 0 / 1 | 2 / 0 | Real Click default_map and stored observation prove COMMANDLINE provenance handling. |
| `test_shared_date_predicate_is_used_by_all_four_operations` | 0 / 1 | 2 / 0 | Wraps actual cs and health predicate aliases; command imports that same cs function locally. |

The four rejection GREEN nodes are `test_rejections_before_reads_and_writes` with source/source-date, existing state both ways: base exits 2 for unknown --date and its error contains those option hints. Saver-without-date exits 2 for unknown --saver. Legacy bytes, the three wrong-schema cases, and the automatic-v3-parser case preserve behavior already present on the base.

Spies suppress only forbidden reads or inject the stated I/O failure; ordinary positive paths use the real validator, readers, saver mapping, atomic writer and Click command. The date predicate wrapper checks delegation as well as behavior. The test's corruption/permission fixtures are correctly distinguished from expiry.

`tests_base2027.json`, `tests_base2028.json`, `tests_branch2027.json`, and `tests_branch2028.json` retain per-function outcomes and exact GREEN node IDs. `control_mutations.json` retains the deliberate regression results.
