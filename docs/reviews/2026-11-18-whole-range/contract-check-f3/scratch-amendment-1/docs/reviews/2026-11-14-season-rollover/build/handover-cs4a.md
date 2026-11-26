# CS4a handover — 2026-11-17

**NOT deployable alone.** This implements only kickoff version 3's dated-observation slice. A day without a dated observation in effect still uses the base's automatic/legacy/model behavior. It is not the opening-stretch guard.

Branch: `sb-cs4a`. Base: `22e3ebf9f087f503a19ae450d711a6a20a4f5bc6`.
One implementation commit: `6fceaa3f7dab12e7669877ae89db889521f5d6f7` — Add contest-date manual observations with dated saver authority.
Its parent is exactly the specified base. Nothing was pushed. No network, box or contest-account access was used. This handover and everything under `.codex-review/` are uncommitted.

The final fast gate ran on that commit, with no source, test or helper edits during the run. Its failing set is exactly `/Users/eric/bts-runs/gate-failed.txt`: 37 failures and 16 errors, no missing or additional names. The clean base gate has exactly the same failing set.

## Contract implementation and evidence

All test names below refer to `tests/test_contest_dated_cs4a.py` unless another file is named. Its 25 test functions expand to 690 cases. No existing test file was changed, and no existing assertion was deleted or loosened.

| Row | Implementation (file and lines) | Tests that pin it |
| --- | --- | --- |
| 1 | `src/bts/cli.py:1512–1575`: legacy setter body, clocks, serialization, deprecated saver handling and auto-earn remain the base implementation. The new branch returns before this body. | `test_legacy_fixed_clock_bytes_and_output_control` checks exact manual and saver bytes and exact stdout at a fixed clock. The existing `tests/test_cli_integration.py` setter tests pass unchanged; all existing saver tests pass. |
| 2 | `src/bts/cli.py:1438–1441` declares Click DateTime/Choice; `1482–1510` validates before any application-state read. Explicit legacy arguments use `ctx.get_parameter_source(...) == ParameterSource.COMMANDLINE`, including the shared boolean parameter. One UTC clock at line 1495 supplies today ET and all downstream timestamps/output. | `test_rejections_before_reads_and_writes` covers each forbidden option, both boolean spellings, explicit defaults, dates and numeric bounds, with/without existing state, and spies on `read_text`, `Path.open` and `open`. `test_missing_options_controls`, `test_unpadded_date_and_reason_only_correction`, `test_legacy_option_defaults_are_not_explicit_commandline_options`, and the command case of `test_one_captured_clock_across_midnight`. |
| 3 | `src/bts/cli.py:1379–1394` constructs the observation; `1410–1424` appends the request, flushes/closes it, then calls `bts.util.atomic_write_text`. Separate OSError messages distinguish append failure from a request appended but observation NOT saved. No flag, automatic or fetch-status path is written. | `test_write_and_noop_order_output_and_isolation` spies on append and atomic write order and checks both payloads. `test_failed_append_or_replacement_preserves_old_authority_and_retry` covers opening, partial writing, flushing, closing and replacement failures beside an existing observation, plus sentinels for the flag, automatic file and actual fetch-status path. `test_readability_errors_and_command_correction` covers a directory replacement failure and its appended log. The interruption and interleaving tests cover retry/crash ordering and same-clock writes with different values. |
| 4 | `src/bts/cli.py:1396–1408` reads raw manual JSON, validates an exact-v3 snapshot with the shared validator, and compares only the five identity fields. Missing reason compares equal to an omitted option; explicit empty reason differs. No selection loader is used. | `test_write_and_noop_order_output_and_isolation` proves no append/atomic call on no-op; `test_noop_compares_valid_raw_snapshot_identity_only` checks valid matching snapshots with different provenance/timestamps and spies against selection-loader use. Reason-only correction, legacy/invalid/other-date replacement, unreadable/non-UTF-8/non-object replacement, and next-day rejection are covered by their named tests. |
| 5 | `src/bts/cli.py:1426–1434` reports the installed observation or the matching read snapshot, never a later reread. It prints reason (including absence/empty string) and calls the shared date predicate for the exact last line. | `test_write_and_noop_order_output_and_isolation` checks exact last lines for write and no-op. `test_deterministic_two_writer_interleavings` checks output for read/append/replace/output interleavings and a no-op whose output is followed by another writer. `test_interruption_retry_boundaries` checks retry before and after replacement. |
| 6 | `src/bts/contest_state.py:221–230` retains the CS1 readable/UTF-8/JSON/object checks and dispatches only exact-v3 MANUAL files. `95–158` contains the ordered validator and the separate aware-only timestamp grammar. All automatic and other-schema parsing below the dispatch remains unchanged. | `test_ordered_invalid_dated_observation`, `test_missing_required_properties`, `test_first_failure_and_ignored_keys_and_offset`, `test_readability_errors_and_command_correction`, wrong-schema controls, automatic-parser control, and the matrix. Invalid properties are tested on their own date and another date; an automatic-read spy proves invalid manual content is checked first. Unreadable/non-JSON/non-object inputs fail before a schema can be identified, using the existing CS1 error path. |
| 7 | `src/bts/contest_state.py:49–50` adds the optional fields; `153–158` maps dated data to state, including `source_date`, `override_expires_at`, and `saver_available=None`. `86–92` is the only observation-in-effect predicate. | `test_first_failure_and_ignored_keys_and_offset`, wrong-schema and automatic controls, `test_shared_date_predicate_is_used_by_all_four_operations`, and boundary tests. |
| 8 | `src/bts/contest_state.py:269–309` captures one clock, parses manual first, returns an in-effect dated observation before reading auto, and removes any other dated observation from fallback consideration. The legacy branch is unchanged. | `test_out_of_date_is_absent_even_with_legacy_expiry`, authority/isolation tests with absent/corrupt/refused auto, midnight tests, reader clock test, and all 3,120 matrix combinations. |
| 9 | `src/bts/contest_state.py:434–438` passes the captured clock down; `469–470` bypasses local freshness reads for dated state; `492–495` constructs the observation's `SaverState` and applies the existing policy mapping without reading the flag. `181–182` trusts best before any legacy expiry comparison. The validator proves the file's season equals `effective_date.year`, so the saver constructor uses exactly that season. | `test_decision_observation_is_authority_and_isolated`: both require modes, all three saver states, streaks 0/9/10/15/16, conflicting/missing/corrupt flags, bad/absent auto, corrupt model and a later settled miss. Spies prohibit both flag and local freshness reads. `test_midnight_dst_year_boundary_and_clock_rollback` checks start/end, adjacent microseconds and rollback on ordinary, March DST, November DST, Dec 31 and Jan 1 dates; best remains trusted at the final microsecond. Decision clock test and three-day sequence also pass. |
| 10 | `src/bts/health/contest_state.py:114–130` excludes exact-v3 manual files from the legacy/expired warning. Loader errors remain CRITICAL at lines 49–51. | Out-of-date and health saver tests; matrix health outcomes for expected both ways, including invalid and other-date observations. |
| 11 | `src/bts/health/contest_state.py:132–150` uses the selected dated saver and returns before the flag block. Only not_earned at 10–15 warns. It names the observation, season, streak and spent policy, supplies three observation corrections with the checked directory, and explains that changing the flag does not correct the observation. The third command uses N/B placeholders with explicit instructions to replace them with MLB's active/best counts. | `test_health_observation_saver_without_flag_read`: both expected settings, all saver states, streaks 0/9/10/12/15/16/57, absent/corrupt flags, and a flag-reader spy. `test_warning_commands_pasted_into_real_cli` pastes both concrete commands into the real CLI (directory contains spaces), checks decision saver changes and warning removal, and checks the third correction's form. |
| 12 | The commit changes only `src/bts/cli.py`, `src/bts/contest_state.py`, `src/bts/health/contest_state.py` and the new test file. The automatic parser below its new manual-only dispatch, fetch, saver module/commands, dashboard, decision record and scheduler source are unchanged. | Legacy/automatic controls, the base matrix's preservation comparisons, unchanged existing tests, and the committed-tree fast gate. Outputs may differ through selection/decision rows 8 and 9 as specified. |

## Base-versus-branch comparison

Frozen script: `.codex-review/cs4a/matrix.py`; recorded output: `matrix.log` (wrapper exit 0).
It runs `git archive 22e3ebf9 src` into `.codex-review/cs4a/base/`, never over tracked source, and launches base and branch workers with separate PYTHONPATH values. Fixtures and clocks are fixed at 2027-07-15 ET.

26 manual cases × 5 automatic cases × 2 require settings × 2 health expected settings × 6 flag cases = **3,120 paired combinations**. Each worker records reader, decision and health outcomes.

Manual cases include absence; legacy unexpired/expired/no-expiry; schema missing/null/unknown; dated yesterday/today/tomorrow; today/tomorrow with ignored far-future legacy expiry; each ordered property failure (checks 2–10); and check-1 malformed JSON, unreadable, non-UTF-8, non-object and directory cases. The automatic cases are absent, valid, undated, refused and corrupt. Flags are absent, not_earned, active, used, stale-season and corrupt.

| Group | Cases | Identical | Required differences | Rows |
| --- | ---: | ---: | ---: | --- |
| Legacy/absent/readability controls | 1,440 | 1,440 | 0 | 1, 6, 12 preservation |
| Dated today, including ignored legacy expiry | 240 | 0 | 240 | 8, 9, 10, 11 |
| Dated yesterday/tomorrow, including ignored legacy expiry | 360 | 96 | 264 | 8, 9, 10, 11 |
| Readable v3 property failures, checks 2–10 | 1,080 | 0 | 1,080 | 6 |
| Total | **3,120** | **1,536** | **1,584** | No unclassified difference |

Every individual differing combination is listed in `.codex-review/cs4a/matrix-summary.json`: its full key `(manual, auto, require, expected, flag)`, changed component(s), and required row(s). The complete before/after outcomes are `matrix-base.json` and `matrix-branch.json` in that directory.

The script does more than allow differences by category: legacy results must equal the base exactly; out-of-date observations must equal the absent-manual branch result; today's reader/decision must equal explicit expected values regardless of flag/auto; its health warning must contain the dated corrections; and every invalid v3 property must produce the numbered error/CRITICAL. All assertions passed.

## Recorded runs and RED classification

Every pytest run used `/Users/eric/bts-runs/yield_run.sh`, a log under `.codex-review/cs4a/`, TZ=America/New_York, PYTHONDONTWRITEBYTECODE=1, `.venv/bin/python -m pytest`, and `-p no:cacheprovider`. Logs were read. No pattern kill was used. Helper scripts stayed under scratch and were not edited after being submitted for execution.

| Run / log | Result |
| --- | --- |
| Initial sandbox related attempt, `baseline-related.log` first summary | 1 failed, 1,307 passed, 8 skipped (31.48s); socket-wiring sandbox failure. Superseded by the next run. |
| Base related suite, same log final summary | **1,308 passed, 8 skipped** (31.23s). The 36-file list is `baseline-files.txt`; these counts existed before creating the new test. |
| Initial overlapping full attempts, `baseline-fast.log` | 40 failed / 5,574 passed / 16 errors, and 43 failed / 5,571 passed / 16 errors; each had 19 skipped, 6 deselected, 1 xfailed. Discarded as gates; see below. |
| Clean base fast suite, `baseline-fast-clean.log` | **5,577 passed, 37 failed, 16 errors, 19 skipped, 6 deselected, 1 xfailed** (1559.02s). Exactly the expected 53 names; zero extras/missing. |
| New tests on unchanged base, `red.log` / `red.xml` | 681 failed, 9 passed (10.70s), including one erroneous new preservation-control expectation, corrected below. Production source was still unchanged. |
| Corrected preservation control on base, `red-controls.log` | 1 passed, 689 deselected (0.29s). Final classification: **680 behavioral RED cases, 10 GREEN controls**. |
| Implementation, `green.log` | **690 passed** (0.97s). |
| Base/branch matrix, `matrix.log` | 3,120 paired combinations; assertions pass, wrapper exit 0. |
| All related files plus new test, `after-related.log` | **1,998 passed, 8 skipped** (31.51s). Same 36 existing files plus the new file. |
| All new tests at fake wall date 2027-07-15, `fake-2027.log` | **690 passed** (0.82s). |
| All new tests at fake wall date 2028-07-15, `fake-2028.log` | **690 passed** (0.83s). Both use the frozen `fake_clock.py` plugin, importing pandas before replacing date/datetime. |
| Commit, `commit.log` | One local commit, only the four intended files; wrapper exit 0. |
| Final fast suite on committed `6fceaa3f`, `final-fast.log` | **6,267 passed, 37 failed, 16 errors, 19 skipped, 6 deselected, 1 xfailed** (1606.65s). Exactly the same expected 53 names; zero extras/missing. Wrapper exit 1 is expected for that failure set. |

The ten GREEN controls are: the saver-without-date rejection (base rejects the unknown option); four source/source-date rejection cases with/without existing state (base rejects unknown --date and its hint already names --source-date); exact legacy bytes/output; three wrong-schema legacy cases; and the corrected automatic-parser control. These are preservation/already-satisfied rejection controls, not claimed behavioral RED cases.

The initial automatic-parser control incorrectly expected a file with a recorded_at timestamp to be undated. The base accepts that as dated. I changed this NEW control to assert acceptance unchanged, then remove recorded_at before asserting the existing undated rejection, and ran the corrected control on the base before implementation. No existing test was changed.

There were two procedural mistakes in the first baselines: after the sandbox nice warning I retried while the original process was still running, and I created the new test while the full baseline was still running. The overlap caused screen-test log-change failures; the uncommitted test caused the real-suite boundary failure. The sandbox run also had socket failures. I moved the prepared test to scratch with the patch tool, waited for both wrappers to exit, and reran the entire fast baseline alone on unchanged tracked source with no new test under tests/. That clean baseline is the authoritative before count. The test was restored only after it finished. The final fast gate likewise ran alone on the committed tree, with no file edits during execution.

All subsequent file authoring used the patch tool, and each approval request was a single yield_run.sh invocation as requested. The prepared historical test copy and unused reporting helper remain scratch evidence, not committed implementation.

## Manual-file readers

Search covered `src` and `scripts` for the manual filename, `_parse_state_file` and `load_contest_streak_state`.

| Reader / consumer | Dated-observation behavior |
| --- | --- |
| `src/bts/contest_state.py:221` `_parse_state_file` | The shared raw reader: exact-v3 manual files use the ordered validator; automatic/other-schema files use the unchanged parser. |
| `src/bts/contest_state.py:269` `load_contest_streak_state` | Parses manual first, selects it only through the shared predicate, and discards an out-of-date dated observation before automatic/fallback selection. |
| `src/bts/contest_state.py:421` `load_decision_streak_state` | Indirect reader through the loader; captured now is passed down. Today's dated observation supplies fresh streak, trusted best and policy saver. Model corruption remains diagnostics rather than authority. |
| `src/bts/cli.py:1401` dated command raw snapshot | Independently reads only the manual path for valid-snapshot identity comparison, using the same validator. It does not use the selection loader or read the flag/auto/fetch status. |
| `src/bts/health/contest_state.py:49,117` | Uses the shared loader, then separately reads manual schema/legacy expiry for its existing warning. Valid dated files are exempt from that legacy warning. When selected/in effect, the dated saver branch returns before any flag read. |
| `src/bts/web.py:150` `saver_dashboard_context` | Unchanged indirect reader through the loader: it can see the selected observation's best, while the displayed saver/buttons still come from the flag. It does not become an observation editor. |

No other direct semantic manual-file reader was found in production src/scripts. The legacy setter remains a writer, not a new reader. Scheduler/pick paths consume the shared decision loader; no alternate manual parser was added.

## Guarantees, limits and unresolved work

The file/argument guarantees concern ordinary on-disk contents and command-line strings, with timezone-aware caller now values as defined by the contract. They are not promises about hostile Python objects. Legacy preservation covers setter invocations accepted by the base that write an override, using the same clock sequence and filesystem outcome. File-system failure guarantees are the stated append/pre-replacement outcomes, not a multi-file transaction or a permanent claim about a path another writer can change.

- **Not deployable alone:** no observation in effect still permits the base automatic/legacy/model decision and flag saver. The three-day sequence explicitly pins last season's auto on day two. CS4b/6 supply the refusal.
- **No wake/re-decision:** a correction after a pick is selected or locked does not re-decide it; after the entry window closes it cannot be acted on. CS5 owns recovery/reselection.
- **Saver authority lasts one date:** the observation controls the decision only on its effective date. Dashboard and saver-state still display/edit the flag. When switched to automatic later, that flag is authoritative again and must be reconciled/set.
- **Log is evidence, not state:** an append can exist for an observation that was never installed; partial/combined lines are possible; no-op adds no line. There is no lock or multi-file transaction. Among successful dated invocations the later atomic replacement wins, not necessarily the later log line.
- **Snapshot-scoped success:** write success means this invocation replaced the file; no-op success means the read snapshot matched. Output need not describe the path at command completion. Midnight/rollback use the one captured clock, with no persisted expiry memory.

No unresolved contract-table mismatch was found. The next step is the requested fresh adversarial review; this commit has not been pushed or deployed.
