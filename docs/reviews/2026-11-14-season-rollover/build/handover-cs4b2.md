# CS4b-2 author handover

Base: `0bbfc8d479a72df4ba62b1bcfa32ca12bab8e3d8` on `main`.
Branch: `sb-cs4b2`. Commit: `8494b6e61fd12e5d32a22c72136336380117590d`
(`Refuse uncovered live season state and add revoke quarantine`). Nothing pushed or deployed.
This handover and `.codex-review/cs4b2/` are deliberately not committed.

## Contract implementation

| Row | Implementation | Evidence |
|---|---|---|
| 1 | `src/bts/season_contract.py:103`, `live_automatic_refusal`: one `switch_state` call, selected snapshot only; corrupt → manual filename → inactive switch → absent observation → schema → season → identifier. | `test_season_contract_cs4b2.py::test_selection_matrix`, `::test_refusal_one_snapshot_no_observation_io`, `::test_schema_diagnostics`; edges `::test_manual_filename_wins_over_claimed_schema_and_source`; recorded comparison. |
| 2 | `season_contract.py:97` shared dated template; refusal recovery text at `:141`. Every printed command uses `shlex.quote` on the picks directory. The alternatives distinguish replacing a legacy override, deliberate corrupt-switch repair/removal/replacement, unswitched evidence collection and switched installation. | `::test_message_complete_recovery_paths` uses a path containing spaces, a quote and `&`, extracts commands with `shlex.split`, fills N/B/S and executes actual CLI commands with captured fetch transports. Schema diagnostics and matrix exercise every reason. |
| 3 | `src/bts/contest_state.py:453`: after the existing selecting loader succeeds and before the existing missing-state/freshness branches. Existing model diagnostics remain earlier. Dated observations bypass the shared refusal and switch. | Selection matrix, optional/dated spies, unchanged loader-error text assertions, midnight tests, and comparison. |
| 4 | The optional branch never enters the new conditional. `load_contest_streak_state` is unchanged. | Optional/dated switch-read spy; all 9,504 optional decisions equal the archived base, including model errors. |
| 5 | `src/bts/health/contest_state.py:48`: `require_dated_auto=expected`; loader exception still returns its sole original alert. Closure at `:55`, invoked at the missing-state return and final return, appends a refusal last on normal completion. Dated saver early return bypasses it. | Matrix exact loader errors, two missing-state CRITICALs, exact manual/coverage/saver/refusal order, post-loader overlong saver JSON exception, optional and dated spies; all 9,504 `expected=False` results equal base. |
| 6 | `season_contract.py:243`: independent live section after saver and before ledger calls the real required decision with the command's captured now. Distinguishes accepted/refused/unavailable. Existing status exit rule remains unchanged. | `::test_status_real_decision_and_read_only` checks actual-year behavior with `--season 2026`, real message equality, source/streak/status/saver, write-open and mkdir spies, and tree equality. Overlong saver JSON demonstrates unavailable. Existing cold-status tests remain GREEN. |
| 7 | `src/bts/cli.py:2170`, `season_contract_revoke`: validate current year and nonblank reason before application reads; one prior switch snapshot; atomic revoked switch write; checked `lstat`/`os.rename` manual then saver; all eight switch fields present. Identifier/switched_at carry only from wholly valid same-year snapshots. One captured UTC instant supplies timestamp and shared microsecond archive stamp. | Usage, snapshot/repeat, failure-prefix/retry, dangling source/target, occupied target, entry-clock tests; edges interrupted-prefix/retry, source/target `lstat` errors and absent-manual saver failure. Protected automatic, ledger and fetch-status bytes are checked. |
| 8 | `cli.py:2236`: prints the written snapshot and per-source outcome; archives kept; new dated and saver templates; explicit uninitialized flag and dated-saver separation; scheduler/fetch quiescence, running operations and delivered/locked pick limits; status command. | `::test_revoke_snapshot_archives_and_repeat`, interruption-before-printing and retry, restore sequence. |
| 9 | No production source changes outside the four permitted files. The CLI's existing functions, including fetch, enable, dated and saver commands, are unchanged; only the group docstring and new revoke function differ. | Commit helper compares preserved CLI and contest-state functions by AST source segments and compares scheduler/orchestrator/web/saver/fetch/decision source against base. Full related suite; real caller controls and grading/delivery controls. |

## Recorded comparison

The immutable helper is `.codex-review/cs4b2/comparison.py`. Base source was extracted with
`git archive 0bbfc8d4 src` into `.codex-review/cs4b2/base/`; no checkout replaced tracked source.
Two separate Python processes load the base and branch source. Both use the same fixture path,
captured 2027-07-15 ET clock and serialized dataclass/error/alert results, including exact messages.

The product is 12 switches (absent, automatic, revoked, other season, and each of eight ordered
corrupt checks) × 9 manual classes (absent; legacy unexpired/expired/no expiry; dated today/yesterday/
tomorrow; invalid dated; unreadable) × 11 automatic classes (absent; v1; missing/null/unknown schema;
covered v2; different ID; other season; refused; corrupt; undated) × 2 model classes (present/corrupt)
× 4 saver states = **9,504 cells**, with both decision settings and both health settings per cell.
The tracked selection matrix additionally includes non-string schema and recorded-at-only cases.

| Operation | Identical cells | Changed cells | Required differences |
|---|---:|---:|---|
| Optional decision | 9,504 | 0 | Row 4 preservation. |
| Health, expected=False | 9,504 | 0 | Row 5 preservation. |
| Required decision | 4,648 | 4,856 | Row 3: uncovered/legacy/absent selections now raise the ordered shared refusal. Includes replacement of the old required-but-missing message. |
| Health, expected=True | 4,168 | 5,336 | Row 5: same 4,856 refusal additions, plus 480 completely undated-auto cells now return only their loader-error CRITICAL, suppressing former downstream alerts. |

Every optional result is asserted exactly equal, including failures. Every required loader error
is unchanged. Every unchanged accepted cell is exactly equal. Every refusal health result consists
of its old alerts in their old order plus exactly the decision's refusal last. For undated automatic
loader errors, health has exactly the required decision loader message and no other alert.
The 9,504 cells factor into 4,856 refusals, 3,552 existing live loader errors and 1,096 accepted controls.
No unexplained difference was found. Raw results and counts are `matrix-base.json`,
`matrix-branch.json`, and `comparison-summary.json` in scratch.

## Tests first and controls

The fast baseline started alone before any test file was created, and its wrapper exited before
the targeted baseline started. Counts from both baselines were recorded before creating tests.
No wrappers overlapped. A completed pytest log was not treated as wrapper exit; sessions were
polled until exit, including the wrapper's one-minute housekeeping interval.

On the unchanged base, `red-contract.log` recorded **969 failed, 592 passed**. Of the failures:

- 816 selection-matrix cells raised no refusal or had the old missing-state/health result; these
  are behavioral RED for the real decision and health paths and their reason ordering.
- 18 recovery-path cases, four schema diagnostics and the year-boundary case failed because
  the base still accepted the uncovered live observation. The alert-order case lacked the final
  refusal. Three status cases lacked the real live-result section.
- 84 direct-helper cases fail on API availability. Revoke tests fail on command availability,
  including usage tests which explicitly reject an unknown-command exit as evidence. These are
  **not** claimed as behavioral RED for revoke's individual persistence assertions; those assertions
  become discriminating execution checks after implementation.
- The restore sequence initially reaches the absent revoke command on base; likewise not proof
  of the individual later steps. Its implemented execution covers fresh activation mismatch,
  tomorrow's refusal, and fetch failure before/after installing replacement bytes.
- The 592 GREEN cells retain existing parser/error/dated/covered behavior. Optional and dated
  switch bypass, covered positive acceptance and existing loader precedence are controls.

The base-versus-base recorded matrix was RED at its requirement that an uncovered required
decision differ from the archived base; this was a behavioral failure, not API availability.
`red-edges.log` recorded **13 failed, 6 passed** against the archived base: revoke availability,
manual filename refusal and season-start refusal RED; existing delivery guards/after-read snapshot
behavior and grading GREEN. The focused real selection/fallback/orchestrator/clean-outcome control
also passed against the archived base (`base-caller-control.log`).

## Existing test migrations (each changed test)

One shared `covered_automatic` factory in `tests/conftest.py` takes the explicitly pinned operation
clock and raw payload. It adds schema/season/ID and a valid switch without changing value, date,
source, best or corruption under test. Optional legacy controls and standalone builders retain v1.

| File | Changed test(s) | Reason |
|---|---|---|
| `test_contest_state_defensive_reads.py` | `test_corrupt_pick_is_skipped_by_both_scans`; `test_pick_with_unhashable_result_is_skipped_by_both_scans`; `test_u1_bad_best_reachable_after_corruption`; automatic cells of `test_manual_not_universal_recovery` | Covered factory preserves their pick scanning, automatic bounds and corrupt-model isolation assertions. |
| same | `test_manual_precedes_corrupt_auto`; manual cells of `test_manual_not_universal_recovery` | Their shared manual recovery helper now installs today's valid dated observation; clock is explicitly pinned. Keeps live recovery before corrupt automatic parsing and independent model diagnostics. |
| same | `test_manual_bounds_unchanged_in_cs1` | Optional acceptance still checks all legacy out-of-bounds meanings; required mode now asserts legacy refusal. Values 58, best<active and best>57 are not converted to automatic. |
| same | `test_dated_version_one_auto_still_accepted` | Keeps optional v1 acceptance and malformed-primary-date behavior; required mode explicitly asserts absent-switch refusal rather than old acceptance. |
| same | `test_missing_required_contest_wins_over_corrupt_model` | Asserts the new absent-switch reason in place of required-but-missing; model corruption remains subordinate. |
| `test_contest_state_cs1_review.py` | `test_chmod000_unreadable_files_across_rows_1_2_5_6_7`; `test_row3_auto_exact_boundaries_accepted`; `test_row5_optional_diagnostics_flow_cleanly_through_dm_decision_and_subtitle` | Covered factory preserves unreadable-pick/model isolation, exact automatic bounds, and diagnostic downstream formatting. The missing observation after removing its covered automatic now reports no automatic observation. |
| same | `test_row3_real_unexpired_manual_v2_bounds_remain_accepted` | Retains every optional legacy bound/trust assertion; required mode asserts legacy refusal. |
| same | `test_fetch_writer_out_of_bounds_best_rejected_and_manual_override_recovers` | Pins the CLI clock and uses today's dated recovery. Writer bound rejection and trusted best=57 remain asserted. |
| `test_contest_state_decoder_limits.py` | `test_row5_large_json_scalar_cannot_block_observation` | Required automatic uses covered factory; required manual uses a valid dated observation. Optional manual remains legacy. Model ValueError diagnostics and single warning are unchanged assertions. |
| same | `test_rows6_7_large_json_scalar_has_domain_error` | Required missing case now asserts absent switch; optional model-error assertion remains. |
| same | `test_deeply_nested_body_is_isolated_everywhere` | Today's dated manual preserves the required model/pick recursion-error isolation purpose and subsequent malformed-manual rejection. |
| `test_contest_dated_cs4a.py` | `test_three_day_intermediate_sequence` | The middle day now asserts live refusal; its optional read retains the old streak=22/uninitialized/spent assertions. Dated days still accept unchanged. |
| `test_season_contract_cs4b1_edges.py` | `test_decision_and_health_never_read_switch` | Optional and dated cells keep the no-read spy. Required non-dated cells now read the switch and assert corrupt/schema refusal, shared with health. |
| `test_contest_state.py` | `test_required_contest_state_missing_raises` | Pins the new absent-switch reason instead of the old generic missing message. |
| `health/test_contest_state.py` | `test_expected_missing_is_critical` | Now requires missing-state CRITICAL then refusal CRITICAL. |
| same | `test_valid_auto_state_is_clean_when_expected`; `test_gap2_coverage_lag_is_warn`; `test_expected_overnight_lag_is_not_critical`; `test_gap1_past_noon_is_info_not_warn`; `test_offday_gap_with_one_new_pick_is_not_critical`; `test_two_day_gap_is_warn_not_critical` | Covered factory leaves every active/best/source-date and settled-pick input intact, preserving exact clean/coverage/no-false-CRITICAL assertions. Each uses its actual explicit now, not source_date.year. |
| `test_saver_cs3.py` | expected cells of `test_health_saver_matrix`; `test_health_reads_current_season_despite_prior_source_date`; real-check cell of `test_saver_warn_gets_first_occurrence_attention_but_coverage_does_not`; `test_uninitialized_warning_gives_a_command_for_each_thing_the_operator_can_find` | Covered factory isolates the saver alert/attention/repair purpose; optional health cells remain unchanged. Every former alert/order/command assertion remains. |
| `test_saver_cs3_review.py` | `test_eod_health_dm_day1_day2_day30_saver_attention_vs_coverage_lag`; `test_mistaken_use_then_undo_never_earned_behavior_at_streaks_5_10_16` | Covered current-year automatic fixtures preserve the 30-day WARN/DM attention assertions and the undo/health/dashboard/policy assertions, including prior-season source dates. |

No test was deleted. No production caller was changed to make a test pass.

## Every live caller and its existing outcome

Both `_contest_state_required` helpers OR `scheduler.contest_state_required` and
`health_checks.contest_state_expected`. Delivery mode alone does not enable the gate.

- `orchestrator.run_and_pick`, `orchestrator.py:283`: required decision raises before selection;
  the error propagates. No/empty predictions return before the state read, unchanged.
- `scheduler.run_single_check`, catch at `scheduler.py:1701`: alerts through existing health
  dispatch, returns no pick/selection and `should_post=False`. A later scheduled check may try
  again; there is no immediate wake or scheduled retry created by this change.
- `scheduler._deliver_and_lock_pick`, `scheduler.py:1098`: undelivered public/DM read catches,
  alerts and returns False before persisting a delivery attempt. Already delivered and private
  paths bypass the loader and lock as before. The prior crash-attempt guard locks without a new
  read/send and emits its existing late-delivery attention.
- `scheduler._refresh_pick_at_fallback_decision`, catch at `scheduler.py:2110`: rethrows the
  `ContestStateError`, rather than delivering cached state. The in-loop catch at `:2990` alerts
  and continues; the final catch at `:3143` alerts and clears `daily`, allowing existing EOD work.
- `orchestrate` initial catch at `orchestrator.py:336`: direct DM attempt and False. Its subsequent
  skip read (`:362`), locked-pick catch-up read (`:386`) and post-save read (`:429`) are outside that
  catch and propagate the error. Post-save refusal can leave a saved undelivered pick.
- `season-contract status` additionally calls the real required decision and renders its error;
  it neither authorizes nor repairs anything. Health uses the shared refusal after its own loader.

The captured real caller control exercises initial selection, both fallback catches, later
orchestrator propagation, EOD skip writing, result polling and a clean `day-outcome` skip fixture.
`test_delivery_snapshot_and_guards` covers refusal after selection, acceptance read before a revoke,
already-delivered/private/crash guards. `test_captured_skip_and_grading_ignore_refusal` exercises
real `check-results`. A real non-dry `bts run` under required-only injected refusal still posts through
a captured transport, preserving and demonstrating the optional-entry-point limitation.

## Runs and counts

All pytest commands used the specified interpreter, TZ, no-bytecode and no-cacheprovider options
inside a single `yield_run.sh` invocation; every wrapper exited before the next started. Logs,
watch logs, immutable comparison/clock/commit helpers and raw results are under `.codex-review/cs4b2/`.

| Log | Result |
|---|---|
| `baseline-fast.log` | 6,510 passed; 37 failed; 16 errors; 19 skipped; 6 deselected; 1 xfailed, 1484.89s. Exactly the known 53 failing/error node IDs in `~/bts-runs/gate-failed.txt`; no extra or missing IDs. |
| `baseline-related.log` | 2,172 passed, 8 skipped. Before creating test files. |
| `red-contract.log` | 969 failed, 592 passed on unchanged base; classified above. |
| `red-comparison.log` | 1 intentionally RED base-versus-base comparison. |
| `green-contract.log` | 1,562 passed (contract plus 9,504-cell comparison). |
| `red-edges.log` | 13 failed, 6 passed against archived base; availability/behavior/control distinction above. |
| `base-caller-control.log` | 1 passed against archived base. |
| `touched-2027.log` | 2,678 passed. |
| `final-touched-2027.log` | 2,680 passed, including comparison. |
| `final-touched-2028.log` | 2,679 passed, every added/touched test file. |
| `final-related.log` | 3,753 passed, 8 skipped; every matching existing/new test file, normal clock. |
| `isolated-callers-2027.log` | 20 passed; final child-process caller isolation at 2027-07-15. |
| `commit.log` | Commit `8494b6e6`; 16 files. Protected-source comparisons passed. |
| `final-fast.log` | On committed tree `8494b6e6`, alone: 8,091 passed; 37 failed; 16 errors; 19 skipped; 6 deselected; 1 xfailed, 1489.05s. Failing/error set exactly equals the known 53 IDs; no extra or missing IDs. Clean under the kickoff's specified environment-failure rule. |

Two exploratory failures are retained, not hidden:

1. `related-2027.log` froze all 36 existing files, including untouched ones: 36 failed, 3,697 passed,
   8 skipped. Thirty-four failures identified the migrated missing-state and saver-health fixtures;
   the two others were an untouched heartbeat test which requires elapsed wall time and an untouched
   recipe fingerprint binding the replaced datetime class. The required fake-clock runs target every
   added/touched test file, and the normal-clock final-related run covers all 38 matching files.
2. The immutable scratch helper `callers.py` / `base-callers.log` had one invalid assertion: a
   non-clean `day-outcome` fixture was assumed to exit 0 on base. It failed on the archived base and
   is not counted as change evidence. It was not edited after requesting its run. The focused tracked
   caller control uses the separately provided clean research-skip fixture, passed on base, and is
   included in both the touched and normal-clock final checks. Historical probe audit hooks are isolated
   in a child process so they do not change later tests' transport semantics.

## Known limits and promise scope

- **Not deployable before change set 6.** No push, deployment, network lookup, box access or contest
  account access was performed. Fetch tests capture authentication, profile, rounds and DM transports.
- No refusal wakes selection or creates a retry. If no scheduled check remains after repair, there is
  no pick that day. The re-selection command belongs to change set 5.
- Selection and delivery are separate reads. A revoke or midnight between them can prevent delivery
  on that attempt. A revoke after a delivery's successful read cannot retract the later send. Accepted
  changed streak/saver data does not reselect the saved candidate. Later orchestrator reads propagate
  rather than receiving a new catch/alert.
- `bts run`, dashboard, preview and other optional-loader entry points remain outside this refusal,
  even if they post. Private/research mode with required/expected flags is gated during selection;
  public/DM mode without those flags is not protected. Change set 6 owns configuration/entry points.
- Revoke is an ordered three-step operation, with no transaction or lock. Completed steps stand on
  failure or interruption, including before printing. A dated manual still present after a failed
  manual step can still authorize today's decision. The command reports actual completed outcomes.
- A target `lstat` check followed by ordinary `os.rename` deliberately retains R31's race: another
  process creating that exact microsecond-stamped target between the two calls can have it replaced.
  The explicit race control demonstrates this; no platform no-replace implementation was added.
- A running fetch may finish using its old activation and recreate the saver flag. A later enable
  changes the identifier, so that old observation is uncovered, but saver continuity still needs the
  operator's reconciliation. Stop the scheduler and wait for every fetch before revoke or changing
  enable, then inspect status. Revoke never cancels a delivered or locked pick.
- Enable remains the operator's statement about endpoint meaning; neither status nor these labels
  verify MLB. Restoring a matching old switch+automatic pair without revoke is undetectable; the
  restore sequence deliberately accepts it. An already covered ordinary stale observation remains
  usable as before. A fetch may replace the observation and later fail, so its exit is not proof of
  whether installation occurred.
- The expected-health refusal is a daily prompt when that check runs, CRITICAL until today's dated
  observation or covered automatic authority is supplied. DM dispatch, scheduling, deduplication and
  timing are unchanged; no midnight or pre-selection alert schedule is promised here.
- File-read and no-write promises concern ordinary file contents, command-line text and stated
  filesystem error cases, not hostile Python objects, arbitrary process failures or concurrent writers.
  Optional preservation covers all compared JSON/error classes. Switch bypass covers optional reads
  and in-effect dated observations. Revoke's protected automatic/ledger/fetch-status promises concern
  this invocation's writes, not another process. Health's final refusal is promised on normal completion,
  not after an existing post-loader exception such as an overlong saver integer.

No implementation row was found impossible to match. The recorded failing exploratory helper and
its invalid fixture assertion are disclosed above, rather than presented as a passing check.

Final state: HEAD remains `8494b6e61fd12e5d32a22c72136336380117590d`; no tracked diff or staged
changes. Only the deliberately untracked handover and scratch evidence remain. The final gate
adds 1,581 passing cases to the baseline without changing its known failing/error set.
