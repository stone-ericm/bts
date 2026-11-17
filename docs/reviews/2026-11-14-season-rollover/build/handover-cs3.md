# CS3 handover

Base: `7a153ec951fff8bb58275b8a6c33c61720094dcb` (CS1 and CS2 merged). Branch: `sb-cs3`.
Implementation commit: `0768a6bd1008fc2f7651ebbb51e79b7f96e118a5` — `fix: bind saver policy and operator controls to the current season`. No push, external network access, box access, or contest-account access performed. The existing web socket test uses loopback only. All transports in the new command tests are captured locally.

## Contract rows and evidence

All new regression cases are in `tests/test_saver_cs3.py` (149 parametrized cases). They were written and run against the unchanged base before production edits: **98 failed, 51 passed** in `.codex-review/cs3/red.log`. The failures include the missing helper/field, wrong-season decisions, the real policy action, saver warning source/attention, dashboard reads, and CLI transitions/hooks. Existing raw `.is_available` tests were not edited.

| Row | Implementation (file:lines) | Pinning tests |
| --- | --- | --- |
| 1 | `src/bts/saver_state.py:31–33`: pure mapping; `.is_available` at 26–28 unchanged. No streak coercion/validation added. | `tests/test_saver_cs3.py:41` `test_policy_saver_table`: 4 states × 0,9,10,15,16,57; unchanged `tests/test_saver_state.py::test_not_earned_and_used_not_available`. |
| 2 | `src/bts/contest_state.py:347,395–406`: caller's ET year, selected contest streak, policy mapping. | `test_decision_does_not_reuse_last_seasons_active_flag` (47), `test_contest_decision_uses_current_flag_and_policy_mapping` (57). Real 2026 observation, 2027 now and flags. |
| 3 | `src/bts/contest_state.py:359–365`: same current year; use the exact model streak being returned. | `test_model_decision_uses_its_returned_streak` (68): 7 and 12, all four flag states, real season-tagged `streak.json`. |
| 4 | `src/bts/contest_state.py:74,365,406`: optional `saver_flag_state` default None, set in both branches. | The decision tests at 47,57,68 assert the raw flag state. No production consumer or serialization of this new field was added. |
| 5 | `src/bts/health/contest_state.py:20,125–149`: season Y; separate `saver_state` source; expected/zone rules and executable season-qualified commands. Non-saver check code unchanged. | `test_health_saver_matrix` (99): all 50 requested combinations; `test_health_reads_current_season_despite_prior_source_date` (117). Each warning checks season, streak, “counting the saver as spent”, and the full command. |
| 6 | `src/bts/health/attention.py:25`: add `saver_state`, leave `contest_state` out. Runner's existing `src/bts/health/runner.py:127–130` collects all returned alerts; it registers checks, not an allowlist of alert source strings. | `test_saver_warn_gets_first_occurrence_attention_but_coverage_does_not` (127): synthetic independent sources and real coverage-lag + saver warnings, fresh attention file. |
| 7 | `src/bts/web.py:137–159,165–186`: ET year helper; display catches ContestStateError and suppresses warning/nudge; POST never reads contest state. OSError from the guarded writer returns 409. | `test_dashboard_post_targets_current_year_without_consuming_prior_flag` (148); `test_dashboard_survives_contest_files_and_post_never_reads_them` (176); `test_bad_contest_suppresses_even_a_likely_save_nudge` (211). Full page rendering is exercised. |
| 8 | `src/bts/web.py:203–210,1102–1104`: pass the decision streak already loaded by the page, or None. Buttons and `_DASHBOARD_TRANSITIONS` unchanged. | `test_page_reserved_label_uses_existing_decision_streak` (227): decision streak 7,10,12,None while the contest fixture itself says 7. |
| 9 | `src/bts/saver_state.py:76–79`; `src/bts/cli.py:1475,1501–1535`: allow not_earned→used; guarded writer once per candidate prior; used is a successful already-used result; uninitialized use and invalid undo fail; errors print before the final resulting state/season line. | `test_cli_transition_matrix_and_one_audit_per_attempt` (255): 4 prior states × use/undo, exact state/exit/output/audit count, including the final output line. Updated existing CLI test now asserts not_earned→used. |
| 10 | `src/bts/saver_state.py:174–189`: `maybe_auto_earn_saver` unchanged. | `test_between_fetch_use_cannot_be_resurrected_by_auto_earn` (273): not_earned→real CLI use→used→best 10 stays used; unchanged saver-state auto-earn tests also pass. |
| 11 | `src/bts/cli.py:1414–1417,1462–1464,2126–2130`: current ET year computed once per command path; only current-year dated observations may call auto-earn. Manual omitted source still defaults to today's date. | `test_fetch_without_settled_source_never_creates_a_flag` (299), `test_other_year_observation_does_not_touch_flag` (309), `test_current_year_observation_initializes_then_earns` (320), `test_manual_omitted_source_is_today_not_null` (329). Real commands, captured transports; previous and next years; absent/current/stale flags; current best 7→10. |
| 12 | `src/bts/saver_state.py:40–42`: `season_for` retained unchanged. | Unchanged `test_season_for_uses_source_date_year_else_now`; final `rg` finds no production call, only the definition and its unit test import/calls. |

Extra ET boundary pin: `tests/test_saver_cs3.py:244`, `test_season_is_the_et_year_at_midnight`, uses 2027-01-01 02:00Z (still 2026 ET) and 05:00Z (2027 ET) for decision and dashboard reads.

## Real policy effect

On **2027-08-29, streak 3, primary probability .80, second probability .80, eligible different-game second pick, doubles allowed**, with a current-season `not_earned` flag:

| Package | Policy saver | Action | Source |
| --- | --- | --- | --- |
| Archived base `7a153ec9` | False | skip | mdp |
| Branch | True | double | mdp |

`test_reserved_saver_changes_the_real_policy_action` (79) pins both actions through `strategy.resolve_policy_decision` and checked-in `data/models/mdp_policy.npz`. `.codex-review/cs3/policy-compare.py` additionally ran the actual archived base package and branch package over independent real flag/contest files; `.codex-review/cs3/policy-compare.log` records `[false,"skip","mdp"]` versus `[true,"double","mdp"]`. No policy artifact was changed.

## Every existing test changed

No assertion was dropped to hide a failure. The assertion of the old CLI behavior was intentionally replaced, and its test renamed, as described below.

| File / test | Reason |
| --- | --- |
| `tests/test_cli_integration.py::TestSaverStateCli::test_use_noop_when_not_active` → `test_use_marks_not_earned_used` | It asserted the defect itself. Now requires use to exit 0 and show used after initializing not_earned. |
| `tests/test_decision_saver_fallback.py::test_active_flag_makes_saver_available` | Pass explicit 2026 now matching the 2026 flag/observation; same assertion. |
| `tests/test_decision_saver_fallback.py::test_used_or_uninitialized_flag_means_unavailable` | Same fixed-now fixture correction for both reads; same assertions. |
| `tests/test_decision_saver_fallback.py::test_model_only_fallback_reads_saver_state_not_streak_json` | Same fixed-now correction; same assertion. |
| `tests/test_contest_fetch.py::test_fetch_auto_earn_cold_file_at_10_stays_uninitialized` | Fixture source dates, resolved-pick filename and season assertion now use current ET year, so the test actually exercises eligible cold-init behavior in every year. Same fail-closed assertion. |
| `tests/test_contest_fetch.py::test_fetch_auto_earn_below_10_inits_not_earned_then_promotes_at_10` | Same current-year fixture correction; retains both not_earned and active assertions. |
| `tests/test_web_saver.py::test_do_post_saver_transition_socket_wiring` | Pin the handler's clock to the 2026 fixture's NOW; retain all real HTTP status, redirect and state assertions. |
| `tests/health/test_contest_state.py::test_valid_auto_state_is_clean_when_expected` | Explicit 2026 now and initialized not_earned saver fixture. Test concerns a healthy contest observation; retain exact empty-alert assertion. |
| `tests/health/test_contest_state.py::test_gap1_past_noon_is_info_not_warn` | Initialize a 2026 not_earned flag to isolate coverage behavior; existing explicit afternoon now and all assertions retained. |
| `tests/health/test_contest_state.py::test_offday_gap_with_one_new_pick_is_not_critical` | Initialize a 2026 not_earned flag to isolate coverage behavior; existing explicit now and all assertions retained. |
| `tests/health/test_contest_state.py::test_saver_flag_uninitialized_in_zone_warns` | Pin now and add assertion that source is saver_state; retain WARN count/level and init-command assertion. |
| `tests/health/test_contest_state.py::test_no_alert_when_not_expected_and_missing` | Explicit 2026 now for deterministic touched-file tests; same assertion. |
| `tests/health/test_contest_state.py::test_expected_missing_is_critical` | Same explicit-now correction; same assertions. |
| `tests/health/test_contest_state.py::test_malformed_existing_state_is_critical` | Same explicit-now correction; same assertions. |
| `tests/health/test_contest_state.py::test_non_object_existing_state_is_critical` | Same explicit-now correction; same assertions. |
| `tests/health/test_contest_state.py::test_gap2_coverage_lag_is_warn` | Same explicit-now correction; same coverage WARN / no CRITICAL assertions. |
| `tests/health/test_contest_state.py::test_two_day_gap_is_warn_not_critical` | Same explicit-now correction; same assertions. |
| `tests/health/test_contest_state.py::test_missing_source_date_is_critical` | Same explicit-now correction; same assertion. |
| `tests/health/test_contest_state.py::test_legacy_manual_present_warns` | Same explicit-now correction; same assertion. |
| `tests/health/test_contest_state.py::test_saver_flag_active_in_zone_is_clean` | Pin now to flag's 2026 season; same exact empty saver-alert assertion. |
| `tests/health/test_contest_state.py::test_saver_flag_uninitialized_below_zone_no_warn` | Pin now; remains silent with expected=False, assertion unchanged. |

`tests/test_saver_state.py` is entirely unchanged, including raw `.is_available` and `season_for` tests. The other two health tests already passed explicit now and were not changed.

## Every saver-season selector

- `load_decision_streak_state`, `src/bts/contest_state.py:346–347`: supplied now or UTC wall clock, converted once to ET year. Both contest and model branches pass this year to the loader.
- Health `check`, `src/bts/health/contest_state.py:44,128`: supplied now or UTC wall clock, converted to ET year for the saver block. Source-date/coverage logic outside that block is unchanged.
- Dashboard display and POST, `src/bts/web.py:137–138`: `_saver_season(now)` returns ET year only, called once by each path. Display reads contest separately for best streak; POST never does.
- Manual command, `src/bts/cli.py:1414–1417,1463–1464`: one ET now yields season and the default observed date; observation year must match season before the hook.
- Fetch command, `src/bts/cli.py:2128–2130`: one ET wall-clock year, passed only when source_date exists in that year.
- Explicit saver command, `src/bts/cli.py:1486`: supplied `--season` remains supported; default is current ET year, unchanged.
- `load_saver_state`, `transition_saver_state`, and `maybe_auto_earn_saver` receive explicit season; they do not select it from observations. A flag's stored season is only checked against the requested year.
- `season_for` has no remaining production caller. The only remaining callers are its unchanged unit test's two assertions. `picks.season_of_date` was not introduced at any of these sites.

Search evidence: `.codex-review/cs3/source-registries.txt` and `.codex-review/cs3/saver-consumers.txt`. `rg` found no additional known-health-source registry requiring an entry. `contest_state` remains absent from both always-attention and repeated-attention source sets.

## Consumers of DecisionStreakState.saver_available

Direct consumers are `src/bts/orchestrator.py:297` and `src/bts/cli.py:1171,1331` (live run / preview): they pass the bit to `strategy.select_pick`. `src/bts/strategy.py:573–574,594,624` uses that explicit bit for the policy and carries it in SelectionResult. Its meaning is now “the policy may count on an unconsumed saver”, including not_earned below 10; it is not a claim that the saver has already been earned.

Downstream uses were inspected without changing their formats or text:

- Scheduler (`src/bts/scheduler.py:699,735,766,828,1731,2816`) retains the bit for selection/skip candidates, pick-bar calculation and decision provenance. New action/bar results follow the mapped input.
- DM text: `src/bts/scheduler.py:1512–1540` formats the resolved skip decision/bar, and pick notifications follow selection. There is no raw saver-state text to relabel; the action/bar and whether a skip DM exists can change. No DM template edits.
- `decision.json`: `src/bts/daily_decision.py:91` writes the same `saver_available` boolean key, now the actual mapped policy input. No `saver_flag_state` key or schema change. Existing decision-record integration tests pass.
- Skip-policy shadow: `src/bts/skip_policy_shadow.py:96` copies the deployed decision's bit. Same schema; a below-10 not_earned input now records True, and changed deployed skips can change which days create divergent shadow records. No shadow implementation change.
- Dashboard: `src/bts/web.py:121–125,1093–1104` loads decision state for the displayed streak; the saver section continues to load the raw flag for controls, now in the current season. It uses the already-loaded decision streak for the reserved label. The live dashboard API uses the decision streak, not the saver bit. No new production reader of saver_flag_state.

Model replay/scoring `streak.json` saver bits are separate diagnostics, unchanged by CS3.

## Recorded base-versus-branch matrix

Base source was extracted by `git archive 7a153ec9 src` into `.codex-review/cs3/base/`; no checkout over tracked files. One script, `.codex-review/cs3/compare-v2.py`, runs isolated workers with base/branch PYTHONPATH and records outcomes in `matrix-v2-base.json`, `matrix-v2-branch.json`, and `matrix-v2-differences.json`. The wrapper entry is `matrix-v2.sh`; log is `matrix-v2.log`.

Matrix dimensions: missing/stale/not_earned/active/used flags; streaks 0,9,10,15,16; explicit 2027/2028 now; null/prior/current source dates and model-only branch; health expected False/True; both saver CLI actions; both observation commands with null/prior/current dates and best 0,7,10,12. CLI auto-earn fixtures use the current ET year of the wall clock. Every branch outcome is checked against its contract rule; non-saver health alerts are compared unchanged after normalizing independent fixture paths.

| Family | Rows | Recorded outcomes per package | Differences | Required reason |
| --- | --- | ---: | ---: | --- |
| Decision | 2,3 | 200 | 36 | Current-season binding and reserve below 10. |
| Health | 5 | 300 | 124 | Current-season binding, separate source, expanded warnings, executable messages. |
| Dashboard display | 7 | 200 | 40 | Read current season despite prior-year source date. |
| Dashboard POST | 7 | 200 | 20 | Current active flag can be used; prior-year active flag cannot be consumed by this year's POST. |
| Saver CLI | 9 | 20 | 14 | not_earned use succeeds; uninitialized use and invalid undo fail. |
| Auto-earn commands | 11 | 120 | 22 | Skip null/other-year observations, use current year for eligible dates. |
| Total | | 1,040 | 256 | All checked against the specified rows. |

First comparison attempt (`compare.py` / `matrix.log`) stopped because non-saver alert messages contained different TemporaryDirectory names. That script was preserved unchanged after its run. V2 only normalizes those fixture paths before comparing; it completed successfully, and the same immutable V2 script passed again after the final CLI output-order fix. No production change was made for this harness issue.

## Verification

Every pytest run used `/Users/eric/bts-runs/yield_run.sh <worktree scratch log> env TZ=America/New_York PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -p no:cacheprovider ...`. Family runners are in `.codex-review/cs3/`. No pattern kill was used. Scratch helpers stayed in that directory; none are staged or committed.

Final review found an output-order defect in my first implementation: ClickException printed its error after the state. A strengthened CLI matrix assertion recorded RED (4 failed, 4 passed, 141 deselected) in `red-output.log`. The command now prints the error before the final state/season line and exits with Click Exit(1). The superseded fast gate was stopped at its verified exact pytest PID 14193 with SIGTERM; `after-fast.log.watch` records rc=143. No result from that partial gate is claimed. The final related suite, both future clocks and fast gate are rerun serially after this fix.

The initial sandboxed baseline-related attempt ran with a nice permission warning and failed only on the socket bind (1 failed, 1,497 passed, 8 skipped). It was rerun outside the sandbox through the same wrapper: 1,498 passed, 8 skipped. Both summaries are preserved in `baseline-related.log`; the successful rerun is the recorded related baseline. The baseline family commands were invoked directly through the wrapper; subsequent check families have immutable runner scripts.

Workflow deviation: I found CLAUDE.md's additional one-job-at-a-time studio rule late. The initial sandboxed related attempt overlapped its escalated rerun, and short related/future-clock/comparison checks were launched as separate wrapped jobs with some overlap. After those jobs completed, the remaining verification families ran serially. Each pytest still used the required nice/yield wrapper. Before the replacement final fast gate, memory_pressure reported 93% free on a 96 GiB machine, above the 15 GiB threshold (`memory-before-final.txt`); the check before the committed-tree gate again reported 93% free (`memory-before-committed.txt`). This is a scheduling deviation, not an assertion that the runs were serialized.

Baseline counts were recorded in `baseline-counts.txt` before creating `tests/test_saver_cs3.py` or changing production code. The baseline gate's set is exactly the supplied `~/bts-runs/gate-failed.txt` (37 failures + 16 errors = 53), with no additions or omissions.

| Check | Result | Evidence |
| --- | --- | --- |
| Before: all test files mentioning the required saver/contest/source terms (`rg -l`, pytest test files) | 1,498 passed, 8 skipped | `baseline-related.log` |
| Before: fast suite, exact requested markers/ignores | 5,371 passed, 37 failed, 16 errors, 19 skipped, 6 deselected, 1 xfailed | `baseline-fast.log`; exact known failing set |
| RED: new CS3 file on unchanged base | 98 failed, 51 passed | `red.log`, `red.sh` |
| Focused: new cases, all changed existing files, unchanged raw saver tests | 308 passed | `focused.log`, `focused.sh` |
| After: all files matching the required saver/contest/source terms | 1,647 passed, 8 skipped (both runs) | `final-related.log`, `related-final.sh`; earlier `after-related.log` |
| Every added/touched test file, clock 2027-07-15 | 292 passed (both runs) | `final-future-2027.log`, `future-2027-final.sh`, `frozen_clock.py` |
| Every added/touched test file, clock 2028-07-15 | 292 passed (both runs) | `final-future-2028.log`, `future-2028-final.sh`, `frozen_clock.py` |
| Actual archived-base / branch policy counterexample | Both asserted actions passed | `policy-compare.log` |
| Archived-base / branch matrix | 1,040 outcomes each; 256 required differences; all assertions passed | `matrix-v2.log` and JSON artifacts |
| After: fast suite on committed tree | 37 failed, 5520 passed, 19 skipped, 6 deselected, 1 xfailed, 16 errors; exact known 53-entry failing set | `committed-fast.log`, `fast-committed.sh` |
| Committed-source boundary assertion, unchanged | 1 passed | `boundary-committed.log`, `boundary-committed.sh` |
| Whitespace / staging / final status | clean tracked tree; empty index; only untracked handover and scratch artifacts | git diff --check; git status --short; parent verified as 7a153ec9 |

The first completed post-change gate (`final-fast.log`) had 5,519 passed, 38 failed, 16 errors, 19 skipped, 6 deselected and 1 xfailed: the known 53 plus `test_the_boundary_admits_the_real_suite`. Its assertion requires importable repository modules to equal HEAD and listed the 12 uncommitted CS3 files. No assertion or fixture was changed for this. The completed change was committed as 0768a6bd, then the unchanged boundary assertion and full fast gate were run against that commit. The tracked tree remains clean; the handover and scratch directory remain uncommitted.

The fake-clock plugin imports pandas before replacing datetime.datetime/date, as requested. Files covered: `tests/test_saver_cs3.py`, `tests/test_decision_saver_fallback.py`, `tests/health/test_contest_state.py`, `tests/test_web_saver.py`, `tests/test_cli_integration.py`, `tests/test_contest_fetch.py`.

## Promise boundaries, test-shape constraints, known limits

The restored “Never raises” filesystem promise for `saver_transition_response` covers normal request arguments (Path, strings, origin boolean, aware datetime) with **any contest file contents or filesystem state**: absent, unreadable, non-UTF-8, non-object, malformed JSON, refused streak 58, or a directory. POST does not call the contest loader at all, so it does not depend on whether a particular file can be decoded. Normal guarded saver-storage OSError failures return 409. This is not a guarantee for hostile Python objects passed as arguments, or a new universal promise for every helper in the storage layer. The saver loader and its write rules were retained; no alternative path was added to replace an unreadable flag.

Display's narrower guarantee: if the contest read raises ContestStateError, it still returns the loaded raw flag and the same button, with no warning or nudge. The full page renders the section for all tested contest-file states. `append_probe_audit` retains its existing best-effort filesystem behavior; CS3 does not change its promise or audit storage semantics.

Two test-shape constraints are explicit:

- Real `set-contest-streak` cannot write a null source_date: omission means today's date. Null/best 0 and null/best 12 are tested through real fetch; manual omission is separately pinned to its unchanged dated behavior. No null-manual mode was invented.
- There is one `account_state/saver_state.json`, not a file per season. The cross-year POST tests therefore separately pin a 2026 active file → 409 with identical bytes, and a 2027 active file with a 2026 contest source → 200 writing used/season 2027. No prior-season file is silently overwritten to manufacture an active current-season flag.

Retained diagnostic limitation: the saver warning's existing detail text uses `no saver_state.json` when the loader returns no stored season, which can also mean an unreadable/invalid file; it is not proof of absence. A readable invalid flag with an integer season can also retain that season and be described as stale-season even if its year matches. The required uninitialized state, target season, policy behavior and recovery command are still reported; no new cause-classification/read path was added.

No production contract row was left unmatched. Contract revocation, new file/schema formats, the manual observation's own saver field, fetch validation, dashboard buttons, and decision/shadow schemas were left out of scope.

Known limits by design: at season start the flag remains uninitialized until operator initialization or an eligible fetch sees a settled round of the current season. Meanwhile the policy counts the saver as spent and expected=True health warns each day when a contest streak is available. The existing cold-file best≥10 fail-closed rule still requires explicit operator initialization; it does not invent an active saver. A saver earned and consumed between fetches remains unobservable; the operator must mark it with `bts saver-state --use --season Y`.
