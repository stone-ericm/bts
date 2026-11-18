# CS5a handover

Base `4cab4212`, branch `sb-cs5a`. Commit `30956cddc4787d5d8dfd844af488b6207a7b2cd8` adds chosen-on provenance and guards saved-pick delivery.

## Runs

| Run | Result |
|---|---|
| Baseline fast suite, alone before test files were created | 8,171 passed, 37 failed, 16 errors, 19 skipped, 6 deselected and 1 xfailed in 25m10s. The 53 failure and error IDs exactly matched `~/bts-runs/gate-failed.txt`. Wrapper exit 1. See `.codex-review/cs5a/baseline-fast.log`. |
| Base targeted and initial RED | 3,676 passed, 185 failed, 12 skipped. Every failure was in the new tests. See `red.log`. |
| Base fixture reruns | `red-fixtures.log` had 9 failures and 2 passes. `red-policy.log` had 8 failures and 2 passes after adjusting the dated streak and prediction probabilities. The base still sent the cached double. The shipped MDP skips at streak 56 in every quality bin on July 15, so the recovery fixture uses streak 30 and a 0.99 primary probability to exercise a real fresh single. Selection code is unchanged. |
| First branch targeted run plus optional LightGBM and web saver | 3,874 passed and 12 skipped, all in one pytest process. See `green.log`. |
| Expanded cases on the archived base | 39 failed, 6 passed, 4 skipped and 50 deselected. See `red-expanded.log`. The expanded cases exercise missing records in both public command branches, real revoke/enable/fetch, four dated retries and the scheduler's reachable reuse returns. |
| Final branch targeted run plus optional LightGBM and web saver | 3,881 passed and 12 skipped, all in one pytest process. See `green-final.log`. |
| Wall clock pinned to 2027-07-15 | All five added or edited test modules ran, with 344 passed and 4 skipped. See `clock-2027.log`. |
| Wall clock pinned to 2028-07-15 | The same five modules ran, with 344 passed and 4 skipped. See `clock-2028.log`. |
| Final fast gate, alone on `30956cdd` | 8,431 passed, 37 failed, 16 errors, 23 skipped, 6 deselected and 1 xfailed in 24m49s. The 53 failure and error IDs exactly matched the baseline and `~/bts-runs/gate-failed.txt`. Wrapper exit 1. See `final-fast.log`. |

All run logs are in `.codex-review/cs5a/`. The RED evidence includes real sends and locks on mismatches, missing serialization and record loss on re-save, missing stale-pick refusal copy, and the missing pure interfaces. The unchanged early guards, optional delivery, automatic drift/restart, genuine skip, and invalid-as-absent loader behavior were GREEN controls on the base. A pure-interface availability failure alone was not used as evidence that the real send paths were guarded.

## Contract rows

| Row | Implementation | Tests |
|---|---|---|
| 1 | `src/bts/picks.py:250`, `:258`, `:397`, `:501`, `:551`, `:608`; archive in `src/bts/scheduler.py:2210`; capture projection in `scripts/live_forward_capture_once.py:190` and inventory in `scripts/live_forward_provenance_inventory.py:65`. | `test_pick_chosen_on_cs5a.py` tests exact earlier encoding, every loader, invalid-as-absent, null/version drift and literal hashes. `test_saved_pick_paths_cs5a.py::test_marker_archive_and_grading_saves_preserve_chosen_on` tests attempt, delivered, posted, grading and archive persistence. |
| 2 | `src/bts/contest_state.py:79` and `:539` classify the selected snapshot. `src/bts/picks.py:298` builds and copies the record. | `test_real_decision_state_identity_and_builder` uses real reads for all four kinds, including a manual file claiming the automatic source, and exercises the required flag and identity copy. |
| 3 | `src/bts/orchestrator.py:282` and `:307`. | `test_fresh_choices_stamp_the_selection_read`, `test_real_policy_outputs_stamp_only_choices`, and `test_real_selection_reuse_preserves_record_or_absence`. |
| 4 | `src/bts/cli.py:1371`. | `test_fresh_choices_stamp_the_selection_read`, `test_uncovered_preview_double_is_refused_then_replaced`, and `test_preview_resaves_live_reuse_without_changing_record`. |
| 5 | `src/bts/cli.py:1239`. | `test_fresh_choices_stamp_the_selection_read` advances the live reading before the second read and checks that the saved stamp is from the first. `test_legacy_public_reuse` checks record preservation through saves. |
| 6 | `src/bts/picks.py:322` uses the same validator as the loaders. | `test_pure_matcher` exercises a dated and automatic oracle, malformed records, both input snapshots in the refusal, and no file or wall-clock access. |
| 7 | `src/bts/scheduler.py:1101`, after the successful required read and before the formatter, cutoff re-check, markers, pick transport and decision. | Same-cycle identity tests for DM and public, all four dated mismatch axes, preview fallback refusal and replacement, valid matches, alert failure, overrun reload, and a full `run_day` restart and recovery sequence. |
| 8 | Existing guards above `src/bts/scheduler.py:1101` and existing optional send branches below it. | `test_early_guards_and_optional_delivery_skip_matcher`, the after-read snapshot control, and the base/branch matrix. |
| 9 | `src/bts/cli.py:1215` and `:1268`; `src/bts/orchestrator.py:395` and `:445`. | `test_legacy_public_reuse` covers locked, strict detailed-status failure, and unlocked Warmup reuse with matching, optional, absent, read-error and applicable window cases. `test_direct_optional_orchestration_preserves_unguarded_reuse` and the fresh-choice tests cover unchanged posts. |
| 10 | `src/bts/scheduler.py:63`; timing, dedup and triggering still use `:85` and the existing health dispatcher. | `test_missed_pick_alert_copy` covers each live-setting source separately and positive, zero, and negative minutes. The restart sequence checks repeat stderr, one contest-state health DM, and a separate candidate-free missed-pick DM. |
| 11 | No edits to selection, fallback refresh or planning, `run_single_check`, health modules, decision schema, dashboard, grading, entry verification, or day-outcome code. | Base/branch targeted runs, grading and hash controls, automatic restart, uncommitted-entry and `no_decision` sequence, and genuine MDP skip exercise these paths. |

Existing test edits retain the original assertions:

- `tests/test_season_contract_cs4b2_edges.py::test_delivery_snapshot_and_guards[after_read]` stamps the cached pick from a captured required snapshot before the delivery read's injected revoke. The delivery read spy still sees one read. The shared opt-in fixture is `stamp_saved_pick` in `tests/conftest.py`.
- `tests/test_live_gate_cs6a.py::test_orchestrate_accepted_cli_loads_once_through_real_core`, `test_run_accepts_dated_and_covered_observations[dated]` and `[covered]`, and `test_run_required_reads_with_real_refusal[new-2]` give their simulated fresh selection a structured policy decision.
- The outside-window half of `tests/test_live_gate_cs6a_review.py::test_bts_run_locked_already_posted_and_dated_corrupt_switch_and_read3_outside_window` gives its simulated fresh single a structured policy decision. It keeps the existing required reread, save and refusal assertions.

## Recorded delivery comparison

The matrix runs the real `_deliver_and_lock_pick` from `git archive 4cab4212 src` and from the branch. The 7,920 cells cross 55 record cases, two live snapshot kinds, three delivery modes, required and optional reads, three marker states, and four selection states. The record cases include missing, null, optional, each kind, each valid dated mismatch, and invalid record combinations. The archived and branch observations are in `.codex-review/cs5a/matrix-base.json` and `matrix-branch.json`. The comparator normalizes only the new pick field.

| Cell group | Identical | Row 7 refusals |
|---|---:|---:|
| Optional configuration, any mode or marker | 3,960 | 0 |
| Required configuration, private mode | 1,320 | 0 |
| Required outward mode, already delivered or attempt marked | 1,760 | 0 |
| Required outward mode, unmarked same-cycle fresh object | 220 | 0 |
| Required outward mode, unmarked saved object | 12 | 648 |
| Total | 7,272 | 648 |

In every changed cell the base sent a DM or public post, marked the pick, locked the day and wrote a delivered decision. The branch returned false and left the exact entry pick bytes in place, with no pick send, attempt, delivery or post marker change, lock or decision write. It printed the refusal and sent a contest-state health alert that updated the existing alert status. Row 7 requires each of these differences. The earlier guard and optional cells were identical, including incident alerts, lock and decision outputs. A branch spy confirmed matcher invocation only in the saved, unmarked, required outward cells.

## Pick-file readers

- The explicit production mapping in `src/bts/picks.py` serves `load_pick` and `load_pick_bytes`. Selection, preview, grading and reconciliation, scheduler reloads, the dashboard commitment check, and health consumers use that mapping. The separate shadow mapping serves scheduler shadow work, shadow evaluation, grading, and `scripts/audit/p03_context_shadow_closeout.py`. `scripts/shadow_predict_once.py` uses the production mapping.
- The dashboard (`src/bts/web.py`), contest freshness (`src/bts/contest_state.py`), calibration (`src/bts/model/calibrate.py`), `bts shadow-compare` and `src/bts/skip_policy_shadow.py` also have readers that decode JSON directly and select known keys. The same applies to the pick-level health readers for calibration, MDP policy alignment, pick entry, pitcher sparsity, post failure, predicted versus realized, projected lineup, realized calibration, and same-team correlation.
- Pick analysis scripts that decode JSON and project known keys include `canonicalize_realized_picks.py`, `evaluate_mdp_dd_guardrail.py`, `production_live_boundary_feasibility.py`, `gate_b_production_metric_rebaseline.py`, `measure_raw_rebin_gate.py`, and `audit_mdp_live_aggressiveness.py` under `scripts/`, and `boundary_shadow_census.py`, `p05_tail_policy_audit.py` and `benchmark_bridge/run.py` under `scripts/audit/`.
- `scripts/cutover/phase2-shadow-diff.py` reads remote JSON and compares selected slot keys. `src/bts/experiment/artifacts.py::load_production_pick_snapshot` retains the entire decoded pick and exact file hash beside its normalized slot projection.
- The live-forward capture and inventory scripts compare decision projections and also track literal bytes or their hashes. The entry receipt emitted by `check-pick-entered` binds the exact bytes returned by `load_pick_bytes`. `scripts/audit/season_ledger/sources/pick_files.py` keeps the raw JSON alongside selected fields and hashes the file bytes for observation identity. Backup copies the file bytes.
- `src/bts/state/export.py` projects historical fields. `src/bts/state/regenerate.py` builds picks from those historical projections or the public timeline. `src/bts/state/verify.py` compares projected names, results and streaks. The lineup evolution writer projects slots and run time. These projections do not contain the chosen-on record.

## Delivery call sites

| Call site | Pick and selection handed to delivery | Applicable rows |
|---|---|---|
| Scheduler lineup check | The current check's `result["pick_result"].daily` and `result.get("selection")`. A same-object policy choice qualifies for the same-cycle exception. An unlocked reuse carries no policy decision. | 7–8 |
| In-loop fallback | The refresh's `daily` and `selection`. A successful fresh refresh can supply the same object and its policy decision. A failed or no-pick refresh can supply the cache, and a reuse carries no policy decision. The overrun path reloads the pick from disk and forwards the check's selection. Production `run_single_check` does not supply the `lock_decision` key that activates that path. | 7–8 |
| Final fallback | The refresh's `daily` and `selection`. The same fresh, reuse and cached distinctions apply. Standing genuine MDP skips suppress delivery. | 7–8 |
| Scheduler classification and restart | Local recovery can lock a previously delivered, attempted or otherwise classified pick with no outward send. A still-enterable saved pick may proceed to a later lineup or fallback send. | 7–8, 11 |
| Scheduler delivery guard exits | An already delivered pick restores the lock and commit. A prior attempt locks as unconfirmed and alerts. Private mode locks locally after the void and cutoff checks. Initial all-void and late cases can archive and alert without sending a pick. | 8 |
| `bts run` and `bts orchestrate` catch-up | A locked, unposted reuse can reach public formatting after a second read independently of the regular posting window. The strict detailed-status failure reuse can arrive here. The scheduler's production check and fallback never enable strict mode. | 9 |
| `bts run` and `bts orchestrate` regular post | The selected object may be fresh or an unlocked reuse. The existing provenance save precedes the second read and window check. | 3, 5, 9 |

## Contract limits

- A refusal does not choose again, lock, or write a decision record. Existing scheduled checks and fallback refreshes may later replace, retry, archive or classify the candidate. Contest-state alerts retain their per-ET-day and per-source deduplication. Missed-pick alerts keep their existing gates. With no later commitment or genuine MDP skip, a verifiable playable slate under valid entry intent yields `no_decision`. An on-demand check is CS5b.
- A same-cycle fresh choice can be delivered after a successful delivery read that saw a different observation. The message can show a different streak from the selection input. An edit can race the send after a successful read.
- Required automatic records match on kind alone. A subsequent fetch, saver-flag edit, or reactivation and fetch may change the reading enough that a previously chosen double goes out at a high streak after a failed refresh. An automatic match can resume after a dated override lapses.
- A newly chosen preview pick has an optional record and cannot match a live saved-pick check, even when its observation agrees with the live read. If live selection keeps failing, the day may have no pick.
- A dated record loses its match at midnight ET or when any of its four values changes.
- Private and optional delivery bypass the matcher. Entry verification, incident alerts and the dashboard may show the saved names.
- Pre-CS5a files have no chosen-on record and cannot match a live saved-pick check until selection makes a fresh choice.
