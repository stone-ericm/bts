# D3 handover

Base `24019a9a`. D3 commit `d3d05abdc4fdde8a534b0fc4e58b466a359a5c06`. Committed-tree gate matched the baseline.

## Box scripts

Exact shell literals: [base deploy](.codex-review/wr-d3/base-deploy.sh), [candidate deploy](.codex-review/wr-d3/candidate-deploy.sh), and [candidate recovery](.codex-review/wr-d3/candidate-recovery.sh).

- Base deploy and its rollback: the sole `with.script` in `git show 24019a9a:.github/workflows/deploy.yml`. It fetches and resets, syncs, cleans legacy units, restarts, waits, then checks the scheduler, dashboard and HTTP. On an unhealthy result it attempts a reset to the old SHA and another restart. Its pass line precedes the stamp.
- Base recovery: no SSH action or shell literal.
- Candidate deploy: the first `with.script` in [.github/workflows/deploy.yml](.github/workflows/deploy.yml). It acquires the root lock, checks the target and unit identities, publishes a same-attempt begin, disables the scheduler's local boot link, resets, syncs, stops and cleans checked legacy units, restarts, verifies HEAD, and runs the four canary checks after the full wait. It stages and fsyncs the stamp, enables and verifies the boot link, renames and directory-syncs the stamp, and verifies a committed record before logging PASS. A detected failure after authorization attempts independent stop proof, atomic heartbeat replacement and an immediate failure POST.
- Candidate recovery: the second literal. It uses the same frame, helpers and containment function, binds recovery to the full target and both attempt IDs, marks absent or another attempt under the lock, leaves a matching committed attempt alone, and rechecks incomplete or contained attempts. The retained `verified` page value is prior HTTP evidence. The failure heartbeat is written and checked again.

F6 pinned action evidence is the supplied local read-only file [.codex-review/wr-d3/ssh-action-0ff4204d-action.yml](.codex-review/wr-d3/ssh-action-0ff4204d-action.yml) from commit `0ff4204d59e8e51228ff73bce53f80d53301dee2`. It declares the `10m` default, `envs`, and `request_pty`. The deploy and recovery override the command limits to 20m and 35m. Both omit `request_pty`.

## Passing transcripts

These are the complete executed shim argv arrays and the exact workflow stdout, with the documented fixed paths mapped into the temporary box. Each delta is a list of `[operation, baseStart, baseEnd, candidateStart, candidateEnd]` spans over the two argv arrays. The test reconstructs the entire candidate order from the base and the delta.

| Legacy scope | Base argv | Candidate argv | Ordered delta |
|---|---|---|---|
| Absent | [31 calls](.codex-review/wr-d3/coverage-final-tmp/test_passing_transcript_order_0/baseline-transcript.json) | [205 calls](.codex-review/wr-d3/coverage-final-tmp/test_passing_transcript_order_0/candidate-transcript.json) | [19 spans](.codex-review/wr-d3/coverage-final-tmp/test_passing_transcript_order_0/delta.json) |
| Present | [35 calls](.codex-review/wr-d3/coverage-final-tmp/test_passing_transcript_order_1/baseline-transcript.json) | [245 calls](.codex-review/wr-d3/coverage-final-tmp/test_passing_transcript_order_1/candidate-transcript.json) | [23 spans](.codex-review/wr-d3/coverage-final-tmp/test_passing_transcript_order_1/delta.json) |

The order adds lock and state preflight before publication, the checked soft hold before reset, checked individual legacy disables and stops before restart, HEAD and the contest-state check, then stamp stage and checked boot-link enable before stamp rename and record commit. The base displayed the deployed line before its canary and stamped after PASS. The candidate displays deploy and PASS after the checked commit.

## Coverage and RED evidence

The initial tests were run on the untouched base before production edits. The RED row run recorded 59 failed and 16 passed. The active and auto-restart cases were green controls. The new workflow shape, healthy wait, rollback removal, fetch line, sync containment, commit record, recovery-first cancellation and operator wording failed. The new heartbeat and canary tests failed as expected. Some of the fuller hostile and parity cases were added during coverage review afterward. The pinned-base replay adds 93 RED cases and the same two green controls. The logs preserve individual test IDs. Ordinary heartbeat parity is a green control against the base file. The new canary scenarios require a CLI command absent on the base.

| Contract area | Shim or pytest coverage |
|---|---|
| F0 and F1, preflight and deadlines | `test_healthy_wait_deadline`, `test_expired_shared_canary_budget_skips_remaining_commands`, `test_preflight_checks_refuse_before_phase_two`, `test_preflight_failure_leaves_checkout_and_services_alone`, `test_preflight_refuses_a_missing_checkout`, `test_bad_inputs_are_refused_in_both_scripts`, `test_actual_lock_contender_cannot_deploy`, `test_invalid_lock_paths_are_refused`, `test_owner_death_does_not_release_lock_while_mutator_is_live`, `test_mutator_does_not_inherit_lock_descriptor`, and the actual EPIPE case. |
| Passing and legacy controls | `test_healthy_scheduler_passes`, `test_passing_controls_for_legacy_fallback_and_publication` and `test_passing_transcript_order_delta` with and without October fragments. The latter records the complete before and after argv and stdout, and a replayable index delta. |
| F2 phase operations | `test_failing_phase_command_contains_first_failure` injects a nonzero result and a timeout per operation, including legacy disable, stop, delete, reload and residual cleanup. `test_head_mismatch_contains_without_rollback` covers wrong HEAD and the legacy-present pass covers pkill exit 1. |
| F3 canary | `test_failed_checks_are_reported_and_contained`, `test_malformed_and_unhealthy_canary_states`, `test_canary_timeout_is_reported`, `test_canary_checks_continue_after_first_failure`, `test_http_failure_leaves_candidate_checkout`, and the tailnet fallback pass control. |
| F4 stamp and commit | `test_stamp_precedes_pass`, `test_stamp_failure_preserves_previous_entry`, `test_stamp_write_and_dirsync_faults_keep_complete_bytes`, the stamp and enable failure matrix, the commit-record failure, and `test_committed_deploy_survives_a_failed_final_log`. |
| F5 containment | `test_stop_verification_requires_all_evidence`, `test_marker_replacement_and_verification`, `test_invalid_env_never_posts`, `test_post_only_uses_a_valid_literal_https_endpoint`, and `test_post_requires_exit_zero_and_2xx`. These include residual PID, job, descendant cgroup, legacy survivor, bulk failure, marker path types, and bad HTTP status or transport. |
| Signals and simulated power loss | `test_owner_signal_at_each_phase_boundary`, `test_signals_at_publication_and_after_commit`, `test_second_signal_does_not_reenter_containment`, `test_literal_epipe_does_not_change_deploy_outcome`, `test_owner_death_does_not_release_lock_while_mutator_is_live`, and `test_boot_link_and_record_loss_at_boundaries`. The power-loss test uses owner death and removal of the temporary `/run` analogue. Actual boot, external runner cancellation and hosted SSH loss are A7. |
| F6 recovery and next attempt | `test_recovery_first_cancels_late_deploy`, `test_attempt_record_controls_authorization`, `test_recovery_does_not_mutate_services_when_mark_fails`, `test_recovery_checks_a_fresh_marker_and_keeps_verified_post`, `test_recovery_rechecks_each_unverified_result`, `test_old_recovery_during_ownership_transfer`, `test_older_recovery_does_not_mutate_newer_commit`, `test_new_attempt_can_deploy_after_containment`, and lock contention. An unreachable host never supplies Bash with a process to test. SSH reachability, Actions step outcome and the provider receipt require the A7 rehearsal. |
| F7 logs | The output assertions in the phase matrix, multiple-failure test, case-category and malformed-output checks, EPIPE case, final-log failure and passing transcript. |
| F8 heartbeat | `tests/test_heartbeat_staleness.py` exercises the new state ahead of timestamps and churn, field sanitization, FIFO, oversize, bad encoding and timestamps, `/fail`, and side-by-side base output, exit and pings under a pinned clock. `tests/test_heartbeat_churn.py` retains the existing controls. |
| F9 contest read | `tests/test_deploy_canary.py` exercises delivery and flag combinations, shared live gate and morning read, refusal matrix, stale/fresh automatic and dated precedence, descriptor replacement, FIFO, midnight and year transition, delivered/uncertain send skips, and the no-write/no-transport case. The existing `test_margin_uses_earliest_planned_target` and `test_ineligible_start_has_no_effects` in `tests/test_morning_state_cs6b1.py` check imminent and off-day suppression. Canary parity compares the read at the same ET instant where the morning read is reached. |
| F10 docs | `test_operator_documentation_records_recovery_eligibility` and the checklist and incident text reviewed against the amendments. |

## Deadlines

The individual budgets include grace. The lower available phase budget applies before command launch. The normal grace is 5 seconds. The exact healthy wait is `timeout -k 2 33 sleep 30` with a 35-second total cap.

| Scope | Total cap and command budgets |
|---|---|
| Deploy preflight | 75s. Fetch 45s and other operations 10s each, including lock filesystem validation, immediate flock, record read, UID, Git config and identity, user units, legacy identity and jobs. |
| Phase 2 to committed | 800s. Reset 45s, sync 480s, restart 75s, legacy operations 20s each, local enable/disable/check 20s each, HEAD and record and stamp operations 10s each, wait 35s with 2s grace. Scheduler/dashboard/tailnet queries 20s, HTTP 15s total and curl's own `-m 10`, contest command 60s. |
| Containment | 280s. Stop calls up to 60s independently per scope. User stop phase ends by 80s, legacy by 170s, proof by 220s, marker by 240s, POST by 270s, and record by 280s. Discovery and disable sub-caps are 10s, most proof, marker, and POST sub-caps 20s. Staged cleanup, publication and bounded read-back each use a 10s individual cap within the remaining allowance. |
| Recovery | 35m action limit. Pre-containment allowance 1580s, including up to 1500s flock wait plus 5s grace. On authorization containment has its own 280s cap. |
| Frame and logs | Monotonic time sample and best-effort log child each use 1s before TERM plus 5s grace. |

The dashboard HTTP wrapper sends TERM at 10s, the same as curl's own end-to-end limit. An otherwise healthy response arriving exactly at that boundary can be a timeout. The published cap stays 15s.

## EnvironmentFile and operator changes

The installed scheduler template reads `EnvironmentFile=/home/bts/projects/bts/.env` and executes `uv run bts schedule --config /home/bts/.bts-orchestrator.toml`. The canary uses the absolute synced venv and opens the same config path. Manager environment overrides can change the installed scheduler's interpreter or semantics even when the canary accepts. A7 verifies the supported candidate runtime, heartbeat path and Debian systemd status/output pairs. The workflow avoids sourcing `.env` for its immediate failure POST and reads one literal bounded URL assignment.

Updated `INCIDENT.md` §§1, 2, 4, 5 and 6, `CLAUDE.md` Deployment and L1–L12, and `docs/ops/2027-season-start.md` A5, A7, B8a, B8b and B8c. The staged CLAUDE.md review restored the cloud and phone scope and cookie re-capture limits, the dispatch default to main and the production gate, and the instruction against manually restarting after pushing. The installed heartbeat cron template invokes `/usr/bin/python3` with the existing arguments, prefix and logging.

## Runs

- Initial RED row run: [red-rows.log](.codex-review/wr-d3/red-rows.log), 59 expected failures and 16 passes.
- Python 3.12 first focused run: [green-python.log](.codex-review/wr-d3/green-python.log), 74 passes.
- Python 3.11.18 standalone: [python311-final.log](.codex-review/wr-d3/python311-final.log), 60 passes including pinned base parity. The first invocation included the project-wide conftest and failed at setup because that interpreter did not have the 3.12 project installed. The standalone runs used `--noconftest` for the stdlib checker tests.
- Corrected signal harness, publication and Python smoke run: [coverage-smoke.log](.codex-review/wr-d3/coverage-smoke.log), 131 passed. The earlier expanded run spanned the stdout correction and reported 266 passes and 39 signal failures, so it was not used to qualify the candidate.
- Full candidate matrix: [coverage-final.log](.codex-review/wr-d3/coverage-final.log), 365 passed and one incorrect harness assertion failed. The test reached the expected contest-state failure, then compared the argv to the shim pathname instead of the injected absolute candidate venv pathname. The corrected assertion passed on the unchanged implementation in [venv-case.log](.codex-review/wr-d3/venv-case.log).
- Pinned-base replay of the expanded phase, canary, stop, signal, boot, record and transcript cases: [red-late.log](.codex-review/wr-d3/red-late.log), 93 expected failures and two passing controls.
- First committed gate on `d3d05abd`: 10,974 passed, 47 failed, 16 errors, 28 skipped, 86 deselected and one xfailed. All 53 expected IDs appeared. My worktree-local `TMPDIR` put scratch paths inside the checkout and lengthened receipt paths. This also explains the audit test which placed its supposed foreign-module probe inside the checkout, so its check did not exit before reading the absent calendar. [First gate's extra IDs](.codex-review/wr-d3/gate-first-delta.txt)
- Exact prescribed gate retry on the same commit, using the wrapper's default environment: 10,984 passed, 37 failed, 16 errors, 28 skipped, 86 deselected and one xfailed in 39m15s. The 53 failing or error IDs matched `~/bts-runs/gate-failed.txt` exactly, with zero extra and zero missing. Pytest exited 1 because the baseline contains failures. [Final gate output](/Users/eric/bts-runs/gate.out.d3-d3d05abd).
