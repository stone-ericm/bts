# K2 author handover

Commit `26c621c7` on `fix/wr-k2`, based on `712351a5`. The staged patch and commit had separate approvals. The handover is outside the commit.

## Contract results

| Row | Base | K2 |
| --- | --- | --- |
| E1, streak 57 | RED. Selection skipped under reach57 with 0 days. | Double under longest_streak at 1 day. |
| E2, streak 120 | RED. Same raw-zero skip. | Double under longest_streak at 1 day. |
| E3, streak 55 and 56, both saver inputs | RED. Both skipped. | Double at 55 and single at 56, both reach57 at 1 day. |
| E4, streak 20 and best 20 or 25 | RED. Both returned the raw-zero skip without a reason. | Double with best 20. Tail stop with best 25 and maximum reachable 22. The resolver probe also doubled with stale best 18, normalized to 20. |
| E5, 2028-09-29 | RED for every selection case. | Same one-day results as 2027, with a fallback-year advisory. |
| E6, eligible morning start | RED. No advisory. | One WARN on first startup and one observation in the journal per qualifying start. Same-date repeats and 1 → 2 → 1 game counts sent only once. |
| E7, makeup walk | RED for all six states. Raw and effective counts were both 0. | All six walk outcomes match the one-day table. Raw count remains 0 and the independent oracle expects effective 1. |
| S1, positive days | Passing control. | 373,188 records match the base digest `7a8f719bafa7017e2f24327c1b5984ca0ed368eaa2838d20906971625e8177c7`, including all 2027 playing dates with positive raw days. |
| S2, health and direct raw callers | Passing controls. | Seven health snapshots, bare contexts and the standalone raw-zero display bar match. The existing no-decision bar regressions passed. |
| S3, positive-day morning | Passing control. | First check at 16:05 ET with no boundary advisory. |
| S4, empty/off-season and nonregular slate | Passing controls. | Empty, postseason-only and spring-only schedules idle. Empty walk has no decision. |
| S5, morning check | Base refusal body and first lineup time recorded. The first scratch check had lost the refusal when boundary transport ran first. | Refusal body matches byte for byte and goes first. An 11-second morning send takes the remaining lead, so only the morning DM goes out. An 11-second boundary send leaves first check at 16:05 ET. |
| S6, dry run | Passing control. | No boundary observation or DM. |
| S7, decision record | Base v3 schema. A raw-zero selection recorded the reach57 skip. | v3 records retain the schema and carry the decision objective. A saved longest-streak pick carries effective 1 in its policy decision, and the tail skip decision also has effective 1 in memory. |

The pre-edit local pytest run reproduced 31 feature failures and 8 passing startup controls. Subsequent pre-edit subsets reproduced the receipt, fallback and delay failures and passed the health and grid comparisons. The empty-walk, real-morning and saved-record tests added after the first implementation use the recorded baseline probe for before behavior. The completed K2 file passed all 59 tests.

## Startup and restart cases

| Case | Before | After |
| --- | --- | --- |
| Missing type, future official date, missing or blank status, null status alongside a good row | No boundary advisory on base. Amendment 1 either counted the invalid row or lost the valid one alongside null status. | Only the explicit regular-season, same-date row with a usable, non-void detailed status counted. |
| Postponed, canceled, or resumed game only | No boundary advisory on base. Version 1 warned on postponed and resumed games. | No boundary advisory. |
| Midnight ET, year boundary and fallback year | No boundary advisory on base. Empty December 31 and January 1 idled. | Empty year-boundary days still idle. Starting after midnight for the old slate journals the old slate date and suppresses transport for insufficient lead. The fallback-year warning labels the fallback explicitly. |
| Malformed slate date and failed schedule read | Existing failure paths and no boundary advisory. | Warning parse failure is contained. An unreadable schedule still raises before the checks. |
| Failed, blank-ID, null-ID, interrupted transport, or receipt write failure | No boundary transport on base. Version 1 suppressed a retry after a null ID. | A later eligible start retries. A good acknowledgment then deduplicates. The worker interruption leaves the main path and next start eligible. |
| Malformed or schema-less receipt, directory, FIFO, symlink and hardlink | No dedicated receipt on base. | Invalid regular receipts can be replaced. Special and aliased entries remain unchanged and do not suppress a retry. |
| D1 → D2 → D1 | No boundary transport on base. Version 1's single receipt could resend D1. | Independent date files suppress a second D1 send. |
| No plan, no recipient, late restart, journal failure | No boundary advisory on base. | Journal gives skip reason when plan, recipient or lead is unavailable. An injected journal failure still allowed transport. |
| Exact ten-minute lead and first-check timing | Base thin-slate plan recorded. Scratch synchronous transport could move the first check and lose an early opener. | Exact ten minutes qualifies. At 9m59s, journal only. With a blocked worker and a test-controlled bounded join, the ordinary first check remained 16:05 ET and the early opener remained 10:15 ET with its game eligible. The older instant-clock-jump probe still advances its shared fake clock when the send returns immediately. |
| Private, DM and public mode | No boundary advisory on base. | All three modes emitted the advisory to the captured recipient. |

## Observed one-day policy

With the shipped artifacts, p=.80, doubles allowed, a different-game partner and a beatable trusted best. Both saver inputs produced the same results.

| Streak | Saver off | Saver on | Objective |
| --- | --- | --- | --- |
| 0–54 | Double | Double | `emax_season_best` |
| 55 | Double | Double | `reach57` |
| 56 | Single | Single | `reach57` |
| 57–60 and 120 | Double | Double | `longest_streak` |

Every table row had effective days 1, null guard and null decision bar. At streak 20, trusted best 25 skips because the maximum reachable streak is 22. The existing availability and double clamps continue to apply. The stop bound uses two potential hits even when only one leg can execute.

## Captured messages

Configured year, 626 characters including the health DM header and source prefix.

```text
BTS health WARN attention:
- [season_boundary] Season boundary behind the schedule for 2027-09-27 ET. At scheduler startup MLB listed at least one eligible regular-season game on that date. The configured exclusive boundary is 2027-09-27, giving a raw count of zero. With the raw count at zero, a fresh pick selection with an available predicted candidate uses the last-day policy. Please send this advisory to the BTS lead or Eric. Check the last regular-season game date on MLB's updated schedule, including pending makeup games, before arranging an update for a later run. Keep any pick already sent or entered as recorded.
```

Unconfigured year, 655 characters including the health DM header and source prefix.

```text
BTS health WARN attention:
- [season_boundary] Season boundary behind the schedule for 2028-09-29 ET. At scheduler startup MLB listed at least one eligible regular-season game on that date. The fallback exclusive boundary for the unconfigured year 2028 is 2028-09-28, giving a raw count of zero. With the raw count at zero, a fresh pick selection with an available predicted candidate uses the last-day policy. Please send this advisory to the BTS lead or Eric. Check the last regular-season game date on MLB's updated schedule, including pending makeup games, before arranging an update for a later run. Keep any pick already sent or entered as recorded.
```

```text
SEASON BOUNDARY BEHIND SCHEDULE — 2027-09-27 ET: 1 eligible regular-season game(s). The configured exclusive boundary is 2027-09-27, giving a raw count of zero.
SEASON BOUNDARY BEHIND SCHEDULE — 2028-09-29 ET: 1 eligible regular-season game(s). The fallback exclusive boundary for the unconfigured year 2028 is 2028-09-28, giving a raw count of zero.
```

## Regression batches

Search-selected pytest files and the tail-health file ran sequentially with the requested 110-second bounds. There were 2,826 passes, 17 skips and one expected failure. The audit driver had the same 33 import failures and 16 setup errors reported by both contract checks for missing GNU rsync. Source scripts and documentation matched by the search were not pytest files. The scheduler and public CLI contract probes also ran with captured transports. The system denied the requested `nice` priority reduction and the commands continued at the available priority.

| Batch | Test files | Result |
| --- | --- | --- |
| 1 | `tests/test_audit_driver.py` | 33 failed, 16 errors in 1.47s |
| 2 | `tests/test_wr_g_guard.py` | 44 passed in 3.57s |
| 3 | `tests/test_wr_h_longest.py` | 89 passed in 15.24s |
| 4 | `tests/test_wr_k2_season_boundary.py` | 59 passed in 9.56s |
| 5 | `tests/experiment/test_factored_runner_bitexact.py`, `tests/experiment/test_integration.py`, `tests/experiment/test_registry.py`, `tests/experiment/test_run_selection_feature_cols.py`, `tests/experiment/test_runner.py`, `tests/experiment/test_runner_nested_split.py`, `tests/experiment/test_strategies.py` | 42 passed, 5 skipped in 4.61s |
| 6 | `tests/health/test_realized_calibration.py`, `tests/health/test_tail_policy_health.py`, `tests/model/test_calibrate.py`, `tests/scripts/season_walk/test_season_walk.py`, `tests/scripts/test_bin_collapse_read.py`, `tests/scripts/test_leaderboard_mechanism_mining.py`, `tests/scripts/test_mining87_inference.py` | 105 passed in 1.81s |
| 7 | `tests/scripts/test_mining87_production.py`, `tests/scripts/test_mining87_units.py`, `tests/scripts/test_p05_tail_policy_audit.py`, `tests/simulate/test_exact.py`, `tests/simulate/test_mdp.py`, `tests/simulate/test_mdp_save_atomic.py`, `tests/simulate/test_monte_carlo.py` | 75 passed in 2.12s |
| 8 | `tests/simulate/test_rare_event_mc.py`, `tests/simulate/test_strategies.py`, `tests/test_alert_status_callers_cs6b2.py`, `tests/test_alert_status_cs6b2_carbon.py`, `tests/test_cli_integration.py`, `tests/test_cli_schedule_date.py`, `tests/test_cold_start_integration.py` | 137 passed, 1 xfailed in 7.75s |
| 9 | `tests/test_contest_dated_cs4a_review.py`, `tests/test_cutoff_candidate_exclusion.py`, `tests/test_decide_action.py`, `tests/test_incident_2026_08_30.py`, `tests/test_incident_register_2026.py`, `tests/test_live_gate_cs6a.py`, `tests/test_live_gate_cs6a_review.py` | 201 passed in 1.37s |
| 10 | `tests/test_morning_state_cs6b1.py`, `tests/test_morning_state_cs6b1_carbon.py`, `tests/test_morning_state_cs6b1_review.py`, `tests/test_orchestrator.py`, `tests/test_pickpath_review_pp1.py`, `tests/test_pickpath_review_pp2.py`, `tests/test_private_mode_transport.py` | 179 passed, 8 skipped in 2.17s |
| 11 | `tests/test_saved_pick_cs5a_review.py`, `tests/test_saved_pick_paths_cs5a.py`, `tests/test_saver_cs3.py`, `tests/test_saver_cs3_review.py`, `tests/test_scheduler.py`, `tests/test_scheduler_decision_record_integration.py`, `tests/test_scheduler_eve_of_break.py` | 461 passed, 4 skipped in 3.13s |
| 12 | `tests/test_scheduler_no_games.py`, `tests/test_scheduler_skip_visibility.py`, `tests/test_scheduler_state_integrity.py`, `tests/test_scheduler_tail_objective.py`, `tests/test_season_end_date.py`, `tests/test_shadow_scheduler.py`, `tests/test_skip_policy_shadow.py` | 107 passed in 75.63s (0:01:15) |
| 13 | `tests/test_stale_delivery_cs4c.py`, `tests/test_stale_delivery_cs4c_carbon.py`, `tests/test_strategy.py`, `tests/test_streak_season_cs2.py`, `tests/test_streak_season_cs2_review.py`, `tests/test_streak_season_cs2b_review.py`, `tests/test_tail_policy_artifact_contract.py` | 533 passed in 2.04s |
| 14 | `tests/test_tail_policy_r3_fixes.py`, `tests/test_tail_policy_strategy.py`, `tests/test_thin_slate_checks.py`, `tests/test_wr_a_calendar.py`, `tests/test_wr_a_calendar_replay_carbon.py`, `tests/test_wr_a_pick.py`, `tests/test_wr_a_review_carbon.py` | 136 passed in 2.22s |
| 15 | `tests/test_wr_a_schedule.py`, `tests/test_wr_c1_history.py`, `tests/test_wr_c2_carbon_v2.py`, `tests/test_wr_c2_r2_carbon.py`, `tests/test_wr_c2_receipt_identity.py`, `tests/test_wr_c2_recovery.py`, `tests/test_wr_c3_r2_entered_pick_blockers.py` | 357 passed in 7.09s |
| 16 | `tests/test_wr_c3v_delivery.py`, `tests/test_wr_e_texts.py`, `tests/test_wr_f3_skip_state_review.py`, `tests/test_wr_h2_caps.py`, `tests/test_wr_k1_refusal_dm.py`, `tests/validate/test_rare_event_mc_eval.py` | 301 passed in 33.56s |

## Known limits

The CLI probe confirmed the contract's owner item. With injected prior-date or postseason predictions and matching game-status IDs, both `bts run` and `bts orchestrate` reached the public double-post branch. The probe captured those posts locally. Candidate type and date validation remain with the producing path. The owner list is outside the authorized worktree.

Each eligible raw-zero slate is evaluated as the last day, including when more than one makeup date remains. The morning advisory does not detect a later finale while the raw count is positive. A worker can outlive its waiting deadline and an unconfirmed attempt can repeat. The shared acknowledgment implementation can read an unbounded regular receipt, does not serialize simultaneous starts, and uses its existing failure label.
