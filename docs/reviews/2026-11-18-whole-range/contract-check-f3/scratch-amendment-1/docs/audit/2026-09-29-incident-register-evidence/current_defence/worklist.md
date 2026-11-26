# Task 5/6 worklist — 30 fixed Tier-A observed incidents (drafts as of 2026-09-29)

## I-004 (E04) plan_named=False — 4/01 compute tiers failed: predict-json stdout carried log lines (Mac 'Invalid JSON output'); Alienware build_
- link 1: fix 1929f5a | deployed unknown | tests touched: none
- link 2: fix cb4c1cc | deployed unknown | tests touched: none
- link 3: fix 64ae365 | deployed unknown | tests touched: none

## I-009 (E09) plan_named=False — 4/04 confirmed pick never locked: a projected batter from a postponed game held the early-lock gap open
- link 1: fix 7638af7 | deployed unknown | tests touched: tests/test_scheduler.py

## I-011 (E11) plan_named=False — 4/05 1am check-results re-applied the scheduler's already-recorded 4/04 result: local streak 4 instead of 2
- link 1: fix 5e02676 | deployed unknown | tests touched: tests/test_cli_integration.py

## I-017 (E17) plan_named=False — 4/10 the Fly migration instance posted picks to the public Bluesky feed as a second production bot (shadow_mod
- link 1: fix 5075fba | deployed unknown | tests touched: none

## I-018 (E18) plan_named=False — 4/11 Hetzner deploy runs reported success while git pull failed (safe.directory) and the restart hit the wrong
- link 1: fix 870cede | deployed candidate_ancestry d5c466c live_by 2026-04-12T15:52:42Z | tests touched: none
- link 2: fix cda8c1c | deployed candidate_ancestry d5c466c live_by 2026-04-12T15:52:42Z | tests touched: none

## I-019 (E19) plan_named=False — 4/11–4/12 every Hetzner cron job died on its first line: `source .env` under /bin/sh (dash)
- link 1: fix d5c466c | deployed candidate_ancestry d5c466c live_by 2026-04-12T15:52:42Z | tests touched: none

## I-020 (E20) plan_named=False — 4/11–4/12 two scheduler units ran concurrently; every killed orphan was respawned by the other unit
- link 1: fix 8193909 | deployed candidate_ancestry 8193909 live_by 2026-04-12T16:37:36Z | tests touched: none
- link 2: fix 8193909 | deployed candidate_ancestry 8193909 live_by 2026-04-12T16:37:36Z | tests touched: none

## I-023 (E23) plan_named=False — 4/12 shadow pick file overwritten with production's pick: a real DISAGREES day recorded as AGREES
- link 1: fix 634768e | deployed candidate_ancestry 634768e live_by 2026-04-12T18:33:34Z | tests touched: tests/test_shadow_scheduler.py, tests/test_strategy.py
- link 2: fix 80fd31f | deployed candidate_ancestry 80fd31f live_by 2026-04-12T18:46:31Z | tests touched: tests/test_shadow_picks.py

## I-025 (E25) plan_named=False — 4/12 fallback posted the 13:19 cached pick although a lineup confirmed since then had produced a stronger cand
- link 1: fix dd25664 | deployed candidate_ancestry dd25664 live_by 2026-04-12T19:25:45Z | tests touched: tests/test_scheduler.py

## I-027 (E27) plan_named=False — 4/13–4/14 the 3am pull/build/preview chain died on `bts data pull` without --start/--end: no fresh blend or pr
- link 1: fix 28732c2 | deployed candidate_ancestry 28732c2 live_by 2026-04-14T15:47:20Z | tests touched: none

## I-028 (E28) plan_named=False — 4/14 deploy run green while git pull failed on box-local uv.lock drift; services restarted on stale code
- link 1: fix 6bffd63 | deployed candidate_ancestry 6bffd63 live_by 2026-04-14T23:47:55Z | tests touched: none
- link 2: fix 6bffd63 | deployed candidate_ancestry 6bffd63 live_by 2026-04-14T23:47:55Z | tests touched: none

## I-029 (E29) plan_named=False — 4/15 02:00 reconcile rewrote the local streak to 0: the backward walk broke on the pre-generated preview pick 
- link 1: fix 1d61908 | deployed candidate_ancestry 1d61908 live_by 2026-04-15T16:05:33Z | tests touched: tests/test_picks.py

## I-032 (E32) plan_named=False — 4/22 new heartbeat monitor paged Healthchecks falsely: no heartbeat during cold-wake predictions and during th
- link 1: fix 0830ef4 | deployed candidate_ancestry 0830ef4 live_by 2026-04-22T17:24:12Z | tests touched: tests/test_heartbeat.py
- link 2: fix c43869d | deployed candidate_ancestry c43869d live_by 2026-04-22T20:30:30Z | tests touched: tests/test_scheduler.py

## I-034 (E34) plan_named=False — 4/22-23 systemd watchdog SIGABRT killed the sleeping scheduler every ~30.5 min overnight (NRestarts=21)
- link 1: fix 684c160 | deployed candidate_ancestry 684c160 live_by 2026-04-23T14:24:40Z | tests touched: tests/test_scheduler.py

## I-035 (E35) plan_named=False — 4/23 end-of-day clean exit relaunched by Restart=always in a loop (NRestarts 0 to 7 in 25 min)
- link 1: fix ddee3db | deployed candidate_ancestry ddee3db live_by 2026-04-23T22:59:15Z | tests touched: tests/test_scheduler.py

## I-040 (E40) plan_named=False — 4/29-30 bpm promoted to FEATURE_COLS but never built in the inference path: every scheduler prediction crashed
- link 1: fix ee4190f | deployed candidate_ancestry ee4190f live_by 2026-04-30T16:27:37Z | tests touched: none
- link 2: fix bb7ccc1 | deployed candidate_ancestry df01882 live_by 2026-06-11T14:36:14Z | tests touched: tests/test_e3_missed_pick_alert.py

## I-042 (E42) plan_named=False — 4/29-5/01 realized_calibration paged a daily CRITICAL on a DD-attribution-biased, iteration-pooled signal
- link 1: fix b08769d | deployed candidate_ancestry b08769d live_by 2026-05-01T13:50:49Z | tests touched: tests/health/test_realized_calibration.py, tests/model/test_calibrate.py
- link 2: fix c5256e6 | deployed candidate_ancestry c5256e6 live_by 2026-05-01T19:12:35Z | tests touched: tests/health/test_realized_calibration.py

## I-043 (E43) plan_named=False — 5/05 unposted pick stayed committed to a postponed game (abstract-status lock); follow-on void grading, pollin
- link 1: fix 7b701c9 | deployed candidate_ancestry 293ce51 live_by 2026-05-09T03:34:05Z | tests touched: tests/test_scheduler.py, tests/test_strategy.py
- link 2: fix a142c14 | deployed candidate_ancestry a142c14 live_by 2026-05-09T23:02:03Z | tests touched: tests/experiment/test_artifacts.py, tests/health/test_predicted_vs_realized.py, tests/health/test_same_team_corr.py, tests/test_cli_integration.py, tests/test_picks.py, tests/test_posting.py, tests/test_scheduler.py, tests/test_shadow_eval.py, tests/test_web_render.py
- link 3: fix 68e4cdb | deployed candidate_ancestry ac4a959 live_by 2026-05-09T23:21:06Z | tests touched: tests/test_scheduler.py
- link 4: fix 0799bf2 | deployed candidate_ancestry 0799bf2 live_by 2026-05-24T11:39:41Z | tests touched: tests/test_cli_integration.py, tests/test_orchestrator.py, tests/test_picks.py, tests/test_scheduler.py, tests/test_shadow_scheduler.py, tests/test_strategy.py
- link 5: fix abbfdc5 | deployed candidate_ancestry d63f60b live_by 2026-05-26T02:37:06Z | tests touched: tests/test_orchestrator.py, tests/test_scheduler.py, tests/test_strategy.py

## I-047 (E47) plan_named=False — 5/08-09 inline shadow generation ran without watchdog pings and re-ran after restarts: scheduler restart loop;
- link 1: fix 150ed4c | deployed candidate_ancestry a142c14 live_by 2026-05-09T23:02:03Z | tests touched: tests/test_shadow_scheduler.py
- link 2: fix 150ed4c | deployed candidate_ancestry a142c14 live_by 2026-05-09T23:02:03Z | tests touched: tests/test_shadow_scheduler.py

## I-056 (E56) plan_named=False — 5/29-6/06 contest state silently froze at a stale manual value (7) while the real streak was 0: decisions ran 
- link 1: fix fa30ecb 68a35d3 | deployed candidate_ancestry eaccaf8 live_by 2026-06-06T20:25:30Z | tests touched: tests/test_cli_integration.py
- link 2: fix 2b4ff1d | deployed candidate_ancestry 9958874 live_by 2026-06-17T21:09:18Z | tests touched: tests/test_contest_state.py
- link 3: fix 365ea40 58c9adc | deployed candidate_ancestry eaccaf8 live_by 2026-06-06T20:25:30Z | tests touched: tests/health/test_contest_state.py

## I-061 (E61) plan_named=False — 6/11 delivered pick never entered in the MLB app; nothing alerted until the operator noticed the frozen streak
- link 1: not_applicable
- link 2: fix 4f13eb3 a6ec548 720651a | deployed log 720651a live_by 2026-07-04T03:07:46Z | tests touched: tests/health/test_contest_state.py, tests/test_cli_integration.py, tests/test_contest_fetch.py
- link 3: fix 4f13eb3 | deployed log 83a5cad live_by 2026-07-02T15:28:10Z | tests touched: tests/health/test_contest_state.py, tests/test_cli_integration.py, tests/test_contest_fetch.py

## I-063 (E63) plan_named=False — 6/11–6/17 decision and dashboard streak inflated above the real MLB streak (max(model, contest)); fetch discar
- link 1: fix 2b4ff1d a9522a4 | deployed log 83a5cad live_by 2026-07-02T15:28:10Z | tests touched: tests/test_contest_state.py
- link 2: fix 9a3e8ed 4bdfdb8 | deployed log 83a5cad live_by 2026-07-02T15:28:10Z | tests touched: tests/test_cli_integration.py, tests/test_contest_fetch.py
- link 3: fix 709e68b | deployed log 83a5cad live_by 2026-07-02T15:28:10Z | tests touched: tests/test_contest_fetch.py
- link 4: fix 56f9726 | deployed log 83a5cad live_by 2026-07-02T15:28:10Z | tests touched: tests/health/test_contest_state.py

## I-071 (E71) plan_named=False — 7/06 check-pick-entered sent a 'not entered — fix it now' DM for an undelivered, later-deferred double-down pr
- link 1: fix af6329f | deployed log 39b02bd live_by 2026-07-06T20:39:41Z | tests touched: tests/test_cli_integration.py
- link 2: fix af6329f 540b1ab | deployed log 39b02bd live_by 2026-07-06T20:39:41Z | tests touched: tests/test_cli_integration.py

## I-072 (E72) plan_named=False — 7/08 double-down leg never entered: check-pick-entered sent one 18:00 alert, then treated 'alerted' as termina
- link 1: not_applicable
- link 2: fix 8bceda1 e5ef7ca | deployed log 49f2538 live_by 2026-07-10T15:20:05Z | tests touched: tests/health/test_attention.py, tests/health/test_f4_exception_propagation.py, tests/health/test_pick_entry_source.py, tests/health/test_realized_calibration.py, tests/test_cli_integration.py, tests/test_heartbeat_churn.py, tests/test_scheduler_state_integrity.py

## I-075 (E75) plan_named=False — 7/12 eve-of-break restart loop: ~47 duplicate CRITICAL health DMs, blind restart_spike, confounded predicted_v
- link 1: fix 9551818 | deployed log 230f65c live_by 2026-07-13T01:12:47Z | tests touched: tests/test_scheduler_eve_of_break.py
- link 2: fix 9551818 | deployed log 230f65c live_by 2026-07-13T01:12:47Z | tests touched: tests/test_scheduler_eve_of_break.py
- link 3: fix ec242da | deployed log 230f65c live_by 2026-07-13T01:12:47Z | tests touched: tests/health/test_alert.py, tests/health/test_attention.py, tests/health/test_restart_spike.py, tests/health/test_runner.py
- link 4: fix ec242da | deployed log 230f65c live_by 2026-07-13T01:12:47Z | tests touched: tests/health/test_alert.py, tests/health/test_attention.py, tests/health/test_restart_spike.py, tests/health/test_runner.py
- link 5: fix 230f65c | deployed log 230f65c live_by 2026-07-13T01:12:47Z | tests touched: tests/health/test_predicted_vs_realized.py

## I-082 (E82) plan_named=True — 8/11 MLB auth flap: one blank HTTP 200 failed the contest-streak fetch and the DM wrongly advised a cookie re-
- link 1: fix 404358d | deployed log 404358d live_by 2026-08-11T20:30:11Z | tests touched: tests/leaderboard/test_auth.py, tests/leaderboard/test_cli.py, tests/test_cli_integration.py, tests/test_contest_fetch.py
- link 2: fix 404358d | deployed log 404358d live_by 2026-08-11T20:30:11Z | tests touched: tests/leaderboard/test_auth.py, tests/leaderboard/test_cli.py, tests/test_cli_integration.py, tests/test_contest_fetch.py

## I-083 (E83) plan_named=True — 8/13 silent pass: MLB Warmup status classification-locked the undelivered pick; no missed-pick alert
- link 1: fix 1b50b78 | deployed log 3e1791c live_by 2026-08-14T19:04:35Z | tests touched: tests/test_picks.py, tests/test_strategy.py, tests/test_warmup_lock_classification.py
- link 2: fix 224ddce | deployed log 3e1791c live_by 2026-08-14T19:04:35Z | tests touched: tests/test_e3_missed_pick_alert.py, tests/test_scheduler_decision_record.py, tests/test_scheduler_decision_record_integration.py

## I-084 (E84) plan_named=True — 8/30 late pick: the fallback deferred an enterable pick on a stale plan and re-picked it after the cutoff (DM 
- link 1: fix a681e89 32057b7 67338cd 3697512 | deployed log 2ff2db9 live_by 2026-08-30T21:10:22Z | tests touched: tests/test_cutoff_candidate_exclusion.py, tests/test_fallback_plan.py, tests/test_incident_2026_08_30.py, tests/test_late_delivery_guard.py, tests/test_lock_decision.py, tests/test_scheduler.py, tests/test_scheduler_decision_record_integration.py, tests/test_scheduler_skip_visibility.py, tests/test_submission_cutoff.py
- link 2: fix 734bdb3 ac0ce8d fa977c3 2ff2db9 | deployed log 2ff2db9 live_by 2026-08-30T21:10:22Z | tests touched: tests/test_cutoff_candidate_exclusion.py, tests/test_e2_delivery_idempotency.py, tests/test_late_delivery_guard.py, tests/test_scheduler_decision_record_integration.py, tests/test_submission_cutoff.py
- link 3: fix c0c0a97 0c8ed32 | deployed log 2ff2db9 live_by 2026-08-30T21:10:22Z | tests touched: tests/data/test_pull.py, tests/test_refresh_memo.py
- link 4: fix 314154d | deployed log 2ff2db9 live_by 2026-08-30T21:10:22Z | tests touched: tests/health/test_fallback_defer.py, tests/health/test_late_delivery.py

## I-085 (E85) plan_named=True — 9/03 idle: the reach-57 policy table was all-skip once 57 became unreachable, so production stopped picking
- link 1: fix 0abf503 eb010fd | deployed log eb010fd live_by 2026-09-03T19:20:50Z | tests touched: tests/health/test_mdp_policy_alignment.py, tests/health/test_tail_policy_health.py, tests/simulate/test_tail_policy.py, tests/test_contest_state_best_trust.py, tests/test_daily_decision_v2.py, tests/test_daily_decision_v3.py, tests/test_picks.py, tests/test_scheduler_tail_objective.py, tests/test_skip_policy_shadow.py, tests/test_tail_policy_artifact_contract.py, tests/test_tail_policy_r3_fixes.py, tests/test_tail_policy_strategy.py, tests/test_tail_provenance.py

## I-113 (E113) plan_named=False — R2 sync omitted mdp_policy.npz: a host restored from R2 (the Fly instance, 4/10) silently ran the heuristic in
- link 1: fix 947fce8 | deployed unknown | tests touched: none

