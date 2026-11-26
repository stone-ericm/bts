# BTS Alert Disposition Walkthrough (2026-05-23)

Status: active disposition log after PRs #102-#112.

This document records the alert and visibility decisions from the
Codex-Claude walkthrough after the alert-policy and calibration-policy work.
The goal is to keep paging surfaces actionable without hiding slow stalls.

## Current live evidence

- Production deploy head after this pass: `67b21df`.
- `bts-scheduler.service`: active/running, `NRestarts=0`.
- `bts-dashboard.service`: active/running, dashboard `/health=200`.
- `bts-live-forward-resolve.service`: cleared from failed state by PR #112.
  Manual verification after deploy returned `Result=success`,
  `ExecMainStatus=0`, and `is-failed=inactive`.
- Canonical live-forward resolved artifacts are not stalled:
  2026-05-16 through 2026-05-21 are `existing_verified`, 2026-05-22 is
  `existing_verified_with_voids`, and 2026-05-23 is `pending_outcomes`.

## 2026-05-24 WARN Attention Follow-up

The 2026-05-24 end-of-day health DM repeated two WARN attention surfaces:
`mdp_policy_alignment` and `dd_pair_realized_shortfall`.

Live verification on production showed:

- `bts-scheduler.service` and `bts-dashboard.service` were active, both with
  `NRestarts=0`; dashboard `/health` returned `ok` and the scheduler was
  sleeping until the next scheduled check.
- `mdp_policy_alignment` matched the health DM exactly: the last 21 primary
  picks and 21 double-down picks all mapped to Q0 below the deployed policy's
  lowest quality boundary. This is the known Gate B condition, not a new policy
  decision point. PR #128 closed the production-live boundary feasibility slice
  as `NOT_FEASIBLE_DIRECT_OR_RECONCILIATION_NEEDS_MORE_LIVE_N`, so the expected
  disposition is to keep the diagnostic visible until enough live support exists
  for a pre-registered reconciliation or re-solve.
- `dd_pair_realized_shortfall` decomposed to model-pair shortfall, not residual
  pair dependence. Over the last 14 resolved double-down days, primary marginal
  hit rate was `7/14`, double-down marginal hit rate was `6/14`, their empirical
  product was `3/14`, and realized both-hit rate was also `3/14`. The same
  health run reported rolling residual gap `0.0` and did not emit
  `dd_pair_residual_corr`.

Conclusion: do not open a policy/deploy path from these WARNs alone. Treat the
MDP warning as a known-accepted diagnostic under the Gate B `NO SWAP` decision,
and treat the DD shortfall warning as calibration-scale evidence unless the
separate residual-correlation check rises. The DD shortfall belongs with the
Gate A marginal-calibration debt and should accumulate toward the existing
`n >= 200` re-check threshold, not start a separate DD policy track.

## 2026-05-26 `analytics_artifacts_missing` Follow-up

The 2026-05-25 end-of-day health run emitted a CRITICAL
`analytics_artifacts_missing` alert for live-forward capture status
`failed_recapture_post_resolution`.

Production evidence showed this was not an active capture outage. The canonical
2026-05-25 artifact was recaptured before outcomes with the delivered picks and
stored a matching production-pick snapshot. After result reconciliation, the
pick file gained nullable `feature_env_schema_version`, `feature_env`, and
`feature_env_hash` fields. Because the capture guard compared the current pick
to the stored at-lock snapshot after removing only `result` and `slot_results`,
absent-vs-null provenance defaults made the valid artifact look stale after the
pick had already resolved.

Disposition: keep `analytics_artifacts_missing` loud, but fix the snapshot
comparison so absent and null optional provenance fields are equivalent while
non-null provenance still participates in drift detection. The 2026-05-26
canonical live-forward capture was independently healthy with
`status=existing_verified`, `snapshot_matches_current_pick=true`, and
`stale_pick_snapshot=false`.

## 2026-05-27 Probability-Scale Refresh

The latest production health state had no CRITICAL alerts. The only WARN
attention surface still active was the known `mdp_policy_alignment` quality-bin
collapse.

Current-ET-capped production-pick inventory on 2026-05-27:

| Surface | n | mean | q10 | q50 | q90 |
|---|---:|---:|---:|---:|---:|
| primary rank-1 | `60` | `0.749184` | `0.716450` | `0.751280` | `0.781160` |
| double-down rank-2 | `56` | `0.723778` | `0.688331` | `0.729671` | `0.749560` |
| recent 21 primary | `21` | `0.750621` | `0.723034` | `0.752266` | `0.781091` |
| recent 21 double-down | `21` | `0.725595` | `0.687434` | `0.729990` | `0.748343` |

Recent 21 primary and double-down picks still mapped entirely to Q0 against the
deployed boundaries `[0.795979, 0.811491, 0.825247, 0.840740]`. Only two
current-date-capped primary rows had `feature_env_hash`, so the stronger
`model_pickle_sha256 + feature_env_hash` stability window introduced by PR
#130 is still far too small for a live-scale reconciliation.

The explanation track is closed enough for alert disposition. Existing Gate B
docs found that deployed MDP bins were built on a historical actual-PA-expanded
surface, while production acts on a lower estimated-PA live surface. The later
production-live feasibility slice found the same roughly three-point downward
scale gap and concluded direct live boundary derivation or backtest-to-live
reconciliation is not feasible yet.

Disposition: no production calibration, MDP policy, or pick-selection change.
This refresh does not alter the 2026-05-25
`NOT_FEASIBLE_DIRECT_OR_RECONCILIATION_NEEDS_MORE_LIVE_N` verdict. Keep the
WARN visible. The next action is support-gated live-scale reconciliation, not
another immediate explanation pass.

## 2026-05-27 Postponed-Game Root Refresh

The scheduler-side postponed-game root fix is now deployed, not just covered by
the `postponed_pick` health symptom check. Production HEAD on 2026-05-27 was
`fac6bdb`, with PR #131 (`d63f60b`) in history. The deployed code includes
strict detailed-status mode for live pick generation and refreshed lock
decisions, while keeping offline/backtest callers on the legacy abstract-status
path unless they explicitly opt in.

Verification:

- production source contains `require_detailed_statuses=True` on the scheduler
  live path;
- tests cover projected candidates from abstract-preview but detailed-postponed
  games being excluded from lock-gap decisions;
- tests cover unposted primary and double-down postponed games regenerating a
  fresh pick, while already posted picks remain locked;
- `tests/test_scheduler.py` and `tests/test_strategy.py` passed locally.

Disposition: keep `postponed_pick` loud as a Tier-1 symptom alert, but close
the separate scheduler root-fix item.

## 2026-05-27 Dashboard Health Canary Follow-up

During PR #136 deploy-safety checks, `bts-dashboard.service` was active and
TCP-accepting, but `/health` timed out. A dashboard-only restart restored
`/health=ok` while `bts-scheduler.service` stayed active with `NRestarts=0`.

The dashboard used Python's single-threaded `HTTPServer`, so one slow or stuck
request could block `/health` and the deploy canary even when the process was
otherwise alive. The follow-up is to run the dashboard on a threaded HTTP
server with daemon request threads.

## Disposition Table

| Surface | Current level/path | Disposition | Rationale | Follow-up |
|---|---|---|---|---|
| `bts-live-forward-resolve.service` stale archive failures / canonical resolver stall | systemd failed unit, resolver status JSON, `live_forward_resolution` health check | Implemented / keep | PR #112 stopped default resolver discovery from scanning `*.stale_pick_snapshot.*` archive directories. The health check now alerts only when canonical preoutcome artifacts remain unresolved beyond the grace window; a 2026-05-27 production check returned `alerts 0`. | Keep explicit `--date` forensic access and keep the grace-window health check visible for future canonical stalls. |
| Canonical live-forward pending outcomes | resolver `pending_outcomes` | Keep non-paging | Missing same-day PA outcomes are expected before games resolve. The runner already exits 0 when pending and `--fail-on-pending` is false. | Alert only on repeated canonical pending/failure after a grace window, not on one transient day. |
| `restart_spike` | CRITICAL DM | Keep loud | Catches automatic service restarts and crash-loop class failures. Manual deploy restarts do not increment `NRestarts`, so planned deploys are already separated. Alert wording now points operators at journal/OOM/watchdog/crash evidence instead of assuming a heartbeat-gap regression. | None unless future evidence supports automatic cause classification. |
| `analytics_artifacts_missing` | WARN attention / CRITICAL with fatal evidence | Keep loud | Correctly makes missing shadow/capture artifacts visible after a locked pick. OOM evidence promotion is the right CRITICAL path. | Confirm stale WARN streaks clear naturally when absent on the next day. |
| `select_pick_returned_none` shadow absence | INFO | Keep low | This is benign shadow abstention, not a production failure. | None unless it becomes frequent enough to suggest shadow model coverage decay. |
| `mdp_policy_alignment` | WARN, repeated attention | Keep as known-accepted diagnostic | It correctly surfaced policy-bin collapse and led to Gate A/B measurement. Gate B closed as `NO SWAP`, and PR #128 found production-live boundary derivation/reconciliation not feasible yet. The collapse is therefore expected to persist until enough live support accumulates. | Auto-clear when recent p distribution again uses multiple policy bins, or after a future pre-registered policy re-solve/reconciliation. Do not re-open the policy path from this WARN alone. |
| Gate A calibration validation | script/doc decision `WAIT_FOR_N` | Document only | Model-policy evidence is underpowered and should not page. Current isotonic check did not beat raw Brier at n=158. | Re-run when resolved pick-slot support reaches n >= 200. |
| Gate B raw re-bin measurement | script/doc decision `INSUFFICIENT_SUPPORT` | Document only | Current-era DD-pair support is only n=49 with thin bins and a backtest distribution mismatch. | Revisit at n >= 200, min per bin >= 30, and off-host policy-file backtest compatibility. |
| `memory_growth` | INFO/WARN/CRITICAL by high absolute RSS plus post-prediction baseline delta; Tuesday digest | Implemented / keep | Code now separates cold sleeping samples from post-prediction RSS. On 2026-05-27, live scheduler RSS was about 2481.5 MB versus a recent post-prediction baseline median of 3023.7 MB, correctly producing no alert. | Keep monitoring weekly digest and treat future WARNs as sustained growth above the post-prediction baseline, not normal model residency. |
| `pooled_training` | dormant unless `pooled_dir` configured | Keep dormant | Pooled models are not currently load-bearing in production. Alerting before deployment would create noise. | Enable only if pooled inference becomes production-critical. |
| `leaderboard_freshness` | dormant unless `leaderboard_dir` configured | Keep dormant | Leaderboard data is not currently load-bearing for pick delivery. | Enable when leaderboard ingestion becomes a dependency. |
| `health_dm_delivery` | persistent dashboard/state indicator on failure | Add secondary visibility | If Bluesky DM delivery itself fails, the current user-facing path may be unavailable by definition. Logging alone may not be noticed quickly, so health-DM attempts now write `data/health_state/health_dm_delivery_status.json`; failed or missing-recipient attempts render a dashboard banner until a successful attempted health DM clears it. | Consider a fully independent out-of-band channel only if dashboard visibility is not enough. |
| E fallback/defer path | deployed behavior, INFO status when observed | Track live validation | The defer path is deployed, but production has not naturally exercised it yet. On 2026-05-23 at prod head `c511a03`, `data/picks` had 176 pick JSON files and 0 `deferred_fallback_*.json` archives; scheduler journal since 2026-05-01 showed fallback force-deliveries but no `FALLBACK DEFERRED` lines. | `fallback_defer` health status now self-announces the rare event at INFO when a defer archive exists and a final pick was delivered; it escalates only if a defer fires without a delivered pick. Close live validation after observing a natural defer and verifying fired/no force-lock/never-miss/better-pool criteria. |
| `projected_lineup` | INFO/WARN | Keep repeated attention | It detects excessive projection use and can catch lineup-confirmation quality decay. It should not page on a single noisy day. | Keep in repeated WARN attention set. |
| `predicted_vs_realized`, `realized_calibration`, `dd_pair_realized_shortfall`, `dd_pair_residual_corr` | INFO/WARN/CRITICAL by evidence | Keep repeated attention | These are model-quality surfaces. `dd_pair_realized_shortfall` tracks model-pair shortfall, while `dd_pair_residual_corr` is the marginal-adjusted pair-dependence signal. On 2026-05-24 the DD shortfall WARN decomposed to marginal/model shortfall with rolling residual gap `0.0`, so it was not pair-correlation evidence. | Keep separated from MDP/gate docs to avoid mixing alerting with policy-change decisions. Attribute DD shortfall to the Gate A calibration/marginal scale track unless `dd_pair_residual_corr` also rises. |
| `disk_fill`, `postponed_pick`, `blend_training`, `post_failure`, `streak_validation` | WARN/CRITICAL depending on check | Keep loud | These are operational integrity surfaces where a miss can directly break pick delivery, scoring, or state validity. The postponed-game scheduler root fix is deployed; live pick generation now uses detailed postponed/cancelled status for stale-pick and lock-gap decisions. | Keep `postponed_pick` loud as a symptom alert for future stale committed-game failures. |

## Next Work Items

1. E fallback/defer live validation: wait for the next natural
   `fallback_defer` INFO event, then inspect the matching
   `data/picks/<date>/deferred_fallback_*.json` archive and final pick file.
   Close only after confirming:
   - the defer fired for `reason=should_lock_false_future_checks_remain`;
   - the unsafe candidate was not force-locked at that fallback deadline;
   - a later check or final fallback delivered a pick for the day;
   - the delivered primary was the same or better by `p_game_hit`, or any
     lower-probability result is explicitly explained.
2. Health-DM delivery visibility: monitor the dashboard/state-file secondary
   visibility path and add an independent out-of-band channel only if that is
   not enough.
3. Probability-scale live-support gate: keep accumulating
   `model_pickle_sha256 + feature_env_hash` production picks. Re-open only for a
   pre-registered reconciliation at enough policy-stable live support, or for
   direct live-boundary derivation once the larger direct-support threshold is
   met. Until then, keep `mdp_policy_alignment` visible but do not treat it as
   a deploy trigger.
