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

## Disposition Table

| Surface | Current level/path | Disposition | Rationale | Follow-up |
|---|---|---|---|---|
| `bts-live-forward-resolve.service` stale archive failures | systemd failed unit, resolver status JSON | Fixed in PR #112 | Default resolver discovery was scanning `*.stale_pick_snapshot.*` archive directories. Some stale archives had `failed_verify_existing`, leaving the one-shot failed even though canonical dates were resolving. | Keep explicit `--date` forensic access. Add a future multi-day canonical resolver-failure health check if canonical dates stop resolving. |
| Canonical live-forward pending outcomes | resolver `pending_outcomes` | Keep non-paging | Missing same-day PA outcomes are expected before games resolve. The runner already exits 0 when pending and `--fail-on-pending` is false. | Alert only on repeated canonical pending/failure after a grace window, not on one transient day. |
| `restart_spike` | CRITICAL DM | Keep loud | Catches automatic service restarts and crash-loop class failures. Manual deploy restarts do not increment `NRestarts`, so planned deploys are already separated. | Improve message wording. "Heartbeat-gap regression suspected" is too narrow when OOM is also a likely cause. |
| `analytics_artifacts_missing` | WARN attention / CRITICAL with fatal evidence | Keep loud | Correctly makes missing shadow/capture artifacts visible after a locked pick. OOM evidence promotion is the right CRITICAL path. | Confirm stale WARN streaks clear naturally when absent on the next day. |
| `select_pick_returned_none` shadow absence | INFO | Keep low | This is benign shadow abstention, not a production failure. | None unless it becomes frequent enough to suggest shadow model coverage decay. |
| `mdp_policy_alignment` | WARN, repeated attention | Keep as known-accepted diagnostic | It correctly surfaced policy-bin collapse and led to Gate A/B measurement. Gate B closed as `NO SWAP`, and PR #128 found production-live boundary derivation/reconciliation not feasible yet. The collapse is therefore expected to persist until enough live support accumulates. | Auto-clear when recent p distribution again uses multiple policy bins, or after a future pre-registered policy re-solve/reconciliation. Do not re-open the policy path from this WARN alone. |
| Gate A calibration validation | script/doc decision `WAIT_FOR_N` | Document only | Model-policy evidence is underpowered and should not page. Current isotonic check did not beat raw Brier at n=158. | Re-run when resolved pick-slot support reaches n >= 200. |
| Gate B raw re-bin measurement | script/doc decision `INSUFFICIENT_SUPPORT` | Document only | Current-era DD-pair support is only n=49 with thin bins and a backtest distribution mismatch. | Revisit at n >= 200, min per bin >= 30, and off-host policy-file backtest compatibility. |
| `memory_growth` | INFO/WARN/CRITICAL by absolute RSS, Tuesday digest | Recalibrate | History shows normal post-prediction RSS often around 2.8-3.6 GB, while current sleeping RSS can be about 140 MB. Absolute 1 GB/3 GB thresholds mix cold and post-prediction baselines. | Convert to growth-rate or delta-over-baseline, or split cold vs post-prediction thresholds. |
| `pooled_training` | dormant unless `pooled_dir` configured | Keep dormant | Pooled models are not currently load-bearing in production. Alerting before deployment would create noise. | Enable only if pooled inference becomes production-critical. |
| `leaderboard_freshness` | dormant unless `leaderboard_dir` configured | Keep dormant | Leaderboard data is not currently load-bearing for pick delivery. | Enable when leaderboard ingestion becomes a dependency. |
| `health_dm_delivery` | persistent dashboard/state indicator on failure | Add secondary visibility | If Bluesky DM delivery itself fails, the current user-facing path may be unavailable by definition. Logging alone may not be noticed quickly, so health-DM attempts now write `data/health_state/health_dm_delivery_status.json`; failed or missing-recipient attempts render a dashboard banner until a successful attempted health DM clears it. | Consider a fully independent out-of-band channel only if dashboard visibility is not enough. |
| E fallback/defer path | deployed behavior, INFO status when observed | Track live validation | The defer path is deployed, but production has not naturally exercised it yet. On 2026-05-23 at prod head `c511a03`, `data/picks` had 176 pick JSON files and 0 `deferred_fallback_*.json` archives; scheduler journal since 2026-05-01 showed fallback force-deliveries but no `FALLBACK DEFERRED` lines. | `fallback_defer` health status now self-announces the rare event at INFO when a defer archive exists and a final pick was delivered; it escalates only if a defer fires without a delivered pick. Close live validation after observing a natural defer and verifying fired/no force-lock/never-miss/better-pool criteria. |
| `projected_lineup` | INFO/WARN | Keep repeated attention | It detects excessive projection use and can catch lineup-confirmation quality decay. It should not page on a single noisy day. | Keep in repeated WARN attention set. |
| `predicted_vs_realized`, `realized_calibration`, `dd_pair_realized_shortfall`, `dd_pair_residual_corr` | INFO/WARN/CRITICAL by evidence | Keep repeated attention | These are model-quality surfaces. `dd_pair_realized_shortfall` tracks model-pair shortfall, while `dd_pair_residual_corr` is the marginal-adjusted pair-dependence signal. On 2026-05-24 the DD shortfall WARN decomposed to marginal/model shortfall with rolling residual gap `0.0`, so it was not pair-correlation evidence. | Keep separated from MDP/gate docs to avoid mixing alerting with policy-change decisions. Attribute DD shortfall to the Gate A calibration/marginal scale track unless `dd_pair_residual_corr` also rises. |
| `disk_fill`, `postponed_pick`, `blend_training`, `post_failure`, `streak_validation` | WARN/CRITICAL depending on check | Keep loud | These are operational integrity surfaces where a miss can directly break pick delivery, scoring, or state validity. | Keep `postponed_pick` loud, but separately fix the scheduler lock-decision path that still uses abstract game statuses where detailed postponed/cancelled status is required. |

## Next Work Items

1. Memory-growth recalibration PR: replace or supplement absolute RSS
   thresholds with a cold/post-prediction aware rule.
2. Multi-day canonical resolver-failure check: alert only when canonical
   live-forward resolution is stale for multiple runs or days, not when same-day
   outcomes are pending.
3. E fallback/defer live validation: wait for the next natural
   `fallback_defer` INFO event, then inspect the matching
   `data/picks/<date>/deferred_fallback_*.json` archive and final pick file.
   Close only after confirming:
   - the defer fired for `reason=should_lock_false_future_checks_remain`;
   - the unsafe candidate was not force-locked at that fallback deadline;
   - a later check or final fallback delivered a pick for the day;
   - the delivered primary was the same or better by `p_game_hit`, or any
     lower-probability result is explicitly explained.
4. Health-DM delivery visibility: decide whether a failed Bluesky DM health
   send needs a non-DM secondary visibility path.
5. Postponed-game root fix: update scheduler lock-decision filtering to use
   detailed postponed/cancelled game status, not only abstract game status.
   The health check catches the symptom; this is the scheduler-side cause.
6. Probability-scale investigation: explain why current production primary
   probabilities are materially lower than the 2021-2025 backtest/policy-bin
   distribution. Plausible branches are model calibration, 2026 distribution
   shift, or backtest-vs-production data differences.
