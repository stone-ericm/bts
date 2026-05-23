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

## Disposition Table

| Surface | Current level/path | Disposition | Rationale | Follow-up |
|---|---|---|---|---|
| `bts-live-forward-resolve.service` stale archive failures | systemd failed unit, resolver status JSON | Fixed in PR #112 | Default resolver discovery was scanning `*.stale_pick_snapshot.*` archive directories. Some stale archives had `failed_verify_existing`, leaving the one-shot failed even though canonical dates were resolving. | Keep explicit `--date` forensic access. Add a future multi-day canonical resolver-failure health check if canonical dates stop resolving. |
| Canonical live-forward pending outcomes | resolver `pending_outcomes` | Keep non-paging | Missing same-day PA outcomes are expected before games resolve. The runner already exits 0 when pending and `--fail-on-pending` is false. | Alert only on repeated canonical pending/failure after a grace window, not on one transient day. |
| `restart_spike` | CRITICAL DM | Keep loud | Catches automatic service restarts and crash-loop class failures. Manual deploy restarts do not increment `NRestarts`, so planned deploys are already separated. | Improve message wording. "Heartbeat-gap regression suspected" is too narrow when OOM is also a likely cause. |
| `analytics_artifacts_missing` | WARN attention / CRITICAL with fatal evidence | Keep loud | Correctly makes missing shadow/capture artifacts visible after a locked pick. OOM evidence promotion is the right CRITICAL path. | Confirm stale WARN streaks clear naturally when absent on the next day. |
| `select_pick_returned_none` shadow absence | INFO | Keep low | This is benign shadow abstention, not a production failure. | None unless it becomes frequent enough to suggest shadow model coverage decay. |
| `mdp_policy_alignment` | WARN, repeated attention | Keep as-is | It correctly surfaced policy-bin collapse and led to Gate A/B measurement. It is diagnostic, not a deploy gate by itself. | Auto-clear when recent p distribution again uses multiple policy bins, or after a future policy re-solve. |
| Gate A calibration validation | script/doc decision `WAIT_FOR_N` | Document only | Model-policy evidence is underpowered and should not page. Current isotonic check did not beat raw Brier at n=158. | Re-run when resolved pick-slot support reaches n >= 200. |
| Gate B raw re-bin measurement | script/doc decision `INSUFFICIENT_SUPPORT` | Document only | Current-era DD-pair support is only n=49 with thin bins and a backtest distribution mismatch. | Revisit at n >= 200, min per bin >= 30, and off-host policy-file backtest compatibility. |
| `memory_growth` | INFO/WARN/CRITICAL by absolute RSS, Tuesday digest | Recalibrate | History shows normal post-prediction RSS often around 2.8-3.6 GB, while current sleeping RSS can be about 140 MB. Absolute 1 GB/3 GB thresholds mix cold and post-prediction baselines. | Convert to growth-rate or delta-over-baseline, or split cold vs post-prediction thresholds. |
| `pooled_training` | dormant unless `pooled_dir` configured | Keep dormant | Pooled models are not currently load-bearing in production. Alerting before deployment would create noise. | Enable only if pooled inference becomes production-critical. |
| `leaderboard_freshness` | dormant unless `leaderboard_dir` configured | Keep dormant | Leaderboard data is not currently load-bearing for pick delivery. | Enable when leaderboard ingestion becomes a dependency. |
| `health_dm_delivery` | logged CRITICAL on DM send failure | Needs visibility design | If Bluesky DM delivery itself fails, the current user-facing path may be unavailable by definition. Logging alone may not be noticed quickly. | Consider secondary notification or persistent dashboard/state indicator for health-alert delivery failures. |
| E fallback/defer path | deployed behavior, not an alert | Track live validation | The defer path is deployed, but 2026-05-23 locked with `should_lock=True`, so the `should_lock=False` fallback-defer branch has not been live-validated yet. | Validate on next fallback day with `should_lock=False` and future checks remaining. |
| `projected_lineup` | INFO/WARN | Keep repeated attention | It detects excessive projection use and can catch lineup-confirmation quality decay. It should not page on a single noisy day. | Keep in repeated WARN attention set. |
| `predicted_vs_realized`, `realized_calibration`, `dd_pair_realized_shortfall`, `dd_pair_residual_corr` | INFO/WARN/CRITICAL by evidence | Keep repeated attention | These are model-quality surfaces. `dd_pair_realized_shortfall` tracks model-pair shortfall, while `dd_pair_residual_corr` is the marginal-adjusted pair-dependence signal. They should accumulate evidence before paging unless they reach CRITICAL thresholds. | Keep separated from MDP/gate docs to avoid mixing alerting with policy-change decisions. |
| `disk_fill`, `postponed_pick`, `blend_training`, `post_failure`, `streak_validation` | WARN/CRITICAL depending on check | Keep loud | These are operational integrity surfaces where a miss can directly break pick delivery, scoring, or state validity. | Keep `postponed_pick` loud, but separately fix the scheduler lock-decision path that still uses abstract game statuses where detailed postponed/cancelled status is required. |

## Next Work Items

1. Memory-growth recalibration PR: replace or supplement absolute RSS
   thresholds with a cold/post-prediction aware rule.
2. Multi-day canonical resolver-failure check: alert only when canonical
   live-forward resolution is stale for multiple runs or days, not when same-day
   outcomes are pending.
3. E fallback/defer live validation: inspect the next day where fallback would
   post but fresh `should_lock` is false and future checks remain.
4. Health-DM delivery visibility: decide whether a failed Bluesky DM health
   send needs a non-DM secondary visibility path.
5. Postponed-game root fix: update scheduler lock-decision filtering to use
   detailed postponed/cancelled game status, not only abstract game status.
   The health check catches the symptom; this is the scheduler-side cause.
6. Probability-scale investigation: explain why current production primary
   probabilities are materially lower than the 2021-2025 backtest/policy-bin
   distribution. Plausible branches are model calibration, 2026 distribution
   shift, or backtest-vs-production data differences.
