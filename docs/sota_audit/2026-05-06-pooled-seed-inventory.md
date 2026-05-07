# Pooled-Seed Inventory

- **Generated**: 2026-05-06
- **Inventory artifact**: `data/validation/pooled_seed_inventory_2026-05-06.json`
- **Generator**: `scripts/inventory_pooled_seed_surfaces.py`
- **Scope**: current local research workspace. Some source surfaces are untracked artifacts and are inventoried here without adding their large payloads to git.

## Verdict

The next bin-side / multi-seed item is not blocked on writing the pooled-policy machinery. That machinery and several validation surfaces already exist. The immediate blocker is provenance and decision certification:

1. The 24-seed raw policy-bin surface is present locally, but not under the `data/simulation/profiles_seed*_season*.parquet` shape expected by the DR-MDP memo. It is path-partitioned under `data/hetzner_results/pooled_bins_run` plus `data/hetzner_results/pooled_bins_run_trackc`.
2. The strongest existing signal is pooled policy / pooled bins, not pooled prediction. Do not conflate this with the rejected 2026-04-29 pooled-prediction overconfidence fix.
3. The raw pooled-bin parquet surfaces do not embed determinism flags in the artifacts. Treat their determinism state as not proven by the local artifact alone. The deterministic 10-seed screening and 100-seed baseline artifacts do embed `BTS_LGBM_DETERMINISTIC=1`.

No deploy is implied by this memo.

## What Exists

### Validation Artifacts

| Artifact | Surface | Headline |
| --- | --- | --- |
| `data/validation/pooled_policy_ab.json` | 16-seed pooled policy A/B | Leave-one-out mean gap `+0.019225` P(57), pooled wins `16/16` |
| `data/validation/pooled_policy_ab_24seed_consolidated.json` | 24-seed pooled policy A/B | Leave-one-out mean gap `+0.019290` P(57), pooled wins `24/24` |
| `data/validation/pooled_policy_ab_trackd_crosspath.json` | 8-seed cross-path policy A/B | Leave-one-out mean gap `+0.015846` P(57), pooled wins `8/8` |
| `data/validation/pooled_policy_mc_replay_ab.json` | 80 seed-season MC replay | Mean MC P(57) gap `+0.025850`, pooled wins `60/80`; replay max-streak gap `+2.3125`, but prod reached 57 once and pool zero times |
| `data/validation/screen_pooled_n10_2026-04-28.json` | Deterministic 10-seed experiment screen | `2` winners, `4` losers, `26` noise; embeds `BTS_LGBM_DETERMINISTIC=1` |
| `data/validation/baseline_n100_deterministic_2026-04-27.json` | Deterministic 100-seed baseline | P(57) MDP mean `0.033647`, std `0.016061`, p95 `0.072737`; embeds `BTS_LGBM_DETERMINISTIC=1` |

### Raw / Derived Surfaces

| Surface | Path | Seeds | Raw backtest parquets | Use |
| --- | --- | ---: | ---: | --- |
| Single-seed canonical backtests | `data/simulation` | 0 path-tagged | 6 | Single-seed 2021-2026 baseline; not a pooled seed surface |
| Default pooled-bin run | `data/hetzner_results/pooled_bins_run` | 16 | 80 | Raw profile source for `pooled_policy_ab.json` |
| Track C pooled-bin run | `data/hetzner_results/pooled_bins_run_trackc` | 8 | 40 | Additional raw profile source that completes the 24-seed surface |
| Track D cross-path run | `data/hetzner_results/pooled_bins_run_trackd` | 8 | 40 | Raw source for `pooled_policy_ab_trackd_crosspath.json` |
| Phase-1 score JSONs | `data/hetzner_results/audit_phase1` | 16 | 0 | Score JSON only; not enough for pooled-bin rebuild |
| Full 48-seed v2 audit | `data/hetzner_results/audit_full_48seed_v2` | 48 | 0 | Experiment scorecards only; not enough for pooled-bin rebuild |
| Deterministic n=100 baseline dirs | `data/det_baselines_n100` | 100 | 0 | Baseline scorecards only; not enough for pooled-bin rebuild |

The raw pooled-bin parquets are nested under directories like `simulation_seed42`, but sampled parquet columns are only:

```text
date, rank, batter_id, p_game_hit, actual_hit, game_time, game_pk
```

Any consumer that loads multiple seed directories must tag seed identity from the path before pairing rank-1 and rank-2 rows. `bts.simulate.pooled_policy.load_pooled_profiles` already does this. `scripts/dr_mdp_gap_measure.py --profiles-glob ...` currently concatenates parquet files directly, so pointing it at multiple raw seed directories would need seed tagging first; otherwise rank-1/rank-2 pairing can cross seed boundaries.

## Interpretation

The 2026-04-29 pooled-prediction rejection remains valid for that hypothesis: averaging per-batter `p_game_hit` across seeds did not fix production overconfidence and worsened 2025 backtest Brier in the local rejection record. That result should block a pooled-prediction cutover unless a new proper-scoring surface reverses it.

The pooled-policy evidence is a different claim. It says that solving and evaluating MDP policy tables against pooled or leave-one-out seed-derived quality-bin manifolds repeatedly beats the current production policy table on the saved 2021-2025 profile surface. That is decision-layer/bin-side evidence, not direct probability-calibration evidence.

The 24-seed A/B artifact is strong enough to deserve the same uncertainty treatment used by the v2.6 harness, but it is not yet a deployment proof. The artifact has no block-bootstrap CI, no production-equivalence proof, and no embedded determinism metadata for the raw profile parquets that produced it.

## Next Work

1. Add a v2.6-style uncertainty layer to `pooled_policy_ab_24seed_consolidated.json`. The load-bearing number is the leave-one-out `+1.929pp` P(57) gap with `24/24` pooled wins; certify whether it survives a dependence-aware bootstrap or downgrade it to a promising screen.
2. If the raw profile provenance is accepted, rerun the DR-MDP gap screen on the 24-seed raw profile surface. Do not pass the raw parquets directly through the current `--profiles-glob` loader until seed tagging is handled.
3. Keep production unchanged until pooled-policy evidence is reconciled with #12 proper scoring, realized-picks calibration, and production data-lineage constraints.
