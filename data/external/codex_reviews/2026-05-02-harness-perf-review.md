# Codex Round 4 — Production-Scale Performance Review

**Date**: 2026-05-02 ~19:25 ET
**Model**: GPT-5.5 (high reasoning)
**Trigger**: After Task 13's harness run hit a 1+ hour O(N*M) bottleneck in `pa_residual_correlation` that prior 3 review rounds didn't catch (they focused on statistical correctness, not big-O scale). User authorized round 4 specifically focused on production-scale performance + memory.
**Cost**: ~$1.50

---

## Summary of findings

| Severity | Function | Issue | Fix |
|---|---|---|---|
| 🚨 High | `pa_residual_correlation` bootstrap | Even after my O(N) groupby patch, inner Python pair-product loops are still 1B ops at production scale | Closed-form `pair_sum = 0.5*(sum_e² - sum_e2)` per group + `bincount` bootstrap |
| 🚨 High | `fit_logistic_normal_random_intercept` outer pair-product loop | Same Python pair appends, no bootstrap but 360K groups × ~10 pairs = 3.6M ops | Closed-form `rho_hat = pair_sum.sum() / pair_n.sum()` |
| 🚨 High | `fitted_q_evaluation` / `dr_ope_full_information` | `counts[horizon, n_states, n_actions, n_states]` allocates 4.9T entries at n_states=103K. Production-impossible. | Sparse aggregation via groupby on observed (t,s,a,sn) |
| 🟡 Med | `_trajectory_dataframe_from_profiles` | `iterrows()` is slow; `bins.classify()` per row | Precompute `qbin = np.digitize(top1_p, bins.boundaries)`, use `itertuples`, integer action ids |
| 🟡 Med | `corrected_audit_pipeline` | `_compute_bins_from_direct_profiles(profiles)` inside per-fold loop = redundant work | Hoist outside fold loop |
| 🟡 Med | `paired_hierarchical_bootstrap_sample` | `season_df[season_df["date"] == source_date]` per resample iteration = O(N) filter × bootstrap | Precompute `{season: {date: chunk}}` once |
| 🟡 Med | `_event_reached_threshold` + CE-IS event_indicators | Python loop per path × 20K paths. Vectorizable. | `np.maximum.accumulate` trick to find max consecutive run |
| 🟢 Low | `estimate_p57_with_ceis` bootstrap | `bs_idx = rng.choice(20K, size=(2K, 20K))` = 320MB int64 alloc | Chunk bootstrap iterations |
| ❌ Disagree | `audit_pipeline` MDP cache across folds | I'd suggested caching; Codex correctly pushed back: each fold has different bins, so per-fold MDP is required | Don't cache; per-fold solve is honest LOSO |

---

## Key insight from Codex

> "Same function, pre-patch line ~64 if still present anywhere: `{bg: df.loc[df['batter_game_id'] == bg, ...] for bg in bg_ids}` is the exact `O(N*M)` bug. **Fix is not just `groupby`; better is no dict at all: `pd.factorize` + `np.bincount`.**"

The closed-form refactor avoids the dict entirely. For each batter_game_id:
- `pair_sum = 0.5*(sum_e² - sum_e2)` (sum of all pair products via algebraic identity)
- `pair_n = n*(n-1)/2` (number of pairs)

Then bootstrap is purely vectorized `bincount` over group counts × pair_sum array. No Python loops, no dict lookups.

---

## Codex output (verbatim)

**High-Severity Hot Paths**

- `src/bts/validate/dependence.py::pa_residual_correlation` lines ~44-73: current bootstrap still does Python pair reconstruction per sampled group. Fix: factorize `batter_game_id`, compute per-group `pair_sum = ((sum_e**2 - sum_e2) / 2)` and `pair_n = n*(n-1)//2` with `np.bincount`, then bootstrap only counts:
  `idx = rng.integers(0, G, G); c = np.bincount(idx, minlength=G); bs = c @ pair_sum / (c @ pair_n)`.
  This preserves the current pair-weighted estimand and removes all inner PA-pair loops.

- Same function, pre-patch line ~64 if still present anywhere: `{bg: df.loc[df["batter_game_id"] == bg, ...] for bg in bg_ids}` is the exact `O(N*M)` bug. Fix is not just `groupby`; better is no dict at all: `pd.factorize` + `np.bincount`.

- `fit_logistic_normal_random_intercept` lines ~141-159: same pair-product append loop. Fix: vectorize residuals and use per-group sums:
  `rho_hat = pair_sum.sum() / pair_n.sum()`, where `pair_sum = 0.5*(sum_e**2 - sum_e2)`.
  No `np.triu_indices` needed; the closed-form pair sum is faster and avoids pair materialization.

- `src/bts/validate/ope.py::fitted_q_evaluation` / `dr_ope_full_information` lines ~37-59 and ~95-116: dense `counts[horizon, n_states, n_actions, n_states]` is production-impossible if `n_states ~= 103K`. Fix: sparse transition aggregation by observed `(t,s,a,sn)` via pandas/groupby or scipy sparse; Bellman backup only over observed transitions.

**Medium-Severity Hot Paths**

- `_trajectory_dataframe_from_profiles` lines ~291-355: `iterrows()` plus `bins.classify()` per row is slow, but I would not fully vectorize the state replay because `streak/saver` are sequential policy state. Fix: sort once globally, precompute `qbin = np.digitize(top1_p, bins.boundaries)`, iterate per group over NumPy arrays or `itertuples(index=False)`, and compare integer action ids instead of `ACTIONS[action_idx]` strings.

- `corrected_audit_pipeline`: yes, `_compute_bins_from_direct_profiles(profiles)` inside the fold loop is pure repeated work. Fix: compute `bins = _compute_bins_from_direct_profiles(profiles)` once before the loop and reuse.

- `audit_pipeline` lines ~430-435: disagree on caching a single MDP across folds if this is honest LOSO pipeline evaluation. Each fold's training set differs, so bins and solved policy differ. Valid optimizations: cache by `tuple(train_seasons)` if repeated elsewhere, parallelize folds, and avoid re-solving only in fixed-policy mode.

- `paired_hierarchical_bootstrap_sample` lines ~181-193: hidden repeated full-season date scans: `season_df[season_df["date"] == source_date]` inside bootstrap/day loops. Fix: precompute `{season: {date: chunk}}` or use `groupby(["season","date"])` once, then lookup chunks.

**CE-IS**

- `rare_event_mc.py::_event_reached_threshold` + `event_indicators` list comprehension: yes, vectorize across all paths. Concrete fix:
  ```python
  zero_pos = np.where(final_paths == 0, np.arange(final_paths.shape[1]), -1)
  last_zero = np.maximum.accumulate(zero_pos, axis=1)
  run_len = np.arange(final_paths.shape[1]) - last_zero
  event_indicators = (run_len >= streak_threshold).any(axis=1).astype(np.int8)
  ```

- `estimate_p57_with_ceis` bootstrap: `bs_idx = rng.choice(n_final, size=(2000, n_final), replace=True)` is a 320MB int64 allocation at `20K x 2000`. Fix: bootstrap in chunks or loop `2000` vectorized draws one at a time; runtime is fine, memory is the issue.

**Lower Priority**

- `_compute_bins_from_direct_profiles`: its 5 bin masks are fine; the problem is repeated calls, not internal complexity.

- `pair_residual_correlation`: permutation/bootstrap loops are okay at daily-profile scale. If rows grow large, chunk permutation indices, but it is not the production bottleneck here.

---

## Verdict robustness vs perf fixes

The Task 13 verdict (HEADLINE_BROKEN, corrected_pipeline_p57=0.0083 [0, 0.0375]) was produced at n_bootstrap=300. Codex's high-severity fixes don't change the verdict — they just make a tighter (n=2000) re-run feasible. The verdict at n=300 is decisive: corrected_pipeline upper bound 0.0375 << half-headline 0.0408, so even a 4× tighter CI couldn't rescue the headline.

Applied (post-review):
- ✅ pa_residual_correlation closed-form pair_sum + bincount bootstrap
- ✅ fit_logistic_normal_random_intercept closed-form rho_hat
- ✅ _event_reached_threshold vectorized across paths
- ⏳ Other findings: documented as v2 follow-up, not applied tonight

Deferred to v2 follow-up:
- corrected_audit_pipeline bin hoist (~10s saved per harness run)
- _trajectory_dataframe_from_profiles itertuples optimization
- paired_hierarchical_bootstrap_sample chunk precompute (only matters if we use sequential DR)
- estimate_p57_with_ceis bootstrap chunking (memory only, runtime fine)
- fitted_q_evaluation sparse transition aggregation (only matters if production scale via dr_ope_full_information)
