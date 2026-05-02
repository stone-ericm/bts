# Codex Adversarial Review — BTS Falsification Harness Implementation

**Date**: 2026-05-02 ~14:05 ET
**Model**: GPT-5.5 (high reasoning) via OpenAI Codex CLI
**Trigger**: User-requested pre-merge adversarial review of the harness implementation after 16-commit branch landed (Tasks 0-12). Goal: verify that the 5 implementer-introduced statistical corrections preserve the underlying statistical guarantees, before running Task 13 (real-data verdict).
**Cost**: ~$2 of GPT-5.5 high-reasoning tokens
**Outcome**: BLOCKED with three concrete blocking issues + four other significant findings. Implementation requires fixes before Task 13.

---

## Codex output (verbatim)

**1. Corrections**

BLOCK ON: Correction 2 as implemented is not DR-OPE, and the driver later treats a 24-trajectory fixed-policy audit as if it can defend an 8% rare-event headline.

- **Correction 1, slot-date bootstrap:** conditionally valid, but narrow. The relabeling in `paired_hierarchical_bootstrap_sample` preserves exactly one 24-seed day-vector per output slot, so for day-indexed statistics it is a reasonable stationary block bootstrap (`src/bts/validate/ope.py`, `chunk["date"] = slot_date`). It does not preserve calendar-date meaning or original seed trajectories. If the statistic depends on coherent per-seed season paths, season phase, or date-specific covariates, coverage is not guaranteed. It is a day-vector bootstrap, not a trajectory bootstrap.

- **Correction 2, terminal-R MC:** locally unbiased only if the profile rows are full-information counterfactual outcomes under the target policy. The estimator is just `terminal_R = traj_df.groupby("trajectory_id")["r"].sum().to_numpy()` and a mean bootstrap over trajectory IDs (`src/bts/validate/ope.py`, `_run_dr_ope_with_bootstrap`). That handles within-trajectory dependence, but it drops the original DR guarantee entirely: no nuisance correction, no action-propensity correction, no protection against off-policy support problems. It also underestimates uncertainty if the 24 seeds within a season share day/player shocks, because it bootstraps `season_seed` as iid units.

- **Correction 3, per-season Z plus CE-IS bypass:** the direct CE-IS sampler is mathematically valid for the target it actually estimates: independent Bernoulli days with proposal `q = sigmoid(logit(p) + theta_0)` and exact likelihood ratios (`src/bts/simulate/rare_event_mc.py`, `_sample_paths` / `_is_weights`). But that target is no longer the `LatentFactorSimulator` target. Yes, `LatentFactorSimulator.sample_season` is effectively dead for CE-IS; `estimate_p57_with_ceis` bypasses it completely. The per-season `Z_season = rng.normal(...)` is a different dependence model than per-day latent shocks and can materially change streak probabilities.

- **Correction 4, tau via cross-pair inversion:** better than the backwards variance heuristic, but not generally consistent as written. The inversion uses one scalar `mean_p` (`src/bts/validate/dependence.py`, `mean_p = float(df[p_col].mean())`) instead of averaging the theoretical cross-product over the actual within-group `(p_i, p_j)` pairs. It is consistent only in a nearly homogeneous-p regime. The asymptotic distribution is a clustered U-statistic / method-of-moments estimator, not something this code estimates. The `[0, 5]` bound is pragmatic, but if `_expected_cross_product(5.0) < rho_hat`, the code silently clamps `tau_hat = tau_max` (`dependence.py`, `if rho_hat >= max_theory: tau_hat = tau_max`), producing a boundary estimate with no warning and no valid normal approximation.

- **Correction 5, Jensen sign flip plus delta p_both:** the Jensen direction is not adequately documented in production code. The docstring says "mean-corrected p_hit," but the code actually shifts `logit_p_pa` by an uncentered logistic-normal random effect (`src/bts/validate/dependence.py:260` onward), so the marginal PA hit probability is not preserved. The delta-based `new_p_both = b.p_both + delta` (`dependence.py`, `delta = rho_pair_cross_game * ...`) matches the original Pearson formula only in the independence limit where `b.p_both == p1 * p2`. Otherwise it treats `rho_pair_cross_game` as an incremental correlation, while `pair_residual_correlation` estimates an absolute residual correlation. That can double-count existing dependence.

**2. Driver Integration**

- `tau_hat ** 2` is the right scale mechanically. `fit_logistic_normal_random_intercept` returns latent Gaussian SD, and `build_corrected_transition_table` expects `tau_squared` then takes `sqrt` (`scripts/run_falsification_harness.py`, Step 6; `dependence.py:260`). The issue is calibration, not squaring.

- The CE-IS profile construction is not sound. `qb = bins_full.bins[d % len(bins_full.bins)]` cycles bins uniformly (`scripts/run_falsification_harness.py`, Step 4), ignoring `QualityBin.frequency`, actual 153-day bin counts, and temporal order. That does not represent a BTS season's empirical bin mix.

- The pair-correlation test is invalid in the driver. Lines 119-129 aggregate `top1_hit` / `top2_hit` across seeds into fractional `y_rank1` and `y_rank2`, then `pair_residual_correlation` computes Bernoulli Pearson residuals on those floats (`src/bts/validate/dependence.py:208`). That changes the variance by roughly the seed count and breaks the permutation null. It tests correlation of seed-averaged daily rates, not pair residual correlation for Bernoulli game outcomes.

- The verdict thresholds are too brittle for the sample size. The fixed and corrected audits use only the held-out season, so about 24 trajectories. For true `P(57) ~= 0.08`, binomial SE is about `sqrt(.08*.92/24) = 0.055`; a 95% interval is enormous and often has lower bound 0. With 120 trajectories it is still about 0.025 SE. The `corrected_pipeline_lo >= 0.05` DEFENDED gate can fail from sampling noise alone. `n_bootstrap=2000` is not the limiting problem; the trajectory count is.

**3. v1 Simplification Risk**

- **Terminal-R MC vs sequential DR:** invalid if logged data are not full-information counterfactuals for skip/single/double, or if policy actions affect future availability beyond observed rows.

- **Per-season Z vs per-day Z:** invalid if dependence is local, game-day, opponent, weather, or player-form driven rather than one season-wide latent shock.

- **Direct deterministic-theta CE-IS vs simulator Z:** invalid if the claimed `P(57)` is under a stochastic latent-factor transition model. The current CE-IS estimates an independent Bernoulli product model.

- **Slot-date bootstrap vs trajectory-coherent bootstrap:** invalid for streak/path statistics where preserving each seed's actual temporal dependence is required.

- **MoM tau vs full GLMM MLE:** invalid under heterogeneous probabilities, sparse groups, calibration error, or when tau is near the brentq boundary.

**4. Hidden Bugs**

- The early `break` at `src/bts/validate/ope.py:298` does not hurt terminal-R bootstrap coverage by itself. Since reward is already summed per trajectory, stopping after success still gives terminal reward 1. It would matter for sequential DR or any row-count-weighted statistic.

- Reusing the same RNG in `pair_residual_correlation` for permutation and bootstrap (`src/bts/validate/dependence.py:208`) does not bias the CI or p-value. It only makes the two Monte Carlo outputs dependent. Use separate streams for cleanliness, but this is not a blocker.

- `rho_PA_within_game` is currently a dead correction input in `build_corrected_transition_table` (`src/bts/validate/dependence.py:260`). The pipeline relies entirely on `tau_squared` to carry PA-dependence information. That is acceptable only if `tau_hat` is the intended sufficient correction. Given the estimator weaknesses above, the raw `rho_PA` diagnostic is effectively lost.

I would not merge as-is. BLOCK ON: fix the driver's pair-correlation aggregation, replace the cyclic CE-IS season profile with an empirical 153-day profile/mix, and stop classifying a 24-trajectory terminal-MC lower bound as sufficient evidence to defend the headline.

---

## Synthesis (Claude, post-review)

### The blocking findings

1. **Driver pair-correlation aggregation is invalid** — averaging hit indicators across seeds produces fractional values, breaking the permutation null in `pair_residual_correlation`. Tests correlation of seed-averaged daily rates, not Bernoulli game-outcome correlation. Real bug.

2. **CE-IS season profile ignores empirical bin frequencies** — cyclic `bins[d % n_bins]` gives uniform 1/5 per bin instead of `qb.frequency`-weighted mix. Real bug.

3. **Verdict gates are statistically uncalibrated for trajectory count** — at n=24 trajectories, binomial SE on P(57)=0.08 is ~0.055; the DEFENDED gate "CI lower ≥ 0.05" can fail purely from sampling noise even when the headline is true. **Most consequential finding.**

### The non-blocker but real bugs

4. **p_both double-count** — Empirical `b.p_both` already includes whatever cross-game correlation exists in the sample. Adding `delta = rho × sqrt(...)` on top double-counts. The bug pushes corrected_p_both higher than reality, biasing the verdict toward HEADLINE_DEFENDED — exactly the failure mode the harness is supposed to prevent. Real bug.

5. **`rho_PA` dead input** — `pa_residual_correlation` and `fit_logistic_normal_random_intercept` compute the same cross-pair Pearson product from the same data; two estimators redundantly. Smell, not bug.

### The fundamental statistical concern

Codex's most important point is that the harness CANNOT, with the current trajectory count and verdict gates, distinguish a TRUE 8% headline from sampling noise. **No amount of bug-fixing makes a 24-trajectory test capable of certifying an 8% rare event with CI lower ≥ 5pp.** The verdict logic needs either:
- More trajectories per audit (aggregate season-month blocks rather than season-seeds)
- Different verdict thresholds (calibrated to actual achievable CI widths)
- A different uncertainty quantification approach (parametric instead of bootstrap on a small n)

### Decision

User chose Option B: fix the three blockers + p_both double-count + `rho_PA` dead input. ~3-4 hours. Verdict gate recalibration is the hardest of these — requires deciding what verdict thresholds are achievable at the system's actual statistical power.
