# Codex Literature Consultation — BTS Falsification Harness Design

**Date**: 2026-05-01 evening (~23:10 ET)
**Model**: GPT-5.5 (high reasoning) via OpenAI Codex CLI
**Purpose**: Technical literature consultation for the design of a falsification harness covering offline policy evaluation (#13), rare-event Monte Carlo (#14), and dependence testing (#15) — the new top-of-execution-order item per the SOTA audit tracker.
**Cost**: ~$1-2 of high-reasoning tokens
**Outcome**: Produced concrete algorithm sketches, citations, and one substantive pushback (variance-inflation alone is insufficient as a dependence correction — mean correction is also required). Informed all three sub-area sections of the design spec at `docs/superpowers/specs/2026-05-01-bts-falsification-harness-design.md`.

---

## Codex output (verbatim)

## Area 1: Doubly Robust Offline Policy Evaluation

**Recommendation:** use cross-fitted sequential DR-OPE from **Jiang & Li 2016** with tabular fitted-Q / dynamic-programming nuisance estimates. Do not use pure per-decision IS as the main estimator. Use **Precup, Sutton & Singh 2000** style per-decision IS only as a diagnostic for overlap/weight instability. Use **FQE / batch policy evaluation** in the spirit of **Le, Voloshin & Yue 2019** as the nuisance model, not as the headline estimator.

Reason: this MDP is small and tabular, so the hard part is not function approximation. The hard part is honest uncertainty under a long-horizon rare terminal event. DR gives you a model-based value with observed-transition residual correction; pure IS will be too noisy unless the behavior policy nearly matches the target policy.

Citations: [Jiang & Li 2016](https://proceedings.mlr.press/v48/jiang16.html), [Precup et al. 2000](https://web.eecs.umich.edu/~baveja/Papers/OffPolicy.pdf), [Le et al. 2019](https://proceedings.mlr.press/v97/le19a.html).

**Data required:** the existing walk-forward daily profiles are not enough if they contain only predicted Q5/Q4/Q3/Q2/Q1 probabilities. The OPE table must include, for each season/day/seed:

- target-state inputs: `streak`, `days_remaining`, `saver`, `quality_bin`
- available actions: `skip`, `rank1`, `rank1_rank2_double`, plus any admissible alternatives
- predicted success probabilities used by the MDP
- realized hit/no-hit outcome for every candidate needed to evaluate those actions
- realized transition under each action, or enough raw outcomes to reconstruct it
- fold identity: season and calendar day

If only one historical action is logged and the unchosen actions' outcomes are unavailable, this becomes ordinary logged-policy OPE and requires behavior propensities. If the target policy sometimes chooses actions with zero logged support, the value is not identifiable from DR.

**Estimator sketch:**

For fold `f`, train everything except evaluation data in `f`:

1. Refit calibration/profile bins and transition tables using only training seasons/days.
2. Solve the target policy `pi_f` by value iteration on training-only transition estimates.
3. Compute nuisance values `Q_hat_t(s,a)` and `V_hat_t(s)=Q_hat_t(s,pi_f(s))`.
4. On held-out season/day trajectories, replay `pi_f` using realized slate outcomes.
5. Estimate terminal reward `R = 1{streak >= 57}` with sequential DR.

Run this twice:

- **fixed-policy audit:** evaluate the already-frozen headline policy on data not used to build it
- **pipeline audit:** in each fold, rebuild bins, solve VI, and evaluate the resulting policy out of fold

The second one is the right falsification target for "this system achieves 8.17%."

**Uncertainty interval:**

Do not bootstrap individual seeds or individual days naively. The unit of dependence is the MLB slate/day, with seeds sharing the same realized baseball outcomes.

Use a paired hierarchical block bootstrap:

1. Resample seasons as top-level clusters.
2. Within each selected season, resample contiguous day blocks or use the stationary bootstrap of **Politis & Romano 1994**.
3. Keep all 24 seeds for a resampled day together, or resample seed bundles only as a secondary model-perturbation layer.
4. Recompute the OPE estimate and paired baselines per replicate.
5. Report percentile or studentized intervals, plus leave-one-season-out sensitivity because 5 seasons is too few for bootstrap comfort.

Citation: [Politis & Romano 1994](https://www.tandfonline.com/doi/abs/10.1080/01621459.1994.10476870). For very small season counts, also report subsampling sensitivity per [Politis, Romano & Wolf 1999](https://link.springer.com/book/10.1007/978-1-4612-1554-7).

**Policy regret:**

Compute every policy with the same estimator, folds, nuisance models, and bootstrap replicates.

Baselines: `always_skip`, `always_rank1`, pre-MDP heuristic.

Use paired differences inside each bootstrap replicate. Do not compare target DR value to baseline naive MC value; that mixes estimator bias and policy effect.

**Watch-outs:**

- If evaluation folds influenced PA calibration, Q-bin construction, or VI transition estimates, the OPE is still in-sample.
- If seeds are treated as independent seasons, the CI will be far too tight.
- If behavior propensities are unknown and only logged actions are observed, DR cannot rescue missing overlap.

---

## Area 2: Rare-Event Monte Carlo With Variance Reduction

**Recommendation:** use cross-entropy importance sampling, not MLMC. Subset simulation is a useful secondary check, but CE-IS is the best fit here.

Reason: the event is a long finite-horizon survival event over about 153 days, with policy-dependent decisions and Bernoulli-like daily outcomes. `P(57) ≈ 8%` is rare enough for naive MC to be inefficient at tail CIs, but not so rare that subset simulation is the first tool I would reach for. MLMC has no natural fidelity hierarchy here.

Citations: [Rubinstein 1997](https://www.sciencedirect.com/science/article/abs/pii/S0377221796003852), [Rubinstein & Kroese 2017](https://www.wiley.com/en-us/Simulation+and+the+Monte+Carlo+Method%2C+3rd+Edition-p-9781118631980), [Au & Beck 2001](https://authors.library.caltech.edu/records/m7b6w-kbc81), [Giles 2008](https://pubsonline.informs.org/doi/pdf/10.1287/opre.1070.0496).

**CE-IS design:** auxiliary distribution that tilts daily Bernoulli logits by streak / days_remaining / action-type. For correlated simulations, tilt latent day/game factors instead of only Bernoulli draws.

Algorithm:

1. Start with `theta=0`.
2. Simulate `M` seasons under `q_theta`.
3. Score each path by max streak or terminal streak.
4. Keep elite paths, e.g. top 5-10%, or all paths reaching an adaptive threshold.
5. Fit `theta` by weighted logistic MLE on the elite paths, minimizing KL to the conditional rare-event distribution.
6. Repeat until the event rate under `q_theta` is roughly 25-50%.
7. Run the final estimator with the ordinary likelihood-ratio.

Burn-in cost should be modest because 8% is not ultra-rare: start with 5-10 CE rounds of 2k-10k simulated seasons each.

**Correlated outcomes:** add latent factors `Z_t ~ N(0,1)` and `G_{tg} ~ N(0,1)` before CE. Tilt the latent means. The likelihood weight includes both Gaussian density ratios and Bernoulli density ratios.

This matters: if you tilt only individual Bernoulli outcomes, the estimator is answering the wrong independence model.

**Expected variance reduction:**

For `p=0.08`, do not promise 100x. If CE raises event frequency under the proposal to 30-50% and weights are stable, a realistic variance reduction is about 5-12x. With strong day-level correlation and weight dispersion, expect 2-5x. If diagnostics show ESS below 20-30% of `N`, the proposal is too aggressive.

**Watch-outs:**

- Validate unbiasedness with `theta=0` and with toy policies where exact DP value is known.
- Report ESS, max weight share, and log-weight variance. A low-variance-looking IS estimate with one dominant weight is invalid.
- CE tuning data and final estimation data must be separated, or the CI is optimistic.

---

## Area 3: PA-Independence And Cross-Game Dependence

**Within-game PA dependence recommendation:** use conditional residual covariance plus a logistic-normal or beta-binomial overdispersion model. I would not use HSIC or a generic copula LRT as the primary test. You have binary outcomes, 5-7 PAs per batter-game, and strong model-based probabilities; the relevant question is whether residuals remain correlated after conditioning on the LightGBM probabilities.

Citations: [Liang & Zeger 1986](https://academic.oup.com/biomet/article/73/1/13/246001), [Williams 1982](https://academic.oup.com/jrsssc/article/31/2/144/6985105), [Self & Liang 1987](https://www.tandfonline.com/doi/abs/10.1080/01621459.1987.10478472). HSIC reference if used as secondary: [Gretton et al. 2007](https://papers.nips.cc/paper/2007/hash/d5cfead94f5350c12c322b5b664544c1-Abstract.html).

Implementation: Pearson residual + within-batter-game correlation + logistic-normal random-intercept fit + boundary-aware LRT for `tau^2 = 0`.

Translate the fitted model into corrected game-hit probability via quadrature integration over the random-intercept distribution.

Positive within-game PA dependence usually lowers `P(at least one hit)` relative to independent PA aggregation.

**Cross-game rank1/rank2 dependence recommendation:** paired residual covariance with stratified permutation, plus a small hierarchical latent-day model. Do not use a high-dimensional independence test; 80-130 picked pairs per season is too small.

Test statistic `T = sum_t e_{t1} e_{t2}`. Permute `e_{t2}` across days within season/month/probability-bin strata. Combine season p-values with Stouffer or Fisher. Report a confidence interval for residual correlation, not just a p-value.

**Feeding dependence into the MDP:**

Pushback: a variance-inflation factor alone is not enough if dependence changes the mean transition probability. You need two knobs:

1. **Mean correction**
   - Within-game PA dependence: replace `1 - prod(1-p_PA)` with the fitted logistic-normal integrated probability.
   - Cross-game doubles: replace `p1*p2` with `p_{12} = p1*p2 + rho*sqrt(p1(1-p1) p2(1-p2))`, clipped to Fréchet bounds.

2. **Uncertainty inflation**
   - If effective sample size shrinks by 30%, use `n_eff = 0.7n`.
   - Transition SE inflates by `sqrt(n / n_eff)`.

Then run the existing value iteration with pessimistic transition tables.

This does not require a new DP derivation; it requires rebuilding the transition table and rerunning the same VI/evaluation code.

**Watch-outs:**

- PA-level dependence must be estimated out of fold; otherwise the correction is tuned to the same data being audited.
- Positive dependence can help double-hit probability but hurt within-game at-least-one-hit probability. Do not apply one generic "correlation penalty."
- Small-n pair tests should report uncertainty intervals and sensitivity bands. A non-significant p-value is not evidence of independence.

---

## Notable Codex pushbacks (vs the original prompt framing)

1. **Q1 — "which DR estimator?"** I'd asked Codex to choose between Jiang & Li 2016, per-decision IS (Precup 2000), and FQE (Le et al. 2019). Codex pushed back: don't pick *one*, use Jiang & Li 2016 as primary with FQE as the nuisance model and per-decision IS as a diagnostic. Rejected the framing that they were alternatives.

2. **Q2 — "what variance reduction factor to expect?"** I'd asked open-endedly. Codex anchored hard against optimism: 5-12x best case, 2-5x with day-level correlation, no 100x promises. The rare-event distribution at p=0.08 isn't rare enough for the dramatic CE wins literature reports for p ≪ 0.001.

3. **Q3.3 — "how to feed dependence into the MDP as variance-inflation?"** Codex flagged this question itself as wrong: variance-inflation alone is not enough because dependence changes the mean transition probability. Two knobs required (mean correction + uncertainty inflation), not one. **This is the design's deepest insight and is reflected prominently in §6 of the spec.**

4. **The deeper insight Codex surfaced**: positive dependence has *opposite signs* across the two aggregations BTS uses (within-game `1 - prod(1-p)` vs cross-game `p1 * p2`). A single "correlation penalty" applied uniformly is wrong. Within-game ρ_PA > 0 → game P(hit) drops; cross-game ρ_pair > 0 → P(both) rises. Net P(57) effect is empirical, not predictable from sign of either.
