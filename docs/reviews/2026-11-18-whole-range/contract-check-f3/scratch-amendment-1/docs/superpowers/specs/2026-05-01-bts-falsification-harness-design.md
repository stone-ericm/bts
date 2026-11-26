# BTS 8.17% Falsification Harness — Design Spec

**Date**: 2026-05-01 (evening session)
**Status**: Draft → pending Eric review → implementation plan
**Author**: Eric Stone (with Claude assist + Codex GPT-5.5 literature consultation)
**Source**: `docs/superpowers/specs/2026-05-01-bts-sota-audit-tracker.md` revised execution order item 1; `data/external/codex_reviews/2026-05-01-sota-audit.md`
**Codex literature consultation**: `data/external/codex_reviews/2026-05-01-falsification-harness.md` (TODO: persist) — informed all three sub-area technical recommendations

## 1. Motivation

The audit's revised top priority is **defending (or breaking) the headline P(57) = 8.17% pooled claim** before any further model upgrades. Three findings motivate this:

1. **16× gap to literature.** Published SOTA P(57) is ~0.5%. Our pooled estimate of 8.17% is plausible-but-extreme. Codex's framing: "this is not 'promising'; it is a red-alert target for leakage, dependence error, or selection bias."
2. **Current "evaluation" is in-sample.** `evaluate_mdp_policy` reuses the same value function the policy was solved against. There is no honest cross-validated policy value computation in the codebase today.
3. **Two unfounded independence assumptions.** Game-level aggregation `1 - prod(1 - p_PA)` assumes conditional PA independence within a game; double-down policy treats two same-day games as independent across pitchers/weather/etc. Neither has ever been tested empirically.

Plus today's reality check: Phase 1's 8 KEEP-feature backlog all FAILED against the bpm-included baseline — feature engineering returns are plateauing. The frontier shifts from "find more signal in features" to "find out whether the signal we already think we have is real."

## 2. Goals & non-goals

**Goals**:
- Produce **honest cross-validated estimates of P(57)** with proper uncertainty intervals that respect the rare-event nature of the streak survival event AND the day-level dependence structure of MLB slates.
- Empirically **test the two independence assumptions** (within-game PA, cross-game pair) used by the current aggregation and decision layer.
- If dependence is non-trivial, produce **mean-corrected and uncertainty-inflated transition tables** that the existing MDP solver can consume without DP redesign.
- Surface a **falsification verdict**: does the 8.17% headline survive? At what CI? Which component (OPE, dependence, MC variance) accounts for any gap?

**Non-goals (v1)**:
- v1 does NOT change production picks, the production blend, or the production policy. Pure offline analysis.
- v1 does NOT redesign the MDP state space or solver. Outputs feed the existing tabular VI.
- v1 does NOT include online/streaming versions of these tests (deferred to Tier 5+ alongside ACI / RCPS work).
- v1 does NOT cover provider determinism (separate ongoing thread; OCI determinism failure already documented).
- v1 does NOT deploy a "skip if fragile" heuristic in production based on the dependence findings (deferred to v2 once OPE confirms it would help).

## 3. Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  src/bts/validate/ope.py  (NEW)                                      │
│                                                                      │
│  cross_fit_dr_ope(profiles, policy_fn, fold_spec)                   │
│    → DROPEResult(value, ci_lower, ci_upper, regret_table)           │
│                                                                      │
│  fitted_q_evaluation(profiles, policy_fn, train_idx, eval_idx)      │
│    → Q_hat, V_hat (nuisance models)                                  │
│                                                                      │
│  paired_block_bootstrap(profiles, n_replicates=2000)                │
│    → bootstrap distribution of DR-OPE estimate                       │
└────────────┬─────────────────────────────────────────────────────────┘
             │
             │  consumes daily profiles + realized outcomes
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  src/bts/simulate/rare_event_mc.py  (NEW)                            │
│                                                                      │
│  CrossEntropyTilter(theta_init=zeros)                                │
│    fit(simulator_fn, n_rounds=8, n_per_round=5000)                  │
│    sample(n_final=20000)                                             │
│    estimate_p57(samples, weights) → IS estimate + CI                │
│                                                                      │
│  LatentFactorSimulator(profiles, lambda_d, lambda_g)                │
│    sample_season() → SeasonResult with day/game factors tilted     │
└────────────┬─────────────────────────────────────────────────────────┘
             │
             │  cross-checks naive MC + provides correlated-event CI
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  src/bts/validate/dependence.py  (NEW)                               │
│                                                                      │
│  pa_residual_correlation(predictions, outcomes)                     │
│    → rho_PA, ci, p_value (cluster bootstrap)                        │
│                                                                      │
│  fit_logistic_normal_random_intercept(pa_data)                      │
│    → tau_hat, integrated_p_at_least_one_fn                         │
│                                                                      │
│  pair_residual_correlation(rank1_pairs, rank2_pairs)                │
│    → rho_pair, ci, p_value (stratified permutation)                 │
│                                                                      │
│  build_corrected_transition_table(profiles, rho_PA, tau,            │
│                                     rho_pair, alpha=0.95)            │
│    → mean-corrected + uncertainty-inflated bin probabilities         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  scripts/run_falsification_harness.py  (NEW — driver)                │
│                                                                      │
│  1. Load 24-seed pooled walk-forward profiles (existing)             │
│  2. Run cross-fit DR-OPE with paired hierarchical block bootstrap    │
│  3. Run CE-IS rare-event MC; cross-check vs naive MC                 │
│  4. Run within-game and cross-game dependence tests                  │
│  5. If dependence ≠ 0: build corrected transition table, re-solve   │
│     MDP, re-run DR-OPE on the corrected policy                       │
│  6. Emit data/validation/falsification_harness_<DATE>.json with     │
│     verdict + 7 numbers (see §6)                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 4. Detailed design — Area 1: Cross-fit DR-OPE

### 4.1 Estimator choice

**Primary**: cross-fitted sequential **Doubly Robust OPE** (Jiang & Li 2016) with **fitted-Q-evaluation** (Le et al. 2019) as the nuisance model. The MDP is small and tabular (103K states), so function approximation is not the hard part — the hard part is honest uncertainty under a long-horizon rare terminal event. DR gives model-based value with observed-transition residual correction; pure per-decision IS (Precup et al. 2000) is too noisy unless the behavior policy nearly matches the target policy.

**Diagnostic**: per-decision IS as overlap/weight-stability check, not as headline.

**Rejected**: pure FQE as headline (no residual correction); deep batch-RL methods (overkill for tabular).

### 4.2 Two audit modes

| Mode | What it asks | Why both |
|---|---|---|
| **Fixed-policy audit** | Does the *frozen* current policy achieve 8.17% on data not used to build it? | Direct headline-defense |
| **Pipeline audit** | If we rebuild bins + solve VI within each fold, what's the typical out-of-fold value? | The right falsification target — captures policy-construction overfit |

Both are required. A small gap means the policy is robust; a large gap means the 8.17% is partly an artifact of the policy seeing its own test set.

### 4.3 Algorithm sketch (per fold f)

1. Refit calibration/profile bins and transition tables using only training seasons/days.
2. Solve the target policy `π_f` by value iteration on training-only transition estimates.
3. Compute nuisance values `Q̂_t(s,a)` and `V̂_t(s) = Q̂_t(s, π_f(s))`.
4. On held-out trajectories, replay `π_f` using realized slate outcomes.
5. Estimate terminal reward `R = 1{streak ≥ 57}` with sequential DR:

```
V̂^DR_i = V̂_1(s_{i,1})
       + Σ_t ρ_{i,1:t} · [r_{i,t} + V̂_{t+1}(s_{i,t+1}) − Q̂_t(s_{i,t}, a_{i,t})]
```

Here γ = 1, rewards are zero until terminal, and `ρ = 1` in the full-information replay setting where the target action's outcome is observed in our data. (We have rank-1 + rank-2 outcomes for every day, so most action paths can be replayed; the "skip" action is trivially evaluatable.)

Final: `P̂(57) = (1/N) Σ_i V̂^DR_i`.

### 4.4 Uncertainty interval — paired hierarchical block bootstrap

**Do not** bootstrap individual seeds or individual days naively. The unit of dependence is the MLB slate/day. Seeds share the same realized baseball outcomes — treating them as independent units inflates effective n by 24× and produces falsely tight CIs.

Procedure (Politis & Romano 1994 stationary bootstrap):
1. Resample seasons as top-level clusters.
2. Within each selected season, resample contiguous day blocks (block length tuned via auto-correlation; default ~7 days).
3. Keep all 24 seeds for a resampled day **together** (same realized outcomes anyway).
4. Recompute the OPE estimate per replicate.
5. Report percentile + studentized intervals + leave-one-season-out sensitivity (5 seasons is too few for bootstrap comfort alone).

### 4.5 Policy regret

Compute every policy with the **same** estimator, folds, nuisance models, and bootstrap replicates:

```
Δ(π, b) = V̂^DR(π) − V̂^DR(b)    [paired difference per replicate]
```

Baselines:
- `always_skip` — should give P(57) = 0
- `always_rank1` — naive maximum-aggression
- `pre_MDP_heuristic` — the threshold-based strategy that predates the MDP

Report absolute regret (P(57) points) and relative lift. **Do not** compare target DR value to baseline naive-MC value; that mixes estimator bias with policy effect.

### 4.6 Watch-outs

- If evaluation folds influenced PA calibration, Q-bin construction, or VI transition estimates, the OPE is still in-sample. Audit the data flow in each fold rigorously.
- If seeds are treated as independent seasons, the CI will be far too tight.
- If behavior propensities are unknown and only logged actions are observed (not our case but flag it), DR cannot rescue missing overlap.

## 5. Detailed design — Area 2: Cross-entropy importance sampling

### 5.1 Technique choice

**Primary**: Cross-entropy importance sampling (Rubinstein 1997, Rubinstein & Kroese 2017). The event (P(57) ≈ 8%) is rare enough for naive MC to give noisy tail CIs but not so rare that subset simulation (Au & Beck 2001) is the first tool to reach for. MLMC (Giles 2008) has no natural fidelity hierarchy here.

**Secondary check**: subset simulation as cross-validation.

### 5.2 CE-IS auxiliary distribution

Tilt the simulator's per-day Bernoulli logit by streak / days_remaining / action-type:

```
q_θ(Y_{t,a} = 1) = σ(logit(p_{t,a}) + θ_0
                                     + θ_1 · I[a = double]
                                     + θ_2 · streak_t
                                     + θ_3 · days_remaining_t)
```

For correlated simulations: tilt **latent day/game factors** instead of (or in addition to) per-Bernoulli logits. See §5.4.

### 5.3 Algorithm

1. Start with `θ = 0`.
2. Simulate `M = 5000` seasons under `q_θ`.
3. Score each path by max streak.
4. Keep elite paths (top 5-10%, or all paths reaching an adaptive threshold).
5. Fit `θ` by weighted logistic MLE on elite paths (minimize KL to conditional rare-event distribution).
6. Repeat for 5-10 rounds until event rate under `q_θ` reaches ~25-50%.
7. Final estimator with `N = 20000`:

```
P̂ = (1/N) Σ_i 1{E_i} · dP(X_i)/dQ_θ(X_i)
```

Use the ordinary likelihood-ratio estimator as primary. Self-normalized estimates as diagnostics.

For independent Bernoulli outcomes:

```
w_i = Π_t [p_{t,a_t}^{y_t} (1 - p_{t,a_t})^{1 - y_t}]
         / [q_{t,a_t}^{y_t} (1 - q_{t,a_t})^{1 - y_t}]
```

### 5.4 Correlated outcomes — latent factors

Add latent factors **before** the CE tilt:

```
Z_t  ~ N(0, 1)         [day factor]
G_{t,g} ~ N(0, 1)      [game factor, nested within day]

logit(p*_{t,g,j}) = logit(p_{t,g,j}) + λ_d · Z_t + λ_g · G_{t,g}
```

CE then tilts the latent means: `Z_t ~ N(μ_d, 1)`, `G_{t,g} ~ N(μ_g, 1)`, possibly residual Bernoulli logits. The likelihood weight includes both Gaussian density ratios and Bernoulli density ratios.

This matters: if you tilt only individual Bernoulli outcomes, the estimator is answering the wrong independence model. The dependence findings from §6 feed directly into `λ_d` and `λ_g`.

### 5.5 Expected variance reduction

For p ≈ 0.08:
- Best case (CE raises event frequency to 30-50%, weights stable): **5-12× variance reduction**
- With strong day-level correlation and weight dispersion: **2-5×**
- If diagnostics show ESS below 20-30% of N: proposal too aggressive, retune.

**Do not promise 100×.** The headline event is not ultra-rare.

### 5.6 Watch-outs

- **Validate unbiasedness with `θ = 0`** and with toy policies where exact DP value is known (we have `bts.simulate.exact` for this — a free oracle).
- Report **ESS, max weight share, log-weight variance**. A low-variance-looking IS estimate dominated by one weight is invalid.
- **CE tuning data and final estimation data must be separated**, or the CI is optimistic.

## 6. Detailed design — Area 3: Dependence testing + MDP corrections

### 6.1 Within-game PA dependence

**Test**: Pearson residual correlation + logistic-normal random-intercept model. (Liang & Zeger 1986 GEE; Williams 1982 beta-binomial; Self & Liang 1987 boundary-aware LRT.)

**Rejected**: HSIC (Gretton et al. 2007) and generic copula likelihood-ratio. Reasons: binary outcomes, only 5-7 PAs per batter-game, strong model-based probabilities → the relevant question is whether residuals remain correlated *after* conditioning on LightGBM blend predictions.

**Implementation**:

1. For each PA, compute Pearson residual:
   ```
   e_{i,j} = (y_{i,j} − p̂_{i,j}) / sqrt(p̂_{i,j} (1 − p̂_{i,j}))
   ```
2. Group by batter-game.
3. Estimate within-game residual covariance:
   ```
   ρ̂_PA = corr(e_{i,j}, e_{i,k}),  j ≠ k
   ```
4. Test with cluster bootstrap (resample batter-games) or stratified permutation within calibrated probability strata.
5. Fit logistic-normal random-intercept:
   ```
   logit(P(y_{i,j} = 1)) = logit(p̂_{i,j}) + u_i,    u_i ~ N(0, τ²)
   ```
6. Test `τ² = 0` using boundary-aware LRT (Self & Liang 1987) or parametric bootstrap.

**Mean correction** (if τ² > 0):
```
P(at least one hit) = 1 − E[Π_j (1 − p*_{i,j}(u))]
```
Computed by quadrature over the fitted Gaussian for `u`.

### 6.2 Cross-game pair dependence (rank-1 + rank-2)

**Test**: paired residual covariance with stratified permutation.

**Constraint**: only ~80-130 picked pairs per season. High-dim independence tests don't work at this n. Methods needing n=10k are excluded.

**Implementation**:

1. Per double-down picked pair:
   ```
   e_{t,1} = (y_{t,1} − p_{t,1}) / sqrt(p_{t,1} (1 − p_{t,1}))
   e_{t,2} = (y_{t,2} − p_{t,2}) / sqrt(p_{t,2} (1 − p_{t,2}))
   ```
2. Test statistic:
   ```
   T = Σ_t e_{t,1} e_{t,2}
   ```
3. Null: rank-1 and rank-2 residuals are conditionally independent given predicted probabilities and slate context.
4. Permutation: permute `e_{t,2}` across days within (season × month × probability-bin) strata; preserve same-day pairing for `e_{t,1}`. Combine season p-values via Stouffer/Fisher.
5. **Report a CI for residual correlation, not just a p-value.** Small-n null-acceptance is not evidence of independence.

Also fit:
```
logit(P(y_{t,j} = 1)) = logit(p_{t,j}) + u_t,    u_t ~ N(0, τ²_pair)
```
where `u_t` is a slate/day factor shared by both picked games.

**Mean correction** (if ρ_pair ≠ 0):
```
P(both hit | rank-1, rank-2) ≈ p_1 · p_2 + ρ_pair · sqrt(p_1 (1 − p_1) p_2 (1 − p_2))
```
Clipped to Fréchet-Hoeffding bounds: `max(0, p_1 + p_2 − 1) ≤ P(both) ≤ min(p_1, p_2)`.

### 6.3 Critical insight: within-game and cross-game dependence have OPPOSITE signs in their effect on P(57)

This is the deepest finding from the Codex consultation and deserves its own section because it changes how the harness is interpreted.

| Aggregation | What dependence > 0 does |
|---|---|
| Within-game `1 − Π(1 − p_PA)` | **Lowers** `P(at least one hit)` — positive PA correlation means hits cluster, no-hit games stay no-hit |
| Cross-game pair `p_1 · p_2` for double | **Raises** `P(both hit)` — positive day correlation means good slates produce two hits together |

So a single "correlation penalty" applied uniformly across the MDP is **wrong**. The harness must apply two separate corrections that point in different directions:

- Within-game ρ_PA > 0 → game-level `p_hit` drops → MDP under-pivots toward `single` action
- Cross-game ρ_pair > 0 → double-down `P(both)` rises → MDP **could** pivot more toward `double`, but with inflated uncertainty

The net effect on P(57) depends on which dominates. Empirical answer needed.

### 6.4 Feeding corrections into the MDP

Codex pushed back on my original framing of "variance-inflation factor as the dependence correction." Two knobs are required, not one:

**Knob 1 — Mean correction**: rebuild the per-bin transition probabilities using:
- Within-game integrated `P(at least one hit)` from logistic-normal model (replaces the bin's mean `p_hit`)
- Cross-game `P(both)` from Fréchet-clipped Pearson correction (replaces the bin's mean `p_both`)

**Knob 2 — Uncertainty inflation**: if effective sample size shrinks (e.g., `n_eff = 0.7n`), inflate transition SE by `√(n / n_eff) = √(1/0.7) ≈ 1.20`. Then run pessimistic VI:

```
p̃(s, a) = p̂(s, a) − z_{0.95} · sqrt(φ · p̂(s, a) (1 − p̂(s, a)) / n)
```

where `φ = n / n_eff`.

This does **not** require a new DP derivation. It rebuilds the transition table and reruns the existing VI/evaluation code.

### 6.5 Watch-outs

- **PA-level dependence must be estimated out-of-fold.** Otherwise the correction is tuned to the same data being audited.
- **Positive dependence has opposite signs across aggregations.** Do not apply one generic "correlation penalty" (see §6.3).
- **Small-n pair tests should report uncertainty intervals and sensitivity bands.** A non-significant p-value is not evidence of independence.

## 7. Validation gates / falsification verdict

The harness emits a single JSON with seven numbers and a verdict:

```json
{
  "date": "YYYY-MM-DD",
  "headline_p57_in_sample": 0.0817,
  "fixed_policy_dr_ope_p57": "x.xx [ci_lo, ci_hi]",
  "pipeline_dr_ope_p57":     "x.xx [ci_lo, ci_hi]",
  "rare_event_ce_p57":       "x.xx [ci_lo, ci_hi]",
  "rho_PA_within_game":      "x.xx [ci_lo, ci_hi]",
  "rho_pair_cross_game":     "x.xx [ci_lo, ci_hi]",
  "corrected_pipeline_p57":  "x.xx [ci_lo, ci_hi]",
  "verdict": "HEADLINE_DEFENDED | HEADLINE_REDUCED | HEADLINE_BROKEN",
  "verdict_rationale": "..."
}
```

Verdict rules:

- **HEADLINE_DEFENDED**: corrected_pipeline_p57 CI lower bound ≥ 5pp AND fixed_policy_dr_ope CI overlaps 8.17%.
- **HEADLINE_REDUCED**: corrected_pipeline_p57 point estimate in [3pp, 6pp] OR CI lower bound in [2pp, 5pp]. Production policy still better than always-rank1; the 8.17% claim is partly artifact.
- **HEADLINE_BROKEN**: corrected_pipeline_p57 point estimate < 3pp OR CI overlaps always-rank1 baseline. Triggers a full rebuild of the policy with the corrected transition tables, plus a memory note documenting the headline retraction.

If `pipeline_dr_ope` ≪ `fixed_policy_dr_ope`, that's separate evidence of policy-construction overfit independent of dependence corrections. Document either way.

## 8. Data flow

```
data/simulation/backtest_{season}.parquet  (existing)
        │
        │  per-day, per-seed, per-rank rows: (date, top_k, p_game_hit, hit, …)
        ▼
load_pooled_profiles  (existing in pooled_policy.py)
        │
        ▼
┌───────┴───────┬──────────────────┬──────────────────┐
│               │                  │                  │
▼               ▼                  ▼                  ▼
DR-OPE         CE-IS              PA dependence     Pair dependence
(ope.py)       (rare_event_mc.py) (dependence.py)   (dependence.py)
│               │                  │                  │
└──────┬────────┴──────────────────┴──────────────────┘
       │
       ▼
Corrected transition table (dependence.py: build_corrected_transition_table)
       │
       ▼
solve_mdp(corrected_transitions)  [existing; we just feed different inputs]
       │
       ▼
DR-OPE on corrected policy (ope.py)
       │
       ▼
data/validation/falsification_harness_<DATE>.json
```

PA-level predictions are required for §6.1. They exist in the walk-forward pipeline at `predict_local`'s pre-aggregation step but are not currently persisted to backtest profiles. Implementation note: the harness needs to either (a) re-run walk-forward with PA-level prediction logging enabled, or (b) reconstruct PA predictions from saved blend models. (a) is cleaner; (b) avoids a re-run.

## 9. Open questions / risks

1. **PA-level data availability**: are per-PA predicted probabilities stored anywhere in the existing walk-forward profiles, or do we need to extend the pipeline to log them? (Leaning: extend with a flag that's off by default, on for harness runs.)
2. **CE-IS correctness validation**: `bts.simulate.exact` provides exact P(57) for fixed strategies on any bin set. We have a free oracle for unbiasedness checks at `θ = 0`. Use it.
3. **Block bootstrap block length**: the right block length for daily MLB outcomes is unclear without checking auto-correlation. Default 7-day blocks; tune per-season.
4. **Behavior propensities**: under what conditions can an actually-deployed-policy log be used as the behavior policy? (Production picks are deterministic given inputs, so propensity = 1 at the chosen action and 0 elsewhere — this is exactly the "no overlap" scenario where DR cannot rescue. The harness's audit modes (§4.2) work because we have rank-1 + rank-2 + skip outcomes available for replay; we don't need historical behavior policy at all.)
5. **Computational cost**: the pipeline audit (refit + re-solve VI per fold) could be expensive on 5+ seasons × 24 seeds. Estimate before committing — possibly start with 3 folds × 24 seeds, expand if signal warrants.
6. **What if the harness verdict is HEADLINE_BROKEN?** The audit has succeeded by finding the truth. Production policy gets rebuilt with corrected transitions. Memory note documents the retraction. No rollback to a worse policy unless the corrected one underperforms; in that case revert to heuristic fallback.

## 10. Implementation plan (separate document)

A TDD-driven plan covering:

- Task 0: branch + dependency adds (statsmodels for logistic-normal MLE; possibly arch for stationary bootstrap)
- Tasks 1-3: `bts.validate.ope` (DR-OPE + paired block bootstrap + policy regret table) — TDD against synthetic small-MDP fixtures with known DR values
- Tasks 4-5: `bts.simulate.rare_event_mc` (CE-IS + latent-factor simulator) — unbiasedness check at `θ = 0` against `bts.simulate.exact` is a hard test gate
- Tasks 6-7: `bts.validate.dependence` (PA residuals + logistic-normal MLE + pair permutation) — synthetic data generators with known ρ
- Task 8: `build_corrected_transition_table` + integration with existing `solve_mdp`
- Task 9: `scripts/run_falsification_harness.py` driver + JSON emit
- Task 10: validation gate run on real data; verdict; memo

The implementation plan goes in `docs/superpowers/plans/2026-05-XX-bts-falsification-harness.md` after Eric reviews this spec.

## 11. Why this is the right next move

1. **It addresses the audit's actual blocker**: every other SOTA upgrade (CVaR-MDP, decision-aware learning, model-class challenge) requires honest baseline measurement. None of them can be evaluated against a headline number that may itself be partly artifact.
2. **It builds reusable infrastructure**. DR-OPE, CE-IS, and dependence-aware transition correction are not one-shot tools — every future audit benefits from them.
3. **It surfaces a concrete deliverable in 1-2 weeks** (the verdict JSON), unlike the open-ended "investigate model class" or "design CVaR-MDP" alternatives.
4. **It's the SOTA-audit framing applied to itself**: rather than "install respectable methods on top of an unverified base," it asks "is the base verified?" first. That's the falsification-orientation Codex pushed for and that the methodological note in the tracker document committed to.
