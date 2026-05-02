"""PA-conditional-independence + cross-game pair-dependence diagnostics + MDP corrections.

References:
- Liang & Zeger 1986, Longitudinal data analysis using GLMs.
- Williams 1982, Extra-binomial variation in logistic linear models.
- Self & Liang 1987, Asymptotic properties of MLE under non-standard conditions.
- Romano 1989, Bootstrap and randomization tests of some nonparametric hypotheses.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pearson_residual(y: int | float, p: float) -> float:
    """Pearson residual for a Bernoulli prediction.

    e = (y - p) / sqrt(p * (1 - p))

    Clips p to [1e-9, 1 - 1e-9] to avoid division by zero at the boundaries.
    """
    p = max(min(p, 1 - 1e-9), 1e-9)
    return float((y - p) / np.sqrt(p * (1 - p)))


def pa_residual_correlation(
    df: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Estimate within-batter-game PA residual correlation.

    For each batter-game with >=2 PAs, compute Pearson residuals and average the
    off-diagonal residual products across pairs. Cluster bootstrap (resampling
    batter-games) gives the CI; p-value is two-sided based on bootstrap quantiles.

    df columns required: batter_game_id, p_pa, actual_hit.

    Returns: (rho_hat, ci_lower, ci_upper, p_value).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["e"] = [pearson_residual(y, p) for y, p in zip(df["actual_hit"], df["p_pa"])]

    # Compute the off-diagonal pair products from each batter-game.
    pair_products = []
    for _, group in df.groupby("batter_game_id"):
        residuals = group["e"].to_numpy()
        if len(residuals) < 2:
            continue
        for i in range(len(residuals)):
            for j in range(i + 1, len(residuals)):
                pair_products.append(residuals[i] * residuals[j])
    pair_products = np.array(pair_products)
    if len(pair_products) == 0:
        return 0.0, 0.0, 0.0, 1.0
    rho_hat = float(pair_products.mean())

    # Cluster bootstrap: resample batter_game_id values with replacement.
    bg_ids = df["batter_game_id"].unique()
    bs_estimates = np.empty(n_bootstrap)
    # Pre-group the residual arrays for speed.
    residuals_by_bg = {bg: df.loc[df["batter_game_id"] == bg, "e"].to_numpy() for bg in bg_ids}
    for b in range(n_bootstrap):
        sample_ids = rng.choice(bg_ids, size=len(bg_ids), replace=True)
        bs_pairs = []
        for bg in sample_ids:
            residuals = residuals_by_bg[bg]
            for i in range(len(residuals)):
                for j in range(i + 1, len(residuals)):
                    bs_pairs.append(residuals[i] * residuals[j])
        bs_estimates[b] = float(np.mean(bs_pairs)) if bs_pairs else 0.0

    ci_lo = float(np.quantile(bs_estimates, 0.025))
    ci_hi = float(np.quantile(bs_estimates, 0.975))
    # Two-sided p-value: how often does the bootstrap distribution agree with H0: rho=0?
    p_value = float(2 * min(np.mean(bs_estimates >= 0), np.mean(bs_estimates <= 0)))

    return rho_hat, ci_lo, ci_hi, p_value


def fit_logistic_normal_random_intercept(
    df: pd.DataFrame,
    *,
    p_col: str = "p_pred",
    y_col: str = "y",
    group_col: str = "group_id",
    n_quad_points: int = 21,
):
    """Fit logit(P(y=1)) = logit(p_pred) + u, u ~ N(0, tau^2). Returns (tau_hat, integrate_fn).

    Method-of-moments estimation via cross-pair Pearson residual inversion
    -----------------------------------------------------------------------
    The estimator works in two stages:

    Stage 1 — Estimate the intra-class correlation (rho_hat):
      Compute Pearson residuals e_ij = (y_ij - p_ij) / sqrt(p_ij*(1-p_ij))
      using the model-predicted p_ij (not the realized probability).  For
      pairs (j, k) within the same group, form e_ij * e_ik.  Average across
      all within-group pairs:
          rho_hat = mean(e_ij * e_ik)
      Under the logistic-normal model with tau > 0, rho_hat > 0 because each
      latent u shifts all outcomes in the same direction.

    Stage 2 — Invert the theoretical rho(tau) curve:
      The theoretical expectation under the model is
          E[e_ij * e_ik] = E_u[(sigmoid(logit(p) + u) - p)^2 / (p*(1-p))]
      which is a monotone increasing function of tau (computed via Gauss-
      Hermite quadrature).  We use scipy.optimize.brentq to invert it and
      recover tau_hat.  This gives tau in the same Gaussian latent scale as
      the data-generating process.

    Deliberate simplification vs full GLMM MLE
    -------------------------------------------
    statsmodels GEE / GLMM MLE have known convergence fragility on small
    synthetic datasets and the naive variance-inflation formula
    (tau^2 ≈ mean_within_var - 1) is incorrect here because Pearson residuals
    use p_pred (the fixed prediction) rather than the realized probability —
    the within-group variance actually decreases with tau, not increases, so
    the naive formula is backwards.  The cross-pair inversion approach is:
      (a) Correct: rho_hat is monotonically related to tau and invertible.
      (b) Robust: no iterative MLE, no convergence failures.
      (c) Handles tau→0 cleanly: rho_hat ≤ 0 → tau_hat = 0.

    Edge cases:
    - All groups have ≤ 1 PA (no cross-pairs): rho_hat = 0 → tau_hat = 0,
      integrate_fn collapses to independence product.
    - Empty df: same as above.

    integrate_fn(p_list) computes P(>=1 hit | tau_hat) by marginalising over
    u via Gauss-Hermite quadrature:
        E_u[1 - prod_j (1 - sigmoid(logit(p_j) + u))],  u ~ N(0, tau_hat^2)

    For tau_hat ≈ 0 the integral collapses to 1 - prod(1 - p) (independence).
    For tau_hat > 0 positive within-game correlation reduces P(>=1 hit) below
    the independence baseline (see BTS spec §6.3).
    """
    from scipy.optimize import brentq

    df = df.copy()
    df["_e"] = [pearson_residual(y, p) for y, p in zip(df[y_col], df[p_col])]
    mean_p = float(df[p_col].mean()) if len(df) > 0 else 0.25
    # Clip mean_p away from 0/1 to keep logit finite.
    mean_p = max(min(mean_p, 1 - 1e-9), 1e-9)

    # Stage 1: cross-pair Pearson residual product (intra-class correlation estimate).
    quad_x, quad_w = np.polynomial.hermite_e.hermegauss(n_quad_points)
    pair_products: list[float] = []
    for _, grp in df.groupby(group_col):
        residuals = grp["_e"].to_numpy()
        n = len(residuals)
        if n < 2:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                pair_products.append(float(residuals[i] * residuals[j]))

    rho_hat = float(np.mean(pair_products)) if pair_products else 0.0

    # Stage 2: invert E[e_i*e_j](tau) = rho_hat to recover tau_hat.
    def _expected_cross_product(tau: float) -> float:
        """E_u[(sigmoid(logit(p)+u) - p)^2 / (p*(1-p))], u ~ N(0, tau^2)."""
        logit_p = float(np.log(mean_p / (1 - mean_p)))
        var_p = mean_p * (1 - mean_p)
        total = 0.0
        for xi, wi in zip(quad_x, quad_w):
            u = tau * xi
            p_r = 1.0 / (1.0 + np.exp(-(logit_p + u)))
            total += wi * ((p_r - mean_p) ** 2 / var_p)
        return total / float(quad_w.sum())

    tau_hat: float
    if rho_hat <= 0.0:
        tau_hat = 0.0
    else:
        tau_max = 5.0
        max_theory = _expected_cross_product(tau_max)
        if rho_hat >= max_theory:
            tau_hat = tau_max
        else:
            try:
                tau_hat = float(brentq(
                    lambda t: _expected_cross_product(t) - rho_hat,
                    0.0, tau_max, xtol=1e-4,
                ))
            except ValueError:
                tau_hat = 0.0

    # Build integrate_fn using the same Gauss-Hermite nodes.
    def integrate_fn(p_list: "list[float]") -> float:
        """E_u[1 - prod_j (1 - sigmoid(logit(p_j) + u))], u ~ N(0, tau_hat^2)."""
        if tau_hat <= 1e-9:
            return float(1.0 - np.prod([1.0 - p for p in p_list]))
        result = 0.0
        for xi, wi in zip(quad_x, quad_w):
            u = tau_hat * xi
            p_at_least_one = 1.0 - float(np.prod([
                1.0 - 1.0 / (1.0 + np.exp(-(np.log(max(min(p, 1-1e-9), 1e-9) / (1 - max(min(p, 1-1e-9), 1e-9))) + u)))
                for p in p_list
            ]))
            result += wi * p_at_least_one
        return float(result / quad_w.sum())

    return tau_hat, integrate_fn


def pair_residual_correlation(
    df: pd.DataFrame,
    *,
    n_permutations: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float, float]:
    """Stratified permutation test on rank-1/rank-2 Pearson residuals.

    For each row (one row per day):
        e_{t,1} = pearson_residual(y_rank1, p_rank1)
        e_{t,2} = pearson_residual(y_rank2, p_rank2)

    Test statistic: T = mean over t of (e_{t,1} * e_{t,2}).

    Null H0: rank-1 and rank-2 residuals are conditionally independent given
    predicted probabilities.

    Permutation: shuffle e_{t,2} across days. The null distribution of T under
    permutation gives the two-sided p-value.

    CI on rho_hat: paired bootstrap on rows.

    df columns required: date, p_rank1, p_rank2, y_rank1, y_rank2.

    Returns: (rho_hat, ci_lower, ci_upper, p_value).
    """
    rng = np.random.default_rng(seed)
    e1 = np.array([pearson_residual(y, p) for y, p in zip(df["y_rank1"], df["p_rank1"])])
    e2 = np.array([pearson_residual(y, p) for y, p in zip(df["y_rank2"], df["p_rank2"])])

    rho_hat = float(np.mean(e1 * e2))

    # Permutation null distribution: shuffle e2 across positions.
    null_distribution = np.empty(n_permutations)
    for k in range(n_permutations):
        shuffled = rng.permutation(e2)
        null_distribution[k] = float(np.mean(e1 * shuffled))
    # Two-sided p-value: how often is the absolute null statistic >= |observed|?
    p_value = float(np.mean(np.abs(null_distribution) >= abs(rho_hat)))

    # CI via paired bootstrap on rows.
    n = len(e1)
    bs = np.empty(n_permutations)
    for k in range(n_permutations):
        idx = rng.integers(0, n, n)
        bs[k] = float(np.mean(e1[idx] * e2[idx]))
    ci_lo = float(np.quantile(bs, 0.025))
    ci_hi = float(np.quantile(bs, 0.975))

    return rho_hat, ci_lo, ci_hi, p_value
