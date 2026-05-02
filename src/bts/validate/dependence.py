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
