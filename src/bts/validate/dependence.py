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
    # Vectorized Pearson residual computation (was a list comp ~1.8M times).
    p = np.clip(df["p_pa"].to_numpy(), 1e-9, 1 - 1e-9)
    y = df["actual_hit"].to_numpy()
    df["e"] = (y - p) / np.sqrt(p * (1 - p))

    # Closed-form per-group pair statistics (Codex round 4):
    # For residuals e_1..e_n in a group:
    #   sum_{i<j} e_i*e_j = (sum_e^2 - sum_e2) / 2
    #   n_pairs = n*(n-1)/2
    # So rho_hat = total_pair_sum / total_pair_n. This avoids materializing
    # any pair products and runs in O(N) instead of O(N*k^2).
    grouped = df.groupby("batter_game_id")["e"]
    sum_e = grouped.sum().to_numpy()
    sum_e2 = grouped.apply(lambda v: float(np.square(v).sum())).to_numpy()
    counts = grouped.size().to_numpy()
    pair_sum = 0.5 * (sum_e**2 - sum_e2)
    pair_n = counts * (counts - 1) // 2

    total_pair_n = pair_n.sum()
    if total_pair_n == 0:
        return 0.0, 0.0, 0.0, 1.0
    rho_hat = float(pair_sum.sum() / total_pair_n)

    # Cluster bootstrap via bincount (Codex round 4): for each rep, sample
    # group indices with replacement, get group-counts via np.bincount, then
    # weighted ratio. Pure numpy, no Python loops over groups.
    G = len(pair_sum)
    bs_estimates = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, G, size=G)
        c = np.bincount(idx, minlength=G)
        num = float(c @ pair_sum)
        den = float(c @ pair_n)
        bs_estimates[b] = num / den if den > 0 else 0.0

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
    # Closed-form per-group sums (Codex round 4 perf fix):
    #   sum_{i<j} e_i*e_j = (sum_e^2 - sum_e2) / 2
    # avoids materializing pair products. O(N) instead of O(N*k^2).
    quad_x, quad_w = np.polynomial.hermite_e.hermegauss(n_quad_points)
    grouped_e = df.groupby(group_col)["_e"]
    sum_e = grouped_e.sum().to_numpy()
    sum_e2 = grouped_e.apply(lambda v: float(np.square(v).sum())).to_numpy()
    counts = grouped_e.size().to_numpy()
    pair_sum = 0.5 * (sum_e**2 - sum_e2)
    pair_n = counts * (counts - 1) // 2
    total_pair_n = pair_n.sum()
    rho_hat = float(pair_sum.sum() / total_pair_n) if total_pair_n > 0 else 0.0

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
    bin_assignment: "pd.Series | np.ndarray | None" = None,
    expected_bin_indices: "np.ndarray | list | None" = None,
    strict_bin_labels: bool = True,
) -> "tuple[float, float, float, float] | dict":
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

    When bin_assignment is None: returns (rho_hat, ci_lower, ci_upper, p_value).

    When bin_assignment is provided (one rank-1 bin index per row in df):
    returns dict with arrays indexed by `expected_bin_indices` (when given) or
    by sorted unique values of bin_assignment otherwise.

    **CRITICAL** (caught by Codex round 1 review): caller MUST pass
    `expected_bin_indices=np.arange(n_bins)` when the consumer indexes the
    output by `bin.index` (as `build_corrected_transition_table` does).
    Otherwise output[i] silently means different bins on different folds when
    a fold's data lacks bin labels.

    Returned dict keys:
        rho_per_bin: shape-(K,) array; rho_per_bin[k] is rho for bin k
            where K = len(expected_bin_indices) if given, else len(unique(bin_assignment))
            NOTE: arrays are position-indexed (not label-indexed). Direct lookup
            `rho_per_bin[b.index]` is safe ONLY when `expected_bin_indices == np.arange(n_bins)`.
            For arbitrary label sets like [10, 20, 30], use `bin_indices` to look up by label.
        ci_lo_per_bin: shape-(K,)
        ci_hi_per_bin: shape-(K,)
        p_value_per_bin: shape-(K,)
        n_per_bin: shape-(K,) — bin observation counts (0 means bin absent in this fold's data)
        bin_indices: shape-(K,) — the bin indices each output row corresponds to
        global_rho, global_ci_lo, global_ci_hi, global_p_value: aggregate scalars

    Empty-bin behavior: when n_per_bin[k] < 2, rho_per_bin[k]=0.0,
    ci_lo/ci_hi=0.0, p_value=1.0. Caller should check n_per_bin and treat
    rho=0 as "uninformative for this bin in this fold."

    Strict label validation (strict_bin_labels=True, default): raises ValueError
    when bin_assignment contains labels not in expected_bin_indices. This
    fail-closed default protects against bin-classification bugs hiding silently.
    Set strict_bin_labels=False to opt into silent exclusion of unexpected
    labels (useful for diagnostic explorations).
    """
    rng = np.random.default_rng(seed)
    e1 = np.array([pearson_residual(y, p) for y, p in zip(df["y_rank1"], df["p_rank1"])])
    e2 = np.array([pearson_residual(y, p) for y, p in zip(df["y_rank2"], df["p_rank2"])])

    rho_hat = float(np.mean(e1 * e2))

    # Permutation null distribution: shuffle e2 across positions.
    null_distribution = np.empty(n_permutations)
    for j in range(n_permutations):
        shuffled = rng.permutation(e2)
        null_distribution[j] = float(np.mean(e1 * shuffled))
    # Two-sided p-value: how often is the absolute null statistic >= |observed|?
    p_value = float(np.mean(np.abs(null_distribution) >= abs(rho_hat)))

    # CI via paired bootstrap on rows.
    n = len(e1)
    bs = np.empty(n_permutations)
    for j in range(n_permutations):
        idx = rng.integers(0, n, n)
        bs[j] = float(np.mean(e1[idx] * e2[idx]))
    ci_lo = float(np.quantile(bs, 0.025))
    ci_hi = float(np.quantile(bs, 0.975))

    if bin_assignment is None:
        return rho_hat, ci_lo, ci_hi, p_value

    # Per-bin path. bin_assignment must have len == len(df).
    bins_arr = np.asarray(bin_assignment)
    if bins_arr.ndim != 1:
        raise ValueError(f"bin_assignment must be 1D, got ndim={bins_arr.ndim}")
    if len(bins_arr) != n:
        raise ValueError(
            f"bin_assignment length {len(bins_arr)} != df length {n}"
        )
    # Use caller-supplied expected indices if given (safe contract); else fall
    # back to sorted unique values (legacy behavior).
    if expected_bin_indices is not None:
        bin_indices = np.asarray(expected_bin_indices)
        if bin_indices.ndim != 1:
            raise ValueError(f"expected_bin_indices must be 1D, got ndim={bin_indices.ndim}")
        if strict_bin_labels:
            unexpected = set(np.unique(bins_arr).tolist()) - set(bin_indices.tolist())
            if unexpected:
                raise ValueError(
                    f"bin_assignment contains labels {sorted(unexpected)} not in "
                    f"expected_bin_indices. Pass strict_bin_labels=False to allow "
                    f"silent exclusion of unexpected labels."
                )
    else:
        bin_indices = np.sort(np.unique(bins_arr))
    K = len(bin_indices)
    rho_per_bin = np.zeros(K)
    ci_lo_per_bin = np.zeros(K)
    ci_hi_per_bin = np.zeros(K)
    p_per_bin = np.ones(K)  # default p=1.0 for empty bins
    n_per_bin = np.zeros(K, dtype=int)

    for k, bin_idx in enumerate(bin_indices):
        mask = bins_arr == bin_idx
        e1_b = e1[mask]
        e2_b = e2[mask]
        n_b = len(e1_b)
        n_per_bin[k] = n_b
        if n_b < 2:
            # Empty/singleton bin: rho=0, p=1.0 (uninformative); rest already
            # zeroed at init. Caller should check n_per_bin.
            continue
        rho_b = float(np.mean(e1_b * e2_b))
        rho_per_bin[k] = rho_b
        # Permutation null within this bin.
        null_b = np.empty(n_permutations)
        for j in range(n_permutations):
            shuffled_b = rng.permutation(e2_b)
            null_b[j] = float(np.mean(e1_b * shuffled_b))
        p_per_bin[k] = float(np.mean(np.abs(null_b) >= abs(rho_b)))
        # Bootstrap CI within this bin.
        bs_b = np.empty(n_permutations)
        for j in range(n_permutations):
            idx_b = rng.integers(0, n_b, n_b)
            bs_b[j] = float(np.mean(e1_b[idx_b] * e2_b[idx_b]))
        ci_lo_per_bin[k] = float(np.quantile(bs_b, 0.025))
        ci_hi_per_bin[k] = float(np.quantile(bs_b, 0.975))

    return {
        "rho_per_bin": rho_per_bin,
        "ci_lo_per_bin": ci_lo_per_bin,
        "ci_hi_per_bin": ci_hi_per_bin,
        "p_value_per_bin": p_per_bin,
        "n_per_bin": n_per_bin,
        "bin_indices": bin_indices,
        "global_rho": rho_hat,
        "global_ci_lo": ci_lo,
        "global_ci_hi": ci_hi,
        "global_p_value": p_value,
    }


def build_corrected_transition_table(
    bins,                          # QualityBins
    *,
    rho_PA_within_game: float,
    tau_squared: float,
    rho_pair_cross_game: float | np.ndarray | list | tuple,
    n_pa_per_game: int = 5,
):
    """Apply two-knob mean corrections to a QualityBins instance.

    Returns a new QualityBins with corrected p_hit (logistic-normal integration
    for within-game PA dependence) and p_both (Pearson + Frechet bounds for
    cross-game pair dependence). Bin index, p_range, frequency, and boundaries
    are preserved from the original.

    Mean correction for p_hit (when tau_squared > 0):
        p_pa_indep = 1 - (1 - p_hit_orig)^(1/n_pa_per_game)
        new_p_hit = E_u[1 - (1 - sigmoid(logit(p_pa_indep) + u))^n_pa_per_game]
                    where u ~ N(0, tau_squared)
        Computed via Gauss-Hermite quadrature.

    Mean correction for p_both (when rho_pair_cross_game != 0):
        new_p_both = p1*p2 + rho_pair * sqrt(p1*(1-p1)*p2*(1-p2))
        Clipped to Frechet-Hoeffding bounds [max(0, p1+p2-1), min(p1, p2)].

    rho_pair_cross_game: scalar or array-like (list, tuple, numpy array).
        - Scalar (float): same rho applied to every bin. Backward-compatible;
          np.asarray(0.05).ravel() gives size-1 array → uniform broadcast.
        - Per-bin vector of length K (where K = len(bins.bins)): each bin b
          receives rho_arr[b.index]. Lists and tuples are coerced via np.asarray,
          so any array-like of length K works.
        Raises ValueError if length is neither 1 nor len(bins.bins).

    NOTE: `rho_PA_within_game` is currently UNUSED in the correction. The PA
    dependence correction goes entirely through `tau_squared`, which is the
    inverted form of the same quantity (rho_PA_within_game ≈ implied rho from
    tau_squared via the logistic-normal model). The parameter is preserved in
    the API for future variance-inflation extensions (where rho_PA might drive
    a separate uncertainty knob), but for v1 it has no effect on the returned
    bins. Pass the diagnostic value for record-keeping in your call site.
    """
    from bts.simulate.quality_bins import QualityBins, QualityBin

    # Normalize rho_pair_cross_game to a 1D array once. Handles float, list,
    # tuple, and ndarray uniformly. Size-1 means "broadcast scalar to all bins."
    rho_arr = np.asarray(rho_pair_cross_game, dtype=float).ravel()
    if rho_arr.size == 1:
        pass  # scalar broadcast — handled in loop via rho_arr[0]
    elif rho_arr.size != len(bins.bins):
        raise ValueError(
            f"rho_pair_cross_game length {rho_arr.size} != number of bins "
            f"{len(bins.bins)}; pass scalar or per-bin vector of length {len(bins.bins)}"
        )

    quad_x, quad_w = np.polynomial.hermite_e.hermegauss(21)
    quad_w_sum = float(quad_w.sum())

    new_bins = []
    for b in bins.bins:
        # Mean correction for p_hit via logistic-normal integration.
        if tau_squared > 1e-12:
            # Invert the game-level hit probability to per-PA probability under independence.
            p_pa_indep = 1.0 - (1.0 - b.p_hit) ** (1.0 / n_pa_per_game)
            # Clip to avoid logit blowup.
            p_pa_indep = max(min(p_pa_indep, 1.0 - 1e-9), 1e-9)
            tau = float(np.sqrt(tau_squared))
            logit_p_pa = float(np.log(p_pa_indep / (1.0 - p_pa_indep)))
            num = 0.0
            for x, w in zip(quad_x, quad_w):
                u = tau * x
                p_pa_tilted = 1.0 / (1.0 + np.exp(-(logit_p_pa + u)))
                p_at_least_one = 1.0 - (1.0 - p_pa_tilted) ** n_pa_per_game
                num += w * p_at_least_one
            new_p_hit = float(num / quad_w_sum)
        else:
            new_p_hit = b.p_hit

        # Mean correction for p_both via Pearson copula reconstruction.
        # Replaces empirical p_both with synthetic from independence + correlation,
        # because empirical p_both already incorporates real-data correlation;
        # adding rho*sqrt(...) on top would double-count. Using p1*p2 as the
        # independence baseline + rho*sqrt(p1(1-p1)p2(1-p2)) as the linear-
        # copula correction matches the standard Pearson formulation.
        #
        # Use new_p_hit (the corrected marginal) for both p1 and p2, NOT
        # b.p_hit (the original). This keeps the returned bin internally
        # consistent — the bin's reported p_hit and the reported p_both share
        # the same marginal assumption. v1 limitation: rank-1 and rank-2 may
        # have different real-world marginals, but we only have one bin-level
        # marginal, so p1 = p2 = new_p_hit is the closest internally-consistent
        # approximation available.
        p1 = new_p_hit
        p2 = new_p_hit
        # Per-bin rho_pair: when scalar, broadcast; when array, index by bin position.
        # b.index is in [0, K-1] matching QualityBin's contract.
        rho_for_bin = float(rho_arr[0]) if rho_arr.size == 1 else float(rho_arr[b.index])
        new_p_both = p1 * p2 + rho_for_bin * np.sqrt(
            p1 * (1.0 - p1) * p2 * (1.0 - p2)
        )
        # Clip to Frechet-Hoeffding bounds.
        lower = max(0.0, p1 + p2 - 1.0)
        upper = min(p1, p2)
        new_p_both = float(min(max(new_p_both, lower), upper))

        new_bins.append(QualityBin(
            index=b.index,
            p_range=b.p_range,
            p_hit=new_p_hit,
            p_both=new_p_both,
            frequency=b.frequency,
        ))

    return QualityBins(bins=new_bins, boundaries=bins.boundaries)
