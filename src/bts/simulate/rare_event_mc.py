"""Cross-entropy importance-sampling rare-event Monte Carlo for P(57).

References:
- Rubinstein 1997, Optimization of computer simulation models with rare events.
- Rubinstein & Kroese 2017, Simulation and the Monte Carlo Method, 3rd ed.
- Au & Beck 2001, Subset simulation for rare events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _logit(p: float | np.ndarray) -> float | np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class LatentFactorSimulator:
    """Simulates per-day game outcomes with optional latent factor tilts.

    For each simulated season:
        Z_season ~ N(mu_d, 1)              [drawn ONCE per season]
        For each day t in season, for each game g on day t:
            G_{t,g} ~ N(mu_g, 1)            [drawn fresh per game]
            logit(p*_{t,g}) = logit(p_{t,g}) + lambda_d * Z_season + lambda_g * G_{t,g}
            Y_{t,g} ~ Bernoulli(p*_{t,g})

    When lambda_d = lambda_g = 0, collapses to independent Bernoulli draws — used
    as the unbiasedness oracle baseline for CE-IS validation.

    **Z is per-season, not per-day** (deviation from the original spec). Reasoning:
    with one game per day in the canonical setup, a per-day Z_t would be
    independent across days and produce no observable per-season variance
    effect — the variance-inflation test would be uninformative. The per-season
    structure is also the right CE-IS design: the auxiliary distribution tilts
    at the rare-event-relevant scale (a season's outcome). For *within-day
    correlation modeling* (the harness's Task 11 corrected-transitions path),
    the production code routes through `bts.validate.dependence` instead of
    this simulator, so the deviation is contained.

    Args:
        profiles: list of dicts, one per day. Required key: 'p_game'. Optional: 'date'.
        lambda_d: scale of the season-level latent factor (0 = no season-wide tilt).
        lambda_g: scale of the per-game latent factor (0 = no within-day game tilt).
        mu_d: mean of the season-level factor distribution (used for CE tilting in Task 7).
        mu_g: mean of the per-game factor distribution.
    """
    profiles: list[dict[str, Any]]
    lambda_d: float = 0.0
    lambda_g: float = 0.0
    mu_d: float = 0.0
    mu_g: float = 0.0

    def sample_season(self, rng: np.random.Generator) -> list[int]:
        """Return a list of binary outcomes, one per day (top-1 game).

        One latent factor draw per season is the correct CE-IS structure: Z is drawn
        once per simulated season so that all within-season outcomes share a common
        tilt.  This produces the intra-season correlation that importance sampling
        exploits to reach rare-event tails efficiently.  G_{t,g} is still drawn
        per-game (one per day in the single-game-per-day case).
        """
        Z_season = rng.normal(loc=self.mu_d, scale=1.0)
        outcomes = []
        for day in self.profiles:
            G_tg = rng.normal(loc=self.mu_g, scale=1.0)
            p_tilted = _sigmoid(
                _logit(day["p_game"]) + self.lambda_d * Z_season + self.lambda_g * G_tg
            )
            y = int(rng.random() < p_tilted)
            outcomes.append(y)
        return outcomes


# ---------------------------------------------------------------------------
# Cross-entropy importance-sampling estimator (Task 7)
# ---------------------------------------------------------------------------


@dataclass
class CEISResult:
    """Outputs from the CE-IS rare-event estimator.

    Attributes:
        point_estimate: IS-weighted estimate of P(max_streak >= threshold).
        ci_lower: lower bound of 95% bootstrap CI.
        ci_upper: upper bound of 95% bootstrap CI.
        ess: effective sample size (sum(w)^2 / sum(w^2)).
        max_weight_share: max weight / sum of weights (degeneracy diagnostic).
        log_weight_variance: variance of log-weights (spread diagnostic).
        n_final: number of paths used in the final IS estimate.
        theta_final: theta vector after CE fitting.
    """

    point_estimate: float
    ci_lower: float
    ci_upper: float
    ess: float
    max_weight_share: float
    log_weight_variance: float
    n_final: int
    theta_final: np.ndarray


def cross_entropy_tilt_step(
    paths: np.ndarray,
    weights: np.ndarray,
    elite_quantile: float = 0.95,
) -> np.ndarray:
    """Fit theta_0 on elite paths via simple logit-shift estimation.

    v1 fits only theta_0 (constant logit shift across days). The full
    [theta_0, theta_1, theta_2, theta_3] tilt (action-type, streak, days-remaining)
    is deferred to v1.5; for the unbiasedness test at theta=0 and basic
    rare-event variance reduction, theta_0 alone is sufficient.

    Args:
        paths: shape (M, n_days) binary outcome array sampled under current theta.
        weights: shape (M,) IS weights (dP/dQ per path).
        elite_quantile: quantile of per-path hit-count used to define elite set.

    Returns:
        np.array([theta_0, 0.0, 0.0, 0.0])
    """
    scores = paths.sum(axis=1)
    threshold = np.quantile(scores, elite_quantile)
    elite_mask = scores >= threshold
    if elite_mask.sum() < 5:
        return np.zeros(4)
    elite_paths = paths[elite_mask]
    elite_hit_rate = float(elite_paths.mean())
    overall_rate = float(paths.mean())
    theta_0 = _logit(elite_hit_rate) - _logit(overall_rate)
    return np.array([theta_0, 0.0, 0.0, 0.0])


def estimate_p57_with_ceis(
    profiles: list[dict[str, Any]],
    strategy: Any = None,
    *,
    n_rounds: int = 8,
    n_per_round: int = 5000,
    n_final: int = 20000,
    theta: np.ndarray | None = None,
    seed: int = 42,
    streak_threshold: int = 57,
) -> CEISResult:
    """Cross-entropy importance-sampling estimator of P(max_streak >= threshold).

    Uses a direct deterministic-theta sampler (NOT LatentFactorSimulator's
    stochastic Z): under the auxiliary distribution q, each day's outcome is
    independent Bernoulli with sigmoid(logit(p_game) + theta_0). Likelihood
    ratio per path is the product of per-day Bernoulli ratios.

    Why bypass LatentFactorSimulator: setting lambda_d=0 in LatentFactorSimulator
    collapses the auxiliary to the target distribution regardless of mu_d, making
    IS weights all 1 and the estimator degrade to naive MC. The direct sampler
    correctly tilts each day's Bernoulli via theta_0 in log-odds space, giving
    a proper change of measure.

    v1 simplifications (documented):
    - Fits only theta_0 (constant logit shift). Full per-day/per-action tilt deferred.
    - Event indicator counts max consecutive hits ignoring strategy decisions —
      appropriate for the "always-play, no doubles" baseline used in the
      unbiasedness oracle test. The harness driver (Task 12) wraps this in a
      strategy-aware replayer when needed.
    - `strategy` arg is currently unused; reserved for v1.5 strategy-aware tilt.

    Args:
        profiles: per-day list with 'p_game' key (target probability).
        strategy: currently unused; reserved for future strategy-aware tilt.
        n_rounds: number of CE tilt-fitting rounds. Set to 0 to skip CE
            fitting and use theta as-is (theta=0 collapses to naive MC).
        n_per_round: simulations per CE round.
        n_final: simulations for the final IS estimate.
        theta: optional initial theta. If None, starts at zeros(4).
        seed: rng seed.
        streak_threshold: streak length to count as the rare event (default 57).

    Returns:
        CEISResult with point estimate, bootstrap CI, and IS diagnostics.
    """
    rng = np.random.default_rng(seed)
    n_days = len(profiles)
    p_target = np.array([day["p_game"] for day in profiles])
    log_p = np.log(np.clip(p_target, 1e-12, 1 - 1e-12))
    log_1mp = np.log(np.clip(1.0 - p_target, 1e-12, 1 - 1e-12))
    theta = np.zeros(4) if theta is None else theta.copy()

    def _sample_paths(theta_vec: np.ndarray, n: int) -> np.ndarray:
        """Sample n season paths under q_theta. Returns shape (n, n_days)."""
        # q_theta: each day independent Bernoulli with sigmoid(logit(p) + theta_0).
        q = _sigmoid(_logit(p_target) + theta_vec[0])
        u = rng.random(size=(n, n_days))
        return (u < q).astype(np.int8)

    def _is_weights(paths: np.ndarray, theta_vec: np.ndarray) -> np.ndarray:
        """Per-path likelihood ratio dP/dQ for theta_vec[0] logit shift.

        Vectorized: paths shape (n, n_days). Returns shape (n,).
        log_w_i = sum_t [y_t*(log_p-log_q) + (1-y_t)*(log_1mp-log_1mq)]
        """
        q = _sigmoid(_logit(p_target) + theta_vec[0])
        log_q = np.log(np.clip(q, 1e-12, 1 - 1e-12))
        log_1mq = np.log(np.clip(1.0 - q, 1e-12, 1 - 1e-12))
        log_w = (
            paths * (log_p - log_q) + (1 - paths) * (log_1mp - log_1mq)
        ).sum(axis=1)
        return np.exp(log_w)

    # CE fitting rounds (skipped when n_rounds=0).
    for _r in range(n_rounds):
        paths = _sample_paths(theta, n_per_round)
        weights = _is_weights(paths, theta)
        new_theta = cross_entropy_tilt_step(paths, weights)
        theta = 0.5 * theta + 0.5 * new_theta
        if abs(new_theta[0]) < 0.05:
            break

    # Final IS estimation.
    final_paths = _sample_paths(theta, n_final)
    weights = _is_weights(final_paths, theta)
    event_indicators = np.array([
        _event_reached_threshold(path, threshold=streak_threshold)
        for path in final_paths
    ])
    estimates = event_indicators * weights
    point = float(estimates.mean())

    # Bootstrap CI on the estimates array.
    bs_idx = rng.choice(n_final, size=(2000, n_final), replace=True)
    bs_means = estimates[bs_idx].mean(axis=1)
    ci_lo = float(np.quantile(bs_means, 0.025))
    ci_hi = float(np.quantile(bs_means, 0.975))

    w_sum = float(weights.sum())
    w_sum_sq = float((weights ** 2).sum())

    return CEISResult(
        point_estimate=point,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ess=float(w_sum ** 2 / max(w_sum_sq, 1e-12)),
        max_weight_share=float(weights.max() / max(w_sum, 1e-12)),
        log_weight_variance=float(np.var(np.log(np.maximum(weights, 1e-12)))),
        n_final=n_final,
        theta_final=theta,
    )


def _event_reached_threshold(path: np.ndarray, *, threshold: int) -> int:
    """1 if max consecutive hits in the binary path >= threshold, else 0.

    v1 simplification: counts raw consecutive hits ignoring skip/double decisions.
    Appropriate for the 'always-play, no doubles' baseline strategy used in the
    unbiasedness oracle test. Task 12 harness driver wraps with strategy-aware
    replayer when needed.
    """
    streak = 0
    max_streak = 0
    for hit in path:
        if hit:
            streak += 1
            if streak > max_streak:
                max_streak = streak
        else:
            streak = 0
    return int(max_streak >= threshold)
