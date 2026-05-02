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
