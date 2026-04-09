"""Phase 1 blend experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bts.experiment.base import ExperimentDef
from bts.experiment.registry import register


def _per_model_columns(profiles_df: pd.DataFrame) -> list[str]:
    """Return list of m_<name> per-model columns in profiles."""
    return [c for c in profiles_df.columns if c.startswith("m_")]


class FWLSExperiment(ExperimentDef):
    """Feature-Weighted Linear Stacking: ridge meta-learner over per-model preds."""

    def __init__(self):
        super().__init__(
            name="fwls",
            phase=1,
            category="blend",
            description="Ridge-penalized linear meta-learner over per-model predictions",
        )

    def requires_per_model_capture(self) -> bool:
        return True

    def modify_strategy(self, profiles_df, quality_bins):
        """Train ridge meta-learner with temporal cross-fitting (no leakage).

        For each day d, train ridge on rank-1 picks from days [0, d-1] only,
        then apply to day d's profiles. After warmup, recalibrate weekly.
        """
        from sklearn.linear_model import Ridge

        df = profiles_df.copy().reset_index(drop=True)
        df["date"] = pd.to_datetime(df["date"])
        m_cols = _per_model_columns(df)
        if not m_cols or len(df) < 100:
            return df, quality_bins

        rank1 = df[df["rank"] == 1].sort_values("date").copy()
        sorted_dates = sorted(df["date"].unique())
        warmup = 30
        if len(sorted_dates) < warmup + 10:
            return df, quality_bins

        recalibrate_every = 7
        meta = None
        new_p = df["p_game_hit"].values.copy()
        m_matrix = df[m_cols].fillna(0.5).values

        for i, date in enumerate(sorted_dates):
            if i < warmup:
                continue
            if meta is None or i % recalibrate_every == 0:
                prior_rank1 = rank1[rank1["date"] < date]
                if len(prior_rank1) < 30:
                    continue
                X_train = prior_rank1[m_cols].fillna(0.5).values
                y_train = prior_rank1["actual_hit"].values
                meta = Ridge(alpha=1.0, positive=True)
                meta.fit(X_train, y_train)

            day_mask = (df["date"] == date).values
            if day_mask.any() and meta is not None:
                preds = meta.predict(m_matrix[day_mask])
                new_p[day_mask] = np.clip(preds, 0.0, 0.9999)

        df["p_game_hit"] = new_p

        # Re-rank within each day
        df = df.sort_values(["date", "p_game_hit"], ascending=[True, False]).reset_index(drop=True)
        df["rank"] = df.groupby("date").cumcount() + 1

        try:
            from bts.simulate.quality_bins import compute_bins
            new_bins = compute_bins(df)
        except Exception:
            new_bins = quality_bins

        return df, new_bins


class FixedShareHedgeExperiment(ExperimentDef):
    """Fixed-Share Hedge: online-adaptive blend weights with seasonal tracking."""

    def __init__(self):
        super().__init__(
            name="fixed_share_hedge",
            phase=1,
            category="blend",
            description="Hedge algorithm with Fixed-Share mixing (alpha=0.05)",
        )

    def requires_per_model_capture(self) -> bool:
        return True

    def modify_strategy(self, profiles_df, quality_bins):
        df = profiles_df.copy().reset_index(drop=True)
        m_cols = _per_model_columns(df)
        if not m_cols:
            return df, quality_bins

        # Online Hedge with Fixed-Share over per-day rank-1 outcomes
        n_models = len(m_cols)
        eta = 0.5  # learning rate
        alpha_share = 0.05  # mixing rate

        # Build per-day weights by iterating days in order
        rank1 = df[df["rank"] == 1].sort_values("date").copy()
        weights = np.ones(n_models) / n_models

        per_day_weights: dict = {}
        for _, row in rank1.iterrows():
            per_day_weights[row["date"]] = weights.copy()
            preds = np.array([
                row[c] if pd.notna(row[c]) else 0.5 for c in m_cols
            ], dtype=float)
            y = float(row["actual_hit"])
            losses = (preds - y) ** 2

            new_w = weights * np.exp(-eta * losses)
            new_w = new_w / new_w.sum()
            uniform = np.ones(n_models) / n_models
            weights = (1 - alpha_share) * new_w + alpha_share * uniform

        # Vectorized recompute of p_game_hit using time-varying weights
        # Build a (n_days, n_models) weight matrix indexed by row's date
        date_to_weight_idx = {d: i for i, d in enumerate(per_day_weights.keys())}
        weight_matrix = np.array(list(per_day_weights.values()))  # (D, M)
        uniform = np.ones(n_models) / n_models

        m_matrix = df[m_cols].fillna(np.nan).values  # (N, M)
        new_p = np.full(len(df), np.nan)

        for i, d in enumerate(df["date"].values):
            w = weight_matrix[date_to_weight_idx[d]] if d in date_to_weight_idx else uniform
            row_preds = m_matrix[i]
            valid = ~np.isnan(row_preds)
            if valid.any():
                w_valid = w[valid] / w[valid].sum()
                new_p[i] = float(np.dot(row_preds[valid], w_valid))

        # Fall back to original p_game_hit where weighted average is NaN
        original_p = df["p_game_hit"].values
        new_p = np.where(np.isnan(new_p), original_p, new_p)
        df["p_game_hit"] = new_p

        # Re-rank within each day
        df = df.sort_values(["date", "p_game_hit"], ascending=[True, False]).reset_index(drop=True)
        df["rank"] = df.groupby("date").cumcount() + 1

        try:
            from bts.simulate.quality_bins import compute_bins
            new_bins = compute_bins(df)
        except Exception:
            new_bins = quality_bins

        return df, new_bins


class CopulaDoublesExperiment(ExperimentDef):
    """Gaussian copula for double-down joint probability."""

    def __init__(self):
        super().__init__(
            name="copula_doubles",
            phase=1,
            category="blend",
            description="Gaussian copula P(both hit) instead of P(A)*P(B)",
        )

    def modify_strategy(self, profiles_df, quality_bins):
        # Estimate latent rho from historical co-occurrence in profiles
        # Then modify quality_bins to use copula-adjusted P(both)
        from scipy.stats import multivariate_normal, norm

        if quality_bins is None:
            return profiles_df, quality_bins

        df = profiles_df.copy()

        # Estimate cross-pick correlation: pairs of (rank1, rank2) on each day
        rank1 = df[df["rank"] == 1].set_index("date")["actual_hit"]
        rank2 = df[df["rank"] == 2].set_index("date")["actual_hit"]
        common_dates = rank1.index.intersection(rank2.index)
        if len(common_dates) < 30:
            return df, quality_bins

        r1 = rank1.loc[common_dates].values
        r2 = rank2.loc[common_dates].values
        rho = float(np.corrcoef(r1, r2)[0, 1])
        if not np.isfinite(rho):
            rho = 0.0
        # Cap to plausible range
        rho = float(np.clip(rho, -0.3, 0.3))

        # Modify quality_bins p_both via Gaussian copula
        # Original: p_both = p_hit * p_hit (independence)
        # New: p_both = Phi_2(Phi^-1(p_hit), Phi^-1(p_hit); rho)
        try:
            for i in range(len(quality_bins.p_hit)):
                p = quality_bins.p_hit[i]
                if 0 < p < 1:
                    z = norm.ppf(p)
                    cov = [[1.0, rho], [rho, 1.0]]
                    p_both = multivariate_normal(mean=[0, 0], cov=cov).cdf([z, z])
                    quality_bins.p_both[i] = float(p_both)
        except (AttributeError, TypeError, ValueError):
            pass

        return df, quality_bins


register(FWLSExperiment())
register(FixedShareHedgeExperiment())
register(CopulaDoublesExperiment())
