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
        from sklearn.linear_model import Ridge

        df = profiles_df.copy()
        m_cols = _per_model_columns(df)
        if not m_cols or len(df) < 100:
            return df, quality_bins

        # Train ridge meta-learner on per-model predictions → actual outcome
        # Use rank-1 picks across all days as the training set
        train_df = df[m_cols + ["actual_hit"]].dropna()
        if len(train_df) < 50:
            return df, quality_bins

        X = train_df[m_cols].values
        y = train_df["actual_hit"].values
        meta = Ridge(alpha=1.0, positive=True)
        meta.fit(X, y)

        # Apply meta-learner to all profiles, replacing p_game_hit
        full_X = df[m_cols].fillna(df[m_cols].mean())
        new_p = meta.predict(full_X.values)
        # Clip to [0, 1] since ridge can extrapolate
        new_p = np.clip(new_p, 0.0, 0.9999)

        df["p_game_hit"] = new_p

        # Re-rank within each day after recomputing scores
        df = df.sort_values(["date", "p_game_hit"], ascending=[True, False])
        df["rank"] = df.groupby("date").cumcount() + 1

        # Recompute quality bins
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
        df = profiles_df.copy()
        m_cols = _per_model_columns(df)
        if not m_cols:
            return df, quality_bins

        # Online Hedge with Fixed-Share over per-day rank-1 outcomes
        n_models = len(m_cols)
        eta = 0.5  # learning rate
        alpha_share = 0.05  # mixing rate

        # Iterate days in order, tracking per-model weights
        rank1 = df[df["rank"] == 1].sort_values("date").copy()
        weights = np.ones(n_models) / n_models

        per_day_weights: dict = {}
        for _, row in rank1.iterrows():
            per_day_weights[row["date"]] = weights.copy()
            # Compute per-model loss on this day's actual outcome
            preds = np.array([row[c] if pd.notna(row[c]) else 0.5 for c in m_cols])
            y = float(row["actual_hit"])
            losses = (preds - y) ** 2  # squared loss

            # Hedge update
            new_w = weights * np.exp(-eta * losses)
            new_w = new_w / new_w.sum()

            # Fixed-Share mixing
            uniform = np.ones(n_models) / n_models
            weights = (1 - alpha_share) * new_w + alpha_share * uniform

        # Recompute p_game_hit using the time-varying weights
        new_p_values = []
        for _, row in df.iterrows():
            d = row["date"]
            if d not in per_day_weights:
                # Use initial uniform weights
                w = np.ones(n_models) / n_models
            else:
                w = per_day_weights[d]
            preds = np.array([row[c] if pd.notna(row[c]) else np.nan for c in m_cols])
            valid = ~np.isnan(preds)
            if valid.any():
                w_valid = w[valid] / w[valid].sum()
                new_p_values.append(float(np.dot(preds[valid], w_valid)))
            else:
                new_p_values.append(row["p_game_hit"])

        df["p_game_hit"] = new_p_values

        # Re-rank within each day
        df = df.sort_values(["date", "p_game_hit"], ascending=[True, False])
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
