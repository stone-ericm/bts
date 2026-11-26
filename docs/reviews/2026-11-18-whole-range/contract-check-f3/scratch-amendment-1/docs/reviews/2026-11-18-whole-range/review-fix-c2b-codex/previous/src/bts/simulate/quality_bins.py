"""Empirical prediction quality bins from backtest profiles.

Bins daily profiles into equal-frequency quintiles by top-pick confidence.
Each bin stores the empirical P(hit) and P(both hit) for use in the
absorbing chain and MDP solvers.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class QualityBin:
    """One quality tier with empirical transition probabilities."""
    index: int
    p_range: tuple[float, float]  # (min, max) of top-1 p_game_hit
    p_hit: float                  # empirical P(rank-1 gets a hit)
    p_both: float                 # empirical P(rank-1 AND rank-2 both hit)
    frequency: float              # fraction of days in this bin


@dataclass
class QualityBins:
    """Collection of quality bins with classification helper."""
    bins: list[QualityBin]
    boundaries: list[float]  # quintile cutpoints (4 values for 5 bins)

    def classify(self, p_game_hit: float) -> int:
        """Return bin index (0-4) for a given confidence value."""
        for i, boundary in enumerate(self.boundaries):
            if p_game_hit < boundary:
                return i
        return len(self.boundaries)  # highest bin


def compute_bins(profiles_df: pd.DataFrame, n_bins: int = 5) -> QualityBins:
    """Compute quality bins from backtest profile DataFrame.

    Args:
        profiles_df: DataFrame with columns [date, rank, p_game_hit, actual_hit].
        n_bins: Number of equal-frequency bins (default 5 = quintiles).

    Returns:
        QualityBins with empirical hit rates per bin.
    """
    merged = _rank_pair_frame(profiles_df)

    # Compute quintile boundaries
    quantiles = [i / n_bins for i in range(1, n_bins)]
    boundaries = [float(merged["p_game_hit"].quantile(q)) for q in quantiles]

    # Assign bins
    merged["bin"] = np.digitize(merged["p_game_hit"], boundaries)

    bins = []
    for i in range(n_bins):
        group = merged[merged["bin"] == i]
        if len(group) == 0:
            continue
        bins.append(QualityBin(
            index=i,
            p_range=(float(group["p_game_hit"].min()), float(group["p_game_hit"].max())),
            p_hit=float(group["actual_hit"].mean()),
            p_both=float((group["actual_hit"] & group["top2_hit"]).mean()),
            frequency=len(group) / len(merged),
        ))

    return QualityBins(bins=bins, boundaries=boundaries)


def compute_bins_with_boundaries(profiles_df: pd.DataFrame, boundaries: list[float]) -> QualityBins:
    """Compute empirical quality bins using fixed confidence boundaries.

    This is for replaying a saved policy table against the same confidence
    cutpoints used when the policy was saved. Unlike compute_bins, empty bins
    are retained with zero frequency so the result remains shape-compatible
    with the saved action table.
    """
    merged = _rank_pair_frame(profiles_df)
    boundaries = [float(b) for b in boundaries]
    merged["bin"] = np.digitize(merged["p_game_hit"], boundaries)

    bins = []
    n_bins = len(boundaries) + 1
    for i in range(n_bins):
        group = merged[merged["bin"] == i]
        if len(group) == 0:
            lower = float("-inf") if i == 0 else boundaries[i - 1]
            upper = float("inf") if i == len(boundaries) else boundaries[i]
            bins.append(QualityBin(
                index=i,
                p_range=(lower, upper),
                p_hit=0.0,
                p_both=0.0,
                frequency=0.0,
            ))
            continue
        bins.append(QualityBin(
            index=i,
            p_range=(float(group["p_game_hit"].min()), float(group["p_game_hit"].max())),
            p_hit=float(group["actual_hit"].mean()),
            p_both=float((group["actual_hit"] & group["top2_hit"]).mean()),
            frequency=len(group) / len(merged),
        ))

    return QualityBins(bins=bins, boundaries=boundaries)


def _rank_pair_frame(profiles_df: pd.DataFrame) -> pd.DataFrame:
    r1 = profiles_df[profiles_df["rank"] == 1].copy()
    r2 = profiles_df[profiles_df["rank"] == 2].copy()

    return r1[["date", "p_game_hit", "actual_hit"]].merge(
        r2[["date", "actual_hit"]].rename(columns={"actual_hit": "top2_hit"}),
        on="date",
    )
