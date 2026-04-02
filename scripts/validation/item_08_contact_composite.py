"""
Item 8: Contact Quality Composite Investigation

Tests whether a single composite contact-quality feature captures interaction
effects that the individual blend members miss. shefBoiRDee on r/beatthestreak
uses xBA, xwOBA, and hard-hit rate as a weighted composite.

We proxy this with the 4 batter contact-quality Statcast features already computed:
  - batter_barrel_rate_30g
  - batter_hard_hit_rate_30g
  - batter_sweet_spot_rate_30g
  - batter_avg_ev_30g

Key question: are these 4 components sufficiently un-correlated that a composite
adds unique information the individual blend members don't have? Or are they so
correlated that the composite is redundant noise?

The 12-model blend already uses each component as a separate blend member. Adding
a composite as a 13th model only helps if it captures something the 4 individual
models don't already collectively vote on.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ── Config ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[2]
PA_PATTERN = str(ROOT / "data" / "processed" / "pa_{season}.parquet")
SEASONS = [2021, 2022, 2023, 2024, 2025]

COMPONENTS = [
    "batter_barrel_rate_30g",
    "batter_hard_hit_rate_30g",
    "batter_sweet_spot_rate_30g",
    "batter_avg_ev_30g",
]

# Correlation thresholds per task spec
HIGH_CORR_THRESHOLD = 0.7   # composite is redundant
LOW_CORR_THRESHOLD = 0.3    # composite might add diversity


# ── Feature computation ───────────────────────────────────────────────────────

def _is_barrel(ev, la):
    """MLB barrel classification from exit velocity + launch angle."""
    if pd.isna(ev) or pd.isna(la) or ev < 98:
        return False
    bonus = (min(ev, 116) - 98) * 2
    la_min = max(8, 26 - bonus)
    la_max = min(50, 30 + bonus)
    return la_min <= la <= la_max


def compute_contact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the 4 batter contact-quality Statcast features with temporal guard.
    All features use shift(1) at date level, so every PA on date D only uses
    data from dates strictly before D.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_pk"]).reset_index(drop=True)

    # PA-level batted ball flags
    df["is_barrel"] = df.apply(
        lambda r: _is_barrel(r["launch_speed"], r["launch_angle"]), axis=1
    )
    df["is_hard_hit"] = df["launch_speed"].notna() & (df["launch_speed"] >= 95)
    df["is_sweet_spot"] = (
        df["launch_angle"].notna()
        & (df["launch_angle"] >= 8)
        & (df["launch_angle"] <= 32)
    )
    df["has_batted_ball"] = df["launch_speed"].notna()

    # Date-level aggregates
    date_batted = (
        df.groupby(["batter_id", "date"])
        .agg(
            barrels=("is_barrel", "sum"),
            hard_hits=("is_hard_hit", "sum"),
            sweet_spots=("is_sweet_spot", "sum"),
            batted_balls=("has_batted_ball", "sum"),
            avg_ev=("launch_speed", lambda x: x.dropna().mean() if x.notna().any() else np.nan),
        )
        .reset_index()
        .sort_values(["batter_id", "date"])
    )

    date_batted["barrel_rate"] = np.where(
        date_batted["batted_balls"] > 0,
        date_batted["barrels"] / date_batted["batted_balls"],
        np.nan,
    )
    date_batted["hard_hit_rate"] = np.where(
        date_batted["batted_balls"] > 0,
        date_batted["hard_hits"] / date_batted["batted_balls"],
        np.nan,
    )
    date_batted["sweet_spot_rate"] = np.where(
        date_batted["batted_balls"] > 0,
        date_batted["sweet_spots"] / date_batted["batted_balls"],
        np.nan,
    )

    # Rolling 30-day features with shift(1) temporal guard
    date_batted["batter_barrel_rate_30g"] = date_batted.groupby("batter_id")["barrel_rate"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_batted["batter_hard_hit_rate_30g"] = date_batted.groupby("batter_id")["hard_hit_rate"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_batted["batter_sweet_spot_rate_30g"] = date_batted.groupby("batter_id")["sweet_spot_rate"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )
    date_batted["batter_avg_ev_30g"] = date_batted.groupby("batter_id")["avg_ev"].transform(
        lambda x: x.shift(1).rolling(30, min_periods=10).mean()
    )

    # Merge back to PA level
    df = df.merge(
        date_batted[["batter_id", "date"] + COMPONENTS].drop_duplicates(subset=["batter_id", "date"]),
        on=["batter_id", "date"],
        how="left",
    )

    return df


def compute_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize each component (z-score across all PA rows with valid data),
    then average to produce the composite. Z-scoring normalizes disparate scales
    (barrel rate ~5-15% vs avg EV ~85-92 mph).
    """
    df = df.copy()
    z_cols = []
    for col in COMPONENTS:
        z_col = f"{col}_z"
        col_mean = df[col].mean()
        col_std = df[col].std()
        if col_std > 0:
            df[z_col] = (df[col] - col_mean) / col_std
        else:
            df[z_col] = np.nan
        z_cols.append(z_col)

    # Composite = mean of available z-scores (NaN-safe)
    df["contact_composite"] = df[z_cols].mean(axis=1, skipna=True)

    # Set composite to NaN if fewer than 2 components are available
    valid_count = df[z_cols].notna().sum(axis=1)
    df.loc[valid_count < 2, "contact_composite"] = np.nan

    return df, z_cols


# ── Data loading ─────────────────────────────────────────────────────────────

def load_and_compute() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        print(f"  Loading season {season}...")
        raw = pd.read_parquet(
            PA_PATTERN.format(season=season),
            columns=["game_pk", "date", "batter_id", "is_hit",
                     "launch_speed", "launch_angle"],
        )
        raw["season"] = season
        computed = compute_contact_features(raw)
        frames.append(computed)
    return pd.concat(frames, ignore_index=True)


# ── Analysis ──────────────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def correlation_with_hit(series: pd.Series, target: pd.Series) -> tuple[float, float]:
    """Point-biserial correlation (equivalent to Pearson for binary target)."""
    mask = series.notna() & target.notna()
    if mask.sum() < 30:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(series[mask], target[mask])
    return r, p


def main() -> dict:
    print("Loading PA data and computing contact features (2021–2025)...")
    df = load_and_compute()
    print(f"  Total PAs: {len(df):,}")

    # ── Compute composite ────────────────────────────────────────────────────
    df, z_cols = compute_composite(df)

    # Restrict to rows where we have at least one component (and hit label)
    has_any_component = df[COMPONENTS].notna().any(axis=1)
    df_full = df[has_any_component & df["is_hit"].notna()].copy()
    print(f"  PAs with >=1 contact feature: {len(df_full):,}")

    # Restrict to rows with all 4 components for correlation matrix
    df_complete = df[df[COMPONENTS].notna().all(axis=1) & df["is_hit"].notna()].copy()
    print(f"  PAs with all 4 components:    {len(df_complete):,}")

    # ── Section 1: Component coverage ───────────────────────────────────────
    section("Component coverage (2021–2025)")
    for col in COMPONENTS:
        n_valid = df[col].notna().sum()
        pct = 100 * n_valid / len(df)
        print(f"  {col:<35s}  {n_valid:>8,}  ({pct:.1f}% of PAs)")
    n_composite = df["contact_composite"].notna().sum()
    pct_composite = 100 * n_composite / len(df)
    print(f"  {'contact_composite':<35s}  {n_composite:>8,}  ({pct_composite:.1f}% of PAs)")

    # ── Section 2: Component correlation matrix ──────────────────────────────
    section("Component correlation matrix (Pearson, all 4 present)")
    short_names = {
        "batter_barrel_rate_30g":    "barrel",
        "batter_hard_hit_rate_30g":  "hard_hit",
        "batter_sweet_spot_rate_30g": "sweet_spot",
        "batter_avg_ev_30g":         "avg_ev",
    }
    corr_matrix = df_complete[COMPONENTS].corr()

    # Print header
    labels = [short_names[c] for c in COMPONENTS]
    col_width = 11
    print(f"  {'':20s}", end="")
    for lbl in labels:
        print(f"  {lbl:>{col_width}}", end="")
    print()

    pairwise_corrs = []
    for i, col_i in enumerate(COMPONENTS):
        print(f"  {short_names[col_i]:<20s}", end="")
        for j, col_j in enumerate(COMPONENTS):
            r = corr_matrix.loc[col_i, col_j]
            print(f"  {r:>{col_width}.3f}", end="")
            if i < j:
                pairwise_corrs.append(r)
        print()

    avg_pairwise = float(np.mean(pairwise_corrs))
    max_pairwise = float(np.max(pairwise_corrs))
    min_pairwise = float(np.min(pairwise_corrs))
    print(f"\n  Pairwise correlations: min={min_pairwise:.3f}, avg={avg_pairwise:.3f}, max={max_pairwise:.3f}")

    # ── Section 3: Correlation with is_hit ───────────────────────────────────
    section("Correlation with is_hit (point-biserial r)")
    results = {}
    target = df["is_hit"].astype(float)

    for col in COMPONENTS:
        r, p = correlation_with_hit(df[col], target)
        n = df[col].notna().sum()
        flag = " *" if (p is not None and not np.isnan(p) and p < 0.05) else ""
        print(f"  {col:<35s}  r={r:>+.4f}  p={p:.3e}  (n={n:,}){flag}")
        results[col] = {"r": r, "p": p, "n": n}

    r_comp, p_comp = correlation_with_hit(df["contact_composite"], target)
    n_comp = df["contact_composite"].notna().sum()
    flag = " *" if (p_comp is not None and not np.isnan(p_comp) and p_comp < 0.05) else ""
    print(f"  {'contact_composite':<35s}  r={r_comp:>+.4f}  p={p_comp:.3e}  (n={n_comp:,}){flag}")
    results["contact_composite"] = {"r": r_comp, "p": p_comp, "n": n_comp}

    # ── Section 4: Unique information analysis ───────────────────────────────
    section("Unique information: does composite outperform best individual?")

    valid_rs = {k: v["r"] for k, v in results.items() if not np.isnan(v["r"])}
    best_individual_col = max(
        (c for c in COMPONENTS if c in valid_rs),
        key=lambda c: abs(valid_rs[c]),
    )
    best_r = valid_rs[best_individual_col]
    comp_r = valid_rs.get("contact_composite", float("nan"))
    improvement = comp_r - best_r if not np.isnan(comp_r) else float("nan")

    print(f"  Best individual component: {best_individual_col}")
    print(f"    r = {best_r:+.4f}")
    print(f"  Composite:")
    print(f"    r = {comp_r:+.4f}")
    print(f"  Improvement: {improvement:+.4f}")

    # Is improvement statistically meaningful? Compare using Fisher z-transform.
    # This gives the significance of the difference in two correlations.
    n_both = df[best_individual_col].notna() & df["contact_composite"].notna() & df["is_hit"].notna()
    n_both = n_both.sum()
    if not np.isnan(best_r) and not np.isnan(comp_r) and n_both > 3:
        # Fisher z-transform test for two independent correlations (conservative)
        z1 = np.arctanh(comp_r)
        z2 = np.arctanh(best_r)
        se = np.sqrt(2 / (n_both - 3))
        z_stat = (z1 - z2) / se if se > 0 else 0.0
        p_diff = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        print(f"  Fisher z-test (composite vs best): z={z_stat:.3f}, p={p_diff:.3e}")
    else:
        p_diff = float("nan")
        z_stat = float("nan")
        print("  Fisher z-test: insufficient data")

    # ── Section 5: Per-season stability ──────────────────────────────────────
    section("Per-season correlation stability")
    print(f"  {'Season':<8}", end="")
    for col in COMPONENTS:
        label = short_names[col]
        print(f"  {label:>11}", end="")
    print(f"  {'composite':>11}")

    season_rs = {}
    for season in SEASONS:
        s = df[df["season"] == season]
        target_s = s["is_hit"].astype(float)
        row = [season]
        rs = []
        for col in COMPONENTS:
            r, _ = correlation_with_hit(s[col], target_s)
            row.append(r)
            rs.append(r)
        r_c, _ = correlation_with_hit(s["contact_composite"], target_s)
        row.append(r_c)
        season_rs[season] = row
        print(f"  {season:<8}", end="")
        for r in row[1:]:
            print(f"  {r:>+11.4f}", end="")
        print()

    # ── Verdict ───────────────────────────────────────────────────────────────
    section("VERDICT")

    # Classify component inter-correlation
    if avg_pairwise >= HIGH_CORR_THRESHOLD:
        corr_level = "HIGH"
        corr_desc = f"avg pairwise r={avg_pairwise:.3f} ≥ {HIGH_CORR_THRESHOLD} — components are highly redundant"
    elif avg_pairwise >= LOW_CORR_THRESHOLD:
        corr_level = "MODERATE"
        corr_desc = f"avg pairwise r={avg_pairwise:.3f} is in moderate range ({LOW_CORR_THRESHOLD}–{HIGH_CORR_THRESHOLD})"
    else:
        corr_level = "LOW"
        corr_desc = f"avg pairwise r={avg_pairwise:.3f} < {LOW_CORR_THRESHOLD} — components are diverse"

    # Does composite beat best individual?
    composite_beats_best = not np.isnan(improvement) and improvement > 0.001
    improvement_significant = not np.isnan(p_diff) and p_diff < 0.05

    print(f"\n  Inter-component correlation: {corr_level}")
    print(f"    {corr_desc}")
    print(f"\n  Composite vs best individual:")
    print(f"    Improvement: {improvement:+.4f}")
    print(f"    Significant: {'YES (p<0.05)' if improvement_significant else 'NO'}")

    if corr_level == "HIGH" and not composite_beats_best:
        verdict = "REJECT"
        reasoning = (
            "Components are highly correlated (blend members already vote near-identically) "
            "AND composite shows no improvement over best individual feature. "
            "Adding composite as a 13th blend model would average noise, not add signal."
        )
    elif corr_level == "HIGH" and composite_beats_best and not improvement_significant:
        verdict = "REJECT"
        reasoning = (
            "Components are highly correlated. Composite shows marginal improvement "
            "but difference is not statistically significant. Not worth 2-3h backtest compute."
        )
    elif corr_level == "MODERATE" and composite_beats_best and improvement_significant:
        verdict = "WORTH_TESTING"
        reasoning = (
            "Components have moderate correlation and composite shows statistically significant "
            "improvement over best individual. A backtest may be warranted."
        )
    elif corr_level == "LOW":
        verdict = "WORTH_TESTING"
        reasoning = (
            "Low inter-component correlation means each component brings diverse signal. "
            "The composite may capture an interaction surface the individual models miss."
        )
    else:
        verdict = "REJECT"
        reasoning = (
            f"Inter-component correlation is {corr_level}, composite improvement "
            f"({improvement:+.4f}) is marginal and/or insignificant. No backtest needed."
        )

    print(f"\n  *** VERDICT: {verdict} ***")
    print(f"  {reasoning}")

    return {
        "avg_pairwise_corr": avg_pairwise,
        "max_pairwise_corr": max_pairwise,
        "min_pairwise_corr": min_pairwise,
        "corr_level": corr_level,
        "component_r_with_hit": {c: results[c]["r"] for c in COMPONENTS},
        "composite_r_with_hit": comp_r,
        "improvement_over_best": improvement,
        "improvement_significant": improvement_significant,
        "z_stat": z_stat,
        "p_diff": p_diff,
        "best_individual": best_individual_col,
        "verdict": verdict,
        "corr_matrix": corr_matrix.to_dict(),
    }


if __name__ == "__main__":
    import os
    os.chdir("/Users/stone/projects/bts")
    results = main()
