"""swing_daily aggregates + leak-free rolling features (campaign Stage 0).

Per-pitch bronze (data/processed/swing_{season}.parquet) is aggregated to
(entity, date) KEEPING denominator rows — "no whiffs", "no swings", and "no
tracking" stay distinguishable — then rolled with the same date-level
shift(1) contract as compute.py. Features are left-joined onto PA rows;
the bronze frame is never merged into the PA frame.

Also home of the pre-registered control builders (missingness placebo,
leaky sentinel) so the controls evolve in lockstep with the features.

Spec: docs/superpowers/specs/2026-06-12-statcast-swing-campaign-design.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bunt attempts excluded throughout: they aren't skill-relevant swings and
# Savant's swing leaderboards exclude them too (QA'd 2026-06-12: leaderboard
# n_swings = bat-tracked swings excl. bunts; matching that definition closes
# a +3.5% count bias to 0.25% median APE).
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | {"foul", "foul_tip", "hit_into_play"}


def daily_swing_aggregates(bronze: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Aggregate bronze per-pitch rows to (entity, date) with denominators.

    entity: "batter" or "pitcher" (bronze column name).
    Output columns: n_swings, n_swings_tracked, n_whiffs, n_whiffs_tracked,
    miss_sum, miss_sumsq, swing_len_sum, attack_angle_sum,
    n_whiff_high, n_whiff_low.
    """
    df = bronze.copy()
    df["date"] = pd.to_datetime(df["game_date"])
    df["_is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["_is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["_miss"] = pd.to_numeric(df["miss_distance"], errors="coerce")
    df["_swing_len"] = pd.to_numeric(df.get("swing_length"), errors="coerce")
    df["_attack"] = pd.to_numeric(df.get("attack_angle"), errors="coerce")
    sz_mid = (pd.to_numeric(df["sz_top"], errors="coerce")
              + pd.to_numeric(df["sz_bot"], errors="coerce")) / 2
    plate_z = pd.to_numeric(df["plate_z"], errors="coerce")
    df["_whiff_high"] = df["_is_whiff"] & df["_miss"].notna() & (plate_z > sz_mid)
    df["_whiff_low"] = df["_is_whiff"] & df["_miss"].notna() & (plate_z <= sz_mid)

    swings = df[df["_is_swing"]]
    agg = swings.groupby([entity, "date"]).agg(
        n_swings=("_is_swing", "sum"),
        n_swings_tracked=("_swing_len", "count"),
        n_whiffs=("_is_whiff", "sum"),
        n_whiffs_tracked=("_miss", "count"),
        miss_sum=("_miss", "sum"),
        miss_sumsq=("_miss", lambda s: float(np.nansum(np.square(s)))),
        swing_len_sum=("_swing_len", "sum"),
        attack_angle_sum=("_attack", "sum"),
        n_whiff_high=("_whiff_high", "sum"),
        n_whiff_low=("_whiff_low", "sum"),
    ).reset_index()
    return agg.sort_values([entity, "date"], kind="mergesort").reset_index(drop=True)


def rolling_swing_features(
    daily: pd.DataFrame,
    entity: str,
    windows: list[int] | None = None,
    min_whiffs: int = 8,
) -> pd.DataFrame:
    """shift(1).rolling(w) ratio features from daily sums (leak-free by construction).

    Ratio-of-rolling-sums (not mean-of-daily-means) so sparse days don't get
    equal weight. Values are NaN until min_whiffs tracked whiffs accumulate
    in the window (denominator reliability; spec 'whiff-denominator' control).
    Column naming: {entity}_{stat}_{w}g.
    """
    windows = windows or [7, 15, 30, 60]
    out = daily[[entity, "date"]].copy()
    g = daily.groupby(entity, sort=False)

    def _roll_sum(col: str, w: int) -> pd.Series:
        return g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).sum())

    for w in windows:
        whiffs_tracked = _roll_sum("n_whiffs_tracked", w)
        miss_sum = _roll_sum("miss_sum", w)
        miss_sumsq = _roll_sum("miss_sumsq", w)
        swings = _roll_sum("n_swings", w)
        whiffs = _roll_sum("n_whiffs", w)
        swings_tracked = _roll_sum("n_swings_tracked", w)
        swing_len = _roll_sum("swing_len_sum", w)
        attack = _roll_sum("attack_angle_sum", w)
        hi = _roll_sum("n_whiff_high", w)
        lo = _roll_sum("n_whiff_low", w)

        enough = whiffs_tracked >= min_whiffs
        mean_miss = (miss_sum / whiffs_tracked).where(enough)
        var_miss = (miss_sumsq / whiffs_tracked - mean_miss**2).where(enough)

        out[f"{entity}_miss_dist_{w}g"] = mean_miss
        out[f"{entity}_miss_std_{w}g"] = np.sqrt(var_miss.clip(lower=0))
        out[f"{entity}_whiff_rate_{w}g"] = (whiffs / swings).where(swings >= min_whiffs)
        out[f"{entity}_whiff_high_share_{w}g"] = (hi / (hi + lo)).where((hi + lo) >= min_whiffs)
        out[f"{entity}_swing_len_{w}g"] = (swing_len / swings_tracked).where(swings_tracked >= min_whiffs)
        out[f"{entity}_attack_angle_{w}g"] = (attack / swings_tracked).where(swings_tracked >= min_whiffs)
    return out


def attach_swing_features(
    pa: pd.DataFrame,
    batter_feats: pd.DataFrame | None,
    pitcher_feats: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left-join rolling swing features onto PA rows by (entity id, date)."""
    out = pa.copy()
    if batter_feats is not None:
        out = out.merge(
            batter_feats.rename(columns={"batter": "batter_id"}),
            on=["batter_id", "date"], how="left",
        )
    if pitcher_feats is not None:
        out = out.merge(
            pitcher_feats.rename(columns={"pitcher": "pitcher_id"}),
            on=["pitcher_id", "date"], how="left",
        )
    return out


def build_missingness_placebo(pa: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Boolean availability flags ONLY (no values, no counts — counts carry
    real playing-time signal). The placebo model must show ~nothing, else the
    eval is confounded by the post-2023 era marker."""
    out = pd.DataFrame(index=pa.index)
    for col in feature_cols:
        out[f"has_{col}"] = pa[col].notna()
    return out


def build_leaky_sentinel(pa: pd.DataFrame, daily: pd.DataFrame, entity: str) -> pd.DataFrame:
    """SAME-DAY (unshifted) mean miss distance — deliberately leaky.

    The known-strong sentinel the harness MUST flag as inflated; proves
    leakage detectability. Never a candidate feature.
    """
    d = daily.copy()
    d["LEAKY_same_day_miss"] = d["miss_sum"] / d["n_whiffs_tracked"]
    key = "batter_id" if entity == "batter" else "pitcher_id"
    out = pa.merge(
        d[[entity, "date", "LEAKY_same_day_miss"]].rename(columns={entity: key}),
        on=[key, "date"], how="left",
    )
    return out
