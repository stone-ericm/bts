"""Spot-check audit M3: serving-lookup staleness vs true as-of feature values.

The serving path (`_build_feature_lookups`) reads each entity's feature value
at its most recent PLAYED date. Because every rolling feature is shift(1),
that stored value excludes the most recent game — serving is one game stale
for every entity, every day (audit 2026-06-09, [P1] "inference lookups stale
by one played day").

This script sizes the gap empirically WITHOUT duplicating any transform:
append one synthetic next-day row per active entity and rerun the real
`compute_all_features`. The shift(1) machinery then emits, at the synthetic
date, exactly the as-of value serving *should* use ("data through the last
played game"). Compare against the current lookups.

Run: uv run python scripts/spotcheck_m3_staleness.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from bts.features.compute import compute_all_features
from bts.model.predict import _build_feature_lookups

SEASONS = [2023, 2024, 2025, 2026]
ACTIVE_WINDOW_DAYS = 14
SYNTH_GAME_PK_BASE = 999_999_000
PRIOR_RATE = 0.2195


def _compare_maps(name: str, stale: dict, fresh: dict, keys, col_std: float):
    rows = []
    for k in keys:
        s, f = stale.get(k), fresh.get(k)
        if s is None or f is None or pd.isna(s) or pd.isna(f):
            continue
        rows.append((k, s, f, f - s))
    if not rows:
        print(f"  {name:38s} (no comparable keys)")
        return None
    deltas = np.array([abs(r[3]) for r in rows])
    changed = (deltas > 1e-12).mean()
    rel = deltas.mean() / col_std if col_std and col_std > 0 else float("nan")
    print(
        f"  {name:38s} n={len(rows):4d}  changed={changed:6.1%}  "
        f"mean|d|={deltas.mean():.5f}  p90|d|={np.percentile(deltas, 90):.5f}  "
        f"max|d|={deltas.max():.5f}  mean|d|/std={rel:.3f}"
    )
    return rows


def main():
    proc = Path("data/processed")
    df = pd.concat(
        [pd.read_parquet(proc / f"pa_{s}.parquet") for s in SEASONS],
        ignore_index=True,
    )
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    synth_date = max_date + pd.Timedelta(days=1)
    cutoff = max_date - pd.Timedelta(days=ACTIVE_WINDOW_DAYS)

    active_b = df.loc[df["date"] >= cutoff, "batter_id"].unique()
    active_p = df.loc[df["date"] >= cutoff, "pitcher_id"].unique()

    df_sorted = df.sort_values(["date", "game_pk"], kind="mergesort")
    last_b = df_sorted[df_sorted["batter_id"].isin(active_b)].groupby("batter_id").tail(1)
    last_p = df_sorted[df_sorted["pitcher_id"].isin(active_p)].groupby("pitcher_id").tail(1)
    synth = pd.concat([last_b, last_p], ignore_index=True)
    synth["date"] = synth_date
    synth["game_pk"] = SYNTH_GAME_PK_BASE + np.arange(len(synth))
    synth["is_hit"] = np.nan  # no outcome; shift(1) keeps the synthetic date out of every window

    print(f"history: {len(df)} PA rows {df['date'].min().date()} -> {max_date.date()}")
    print(f"synthetic date {synth_date.date()}: {len(last_b)} active batters, {len(last_p)} active pitchers")

    print("computing features (original frame)...")
    feat1 = compute_all_features(df)
    print("computing features (augmented frame)...")
    feat2 = compute_all_features(pd.concat([df, synth], ignore_index=True))

    lk_stale = _build_feature_lookups(feat1)
    lk_fresh = _build_feature_lookups(feat2)

    # Sanity: augmentation must not change what serving reads from HISTORICAL rows.
    lk_check = _build_feature_lookups(feat2[feat2["date"] < synth_date])
    mismatches = 0
    for col, m1 in lk_stale["batter"].items():
        m2 = lk_check["batter"].get(col, {})
        for b in active_b:
            v1, v2 = m1.get(b), m2.get(b)
            if v1 is not None and v2 is not None and not (pd.isna(v1) and pd.isna(v2)):
                if abs(v1 - v2) > 1e-12:
                    mismatches += 1
    print(f"sanity (augmented run leaves historical lookups identical): {mismatches} mismatches")

    std_frame = feat1[feat1["date"] >= pd.Timestamp(f"{max_date.year - 1}-01-01")]

    print("\n=== batter rolling features (stale .last() vs as-of) ===")
    for col in sorted(lk_stale["batter"]):
        col_std = float(std_frame[col].std()) if col in std_frame.columns else float("nan")
        _compare_maps(col, lk_stale["batter"][col], lk_fresh["batter"][col], active_b, col_std)

    print("\n=== pitcher features ===")
    for name, key in [("pitcher_hr_30g", "pitcher_hr"), ("pitcher_entropy_30g", "pitcher_ent"),
                      ("pitcher_catcher_framing", "pitcher_framing")]:
        if key in lk_stale and key in lk_fresh:
            col_std = float(std_frame[name].std()) if name in std_frame.columns else float("nan")
            _compare_maps(name, lk_stale[key], lk_fresh[key], active_p, col_std)
    for col, m in lk_stale.get("pitcher_statcast", {}).items():
        col_std = float(std_frame[col].std()) if col in std_frame.columns else float("nan")
        _compare_maps(col, m, lk_fresh["pitcher_statcast"][col], active_p, col_std)

    print("\n=== batter_pitcher_shrunk_hr (bpm) — the audit's worst case ===")
    pair_dates = df.groupby(["batter_id", "pitcher_id"])["date"].nunique()
    pairs = list(last_b[["batter_id", "pitcher_id"]].itertuples(index=False, name=None))
    col_std = float(std_frame["batter_pitcher_shrunk_hr"].std())
    rows = _compare_maps("bpm (active batter x last pitcher)", lk_stale["batter_pitcher_hr"],
                         lk_fresh["batter_pitcher_hr"], pairs, col_std)
    if rows:
        single = [(k, s, f) for k, s, f, _ in rows if pair_dates.get(k, 0) == 1]
        multi = [(k, s, f) for k, s, f, _ in rows if pair_dates.get(k, 0) >= 2]
        collapsed = [r for r in single if abs(r[1] - PRIOR_RATE) < 1e-9]
        print(f"  single-meeting pairs: {len(single)}; of those, stale == league prior exactly: {len(collapsed)} "
              f"({len(collapsed) / len(single):.0%} collapse rate)" if single else "  no single-meeting pairs")
        if collapsed:
            fresh_vals = np.array([f for _, _, f in collapsed])
            print(f"  collapsed pairs' TRUE as-of values: mean={fresh_vals.mean():.4f} "
                  f"min={fresh_vals.min():.4f} max={fresh_vals.max():.4f} (vs served {PRIOR_RATE})")
        if multi:
            d = np.array([abs(f - s) for _, s, f in multi])
            print(f"  multi-meeting pairs (n={len(multi)}): mean|d|={d.mean():.5f} max|d|={d.max():.5f}")

    # How many games stale is each batter's served window, really?
    n_dates = df[df["batter_id"].isin(active_b)].groupby("batter_id")["date"].nunique()
    print(f"\nactive batters' career played dates: median={n_dates.median():.0f} "
          f"(staleness is always exactly 1 played date per entity, by construction)")


if __name__ == "__main__":
    main()
