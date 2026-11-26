#!/usr/bin/env python
"""Reproduce the discrimination results in docs/audit/2026-06-29-skip-threshold-and-discrimination.md.

Skip-vs-play hit rates (§4), the corrected AUC table + reverse-causality decomposition, and the
within-day PA-tilt policy test that dissolved under per-season + McNemar (§7).

Run from the repo root, where the data lives (in practice: bts-hetzner):
    .venv/bin/python scripts/audit/discrimination_tests.py \
        --estpa-dir data/validation/estpa_profiles_2026-06-29

Needs: <estpa-dir>/backtest_*.parquet (with est_pas), data/simulation/backtest_*.parquet (actual_pa),
and data/processed/pa_*.parquet (for lineup_position).

NB the AUC must use original-order ranks indexed by the labels (scipy.rankdata). An earlier bug
(sort labels by score, index with original-order ranks) returned ~0.50 for everything; see doc §7/§10.
"""
import argparse, glob
import numpy as np, pandas as pd
from scipy.stats import rankdata, binomtest


def auc(score, y):
    """Mann-Whitney AUC. Sanity: informative -> ~1.0, pure noise -> ~0.5."""
    score = np.asarray(score, float); y = np.asarray(y, float); m = ~np.isnan(score)
    score, y = score[m], y[m]; n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estpa-dir", default="data/validation/estpa_profiles_2026-06-29")
    args = ap.parse_args()

    # sanity check on the AUC implementation (guards against the original bug)
    rng = np.random.default_rng(0); yy = rng.integers(0, 2, 4000).astype(float)
    print(f"[auc sanity] informative={auc(yy + rng.normal(0, .3, 4000), yy):.3f}  noise={auc(rng.random(4000), yy):.3f}")

    EST = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{args.estpa_dir}/backtest_*.parquet"))], ignore_index=True)
    APR = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/simulation/backtest_*.parquet"))], ignore_index=True)
    r1 = EST[EST["rank"] == 1].dropna(subset=["p_game_hit", "actual_hit", "est_pas"]).copy()
    r1["season"] = r1["date"].apply(lambda d: d.year)
    y = r1["actual_hit"].to_numpy().astype(float)

    # join lineup_position (pure pre-game batting slot)
    pa = pd.concat([pd.read_parquet(f)[["game_pk", "batter_id", "lineup_position"]] for f in glob.glob("data/processed/pa_*.parquet")], ignore_index=True)
    slot = pa.groupby(["game_pk", "batter_id"])["lineup_position"].agg(lambda x: x.mode().iloc[0]).reset_index()
    r1 = r1.merge(slot, on=["game_pk", "batter_id"], how="left")

    print("\n=== §4 skip vs play (split at 0.796), estimated_pa ===")
    for lab, thr in [("at threshold 0.796", 0.796)]:
        play = r1[r1["p_game_hit"] >= thr]; skip = r1[r1["p_game_hit"] < thr]
        print(f"  PLAY (>={thr}) n={len(play):3d} hit={play['actual_hit'].mean():.3f} | "
              f"SKIP n={len(skip):3d} hit={skip['actual_hit'].mean():.3f} | "
              f"diff {(play['actual_hit'].mean()-skip['actual_hit'].mean())*100:+.1f}pp")
    bands = [(0, .76), (.76, .78), (.78, .796), (.796, .82), (.82, .84), (.84, 1)]
    print("  hit rate by band (flat right AT the threshold):")
    for lo, hi in bands:
        g = r1[(r1["p_game_hit"] >= lo) & (r1["p_game_hit"] < hi)]
        if len(g):
            print(f"    [{lo:.3f},{hi:.3f}) n={len(g):3d} hit={g['actual_hit'].mean():.3f}")

    print("\n=== §7 corrected AUC vs actual hit (the reverse-causality test) ===")
    ap1 = APR[APR["rank"] == 1].dropna(subset=["p_game_hit", "actual_hit"])
    print(f"  estimated_pa: p={auc(r1['p_game_hit'],y):.3f}  est_pas(pre-game)={auc(r1['est_pas'],y):.3f}  "
          f"lineup_slot={auc(-r1['lineup_position'],y):.3f}  realized_n_pas(contaminated)={auc(r1['n_pas'],y):.3f}")
    print(f"  actual_pa(hindsight): p={auc(ap1['p_game_hit'],ap1['actual_hit'].astype(float)):.3f}")
    print("  -> predictable PA (est_pas / slot) ~random; the realized-n_pas 0.64 is reverse-causal.")

    print("\n=== §7 within-day PA-tilt policy test (does picking the higher-PA candidate help?) ===")
    prof = EST.dropna(subset=["p_game_hit", "actual_hit", "est_pas"]).merge(slot, on=["game_pk", "batter_id"], how="left")
    prof["season"] = prof["date"].apply(lambda d: d.year)
    def policy(df, col, asc=False):
        return df.loc[df.sort_values(col, ascending=asc).groupby("date").head(1).index]
    base = policy(prof, "p_game_hit"); alt = policy(prof, "est_pas"); slt = policy(prof, "lineup_position", asc=True)
    print(f"  baseline(max p)={base['actual_hit'].mean():.3f}  max_est_pas={alt['actual_hit'].mean():.3f}  min_slot={slt['actual_hit'].mean():.3f}")
    print("  per-season baseline / max_est_pas (helps 2/5, hurts 2/5 => noise):")
    for s in sorted(prof["season"].unique()):
        print(f"    {s}: {base[base['season']==s]['actual_hit'].mean():.3f} / {alt[alt['season']==s]['actual_hit'].mean():.3f}")
    bd = base.set_index("date")["actual_hit"]; ad = alt.set_index("date")["actual_hit"]
    both = pd.DataFrame({"b": bd, "a": ad}).dropna()
    n01 = int(((both.a == 1) & (both.b == 0)).sum()); n10 = int(((both.a == 0) & (both.b == 1)).sum())
    print(f"  paired McNemar (est_pas-pick vs baseline): {n01} vs {n10}  net {n01-n10:+d}  2-sided p={binomtest(n01, n01+n10, 0.5).pvalue:.3f}")
    print("  -> not significant; the headline +1.7pp was an in-sample/tie-break artifact. No validated PA lever.")


if __name__ == "__main__":
    main()
