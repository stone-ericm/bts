"""W1.2 metrics and paired comparisons over the long bridge table (design §6, frozen rev 3).

The table has one row per (date, batter_id, game_pk) slate candidate, with columns: date, row_order, outcome
(hit / no_hit / no_pa / unknown), one score column per surface, and boolean pool columns (``pool_verified``,
``pool_surrogate``, ``pool_all``). Ranking always happens before outcomes are consulted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.audit.benchmark_bridge import core

KNOWN = ("hit", "no_hit")


def _winners(df: pd.DataFrame, score: str, pool: str) -> pd.DataFrame:
    rows = []
    for d, g in df[df[pool].fillna(False).astype(bool)].groupby("date", sort=True):
        g = g.assign(eligible=True)
        i = core.rank1(g, score)
        if i is not None:
            rows.append(df.loc[i])
    return pd.DataFrame(rows)


def surface_metrics(df: pd.DataFrame, score: str, pool: str, bins: int = 10) -> dict:
    sub = df[df[pool].fillna(False).astype(bool) & df[score].notna()]
    known = sub[sub["outcome"].isin(KNOWN)]
    y = (known["outcome"] == "hit").astype(int)
    win = _winners(df, score, pool)
    wout = win["outcome"].value_counts().to_dict() if not win.empty else {}
    n_known_win = wout.get("hit", 0) + wout.get("no_hit", 0)
    auc, n_auc, n_auc_omitted = core.equal_date_auc(known.assign(y=y), score, "y")
    edges = np.linspace(0, 1, bins + 1)
    rel = []
    if len(known):
        b = np.clip(np.digitize(known[score], edges) - 1, 0, bins - 1)
        for k in range(bins):
            m = b == k
            if m.any():
                rel.append({"bin": [float(edges[k]), float(edges[k + 1])], "n": int(m.sum()),
                            "mean_p": float(known[score][m].mean()), "hit_rate": float(y[m].mean())})
    return {
        "surface": score, "pool": pool, "n_dates": int(sub["date"].nunique()), "n_rows": int(len(sub)),
        "n_known_rows": int(len(known)),
        "top1": {"hit": int(wout.get("hit", 0)), "no_hit": int(wout.get("no_hit", 0)),
                 "no_pa": int(wout.get("no_pa", 0)), "unknown": int(wout.get("unknown", 0)),
                 "hit_rate": (wout.get("hit", 0) / n_known_win) if n_known_win else None},
        "auc_equal_date": auc, "auc_dates_used": n_auc, "auc_dates_omitted": n_auc_omitted,
        "brier": core.brier(known[score], y) if len(known) else None,
        "log_loss": core.log_loss(known[score], y) if len(known) else None,
        "mean_stated_minus_realized": float(known[score].mean() - y.mean()) if len(known) else None,
        "reliability": rel,
    }


def pair_metrics(df: pd.DataFrame, a: str, b: str, pool: str, n_resamples: int = 10_000, seed: int = 20261004) -> dict:
    base = df[df[pool].fillna(False).astype(bool)].assign(eligible=True)
    pp, excl = core.paired_pool(base, a, b)
    diff = pp[b] - pp[a]
    out = {"pair": [a, b], "pool": pool, "exclusions": excl, "n_rows": int(len(pp)), "n_dates": int(pp["date"].nunique()),
           "score_change": {"mean": float(diff.mean()) if len(pp) else None,
                            "median": float(diff.median()) if len(pp) else None,
                            "mean_abs": float(diff.abs().mean()) if len(pp) else None,
                            "corr": float(np.corrcoef(pp[a], pp[b])[0, 1]) if len(pp) > 2 else None}}
    # winners on the identical pool; never reselected for outcome
    wa, wb, per_date = {}, {}, []
    for d, g in pp.groupby("date", sort=True):
        ia, ib = core.rank1(g, a), core.rank1(g, b)
        oa, ob = g.loc[ia, "outcome"], g.loc[ib, "outcome"]
        per_date.append({"date": d, "changed": bool(ia != ib), "oa": oa, "ob": ob})
    pdf = pd.DataFrame(per_date)
    out["rank1_changed_dates"] = int(pdf["changed"].sum()) if len(pdf) else 0
    both = pdf[pdf["oa"].isin(KNOWN) & pdf["ob"].isin(KNOWN)] if len(pdf) else pdf
    out["paired_top1_dates"] = int(len(both))
    out["unilateral_void_or_unknown_dates"] = int(len(pdf) - len(both)) if len(pdf) else 0
    if len(both):
        ha, hb = (both["oa"] == "hit").astype(int), (both["ob"] == "hit").astype(int)
        out["discordant_dates"] = {"a_hit_b_miss": int(((ha == 1) & (hb == 0)).sum()),
                                   "a_miss_b_hit": int(((ha == 0) & (hb == 1)).sum())}
        tdf = pd.DataFrame({"date": both["date"].values, "d": (hb - ha).values})
        out["top1_diff_b_minus_a"] = float(tdf["d"].mean())
        out["top1_diff_ci95"] = core.date_block_bootstrap(tdf, lambda f: float(f["d"].mean()), n_resamples, seed)
    known = pp[pp["outcome"].isin(KNOWN)]
    if len(known):
        y = (known["outcome"] == "hit").astype(int)
        bdf = known.assign(y=y)
        stat = lambda f: core.brier(f[b], f["y"]) - core.brier(f[a], f["y"])
        out["brier_diff_b_minus_a"] = float(stat(bdf))
        out["brier_diff_ci95"] = core.date_block_bootstrap(bdf, stat, n_resamples, seed)
    return out
