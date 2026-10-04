"""W2.3 MLB forecast benchmark: scoring on the frozen shared pool (design docs/superpowers/specs/
2026-10-04-mlb-forecast-benchmark-design.md rev 3, gates 2, 5 and 6).

The pool is frozen before labels: identity-resolved rows with valid probabilities in both arms (``ours``, ``mlb``)
in one declared stratum. Every equal-date statistic groups by a block column; the bootstrap gives each drawn copy of
a date its own block, so repeated draws are never collapsed back into one date."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from scripts.audit.benchmark_bridge import core as bridge

EPS = 1e-15
TARGET_KNOWN = {"T1": {"hit", "no_hit", "no_pa"}, "T2": {"hit", "no_hit"}}


def target(df: pd.DataFrame, t: str) -> pd.DataFrame:
    """Rows with a known label under target ``t`` (T1: no_pa counts as no hit; T2: no_pa excluded), with ``y``.
    unknown is never known."""
    out = df[df["outcome"].isin(TARGET_KNOWN[t])].copy()
    out["y"] = (out["outcome"] == "hit").astype(int)
    return out


def brier_rows(p: pd.Series, y: pd.Series) -> float:
    return float(((p - y) ** 2).mean())


def log_loss_rows(p: pd.Series, y: pd.Series) -> float:
    q = p.clip(EPS, 1 - EPS)
    return float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean())


def residual_rows(p: pd.Series, y: pd.Series) -> float:
    return float(p.mean() - y.mean())


def equal_date_mean(df: pd.DataFrame, score: str, f: Callable, block: str = "_block") -> float:
    """Mean over blocks of the within-block row statistic ``f(p, y)``; the point estimate uses dates as blocks."""
    col = block if block in df else "date"
    vals = [f(g[score], g["y"]) for _, g in df.groupby(col, sort=True) if len(g)]
    return float(np.mean(vals)) if vals else float("nan")


def equal_date_auc(df: pd.DataFrame, score: str, block: str = "_block") -> tuple[float, int, int]:
    """Tie-aware within-block AUC averaged over blocks; single-class blocks omitted and counted (bridge helper)."""
    col = block if block in df else "date"
    return bridge.equal_date_auc(df, score, "y", date_col=col)


def top1(df: pd.DataFrame, score: str, block: str = "_block") -> pd.DataFrame:
    """Each block's argmax of ``score`` over the frozen pool, chosen before labels; exact ties go to the lower archived
    ``row_order``. A winner whose label is no_pa or unknown is kept, never replaced."""
    col = block if block in df else "date"
    rows = []
    for _, g in df.groupby(col, sort=True):
        top = g[g[score] == g[score].max()]
        rows.append(top.sort_values("row_order").iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _winner_pairs(df: pd.DataFrame, a: str, b: str, block: str) -> pd.DataFrame:
    col = block if block in df else "date"
    wa, wb = top1(df, a, block), top1(df, b, block)
    keep = list(dict.fromkeys([col, "date", "batter_id", "game_pk", "outcome"]))
    return wa[keep].merge(
        wb[[col, "batter_id", "game_pk", "outcome"]], on=col, suffixes=("_a", "_b"))


def _paired(pairs: pd.DataFrame, t: str, a: str, b: str, col: str) -> dict:
    known = TARGET_KNOWN[t]
    both = pairs[pairs["outcome_a"].isin(known) & pairs["outcome_b"].isin(known)]
    excl = {a: pairs.loc[~pairs["outcome_a"].isin(known), "outcome_a"].value_counts().to_dict(),
            b: pairs.loc[~pairs["outcome_b"].isin(known), "outcome_b"].value_counts().to_dict()}
    return {"dates_common": list(both[col]), "n_common": int(len(both)),
            f"{a}_rate": float((both["outcome_a"] == "hit").mean()) if len(both) else float("nan"),
            f"{b}_rate": float((both["outcome_b"] == "hit").mean()) if len(both) else float("nan"),
            "excluded": excl}


def paired_top1(df: pd.DataFrame, a: str, b: str, t: str, block: str = "_block") -> dict:
    """Both arms' top-1 hit rates on the identical blocks where both previously chosen winners are known under ``t``;
    each side's unilateral exclusions are reported."""
    col = block if block in df else "date"
    return _paired(_winner_pairs(df, a, b, block), t, a, b, col)


def disagreement(df: pd.DataFrame, a: str, b: str, t: str, block: str = "_block") -> dict:
    """Blocks whose two winners differ by (batter_id, game_pk), with the common-known paired rates on them."""
    col = block if block in df else "date"
    pairs = _winner_pairs(df, a, b, block)
    diff = pairs[(pairs["batter_id_a"] != pairs["batter_id_b"]) | (pairs["game_pk_a"] != pairs["game_pk_b"])]
    return {"dates": list(diff[col]), **_paired(diff, t, a, b, col)}


def block_bootstrap_many(df: pd.DataFrame, stats: dict, n_resamples: int = 10_000, seed: int = 20261004) -> dict:
    """Whole dates resampled with replacement, every statistic in ``stats`` computed on the same draws. Draw copy k
    of a date gets block id k, so equal-date statistics weight every copy. NaN/inf draws are failures, counted per
    statistic and excluded from its percentiles."""
    dates = np.array(sorted(df["date"].unique()))
    groups = {d: g for d, g in df.groupby("date", sort=False)}
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_resamples) for k in stats}
    for i in range(n_resamples):
        pick = rng.choice(dates, size=len(dates), replace=True)
        sample = pd.concat([groups[d].assign(_block=k) for k, d in enumerate(pick)], ignore_index=True)
        for k, f in stats.items():
            draws[k][i] = f(sample)
    out = {}
    for k, v in draws.items():
        ok = v[np.isfinite(v)]
        out[k] = {"lo": float(np.percentile(ok, 2.5)) if len(ok) else float("nan"),
                  "hi": float(np.percentile(ok, 97.5)) if len(ok) else float("nan"),
                  "n_ok": int(len(ok)), "n_failed": int(n_resamples - len(ok)), "seed": seed,
                  "n_resamples": n_resamples}
    return out


def summary_bootstrap(per_date: pd.DataFrame, stats: dict, n_resamples: int = 10_000, seed: int = 20261004) -> dict:
    """The same date draws as ``block_bootstrap_many`` (sorted dates, same generator), applied to a per-date summary
    table: for an equal-date mean, a drawn copy's block statistic is exactly that date's summary value, so resampling
    summary rows (repeats kept) equals the copy-block bootstrap without rebuilding row frames."""
    per_date = per_date.sort_index()
    rng = np.random.default_rng(seed)
    n = len(per_date)
    draws = {k: np.empty(n_resamples) for k in stats}
    for i in range(n_resamples):
        pick = rng.choice(np.arange(n), size=n, replace=True)
        sample = per_date.iloc[pick]
        for k, f in stats.items():
            draws[k][i] = f(sample)
    out = {}
    for k, v in draws.items():
        ok = v[np.isfinite(v)]
        out[k] = {"lo": float(np.percentile(ok, 2.5)) if len(ok) else float("nan"),
                  "hi": float(np.percentile(ok, 97.5)) if len(ok) else float("nan"),
                  "n_ok": int(len(ok)), "n_failed": int(n_resamples - len(ok)), "seed": seed,
                  "n_resamples": n_resamples}
    return out


def block_bootstrap(df: pd.DataFrame, stat: Callable[[pd.DataFrame], float], n_resamples: int = 10_000,
                    seed: int = 20261004) -> dict:
    """One statistic through ``block_bootstrap_many``."""
    return block_bootstrap_many(df, {"stat": stat}, n_resamples, seed)["stat"]


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    """Overflow-free logistic function."""
    return 0.5 * (1.0 + np.tanh(0.5 * eta))


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def _fit(X: np.ndarray, y: np.ndarray, w: np.ndarray, max_iter: int = 100, tol: float = 1e-10):
    """Unpenalized weighted logistic MLE by Newton's method. Returns (beta, None) or (None, reason)."""
    if np.linalg.matrix_rank(X) < X.shape[1]:
        return None, "nonidentifiable"
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        mu = _sigmoid(eta)
        grad = X.T @ (w * (y - mu))
        hess = X.T @ (X * (w * mu * (1 - mu))[:, None])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None, "singular_information"
        beta = beta + step
        if not np.all(np.isfinite(beta)) or np.max(np.abs(beta)) > 1e3:
            return None, "separation_or_divergence"
        if np.max(np.abs(step)) < tol:
            mu = _sigmoid(X @ beta)
            if np.any(mu < 1e-12) or np.any(mu > 1 - 1e-12):
                return None, "separation_or_divergence"
            return beta, None
    return None, "nonconvergence"


def encompassing(df: pd.DataFrame, block: str = "_block") -> dict:
    """Descriptive residual-information check on a T2 population (gate 6): intercept + logit(ours) versus
    intercept + logit(ours) + logit(mlb), equal-block weights (each block sums to 1), no interactions or selection.
    Reports the MLB coefficient and the in-sample weighted log-loss change against the ours-only recalibration fit;
    any failed fit makes the result unavailable rather than triggering another model."""
    col = block if block in df else "date"
    y = df["y"].to_numpy(dtype=float)
    w = (1.0 / df.groupby(col)[col].transform("size")).to_numpy(dtype=float)
    lo, lm = _logit(df["ours"].to_numpy(float)), _logit(df["mlb"].to_numpy(float))
    boundary = int(((df["ours"] <= EPS) | (df["ours"] >= 1 - EPS) | (df["mlb"] <= EPS) | (df["mlb"] >= 1 - EPS)).sum())
    X0 = np.column_stack([np.ones(len(df)), lo])
    X1 = np.column_stack([np.ones(len(df)), lo, lm])
    b0, r0 = _fit(X0, y, w)
    b1, r1 = _fit(X1, y, w)
    if b0 is None or b1 is None:
        return {"available": False, "reason": r0 or r1, "boundary_rows": boundary}

    def wll(X, b):
        mu = np.clip(_sigmoid(X @ b), EPS, 1 - EPS)
        return float(-(w * (y * np.log(mu) + (1 - y) * np.log(1 - mu))).sum() / w.sum())
    ll0, ll1 = wll(X0, b0), wll(X1, b1)
    return {"available": True, "reason": None, "boundary_rows": boundary,
            "intercept_ours_only": float(b0[0]), "slope_ours_only": float(b0[1]),
            "coef_ours_full": float(b1[1]), "coef_mlb": float(b1[2]),
            "log_loss_ours_only": ll0, "log_loss_full": ll1, "delta_log_loss": ll1 - ll0}
