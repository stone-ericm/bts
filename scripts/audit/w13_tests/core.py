"""W1.3 distinguishing tests, pure core (design docs/superpowers/specs/2026-10-04-w13-distinguishing-tests-design.md
rev 2). Component diagnostics over the frozen W1.2 run: nothing is re-scored, and every interval is a pointwise,
unadjusted exploratory diagnostic."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from scripts.audit.benchmark_bridge import core as bridge
from scripts.audit.mlb_benchmark import metrics as m

REPRO_TOL = 1e-9
KNOWN = {"hit", "no_hit"}
EPS = 1e-15


def _key(row: dict | None):
    try:
        return int(row["batter_id"]), int(row["game_pk"])
    except (KeyError, TypeError, ValueError):
        return None


def selected_primary(decision: dict | None, pick: dict | None) -> dict:
    """The selected primary (design §2). An authoritative skip means no selection and blocks the pick fallback; its
    candidate is kept only as ``declined``. A decision's primary otherwise wins; a pick primary is used only when no
    decision exists, with the action unknown."""
    if decision is not None:
        action = decision.get("action") or "unknown"
        prim = decision.get("primary")
        if action == "skip":
            return {"selected": None, "declined": _key(prim), "action": "skip", "source": "decision", "p": None}
        return {"selected": _key(prim), "declined": None, "action": action, "source": "decision",
                "p": (prim or {}).get("p_game_hit")}
    prim = (pick or {}).get("pick")
    if prim:
        return {"selected": _key(prim), "declined": None, "action": "unknown", "source": "pick",
                "p": prim.get("p_game_hit")}
    return {"selected": None, "declined": None, "action": "unknown", "source": None, "p": None}


def serialized(p: float) -> float:
    return bridge.serialized_probability(p)


def repro_class(row: dict) -> str:
    """The runner's precedence on serialized scores: exact_final_feed, then inferred_weather_absent_at_serve, else
    unexplained; rows without a C-served or D score are not reproducible (kept out of the numeric denominator)."""
    cs, d, wb = row.get("C_served"), row.get("D"), row.get("C_served_wblank")
    if cs is None or d is None or not math.isfinite(cs) or not math.isfinite(d):
        return "not_reproducible"
    if abs(serialized(cs) - d) <= REPRO_TOL:
        return "exact_final_feed"
    if wb is not None and math.isfinite(wb) and abs(serialized(wb) - d) <= REPRO_TOL:
        return "inferred_weather_absent_at_serve"
    return "unexplained"


def from_bridge_b_minus_a(est: float, interval: tuple) -> tuple:
    """W1.2's pair helper reports later minus earlier; G is earlier minus later."""
    lo, hi = interval
    return -est, (-hi, -lo)


def _winners(common: pd.DataFrame, score: str) -> pd.DataFrame:
    return m.top1(common.assign(_block=common["date"]), score)


def paired_reduction(df: pd.DataFrame, a: str, b: str, pool: str, n_resamples: int = 10_000,
                     seed: int = 20261004) -> dict:
    """G(a, b) = top-1 hit rate of a minus that of b, both ranked on the same frozen pool (rows with finite scores in
    both arms), evaluated on dates where both previously chosen winners are known; unilateral exclusions counted."""
    common = df[df[pool].fillna(False).astype(bool) & np.isfinite(df[a]) & np.isfinite(df[b])]
    if common.empty:
        return {"available": False, "reason": "empty_common_pool"}
    wa, wb = _winners(common, a), _winners(common, b)
    per = wa[["date", "outcome"]].merge(wb[["date", "outcome"]], on="date", suffixes=("_a", "_b")).set_index("date")
    both = per["outcome_a"].isin(KNOWN) & per["outcome_b"].isin(KNOWN)
    excluded = {a: per.loc[~per["outcome_a"].isin(KNOWN), "outcome_a"].value_counts().to_dict(),
                b: per.loc[~per["outcome_b"].isin(KNOWN), "outcome_b"].value_counts().to_dict()}
    k = per[both].assign(ha=lambda x: (x["outcome_a"] == "hit").astype(float),
                         hb=lambda x: (x["outcome_b"] == "hit").astype(float))
    if k.empty:
        return {"available": False, "reason": "no_common_known_dates", "excluded": excluded}
    g = lambda s: float(s["ha"].mean() - s["hb"].mean()) if len(s) else float("nan")
    boot = m.summary_bootstrap(k, {"G": g}, n_resamples=n_resamples, seed=seed)["G"]
    return {"available": True, "estimate": g(k), "interval": boot, "n_common_dates": int(len(k)),
            "n_pool_dates": int(per.shape[0]), "excluded": excluded,
            "discordant_dates": int((k["ha"] != k["hb"]).sum())}


def half_split(dates: list[str]) -> tuple[list[str], list[str]]:
    """Sorted registered dates; the median date (lower middle for an even count) and everything before it form the
    first half. Outcome-independent."""
    d = sorted(dates)
    med = d[(len(d) - 1) // 2]
    return [x for x in d if x <= med], [x for x in d if x > med]


def half_contrast_bootstrap(per_date: pd.DataFrame, first: list[str], second: list[str], stats: dict,
                            n_resamples: int = 10_000, seed: int = 20261004) -> dict:
    """Each half resampled separately at its own size, jointly across the per-date columns (surfaces); each stat is
    f(first_sample, second_sample). Failed draws counted."""
    a, b = per_date.loc[first], per_date.loc[second]
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_resamples) for k in stats}
    for i in range(n_resamples):
        sa = a.iloc[rng.choice(len(a), size=len(a), replace=True)]
        sb = b.iloc[rng.choice(len(b), size=len(b), replace=True)]
        for k, f in stats.items():
            draws[k][i] = f(sa, sb)
    out = {}
    for k, v in draws.items():
        ok = v[np.isfinite(v)]
        out[k] = {"lo": float(np.percentile(ok, 2.5)) if len(ok) else float("nan"),
                  "hi": float(np.percentile(ok, 97.5)) if len(ok) else float("nan"),
                  "n_ok": int(len(ok)), "n_failed": int(n_resamples - len(ok)), "seed": seed, "n_resamples": n_resamples}
    return out


def directional_flag(interval: dict) -> str:
    """Design §3 failure rule: a flag needs every draw defined. Wholly positive → consistent; wholly negative →
    contradicting; zero inclusion → undetermined. Pointwise and unadjusted; never equivalence."""
    if interval.get("n_failed", 1) > 0 or not interval.get("n_ok"):
        return "unavailable"
    if interval["lo"] > 0:
        return "consistent"
    if interval["hi"] < 0:
        return "contradicting"
    return "undetermined"


def recalibration_fit(p, y) -> dict:
    """Unpenalized logistic outcome ~ intercept + slope × logit(p), p clipped to [1e-15, 1−1e-15] BEFORE the logit
    (design E5). Invalid when one class is absent, the design is rank deficient, or the fit separates, diverges,
    fails to converge or has singular information; no penalty or alternative fit is substituted."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    boundary = int(((p <= EPS) | (p >= 1 - EPS)).sum())
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "one_class", "boundary_rows": boundary, "n": int(len(y))}
    q = np.clip(p, EPS, 1 - EPS)
    X = np.column_stack([np.ones(len(q)), np.log(q / (1 - q))])
    if np.linalg.matrix_rank(X) < 2:
        return {"valid": False, "reason": "rank_deficient", "boundary_rows": boundary, "n": int(len(y))}
    beta, why = m._fit(X, y, np.ones(len(y)))
    if beta is None:
        return {"valid": False, "reason": why, "boundary_rows": boundary, "n": int(len(y))}
    return {"valid": True, "reason": None, "intercept": float(beta[0]), "slope": float(beta[1]),
            "boundary_rows": boundary, "n": int(len(y))}


def slate_drag(cands: pd.DataFrame, drag: pd.DataFrame) -> pd.DataFrame:
    """Registered-slate as-of drag per date (design E6): the equal-weight mean of exact (venue, date) values over the
    distinct venues in that date's declared support. Available only when every contributing venue has a finite exact
    value; no zero imputation and no forward or back fill."""
    venues = cands[["date", "venue_id"]].drop_duplicates()
    j = venues.merge(drag[["venue_id", "date", "park_drag_delta"]], on=["venue_id", "date"], how="left")
    rows = []
    for d, g in j.groupby("date", sort=True):
        ok = g["venue_id"].notna() & np.isfinite(g["park_drag_delta"].astype(float))
        complete = bool(ok.all())
        rows.append({"date": d, "n_venues": int(len(g)), "missing_venues": int((~ok).sum()), "complete": complete,
                     "drag_mean": float(g["park_drag_delta"].mean()) if complete else float("nan")})
    return pd.DataFrame(rows).set_index("date")
