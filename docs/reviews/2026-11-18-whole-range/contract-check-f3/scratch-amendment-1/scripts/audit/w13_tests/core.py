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


def valid(s) -> pd.Series:
    """The one probability-validity mask (code review r1 F10): finite and within [0, 1]."""
    v = pd.to_numeric(pd.Series(s), errors="coerce").astype(float)
    return np.isfinite(v) & (v >= 0.0) & (v <= 1.0)


def _valid1(x) -> bool:
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x) and 0.0 <= x <= 1.0


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
    if not _valid1(cs) or not _valid1(d):
        return "not_reproducible"
    if abs(serialized(cs) - d) <= REPRO_TOL:
        return "exact_final_feed"
    if _valid1(wb) and abs(serialized(wb) - d) <= REPRO_TOL:
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
    inpool = df[pool].fillna(False).astype(bool)
    invalid = {a: int((inpool & df[a].notna() & ~valid(df[a]).values).sum()),
               b: int((inpool & df[b].notna() & ~valid(df[b]).values).sum())}
    common = df[inpool & valid(df[a]).values & valid(df[b]).values]
    if common.empty:
        return {"available": False, "reason": "empty_common_pool", "invalid_scores": invalid}
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
            "invalid_scores": invalid, "n_common_rows": int(len(common)),
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
    return {k: m.collect(v, n_resamples, seed) for k, v in draws.items()}


def directional_flag(interval: dict) -> str:
    """Design §3 failure rule: a flag needs every draw defined. Wholly positive → consistent; wholly negative →
    contradicting; zero inclusion → undetermined. Pointwise and unadjusted; never equivalence."""
    if interval.get("n_failed", 1) > 0 or not interval.get("n_ok") or interval.get("lo") is None:
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
    lo_n, hi_n = int((p <= EPS).sum()), int((p >= 1 - EPS).sum())
    boundary = {"boundary_rows": lo_n + hi_n, "clipped_low": lo_n, "clipped_high": hi_n}
    if len(np.unique(y)) < 2:
        return {"valid": False, "reason": "one_class", **boundary, "n": int(len(y))}
    q = np.clip(p, EPS, 1 - EPS)
    X = np.column_stack([np.ones(len(q)), np.log(q / (1 - q))])
    if np.linalg.matrix_rank(X) < 2:
        return {"valid": False, "reason": "rank_deficient", **boundary, "n": int(len(y))}
    beta, why = m._fit(X, y, np.ones(len(y)))
    if beta is None:
        return {"valid": False, "reason": why, **boundary, "n": int(len(y))}
    return {"valid": True, "reason": None, "intercept": float(beta[0]), "slope": float(beta[1]),
            **boundary, "n": int(len(y))}


def slate_drag(cands: pd.DataFrame, drag: pd.DataFrame) -> pd.DataFrame:
    """Registered-slate as-of drag per date (design E6): the equal-weight mean of exact (venue, date) values over the
    distinct venues in that date's declared support. Available only when every contributing venue has a finite exact
    value; no zero imputation and no forward or back fill."""
    venues = cands[["date", "venue_id"]].drop_duplicates()
    venues = venues.assign(venue_id=pd.to_numeric(venues["venue_id"], errors="coerce").astype(float))
    dg = drag[["venue_id", "date", "park_drag_delta"]]
    dg = dg.assign(venue_id=pd.to_numeric(dg["venue_id"], errors="coerce").astype(float))
    nuniq = dg.groupby(["venue_id", "date"])["park_drag_delta"].nunique(dropna=False)
    conflicts = set(nuniq[nuniq > 1].index)             # conflicting duplicate keys void that venue/date
    dg = dg.drop_duplicates(["venue_id", "date"])        # identical duplicates collapse to one value
    dg = dg.loc[np.array([(v, d) not in conflicts for v, d in zip(dg["venue_id"], dg["date"])], dtype=bool)]
    j = venues.merge(dg, on=["venue_id", "date"], how="left", validate="many_to_one")
    rows = []
    for d, g in j.groupby("date", sort=True):
        ok = g["venue_id"].notna() & np.isfinite(g["park_drag_delta"].astype(float))
        complete = bool(ok.all())
        rows.append({"date": d, "n_venues": int(len(g)), "missing_venues": int((~ok).sum()), "complete": complete,
                     "conflicting_keys": int(sum((v, d) in conflicts for v in g["venue_id"])),
                     "drag_mean": float(g["park_drag_delta"].mean()) if complete else float("nan")})
    cols = ["n_venues", "missing_venues", "complete", "conflicting_keys", "drag_mean"]
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=cols, index=pd.Index([], name="date"))
