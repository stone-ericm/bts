"""W1.3 distinguishing tests driver (design docs/superpowers/specs/2026-10-04-w13-distinguishing-tests-design.md rev 3;
code review docs/audit/2026-10-04-w13-w23-code-codex-r1.md).

Reads the ACCEPTED W1.2 run (verified bytes), the manifest-hashed decision and pick files, the archived final feeds
(venue only) and the as-of park-drag export with its sibling manifest; every input is read once and hashed from the
bytes used. Writes component diagnostics for the primary stratum (selection_consistent × pool_verified), each with
its measured component, an unadjusted pointwise flag, the full explanation's status, denominators and intervals,
plus separately labelled sensitivity strata that never carry a flag. Refuses to run until exposure row X-29 is in
HEAD."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit.benchmark_bridge import core as bridge
from scripts.audit.mlb_benchmark import metrics as m
from scripts.audit.w13_tests import core

REPO = Path(__file__).resolve().parents[3]
X29_COMMIT = "8c47fb1"                # the register commit that publishes X-29
SURFACES = ["A26", "A26_count", "B26", "C_frozen", "C_served", "D"]
X12_WINDOW = ("2026-09-19", "2026-09-27")
UNEXPLAINED_FLAG_SHARE = 0.05
PRIMARY = ("selection_consistent", "pool_verified")
SENSITIVITIES = {"all_dates/pool_verified": (None, "pool_verified"),
                 "selection_consistent/pool_surrogate": ("selection_consistent", "pool_surrogate"),
                 "selection_consistent/pool_all": ("selection_consistent", "pool_all")}
NO_FLAG = "sensitivity (no flag)"


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", file=sys.stderr, flush=True)


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def x29_gate() -> str:
    head = git("rev-parse", "HEAD")
    if X29_COMMIT is None:
        raise SystemExit("X-29 gate: X29_COMMIT is unset (publish X-29 first)")
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", X29_COMMIT, head]).returncode != 0:
        raise SystemExit(f"X-29 gate: {X29_COMMIT} is not an ancestor of HEAD {head[:7]}")
    if "| X-29 |" not in (REPO / "docs/audit/2026-09-22-exposure-register.md").read_text():
        raise SystemExit("X-29 gate: the register in this checkout has no X-29 row")
    return head


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stratum(tab: pd.DataFrame, sel: str | None, pool: str) -> pd.DataFrame:
    mask = tab[pool].fillna(False).astype(bool)
    if sel is not None:
        mask &= tab["sel_state"] == sel
    return tab[mask]


def counts(df: pd.DataFrame) -> dict:
    return {"dates": int(df["date"].nunique()) if len(df) else 0, "rows": int(len(df)),
            "outcomes": {str(k): int(v) for k, v in df["outcome"].value_counts().to_dict().items()} if len(df) else {}}


def _known(df: pd.DataFrame, score: str) -> pd.DataFrame:
    if df.empty or score not in df:
        return df.iloc[0:0].assign(y=pd.Series(dtype=int))
    k = df[df["outcome"].isin(core.KNOWN) & core.valid(df[score]).values]
    return k.assign(y=(k["outcome"] == "hit").astype(int))


def _per_date(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """Per-date residual (mean p − mean y) and tie-aware AUC over known valid rows of ``score``; schema preserved
    when there are none."""
    rows = []
    for d, g in _known(df, score).groupby("date", sort=True):
        a = bridge._rank_auc(g.loc[g["y"] == 1, score], g.loc[g["y"] == 0, score])
        rows.append({"date": d, f"resid_{score}": float(g[score].mean() - g["y"].mean()),
                     f"auc_{score}": float("nan") if a is None else float(a), f"n_{score}": int(len(g))})
    cols = [f"resid_{score}", f"auc_{score}", f"n_{score}"]
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=cols, index=pd.Index([], name="date"))


def _winners(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """Each date's argmax of a valid ``score`` (row-order ties), chosen before labels; schema preserved when empty."""
    v = df[core.valid(df[score]).values] if len(df) and score in df else df.iloc[0:0]
    if v.empty:
        return pd.DataFrame(columns=list(df.columns))
    return m.top1(v.assign(_block=v["date"]), score)


def _nanmean(s: pd.Series) -> float:
    return float(s.mean()) if len(s) and s.notna().any() else float("nan")


def _label(flag: str, positive: str, negative: str, primary: bool = True) -> str:
    if not primary:
        return NO_FLAG
    return {"consistent": positive, "contradicting": negative}.get(flag, flag)


# --- E1 ----------------------------------------------------------------------------------------------------------
def e1(df: pd.DataFrame, pool: str, summary: dict, n: int, primary: bool = True, cohort_key: str = "selection_consistent",
       selected_rows: pd.DataFrame | None = None) -> dict:
    chain = []
    for pr in summary.get("metrics", {}).get(cohort_key, {}).get(pool, {}).get("pairs", []):
        a, b = pr["pair"]
        if pr.get("top1_diff_b_minus_a") is not None:
            est, (lo, hi) = core.from_bridge_b_minus_a(pr["top1_diff_b_minus_a"], tuple(pr["top1_diff_ci95"]))
            chain.append({"pair": [a, b], "G": est, "G_ci95": [lo, hi], "paired_dates": pr["paired_top1_dates"],
                          "rank1_changed_dates": pr["rank1_changed_dates"], "discordant": pr.get("discordant_dates"),
                          "exclusions": pr["exclusions"], "composite": True})
        else:
            chain.append({"pair": [a, b], "G": None, "reason": "no paired top-1 dates", "exclusions": pr.get("exclusions")})
    g = core.paired_reduction(df, "A26", "B26", pool, n_resamples=n) if len(df) else {"available": False, "reason": "empty"}
    flag = core.directional_flag(g["interval"]) if g.get("available") else "unavailable"
    native = {}
    for s in SURFACES:
        w = _winners(df, s)
        native[s] = ({"n_winners": int(len(w)), "scoring_n_pa_mean": float(w["n_pa"].mean()),
                      "scoring_n_pa_ge5": float((w["n_pa"] >= 5).mean()),
                      "winner_outcomes": w["outcome"].value_counts().to_dict()} if len(w) else {"n_winners": 0})
    sel_pa = None
    if selected_rows is not None and len(selected_rows):
        sel_pa = {"n": int(len(selected_rows)),
                  "n_pa_distribution": {int(k): int(v) for k, v in selected_rows["n_pa"].value_counts().sort_index().items()},
                  "n_pa_mean": float(selected_rows["n_pa"].mean())}
    csd = next((c for c in chain if c["pair"] == ["C_served", "D"]), None)
    return {"component": "restricted optimism reduction G(A26,B26) on its common pool",
            "flag": _label(flag, "consistent", "weakened", primary), "support": counts(df), "G_A26_B26": g,
            "chain_carried_from_W12": chain, "surviving_C_served_D": csd, "native_winner_scoring_pa": native,
            "selected_primary_scoring_pa": sel_pa,
            "limits": "oracle diagnostic; composite chain; no numerical attribution to the headline gap"}


# --- E2 ----------------------------------------------------------------------------------------------------------
def _repro(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    return pd.Series([core.repro_class(r) for r in df[["C_served", "C_served_wblank", "D"]].to_dict("records")],
                     index=df.index)


def _repro_block(df: pd.DataFrame) -> dict:
    cls = _repro(df)
    rep = df[(cls != "not_reproducible").values] if len(df) else df
    rcls = cls[rep.index] if len(rep) else cls
    resid = (rep["C_served"].map(core.serialized) - rep["D"]) if len(rep) else pd.Series(dtype=float)
    share = float((rcls == "unexplained").mean()) if len(rep) else float("nan")

    def by(col):
        return {str(k): rcls[g.index].value_counts().to_dict() for k, g in rep.groupby(col)} if len(rep) else {}
    return {"rows": int(len(df)), "classes": {str(k): int(v) for k, v in cls.value_counts().items()},
            "reproducible_rows": int(len(rep)), "unexplained_share": share,
            "signed_resid_quantiles": {q: float(resid.quantile(q)) for q in (0.01, 0.5, 0.99)} if len(rep) else {},
            "abs_resid_quantiles": {q: float(resid.abs().quantile(q)) for q in (0.5, 0.9, 1.0)} if len(rep) else {},
            "by_month": by(rep["date"].str[:7]) if len(rep) else {},
            "by_projected": by(rep["projected"].astype(str)) if len(rep) else {}}


def e2(tab: pd.DataFrame, summary: dict) -> dict:
    """Primary flag from selection-consistent verified rows only (code review r1 F4); other populations are labelled
    sensitivities. A nonzero unexplained share below the threshold is reported as such, never as parity."""
    block = _repro_block(stratum(tab, *PRIMARY))
    share, nrep = block["unexplained_share"], block["reproducible_rows"]
    flag = ("not_testable_empty_support" if nrep == 0 else
            "unexplained_reconstruction_gap" if share >= UNEXPLAINED_FLAG_SHARE else
            "subthreshold_unexplained" if share > 0 else "numerical_parity_conditional")
    served = summary.get("reproduction", {}).get("served_status", {})
    return {"component": "numerical reconstruction of served scores (C-served vs D), primary support", "flag": flag,
            "primary": block,
            "sensitivities": {"selection_consistent/all_pools": _repro_block(tab[tab["sel_state"] == "selection_consistent"]),
                              **{f"selection_consistent/{p}": _repro_block(stratum(tab, "selection_consistent", p))
                                 for p in ("pool_surrogate", "pool_all")}},
            "served_status_from_W12": dict(Counter(served.values())) if isinstance(served, dict) else served,
            "full_explanation": "historical feature age, lineup/pitcher changes and causal staleness not testable here",
            "limits": "a reconstruction gap is not measured live drift; M3 (X-03) carried, not rerun"}


# --- E3 ----------------------------------------------------------------------------------------------------------
def e3(df: pd.DataFrame, pool: str, n: int, primary: bool = True) -> dict:
    empty = {"available": False, "reason": "empty"}
    g = core.paired_reduction(df, "A26", "A26_count", pool, n_resamples=n) if len(df) else empty
    comp = core.paired_reduction(df, "A26_count", "B26", pool, n_resamples=n) if len(df) else empty
    common = df[core.valid(df["A26"]).values & core.valid(df["A26_count"]).values] if len(df) else df
    flag = core.directional_flag(g["interval"]) if g.get("available") else "unavailable"
    return {"component": "oracle realized-count contribution G(A26, A26_count)",
            "flag": (NO_FLAG if not primary else "oracle_count_consistent" if flag == "consistent" else flag),
            "support": counts(df), "G_A26_A26count": g, "G_A26count_B26_composite": comp,
            "score_change_A26_minus_count": float((common["A26"] - common["A26_count"]).mean()) if len(common) else None,
            "scoring_n_pa_vs_A26_n_pas_actual_mismatch_rows": int((common["n_pa"] != common["n_pas_actual"]).sum())
            if len(common) and "n_pas_actual" in common else None,
            "full_explanation": "not testable here (no at-lock PA forecast is archived)",
            "limits": "oracle; never an achievable improvement; zero change does not weaken forecast error"}


# --- E4 ----------------------------------------------------------------------------------------------------------
def e4(prim: pd.DataFrame, tab: pd.DataFrame, dates: list[str], n: int, primary: bool = True) -> dict:
    """Half contrasts (design E4). The flag comes only from the C-frozen residual contrast on reproduced support:
    rows with valid C-frozen and an E2 class of exact or inferred-weather; each half resamples only its dates with at
    least one known supported row, at that effective count. Full contrasts are descriptions."""
    if len(dates) < 2:
        return {"component": "half contrasts", "flag": "unavailable" if primary else NO_FLAG,
                "reason": "fewer than two registered dates"}
    first, second = core.half_split(dates)
    cls = _repro(prim)
    rsup = (prim[cls.isin(["exact_final_feed", "inferred_weather_absent_at_serve"]).values
                 & core.valid(prim["C_frozen"]).values] if len(prim) else prim)
    per = pd.DataFrame(index=sorted(dates))
    for s in ("C_frozen", "B26", "D"):
        per = per.join(_per_date(prim, s), how="left")
    stats = {f"{k}_{s}": (lambda a, b, c=f"{k}_{s}": _nanmean(b[c]) - _nanmean(a[c]))
             for s in ("C_frozen", "B26", "D") for k in ("resid", "auc")}
    point = {k: f(per.loc[first], per.loc[second]) for k, f in stats.items()}
    boot = core.half_contrast_bootstrap(per, first, second, stats, n_resamples=n)
    rper = _per_date(rsup, "C_frozen")
    f_r = [d for d in first if d in rper.index]
    s_r = [d for d in second if d in rper.index]
    support = {h: {"dates": len(hd), "rows": int(rsup["date"].isin(hd).sum()) if len(rsup) else 0,
                   "classes": cls[rsup.index][rsup["date"].isin(hd).values].value_counts().to_dict() if len(rsup) else {}}
               for h, hd in (("first", f_r), ("second", s_r))}
    if f_r and s_r:
        rstat = {"resid_C_frozen_reproduced": lambda a, b: _nanmean(b["resid_C_frozen"]) - _nanmean(a["resid_C_frozen"])}
        point["resid_C_frozen_reproduced"] = rstat["resid_C_frozen_reproduced"](rper.loc[f_r], rper.loc[s_r])
        boot.update(core.half_contrast_bootstrap(rper, f_r, s_r, rstat, n_resamples=n))
        flag = core.directional_flag(boot["resid_C_frozen_reproduced"])
    else:
        point["resid_C_frozen_reproduced"] = None
        flag = "unavailable"
    return {"component": "second-half minus first-half C-frozen residual on numerically reproduced support",
            "flag": _label(flag, "increased_overprediction_conditional", "reduced_overprediction", primary),
            "halves": {"first": [first[0], first[-1], len(first)], "second": [second[0], second[-1], len(second)]},
            "support": counts(prim), "estimates": point, "intervals": boot, "reproduced_support": support,
            "full_explanation": "conditional hit-model drift not identified (one evolving season; PA-level evidence unavailable)",
            "limits": "conditional on numerical reproduction; does not establish historical input parity"}


# --- E5 ----------------------------------------------------------------------------------------------------------
def _refit_bootstrap(rows: pd.DataFrame, score: str, n: int, seed: int = 20261004) -> dict:
    dates = np.array(sorted(rows["date"].unique()))
    groups = {d: g for d, g in rows.groupby("date", sort=False)}
    rng = np.random.default_rng(seed)
    a, b, fails = np.full(n, np.nan), np.full(n, np.nan), {}
    for i in range(n):
        s = pd.concat([groups[d] for d in rng.choice(dates, size=len(dates), replace=True)], ignore_index=True)
        f = core.recalibration_fit(s[score].to_numpy(float), s["y"].to_numpy(int))
        if f["valid"]:
            a[i], b[i] = f["intercept"], f["slope"]
        else:
            fails[f["reason"]] = fails.get(f["reason"], 0) + 1
    return {"intercept": m.collect(a, n, seed, fails), "slope": m.collect(b, n, seed, fails)}


def e5(df: pd.DataFrame, summary: dict, n: int, primary: bool = True, cohort_key: str = "selection_consistent",
       pool: str = "pool_verified") -> dict:
    carried = {s["surface"]: {k: s.get(k) for k in ("brier", "log_loss", "mean_stated_minus_realized", "reliability",
                                                     "n_known_rows")}
               for s in summary.get("metrics", {}).get(cohort_key, {}).get(pool, {}).get("surfaces", [])}
    fits = {}
    for s in SURFACES:
        k = _known(df, s)
        base = (core.recalibration_fit(k[s].to_numpy(float), k["y"].to_numpy(int)) if len(k)
                else {"valid": False, "reason": "empty"})
        entry = {"base": base, "support": {"known_rows": int(len(k)), "dates": int(k["date"].nunique()) if len(k) else 0}}
        if base["valid"]:
            iv = _refit_bootstrap(k, s, n)
            entry["intervals"] = iv
            full = iv["intercept"]["n_failed"] == 0 and iv["intercept"]["lo"] is not None
            off = full and (iv["intercept"]["lo"] > 0 or iv["intercept"]["hi"] < 0
                            or iv["slope"]["lo"] > 1 or iv["slope"]["hi"] < 1)
            entry["calibration_flag"] = ("flagged" if off else "undetermined") if full else "unavailable"
        else:
            entry["calibration_flag"] = "not_testable"
        if not primary:
            entry["calibration_flag"] = NO_FLAG
        fits[s] = entry
    both = df[core.valid(df["D"]).values & core.valid(df["B26"]).values] if len(df) else df
    pd_ = _per_date(both, "D").join(_per_date(both, "B26"), how="inner")
    auc = (m.summary_bootstrap(pd_, {"D_minus_B26": lambda x: _nanmean(x["auc_D"]) - _nanmean(x["auc_B26"])},
                               n_resamples=n)["D_minus_B26"] if len(pd_) else {"status": "unavailable_no_support"})
    return {"component": "calibration (recalibration intercept/slope) and ranking (D−B26 AUC), reported separately",
            "flag": fits["D"]["calibration_flag"], "support": counts(df), "original_scores_carried_from_W12": carried,
            "recalibration": fits,
            "auc_D_minus_B26": {"estimate": (_nanmean(pd_["auc_D"]) - _nanmean(pd_["auc_B26"])) if len(pd_) else None,
                                "dates": int(len(pd_)), "interval": auc},
            "full_explanation": "undetermined (no predeclared material-loss/equivalence criterion)",
            "limits": "same-row fits are calibration diagnostics, not held-out corrections"}


# --- E6 ----------------------------------------------------------------------------------------------------------
def _spearman(x: pd.Series, y: pd.Series) -> float:
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return float("nan")
    return float(x[ok].rank().corr(y[ok].rank()))


def e6(prim: pd.DataFrame, venues: dict, drag: pd.DataFrame | None, n: int, primary: bool = True,
       drag_manifest_ok: bool = True) -> dict:
    """Registered-slate as-of drag (design E6). Venue membership is frozen from the common valid D/C-frozen rows before
    any outcome filtering; residuals use the known rows of that support. Only the D all-candidate correlation supplies
    the flag; C-frozen and D rank-1 are companions."""
    base = {"component": "registered-slate as-of drag vs D all-candidate per-date residual (Spearman)",
            "full_explanation": "not testable here (no independently dated regime witnesses in these inputs)",
            "limits": "association only; venue membership, shrinkage, source revisions, trends and temporal dependence"}
    if drag is None or not drag_manifest_ok:
        return {**base, "flag": "unavailable" if primary else NO_FLAG,
                "reason": "drag export or its sibling manifest missing"}
    pool = prim[core.valid(prim["D"]).values & core.valid(prim["C_frozen"]).values].copy() if len(prim) else prim.copy()
    pool["venue_id"] = pool["game_pk"].map(venues) if len(pool) else pd.Series(dtype=float)
    sd = core.slate_drag(pool[["date", "venue_id"]], drag)
    per = sd.join(_per_date(pool, "D"), how="left").join(_per_date(pool, "C_frozen"), how="left")
    w = _winners(prim, "D")
    wk = w[w["outcome"].isin(core.KNOWN)] if len(w) else w
    rank1 = pd.Series((wk["D"].astype(float).values - (wk["outcome"] == "hit").astype(int).values) if len(wk) else [],
                      index=wk["date"].values if len(wk) else [], name="resid_D_rank1", dtype=float)
    per = per.join(rank1, how="left")
    comp = per[per["complete"].astype(bool)] if len(per) else per
    stats = {"rho_D": lambda x: _spearman(x["drag_mean"], x["resid_D"]),
             "rho_C_frozen_companion": lambda x: _spearman(x["drag_mean"], x["resid_C_frozen"]),
             "rho_D_rank1_companion": lambda x: _spearman(x["drag_mean"], x["resid_D_rank1"])}
    est = {k: f(comp) for k, f in stats.items()} if len(comp) else {}
    boot = m.summary_bootstrap(comp, stats, n_resamples=n) if len(comp) >= 3 else {}
    flag = core.directional_flag(boot["rho_D"]) if boot else "unavailable"
    return {**base, "flag": _label(flag, "narrow_association_consistent", "contradicts_narrow_direction", primary),
            "support": {**counts(pool), "dates_complete": int(sd["complete"].astype(bool).sum()) if len(sd) else 0,
                        "dates_incomplete": int((~sd["complete"].astype(bool)).sum()) if len(sd) else 0,
                        "dates_used_D": int(comp["resid_D"].notna().sum()) if len(comp) else 0,
                        "conflicting_drag_keys": int(sd["conflicting_keys"].sum()) if len(sd) else 0},
            "estimates": est, "intervals": boot}


# --- E7 ----------------------------------------------------------------------------------------------------------
def e7(prim: pd.DataFrame, tab: pd.DataFrame, day_meta: list, n: int) -> dict:
    k = _known(prim, "D")
    pdD = _per_date(prim, "D")
    pooled = lambda x: (lambda a: float("nan") if a is None else float(a))(
        bridge._rank_auc(x.loc[x["y"] == 1, "D"], x.loc[x["y"] == 0, "D"]))
    rows_boot = m.block_bootstrap_many(k, {"pooled_auc": pooled}, n_resamples=n) if len(k) else {}
    wd = m.summary_bootstrap(pdD, {"within_auc": lambda x: _nanmean(x["auc_D"])}, n_resamples=n) if len(pdD) else {}
    w = _winners(prim, "D")
    wk = w[w["outcome"].isin(core.KNOWN)] if len(w) else w
    meta = pd.DataFrame(day_meta).set_index("date") if day_meta else pd.DataFrame(columns=["action", "n_rows"])
    allv = tab[tab["pool_verified"].fillna(False).astype(bool)]
    allv = allv.assign(action=allv["date"].map(meta["action"]).fillna("unknown"),
                       x12_window=allv["date"].between(*X12_WINDOW))
    by_action = {}
    for (act, x12), g in allv.groupby(["action", "x12_window"]):
        kk = _known(g, "D")
        by_action[f"{act}{'/x12_window' if x12 else ''}"] = {
            "dates": int(g["date"].nunique()), "rows": int(len(g)), "known": int(len(kk)),
            "outcomes": g["outcome"].value_counts().to_dict(), "mean_p": float(g["D"].mean()),
            "resid": float(kk["D"].mean() - kk["y"].mean()) if len(kk) else None}
    med = float(meta["n_rows"].median()) if len(meta) else float("nan")
    size = (prim["date"].map(meta["n_rows"]) > med) if len(prim) else pd.Series(dtype=bool)
    proj = prim["projected"].fillna(False).astype(bool) if len(prim) else pd.Series(dtype=bool)
    strata = {}
    for name, mask in (("projected", proj), ("confirmed", ~proj), ("slate_above_median", size),
                       ("slate_at_or_below_median", ~size)):
        sub = prim[mask.values] if len(prim) else prim
        kk = _known(sub, "D")
        strata[name] = {**counts(sub), "known": int(len(kk)), "mean_p": float(sub["D"].mean()) if len(sub) else None,
                        "resid": float(kk["D"].mean() - kk["y"].mean()) if len(kk) else None}
    return {"component": "selection/composition descriptives (no automatic label)", "flag": "descriptive",
            "support": counts(prim),
            "within_date_auc": {"estimate": _nanmean(pdD["auc_D"]) if len(pdD) else None,
                                "dates": int(pdD["auc_D"].notna().sum()) if len(pdD) else 0,
                                "interval": wd.get("within_auc")},
            "pooled_auc": {"estimate": pooled(k) if len(k) else None, "known_rows": int(len(k)),
                           "interval": rows_boot.get("pooled_auc")},
            "rank1_resid": float(wk["D"].astype(float).mean() - (wk["outcome"] == "hit").mean()) if len(wk) else None,
            "rank1_known_dates": int(len(wk)),
            "all_candidate_resid": float(k["D"].mean() - k["y"].mean()) if len(k) else None,
            "by_action_all_dates": by_action, "strata_primary": strata, "slate_size_median_rows": med,
            "limits": "action is the recommendation, not contest entry; the X-12 window label is a date range, not "
                      "evidence of research-only selection provenance"}


# --- E8 ----------------------------------------------------------------------------------------------------------
def selected_rows(pop: pd.DataFrame, selections: dict) -> tuple[pd.DataFrame, dict]:
    """Rows of ``pop`` that are a genuine selected primary with a serialized-probability match; statuses counted."""
    rows, status = [], Counter()
    for d, sel in selections.items():
        if sel.get("selected") is None:
            status[f"no_selection:{sel.get('action')}"] += 1
            continue
        g = pop[(pop["date"] == d) & (pop["batter_id"] == sel["selected"][0]) & (pop["game_pk"] == sel["selected"][1])]
        if len(g) != 1:
            status["not_in_population"] += 1
            continue
        r = g.iloc[0]
        if not core._valid1(r["D"]):
            status["invalid_D"] += 1
            continue
        if sel.get("p") is None or core.serialized(float(sel["p"])) != r["D"]:
            status["p_mismatch"] += 1
            continue
        status[f"matched:{r['outcome']}"] += 1
        rows.append(r)
    return (pd.DataFrame(rows) if rows else pop.iloc[0:0]), dict(status)


def e8(pop: pd.DataFrame, selections: dict, n: int, primary: bool = True, seed: int = 20261004) -> dict:
    sel, status = selected_rows(pop, selections)
    use = sel[sel["outcome"].isin(core.KNOWN)] if len(sel) else sel
    base = {"component": "selected-primary matched-probability null (independent Bernoulli at served p)",
            "support": status,
            "limits": "assumes calibrated served p and conditional independence across dates; not a luck verdict"}
    if not len(use):
        return {**base, "flag": "not_testable" if primary else NO_FLAG}
    rng = np.random.default_rng(seed)
    sims = (rng.uniform(size=(n, len(use))) < use["D"].astype(float).to_numpy()).sum(axis=1)
    obs = int((use["outcome"] == "hit").sum())
    lo, hi = np.percentile(sims, [2.5, 97.5])
    flag = "null_compatible" if lo <= obs <= hi else "null_incompatible"
    return {**base, "flag": flag if primary else NO_FLAG, "n": int(len(use)), "observed_hits": obs,
            "expected_hits": float(use["D"].astype(float).sum()), "envelope95": [float(lo), float(hi)]}


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


# --- inputs --------------------------------------------------------------------------------------------------------
def validate_inventory(tab: pd.DataFrame, summary: dict) -> list[str]:
    """The accepted run's registered inventory (code review r1 F8): unique registered dates, table rows only on those
    dates, unique candidate keys, and day metadata for exactly those dates."""
    dates = list(summary["registered_dates"])
    if len(set(dates)) != len(dates):
        raise SystemExit("registered_dates contains duplicates")
    extra = set(tab["date"]) - set(dates)
    if extra:
        raise SystemExit(f"table rows outside the registered dates: {sorted(extra)[:5]}")
    if tab.duplicated(["date", "batter_id", "game_pk"]).any():
        raise SystemExit("duplicate (date, batter_id, game_pk) candidate keys")
    if {d["date"] for d in summary["day_meta"]} != set(dates):
        raise SystemExit("day_meta dates differ from registered_dates")
    return dates


def load_selections(data: Path, dates: list[str], manifest: dict) -> tuple[dict, dict]:
    """Decision and pick bytes read once and verified against the W1.2 manifest before use (code review r1 F7). A
    decision that is expected but missing, or present but unmanifested, makes the date's selection unavailable (no
    pick fallback); an unmanifested pick is not used."""
    out, files = {}, {}
    for d in dates:
        docs, status = {}, []
        for kind, path, want in (("decision", data / "picks" / d / "decision.json", manifest.get("decisions", {}).get(d)),
                                 ("pick", data / "picks" / f"{d}.json", manifest.get("picks", {}).get(d))):
            exists = path.exists()
            if want is None and exists:
                status.append(f"{kind}_unmanifested")
                continue
            if want is not None and not exists:
                status.append(f"{kind}_missing")
                continue
            if not exists:
                continue
            raw = path.read_bytes()
            got = _sha(raw)
            if got != want:
                raise SystemExit(f"{kind} file for {d} differs from the W1.2 manifest")
            files[f"{kind}:{d}"] = got
            docs[kind] = json.loads(raw)
        if {"decision_missing", "decision_unmanifested"} & set(status):
            sel = {"selected": None, "declined": None, "action": "unavailable", "source": None, "p": None}
        else:
            sel = core.selected_primary(docs.get("decision"), docs.get("pick"))
        dk = core._key((docs.get("decision") or {}).get("primary"))
        pk = core._key((docs.get("pick") or {}).get("pick"))
        sel["evidence_status"] = status
        sel["decision_pick_conflict"] = bool(dk and pk and dk != pk)
        out[d] = sel
    return out, files


def per_test_table(results: dict) -> pd.DataFrame:
    """One row per reported interval (design §5 ``per_test.parquet``)."""
    rows = []

    def walk(obj, path):
        if isinstance(obj, dict):
            if "n_resamples" in obj and "status" in obj:
                rows.append({"path": "/".join(path), **{k: obj.get(k) for k in
                             ("lo", "hi", "conditional_lo", "conditional_hi", "n_ok", "n_failed", "status", "seed")}})
                return
            for k, v in obj.items():
                walk(v, path + [str(k)])
    walk(results, [])
    return pd.DataFrame(rows, columns=["path", "lo", "hi", "conditional_lo", "conditional_hi", "n_ok", "n_failed",
                                       "status", "seed"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w12-run", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--drag", type=Path, required=True, help="park_drag_export.csv (its sibling manifest is required)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    args = ap.parse_args(argv)
    head = x29_gate()
    w12 = args.w12_run.expanduser().resolve()
    data = args.data_root.expanduser().resolve()
    w12_bytes, w12_acc = bridge.read_accepted_run(w12)
    tab = pd.read_parquet(io.BytesIO(w12_bytes["table.parquet"]))
    summary = json.loads(w12_bytes["summary.json"])
    w12_manifest = json.loads(w12_bytes["manifest.json"])
    dates = validate_inventory(tab, summary)
    used: dict = {"w12_accepted_files": {k: _sha(v) for k, v in w12_bytes.items()}}

    selections, used["selection_files"] = load_selections(data, dates, w12_manifest)
    venues, feed_hashes = {}, {}
    for pk in sorted(set(tab["game_pk"].astype(int))):
        fp = data / "raw" / "2026" / f"{pk}.json"
        if fp.exists():
            raw = fp.read_bytes()
            feed_hashes[str(pk)] = _sha(raw)
            vid = json.loads(raw).get("gameData", {}).get("venue", {}).get("id")
            if isinstance(vid, int) and not isinstance(vid, bool):
                venues[pk] = vid
    used["feeds"] = feed_hashes
    drag, sib_ok = None, False
    if args.drag.exists():
        raw = args.drag.read_bytes()
        used["drag"] = {args.drag.name: _sha(raw)}
        drag = pd.read_csv(io.BytesIO(raw), dtype={"venue_id": "Int64"})
        drag["date"] = pd.to_datetime(drag["date"]).dt.strftime("%Y-%m-%d")
        sib = args.drag.parent / "park_drag_manifest.json"
        if sib.exists():
            used["drag"][sib.name] = _sha(sib.read_bytes())
            sib_ok = True

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out.expanduser().resolve() / f"{head[:7]}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log(f"run dir {run_dir}; code {head[:7]}; W1.2 run {w12.name}; {len(dates)} registered dates")

    prim = stratum(tab, *PRIMARY)
    n = args.n_resamples
    sel_prim, _ = selected_rows(prim, selections)
    results = {"schema": "w13_tests_v2", "code": head, "x29_commit": X29_COMMIT, "w12_run": str(w12),
               "w12_accepted": {k: v for k, v in w12_acc.items() if k != "files"},
               "registered_dates": len(dates), "primary": {"stratum": "/".join(PRIMARY), **counts(prim)},
               "interpretation": "pointwise unadjusted exploratory diagnostics; sensitivities never carry a flag",
               "selections": {"actions": dict(Counter(s["action"] for s in selections.values())),
                              "evidence_status": dict(Counter(x for s in selections.values() for x in s["evidence_status"])),
                              "decision_pick_conflicts": int(sum(s["decision_pick_conflict"] for s in selections.values()))}}
    steps = (("E2", lambda: e2(tab, summary)),
             ("E1", lambda: e1(prim, "pool_verified", summary, n, selected_rows=sel_prim)),
             ("E3", lambda: e3(prim, "pool_verified", n)), ("E4", lambda: e4(prim, tab, dates, n)),
             ("E5", lambda: e5(prim, summary, n)), ("E6", lambda: e6(prim, venues, drag, n, drag_manifest_ok=sib_ok)),
             ("E7", lambda: e7(prim, tab, summary["day_meta"], n)), ("E8", lambda: e8(prim, selections, n)))
    for name, fn in steps:
        log(f"computing {name}")
        results[name] = fn()
    results["E8"]["sensitivity_all_registered_rows"] = e8(tab, selections, n, primary=False)
    results["E8"]["carried_brief_wilson_X09"] = {"96/141": wilson(96, 141), "79/109": wilson(79, 109),
                                                 "note": "previously consumed brief counts (X-09); carried context, not this window"}
    results["sensitivities"] = {}
    for key, (sel, pool) in SENSITIVITIES.items():
        log(f"sensitivity {key}")
        s = stratum(tab, sel, pool)
        ck = "all_dates" if sel is None else "selection_consistent"
        results["sensitivities"][key] = {
            "support": counts(s), "E1": e1(s, pool, summary, n, primary=False, cohort_key=ck),
            "E3": e3(s, pool, n, primary=False), "E4": e4(s, tab, dates, n, primary=False),
            "E5": e5(s, summary, n, primary=False, cohort_key=ck, pool=pool),
            "E6": e6(s, venues, drag, n, primary=False, drag_manifest_ok=sib_ok)}
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    per_test_table(results).to_parquet(run_dir / "per_test.parquet", index=False)
    (run_dir / "manifest.json").write_text(json.dumps(used, indent=1) + "\n")
    log(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
