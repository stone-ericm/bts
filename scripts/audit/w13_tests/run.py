"""W1.3 distinguishing tests driver (design docs/superpowers/specs/2026-10-04-w13-distinguishing-tests-design.md).

Reads the accepted W1.2 run (table, summary, manifest), the manifest-hashed decision and pick files, the archived final
feeds (venue only) and the as-of park-drag export. Writes component diagnostics: each carries its measured component,
an unadjusted pointwise flag, the full explanation's status, supports and intervals. Refuses to run until exposure row
X-29 is in HEAD."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
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


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _known(df: pd.DataFrame, score: str) -> pd.DataFrame:
    k = df[df["outcome"].isin(core.KNOWN) & np.isfinite(df[score].astype(float))]
    return k.assign(y=(k["outcome"] == "hit").astype(int))


def _per_date(df: pd.DataFrame, score: str) -> pd.DataFrame:
    """Per-date residual (mean p − mean y) and tie-aware AUC over known finite rows of ``score``."""
    rows = []
    for d, g in _known(df, score).groupby("date", sort=True):
        a = bridge._rank_auc(g.loc[g["y"] == 1, score], g.loc[g["y"] == 0, score])
        rows.append({"date": d, f"resid_{score}": float(g[score].mean() - g["y"].mean()),
                     f"auc_{score}": float("nan") if a is None else float(a), f"n_{score}": int(len(g))})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def _nanmean(s: pd.Series) -> float:
    return float(s.mean()) if s.notna().any() else float("nan")


def _label(flag: str, positive: str, negative: str) -> str:
    return {"consistent": positive, "contradicting": negative}.get(flag, flag)


# --- E1 ----------------------------------------------------------------------------------------------------------
def e1(prim: pd.DataFrame, summary: dict, n: int) -> dict:
    chain = []
    for pr in summary["metrics"]["selection_consistent"]["pool_verified"]["pairs"]:
        a, b = pr["pair"]
        if "top1_diff_b_minus_a" in pr:
            est, (lo, hi) = core.from_bridge_b_minus_a(pr["top1_diff_b_minus_a"], tuple(pr["top1_diff_ci95"]))
            chain.append({"pair": [a, b], "G": est, "G_ci95": [lo, hi], "paired_dates": pr["paired_top1_dates"],
                          "rank1_changed_dates": pr["rank1_changed_dates"], "discordant": pr.get("discordant_dates"),
                          "exclusions": pr["exclusions"], "composite": True})
        else:
            chain.append({"pair": [a, b], "G": None, "reason": "no paired top-1 dates", "exclusions": pr["exclusions"]})
    g = core.paired_reduction(prim, "A26", "B26", "pool_verified", n_resamples=n)
    flag = core.directional_flag(g["interval"]) if g.get("available") else "unavailable"
    native = {}
    for s in SURFACES:
        w = m.top1(prim[np.isfinite(prim[s].astype(float))].assign(_block=lambda x: x["date"]), s) \
            if np.isfinite(prim[s].astype(float)).any() else pd.DataFrame()
        native[s] = ({"n_winners": int(len(w)), "scoring_n_pa_mean": float(w["n_pa"].mean()),
                      "scoring_n_pa_ge5": float((w["n_pa"] >= 5).mean())} if len(w) else {"n_winners": 0})
    csd = next((c for c in chain if c["pair"] == ["C_served", "D"]), None)
    return {"component": "restricted optimism reduction G(A26,B26) on its common pool",
            "flag": _label(flag, "consistent", "weakened"), "G_A26_B26": g, "chain_carried_from_W12": chain,
            "surviving_C_served_D": csd, "native_winner_scoring_pa": native,
            "limits": "oracle diagnostic; composite chain; no numerical attribution to the headline gap"}


# --- E2 ----------------------------------------------------------------------------------------------------------
def e2(tab: pd.DataFrame) -> dict:
    sc = tab[tab["sel_state"] == "selection_consistent"].copy()
    sc["repro"] = [core.repro_class(r) for r in sc[["C_served", "C_served_wblank", "D"]].to_dict("records")]
    rep = sc[sc["repro"] != "not_reproducible"]
    resid = rep["C_served"].map(core.serialized) - rep["D"]
    counts = sc["repro"].value_counts().to_dict()
    share = float((rep["repro"] == "unexplained").mean()) if len(rep) else float("nan")
    by = lambda col: {str(k): g["repro"].value_counts().to_dict() for k, g in rep.groupby(col)}
    flag = ("not_testable_empty_support" if not len(rep) else
            "unexplained_reconstruction_gap" if share >= UNEXPLAINED_FLAG_SHARE else "numerical_parity_conditional")
    return {"component": "numerical reconstruction of served scores (C-served vs D)", "flag": flag,
            "classes": {str(k): int(v) for k, v in counts.items()}, "reproducible_rows": int(len(rep)),
            "unexplained_share": share,
            "signed_resid_quantiles": {q: float(resid.quantile(q)) for q in (0.01, 0.5, 0.99)} if len(rep) else {},
            "abs_resid_quantiles": {q: float(resid.abs().quantile(q)) for q in (0.5, 0.9, 1.0)} if len(rep) else {},
            "by_month": by(rep["date"].str[:7]), "by_projected": by(rep["projected"].astype(str)),
            "by_pool": {p: rep.loc[rep[p].fillna(False).astype(bool), "repro"].value_counts().to_dict()
                        for p in ("pool_verified", "pool_surrogate", "pool_all")},
            "full_explanation": "historical feature age, lineup/pitcher changes and causal staleness not testable here",
            "limits": "a reconstruction gap is not measured live drift; M3 (X-03) carried, not rerun"}


# --- E3 ----------------------------------------------------------------------------------------------------------
def e3(prim: pd.DataFrame, n: int) -> dict:
    g = core.paired_reduction(prim, "A26", "A26_count", "pool_verified", n_resamples=n)
    comp = core.paired_reduction(prim, "A26_count", "B26", "pool_verified", n_resamples=n)
    common = prim[np.isfinite(prim["A26"].astype(float)) & np.isfinite(prim["A26_count"].astype(float))]
    flag = core.directional_flag(g["interval"]) if g.get("available") else "unavailable"
    return {"component": "oracle realized-count contribution G(A26, A26_count)",
            "flag": "oracle_count_consistent" if flag == "consistent" else flag,
            "G_A26_A26count": g, "G_A26count_B26_composite": comp,
            "score_change_A26_minus_count": float((common["A26"] - common["A26_count"]).mean()) if len(common) else None,
            "scoring_n_pa_vs_A26_n_pas_actual_mismatch_rows": int((common["n_pa"] != common["n_pas_actual"]).sum())
            if "n_pas_actual" in common else None,
            "full_explanation": "not testable here (no at-lock PA forecast is archived)",
            "limits": "oracle; never an achievable improvement; zero change does not weaken forecast error"}


# --- E4 ----------------------------------------------------------------------------------------------------------
def e4(prim: pd.DataFrame, tab: pd.DataFrame, dates: list[str], n: int) -> dict:
    """Half contrasts (design E4). The flag comes only from the C-frozen residual contrast on reproduced support:
    primary rows with finite C-frozen and an E2 class of exact or inferred-weather; each half resamples only its dates
    with at least one known supported row, at that effective count. Full-primary contrasts are descriptions."""
    first, second = core.half_split(dates)
    sc = tab[tab["sel_state"] == "selection_consistent"]
    repro = {tuple(k): core.repro_class(r) for k, r in
             zip(sc[["date", "batter_id", "game_pk"]].itertuples(index=False),
                 sc[["C_served", "C_served_wblank", "D"]].to_dict("records"))}
    cls = pd.Series([repro.get(k) for k in zip(prim["date"], prim["batter_id"], prim["game_pk"])], index=prim.index)
    rsup = prim[cls.isin(["exact_final_feed", "inferred_weather_absent_at_serve"])
                & np.isfinite(prim["C_frozen"].astype(float))]
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
    support = {h: {"dates": len(hd), "rows": int(rsup["date"].isin(hd).sum()),
                   "classes": cls[rsup.index][rsup["date"].isin(hd)].value_counts().to_dict()}
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
            "flag": _label(flag, "increased_overprediction_conditional", "reduced_overprediction"),
            "halves": {"first": [first[0], first[-1], len(first)], "second": [second[0], second[-1], len(second)]},
            "estimates": point, "intervals": boot, "reproduced_support": support,
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
    out = {}
    for k, v in (("intercept", a), ("slope", b)):
        ok = v[np.isfinite(v)]
        out[k] = {"lo": float(np.percentile(ok, 2.5)) if len(ok) else float("nan"),
                  "hi": float(np.percentile(ok, 97.5)) if len(ok) else float("nan"),
                  "n_ok": int(len(ok)), "n_failed": int(n - len(ok)), "seed": seed, "n_resamples": n}
    out["failure_reasons"] = fails
    return out


def e5(prim: pd.DataFrame, summary: dict, n: int) -> dict:
    carried = {s["surface"]: {k: s.get(k) for k in ("brier", "log_loss", "mean_stated_minus_realized", "reliability",
                                                     "n_known_rows")}
               for s in summary["metrics"]["selection_consistent"]["pool_verified"]["surfaces"]}
    fits = {}
    for s in SURFACES:
        k = _known(prim, s)
        base = core.recalibration_fit(k[s].to_numpy(float), k["y"].to_numpy(int)) if len(k) else {"valid": False, "reason": "empty"}
        entry = {"base": base}
        if base["valid"]:
            iv = _refit_bootstrap(k, s, n)
            entry["intervals"] = iv
            full = iv["intercept"]["n_failed"] == 0
            off = full and (iv["intercept"]["lo"] > 0 or iv["intercept"]["hi"] < 0 or iv["slope"]["lo"] > 1 or iv["slope"]["hi"] < 1)
            entry["calibration_flag"] = "flagged" if off else ("undetermined" if full else "unavailable")
        else:
            entry["calibration_flag"] = "not_testable"
        fits[s] = entry
    both = prim[np.isfinite(prim["D"].astype(float)) & np.isfinite(prim["B26"].astype(float))]
    pd_ = _per_date(both, "D").join(_per_date(both, "B26"), how="inner")
    auc = m.summary_bootstrap(pd_, {"D_minus_B26": lambda x: _nanmean(x["auc_D"]) - _nanmean(x["auc_B26"])},
                              n_resamples=n)["D_minus_B26"] if len(pd_) else {"n_ok": 0}
    return {"component": "calibration (recalibration intercept/slope) and ranking (D−B26 AUC), reported separately",
            "flag": fits["D"]["calibration_flag"], "original_scores_carried_from_W12": carried, "recalibration": fits,
            "auc_D_minus_B26": {"estimate": (_nanmean(pd_["auc_D"]) - _nanmean(pd_["auc_B26"])) if len(pd_) else None,
                                "interval": auc},
            "full_explanation": "undetermined (no predeclared material-loss/equivalence criterion)",
            "limits": "same-row fits are calibration diagnostics, not held-out corrections"}


# --- E6 ----------------------------------------------------------------------------------------------------------
def _spearman(x: pd.Series, y: pd.Series) -> float:
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return float("nan")
    return float(x[ok].rank().corr(y[ok].rank()))


def e6(prim: pd.DataFrame, venues: dict, drag: pd.DataFrame, n: int) -> dict:
    """Registered-slate as-of drag (design E6). Venue membership is frozen from the primary pool's common finite
    D/C-frozen rows before any outcome filtering; residuals use the known rows of that support. Only the D
    all-candidate correlation supplies the flag; C-frozen and D rank-1 are companions."""
    pool = prim[np.isfinite(prim["D"].astype(float)) & np.isfinite(prim["C_frozen"].astype(float))].copy()
    pool["venue_id"] = pool["game_pk"].map(venues)
    sd = core.slate_drag(pool[["date", "venue_id"]], drag)
    per = sd.join(_per_date(pool, "D"), how="left").join(_per_date(pool, "C_frozen"), how="left")
    w = m.top1(prim[np.isfinite(prim["D"].astype(float))].assign(_block=lambda x: x["date"]), "D")
    wk = w[w["outcome"].isin(core.KNOWN)]
    per = per.join(pd.Series(wk["D"].values - (wk["outcome"] == "hit").astype(int).values,
                             index=wk["date"].values, name="resid_D_rank1"), how="left")
    comp = per[per["complete"]]
    stats = {"rho_D": lambda x: _spearman(x["drag_mean"], x["resid_D"]),
             "rho_C_frozen_companion": lambda x: _spearman(x["drag_mean"], x["resid_C_frozen"]),
             "rho_D_rank1_companion": lambda x: _spearman(x["drag_mean"], x["resid_D_rank1"])}
    est = {k: f(comp) for k, f in stats.items()}
    boot = m.summary_bootstrap(comp, stats, n_resamples=n) if len(comp) >= 3 else {}
    flag = core.directional_flag(boot["rho_D"]) if boot else "unavailable"
    return {"component": "registered-slate as-of drag vs D all-candidate per-date residual (Spearman)",
            "flag": _label(flag, "narrow_association_consistent", "contradicts_narrow_direction"),
            "dates_complete": int(sd["complete"].sum()), "dates_incomplete": int((~sd["complete"]).sum()),
            "estimates": est, "intervals": boot,
            "full_explanation": "not testable here (no independently dated regime witnesses in these inputs)",
            "limits": "association only; venue membership, shrinkage, source revisions, trends and temporal dependence"}


# --- E7 ----------------------------------------------------------------------------------------------------------
def e7(prim: pd.DataFrame, tab: pd.DataFrame, day_meta: list, n: int) -> dict:
    k = _known(prim, "D")
    pdD = _per_date(prim, "D")
    pooled = lambda x: (lambda a: float("nan") if a is None else float(a))(
        bridge._rank_auc(x.loc[x["y"] == 1, "D"], x.loc[x["y"] == 0, "D"]))
    rows_boot = m.block_bootstrap_many(k, {"pooled_auc": pooled}, n_resamples=n) if len(k) else {}
    wd = m.summary_bootstrap(pdD, {"within_auc": lambda x: _nanmean(x["auc_D"])}, n_resamples=n) if len(pdD) else {}
    w = m.top1(prim[np.isfinite(prim["D"].astype(float))].assign(_block=lambda x: x["date"]), "D")
    wk = w[w["outcome"].isin(core.KNOWN)]
    meta = pd.DataFrame(day_meta).set_index("date")
    allv = tab[tab["pool_verified"].fillna(False).astype(bool)]
    allv = allv.assign(action=allv["date"].map(meta["action"]).fillna("unknown"),
                       x12=allv["date"].between(*X12_WINDOW))
    by_action = {}
    for (act, x12), g in allv.groupby(["action", "x12"]):
        kk = _known(g, "D")
        by_action[f"{act}{'/x12' if x12 else ''}"] = {"dates": int(g["date"].nunique()), "rows": int(len(g)),
                                                      "known": int(len(kk)), "mean_p": float(g["D"].mean()),
                                                      "resid": float(kk["D"].mean() - kk["y"].mean()) if len(kk) else None}
    med = float(meta["n_rows"].median())
    size = prim["date"].map(meta["n_rows"]) > med
    strata = {}
    for name, mask in (("projected", prim["projected"].fillna(False).astype(bool)),
                       ("confirmed", ~prim["projected"].fillna(False).astype(bool)),
                       ("slate_above_median", size), ("slate_at_or_below_median", ~size)):
        kk = _known(prim[mask], "D")
        strata[name] = {"rows": int(mask.sum()), "known": int(len(kk)), "mean_p": float(prim.loc[mask, "D"].mean()),
                        "resid": float(kk["D"].mean() - kk["y"].mean()) if len(kk) else None}
    return {"component": "selection/composition descriptives (no automatic label)", "flag": "descriptive",
            "within_date_auc": {"estimate": _nanmean(pdD["auc_D"]) if len(pdD) else None, "interval": wd.get("within_auc")},
            "pooled_auc": {"estimate": pooled(k) if len(k) else None, "interval": rows_boot.get("pooled_auc")},
            "rank1_resid": float(wk["D"].mean() - (wk["outcome"] == "hit").mean()) if len(wk) else None,
            "all_candidate_resid": float(k["D"].mean() - k["y"].mean()) if len(k) else None,
            "by_action_all_dates": by_action, "strata_primary": strata, "slate_size_median_rows": med,
            "limits": "action is the recommendation, not contest entry"}


# --- E8 ----------------------------------------------------------------------------------------------------------
def e8(tab: pd.DataFrame, selections: dict, n: int, seed: int = 20261004) -> dict:
    rows = []
    for d, sel in selections.items():
        if sel["selected"] is None:
            continue
        g = tab[(tab["date"] == d) & (tab["batter_id"] == sel["selected"][0]) & (tab["game_pk"] == sel["selected"][1])]
        if len(g) != 1:
            rows.append({"date": d, "status": "no_unique_slate_row"})
            continue
        r = g.iloc[0]
        status = "matched" if sel["p"] is not None and core.serialized(float(sel["p"])) == r["D"] else "p_mismatch"
        rows.append({"date": d, "status": status, "p": float(r["D"]), "outcome": r["outcome"]})
    df = pd.DataFrame(rows)
    use = df[(df["status"] == "matched") & df["outcome"].isin(core.KNOWN)] if len(df) else df
    if not len(use):
        return {"component": "selected-primary matched-probability null", "flag": "not_testable",
                "support": df["status"].value_counts().to_dict() if len(df) else {}}
    rng = np.random.default_rng(seed)
    sims = (rng.uniform(size=(n, len(use))) < use["p"].to_numpy()).sum(axis=1)
    obs = int((use["outcome"] == "hit").sum())
    lo, hi = np.percentile(sims, [2.5, 97.5])
    return {"component": "selected-primary matched-probability null (independent Bernoulli at served p)",
            "flag": "null_compatible" if lo <= obs <= hi else "null_incompatible",
            "n": int(len(use)), "observed_hits": obs, "expected_hits": float(use["p"].sum()),
            "envelope95": [float(lo), float(hi)], "support": df["status"].value_counts().to_dict(),
            "limits": "assumes calibrated served p and conditional independence across dates; not a luck verdict"}


def wilson(k: int, n: int, z: float = 1.959964) -> list:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--w12-run", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--drag", type=Path, required=True, help="park_drag_export.csv (its sibling manifest is hashed)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-resamples", type=int, default=10_000)
    args = ap.parse_args(argv)
    head = x29_gate()
    w12 = args.w12_run.expanduser().resolve()
    data = args.data_root.expanduser().resolve()
    tab = pd.read_parquet(w12 / "table.parquet")
    summary = json.loads((w12 / "summary.json").read_text())
    w12_manifest = json.loads((w12 / "manifest.json").read_text())
    dates = list(summary["registered_dates"])
    used: dict = {"w12": {f: sha(w12 / f) for f in ("table.parquet", "summary.json", "manifest.json")}}

    selections, sel_files = {}, {}
    for d in dates:
        dp, pp = data / "picks" / d / "decision.json", data / "picks" / f"{d}.json"
        dec = json.loads(dp.read_bytes()) if dp.exists() else None
        pick = json.loads(pp.read_bytes()) if pp.exists() else None
        for kind, path, want in (("decision", dp, w12_manifest.get("decisions", {}).get(d)),
                                 ("pick", pp, w12_manifest.get("picks", {}).get(d))):
            if path.exists():
                got = sha(path)
                sel_files[f"{kind}:{d}"] = got
                if want is not None and got != want:
                    raise SystemExit(f"{kind} file for {d} differs from the W1.2 manifest")
        selections[d] = core.selected_primary(dec, pick)
    used["selection_files"] = sel_files

    venues, feed_hashes = {}, {}
    for pk in sorted(set(tab["game_pk"].astype(int))):
        fp = data / "raw" / "2026" / f"{pk}.json"
        if fp.exists():
            raw = fp.read_bytes()
            feed_hashes[str(pk)] = hashlib.sha256(raw).hexdigest()
            vid = json.loads(raw).get("gameData", {}).get("venue", {}).get("id")
            if vid is not None:
                venues[pk] = int(vid)
    used["feeds"] = feed_hashes
    drag = pd.read_csv(args.drag, dtype={"venue_id": "Int64"})
    drag["date"] = pd.to_datetime(drag["date"]).dt.strftime("%Y-%m-%d")
    used["drag"] = {args.drag.name: sha(args.drag)}
    sib = args.drag.parent / "park_drag_manifest.json"
    if sib.exists():
        used["drag"][sib.name] = sha(sib)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.out.expanduser().resolve() / f"{head[:7]}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log(f"run dir {run_dir}; code {head[:7]}; W1.2 run {w12.name}; {len(dates)} registered dates")

    prim = tab[(tab["sel_state"] == "selection_consistent") & tab["pool_verified"].fillna(False).astype(bool)]
    n = args.n_resamples
    results = {"schema": "w13_tests_v1", "code": head, "x29_commit": X29_COMMIT, "w12_run": str(w12),
               "registered_dates": len(dates), "primary_rows": int(len(prim)),
               "primary_dates": int(prim["date"].nunique()), "interpretation": "pointwise unadjusted exploratory diagnostics",
               "selections": pd.Series([s["action"] if s["selected"] or s["action"] == "skip" else "none"
                                        for s in selections.values()]).value_counts().to_dict()}
    for name, fn in (("E2", lambda: e2(tab)), ("E1", lambda: e1(prim, summary, n)), ("E3", lambda: e3(prim, n)),
                     ("E4", lambda: e4(prim, tab, dates, n)), ("E5", lambda: e5(prim, summary, n)),
                     ("E6", lambda: e6(prim, venues, drag, n)), ("E7", lambda: e7(prim, tab, summary["day_meta"], n)),
                     ("E8", lambda: e8(tab, selections, n))):
        log(f"computing {name}")
        results[name] = fn()
    results["E8"]["carried_brief_wilson_X09"] = {"96/141": wilson(96, 141), "79/109": wilson(79, 109),
                                                 "note": "previously consumed brief counts (X-09); carried context, not this window"}
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    (run_dir / "manifest.json").write_text(json.dumps(used, indent=1) + "\n")
    log(f"done: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
