"""C1 4b historical screen: orchestration (registration `docs/sota_audit/2026-10-04-prereg-c1-longest-streak-policy.md`).

The run proceeds in this order:
1. **X-31 gate.**
2. **Outcome-free provenance manifest:** profile bytes against the retained W0.1 manifest, the A0 digests and their
   pairing, the schedules and calendars, the code commit and `uv.lock`.
3. **July parity at Δ = 0** through the original module; a failure stops the run.
4. **Profiles decoded and validated.**
5. **Five leave-one-season-out folds:** fit A1/A2, replay every arm under the Δ grid with shared nested masks, and
   compute the decision consequences and the fixed-policy projections on the common refinement.
6. **The final all-season decision object** and its exposed-fit projections on every calendar.
7. **Aggregation** (seeds within season, then seasons equally) and the §6 disposition with the Monte Carlo
   precedence rule.

Every 403/429-free, launcher-run execution writes into a fresh run directory under
`data/hetzner_results/c1/r4b/runs/` (restic-backed).

    python -m scripts.audit.c1_r4b.run --schedules ~/projects/bts/data/hetzner_results/c1/r3/schedules
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit.c1_r4b import data as D
from scripts.audit.c1_r4b import fit as F
from scripts.audit.c1_r4b import project as P
from scripts.audit.c1_r4b import replay as R
from scripts.audit.c1_r4b import solvers as S

REPO = Path(__file__).resolve().parents[3]
X31_COMMIT: str | None = None          # the register commit that publishes X-31; set before any execution
SEASONS = (2021, 2022, 2023, 2024, 2025)
DELTAS = (0.0, 0.05, 0.10, 0.139)
REPS = 200
C_GRID = (0.0, 0.02, 0.04)
H_GRID = (0.0, 0.02, 0.04)
PROJ_DELTAS = (0.0, 0.10)
LATE_DAYS = 30
A0_BASE_SHA = "66d154717ae51afb3343ee4bec8138c60bd1056e46a3de449043f4e9f76b93b4"
A0_TAIL_SHA = "dc5d0c9924431c11c43e720bf9e02903b709a157c0ebb699fdea43e7aaf949f4"
W0_MANIFEST_SHA = "d63fefffff0a8868de4d21c1aa680e352f9ee4badff158bd2a8a01c5e359e7c6"
DATA = Path.home() / "projects" / "bts" / "data"


class ProvenanceError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- pure logic
def x31_gate() -> str:
    if X31_COMMIT is None:
        raise SystemExit("X-31 gate: X31_COMMIT is unset (publish X-31 first)")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    if subprocess.run(["git", "-C", str(REPO), "merge-base", "--is-ancestor", X31_COMMIT, head]).returncode != 0:
        raise SystemExit(f"X-31 gate: {X31_COMMIT} is not an ancestor of HEAD {head[:7]}")
    if "| X-31 |" not in (REPO / "docs/audit/2026-09-22-exposure-register.md").read_text():
        raise SystemExit("X-31 gate: the register in this checkout has no X-31 row")
    return head


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify_profiles(root: Path, manifest: Path, files: list[Path]) -> dict:
    """Every consumed profile's bytes must equal its line in the retained W0.1 manifest."""
    want = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        rel = rel.strip().removeprefix("./").removeprefix(f"{root.name}/")
        want[rel] = digest
    got = {}
    for f in files:
        rel = str(Path(f).relative_to(root))
        h = sha256(Path(f).read_bytes())
        if want.get(rel) != h:
            raise ProvenanceError(f"profile {rel}: sha256 {h[:12]} does not match the W0.1 manifest")
        got[rel] = h
    return got


def mc_se(per_rep: np.ndarray) -> float:
    per_rep = np.asarray(per_rep, float)
    return float(np.std(per_rep, ddof=1) / np.sqrt(per_rep.size)) if per_rep.size > 1 else 0.0


def equal_season_mean(per_season_values: dict) -> float:
    return float(np.mean([np.mean(v) for v in per_season_values.values()]))


def _ambiguous(stat: float, threshold: float, se: float) -> bool:
    """§4: a stochastic gate within two Monte Carlo SEs of its threshold is inconclusive; a sampled zero SE at exact
    equality is too."""
    return abs(stat - threshold) <= 2.0 * se


def disposition(sm: dict) -> dict:
    """§6 with the r2 precedence: coverage/validation/MC ambiguity -> inconclusive first; then positive (all six);
    then negative (a mean contrast <= 0 at Δ=0, or a registered guardrail breached); else inconclusive."""
    reasons = []
    con, r20 = sm["contrast"], sm["reach20"]
    if not sm["coverage_complete"]:
        reasons.append("incomplete coverage (unknown playable dates)")
    if not sm["validation_ok"]:
        reasons.append("a required validation/provenance check failed")
    amb = []
    for k in ("objective", "switch"):
        c = con[k]["d10"]
        if _ambiguous(c["mean"], 0.0, c["mc_se"]):
            amb.append(f"{k} contrast at Δ=0.10 within the MC band")
    for b in ("A1", "A0"):
        diff = r20["A2"]["d10"] - r20[b]["d10"]
        if _ambiguous(diff, -0.02, r20["mc_se_d10"][b]):
            amb.append(f"reach-20 vs {b} at Δ=0.10 within the MC band")
    reasons += amb
    if reasons:
        return {"disposition": "inconclusive", "reasons": reasons}
    g = {
        "g1_mean_bar": all(con[k]["d0"]["mean"] >= 0.25 for k in ("objective", "switch")),
        "g2_four_of_five": all(sum(x > 0 for x in con[k]["d0"]["seasons"]) >= 4 for k in ("objective", "switch")),
        "g3_robust_d10": all(con[k]["d10"]["mean"] >= 0 for k in ("objective", "switch")),
        "g4_reach20": all(r20["A2"][dd] >= r20[b][dd] - 0.02 for dd in ("d0", "d10") for b in ("A1", "A0")),
        "g5_no_season_below": all(min(con[k]["d0"]["seasons"]) >= -0.25 for k in ("objective", "switch")),
        "g6_validation_projections": bool(sm["validation_ok"] and sm["projections_ok"]),
    }
    if all(g.values()):
        return {"disposition": "positive", "gates": g}
    neg = (any(con[k]["d0"]["mean"] <= 0 for k in ("objective", "switch"))
           or not g["g4_reach20"] or not g["g5_no_season_below"])
    return {"disposition": "negative" if neg else "inconclusive", "gates": g}


# ----------------------------------------------------------------------------------------------- environments
def refined_environment(season_days: list[dict], fold_cuts: np.ndarray, a0_bounds, *, late_days: int = LATE_DAYS):
    """The common refinement (§5): types by phase × A0 base bin × fold bin × partner availability, with primary and
    joint rates from the same stratum. q indexes `pairs` = [(a0_bin, fold_bin)]. Zero-frequency cells are absent."""
    pairs, phases = [], {}
    for name, late in (("early", False), ("late", True)):
        rows = []
        for d in season_days:
            k = d["opp"] & d["known"] & ((d["d_raw"] <= late_days) if late else (d["d_raw"] > late_days))
            rows.append(pd.DataFrame({"p1": d["p1"][k], "hit1": d["hit1"][k], "partner": d["partner"][k],
                                      "hit2": d["hit2"][k]}))
        df = pd.concat(rows, ignore_index=True)
        df["q0"] = [R._bin_ge(p, a0_bounds) for p in df["p1"]]
        df["qf"] = F.classify(df["p1"].to_numpy(), fold_cuts)
        n, types = len(df), []
        for (q0, qf, partner), g in df.groupby(["q0", "qf", "partner"]):
            if (q0, qf) not in pairs:
                pairs.append((int(q0), int(qf)))
            ph = float(g["hit1"].mean())
            pb = float((g["hit1"] & g["hit2"]).mean()) if partner else 0.0
            types.append(S.DayType(freq=len(g) / n, q=pairs.index((int(q0), int(qf))), partner=bool(partner),
                                   p_hit=ph, p_both=pb))
        phases[name] = tuple(types)
    return S.Environment(n_bins=len(pairs), early=phases["early"], late=phases["late"]), pairs


def projection_policies(a0: R.A0, reach: S.Solution, emax: S.Solution, pairs, target=S.TARGET):
    """(d, q) -> (T+1, T+1, 2) raw-action tables for A0, A1 and A2, each through its own classifier (pairs[q])."""
    T1 = target + 1
    Sg, Mg, SVg = np.meshgrid(np.arange(T1), np.arange(T1), np.arange(2), indexing="ij")

    def a2(d, q):
        return emax.policy[:, :, min(d, emax.horizon), :, pairs[q][1]]

    def a1(d, q):
        dd = min(d, emax.horizon)
        return np.where(Sg + 2 * dd >= target, reach.policy[:, :, dd, :, pairs[q][1]], emax.policy[:, :, dd, :, pairs[q][1]])

    def a0f(d, q):
        q0 = pairs[q][0]
        if d <= 0:
            return np.zeros((T1, T1, 2), np.int8)
        d_eff = min(d, a0.base_season_length)
        base = a0.base[np.minimum(Sg, target - 1), min(d, a0.base_season_length), SVg, q0]
        tail = a0.tail[Sg, np.minimum(target, np.maximum(Sg, Mg)), min(d_eff, a0.tail.shape[2] - 1), SVg, 0]
        out = np.where(Sg + 2 * d_eff < target, tail, base)
        return np.where(Sg >= target, 0, out)

    return {"A0": a0f, "A1": a1, "A2": a2}


def project_grid(policies: dict, env: S.Environment, cal: D.Calendar) -> list[dict]:
    days = [(opp, left) for _, opp, left in cal.days()]
    out = []
    for delta in PROJ_DELTAS:
        for c in C_GRID:
            for h in H_GRID:
                st = P.Stress(c=c, h=h, delta=delta)
                cell = {"delta": delta, "c": c, "h": h}
                for arm, pol in policies.items():
                    r = P.project(pol, env, days, target=S.TARGET, saver_zone=S.SAVER_ZONE, late_days=LATE_DAYS,
                                  stress=st)
                    cell[arm] = {"p57": r["p_reach"], "e_best": r["e_best"]}
                cell["A2_minus_A0_p57"] = cell["A2"]["p57"] - cell["A0"]["p57"]
                cell["A2_minus_A1_p57"] = cell["A2"]["p57"] - cell["A1"]["p57"]
                out.append(cell)
    return out


# ----------------------------------------------------------------------------------------------- execution
def _load_a0(base_path: Path, tail_path: Path) -> tuple[R.A0, dict]:
    bb, tb = base_path.read_bytes(), tail_path.read_bytes()
    if sha256(bb) != A0_BASE_SHA or sha256(tb) != A0_TAIL_SHA:
        raise ProvenanceError("A0 artifact digests do not match the registered pins")
    base, tail = np.load(io.BytesIO(bb)), np.load(io.BytesIO(tb))
    if str(tail["base_policy_sha256"]) != A0_BASE_SHA:
        raise ProvenanceError("A0 tail is not paired with the pinned base")
    a0 = R.A0(base=base["policy_table"], base_bounds=[float(x) for x in base["boundaries"]],
              base_season_length=int(base["season_length"]), tail=tail["policy_table"],
              tail_bounds=[float(x) for x in tail["boundaries"]])
    return a0, {"base_sha256": A0_BASE_SHA, "tail_sha256": A0_TAIL_SHA, "tail_base_pair": True}


def _profiles(root: Path) -> dict:
    out = {}
    for f in sorted(root.glob("*/simulation_seed*/backtest_*.parquet")):
        season, seed = int(f.stem.rsplit("_", 1)[1]), int(f.parent.name.removeprefix("simulation_seed"))
        if season in SEASONS:
            if (season, seed) in out:
                raise ProvenanceError(f"duplicate profile for season {season} seed {seed}")
            out[(season, seed)] = f
    seeds = {s for _, s in out}
    if len(seeds) != 24 or len(out) != 24 * len(SEASONS):
        raise ProvenanceError(f"expected 24 seeds x {len(SEASONS)} seasons, found {len(seeds)} seeds / {len(out)} files")
    return out


def _arm_metrics(res: dict) -> dict:
    return {k: np.asarray(res[k], float) for k in ("max", "resets", "reach20", "reach30", "reach40", "reach57")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", type=Path, default=DATA / "hetzner_results" / "mdp_estpa_run")
    ap.add_argument("--w0-manifest", type=Path, default=DATA / "hetzner_results" / "mdp_estpa_run.sha256")
    ap.add_argument("--schedules", type=Path, required=True)
    ap.add_argument("--a0-base", type=Path, default=DATA / "models" / "mdp_policy.npz")
    ap.add_argument("--a0-tail", type=Path, default=DATA / "models" / "mdp_tail_policy.npz")
    ap.add_argument("--out", type=Path, default=DATA / "hetzner_results" / "c1" / "r4b" / "runs")
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args(argv)
    head = x31_gate()
    run_dir = args.out / f"{head[:7]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True, exist_ok=False)
    log = lambda msg: print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", file=sys.stderr, flush=True)  # noqa: E731

    # 2. provenance (outcome-free)
    if sha256(args.w0_manifest.read_bytes()) != W0_MANIFEST_SHA:
        raise ProvenanceError("the W0.1 manifest's own digest does not match the registered pin")
    files = _profiles(args.profiles)
    prof_sha = verify_profiles(args.profiles, args.w0_manifest, list(files.values()))
    a0, a0_meta = _load_a0(args.a0_base, args.a0_tail)
    cals, sched_sha = {}, {}
    for s in SEASONS:
        b = (args.schedules / f"sched_{s}.json").read_bytes()
        sched_sha[s] = sha256(b)
        cals[s] = D.calendar_from_schedule(json.loads(b), s)
    manifest = {"code": head, "x31_commit": X31_COMMIT, "uv_lock_sha256": sha256((REPO / "uv.lock").read_bytes()),
                "w0_manifest_sha256": W0_MANIFEST_SHA, "profiles": prof_sha, "a0": a0_meta, "schedules": sched_sha,
                "calendars": {s: {"opening": str(c.opening), "final": str(c.final), "exclusive_end": str(c.exclusive_end),
                                  "horizon": c.horizon, "no_opportunity": sorted(map(str, c.no_opportunity))}
                              for s, c in cals.items()},
                "grid": {"deltas": DELTAS, "reps": args.reps, "c": C_GRID, "h": H_GRID, "proj_deltas": PROJ_DELTAS}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    log(f"manifest written: {len(prof_sha)} profiles verified against W0.1")

    # 3. July parity at Δ=0 (stops on failure)
    from scripts.audit import dd_p_policy_value_sensitivity as ddp
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            ddp.main(["--root", str(args.profiles), "--policy", str(args.a0_base), "--stage", "l2",
                      "--l2-deltas", "0", "--reps", "1", "--out", str(run_dir / "parity.json")])
    except AssertionError as e:
        (run_dir / "STOPPED_parity.txt").write_text(f"{e}\n{buf.getvalue()[-4000:]}")
        log("PARITY FAILED: stopped before any corrected result")
        return 2
    log("parity: July Δ=0 anchors reproduced")

    # 4. decode and validate
    days = {(s, seed): D.season_days(D.validate_profile(pd.read_parquet(f)), cals[s]) for (s, seed), f in files.items()}
    seeds = sorted({seed for _, seed in days})
    unknown = {s: sorted(map(str, days[(s, seeds[0])]["unknown_dates"])) for s in SEASONS}
    log(f"profiles decoded; unknown-coverage dates: {unknown}")

    # 5. folds
    per = {}            # (arm, delta, season) -> list over seeds of per-rep metric dicts
    consequences, folds_meta, projections = {}, {}, {}
    for H in SEASONS:
        fit_days = [days[(s, seed)] for s in SEASONS if s != H for seed in seeds]
        p1 = np.concatenate([d["p1"][d["opp"] & d["known"]] for d in fit_days])
        cuts = F.cutpoints(p1)
        env, stats = F.fit_environment(fit_days, cuts, late_days=LATE_DAYS)
        rb = F.r_bar(fit_days)
        reach = S.solve(env, horizon=cals[H].horizon, objective="reach")
        emax = S.solve(env, horizon=cals[H].horizon, objective="emax")
        arms = {"A0": a0, "A1": R.HybridArm(reach=reach, emax=emax, cuts=cuts), "A2": R.Table(emax, cuts),
                "single": R.const(S.SINGLE), "double": R.const(S.DOUBLE)}
        folds_meta[H] = {"cuts": cuts.tolist(), "r_bar": rb, "fit_stats": stats, "horizon": cals[H].horizon}
        cons_h = {}
        for seed in seeds:
            sd = days[(H, seed)]
            for delta in DELTAS:
                masks = R.thin_masks(sd["hit2"] & sd["partner"], delta=delta, r_bar=rb, reps=args.reps, identity=(H, seed))
                res = R.replay(sd, arms, hit2_masks=masks, compare=("A0", "A2") if delta == 0 else None)
                for arm in arms:
                    per.setdefault((arm, delta, H), []).append(_arm_metrics(res[arm]))
                if delta == 0:
                    for k, v in res["_consequences"].items():
                        acc = cons_h.setdefault(k, {})
                        for kk, vv in v.items():
                            acc[kk] = acc.get(kk, 0) + vv
        consequences[H] = cons_h
        env_fit, pairs_fit = refined_environment(fit_days, cuts, a0.base_bounds)
        env_held, pairs_held = refined_environment([days[(H, seed)] for seed in seeds], cuts, a0.base_bounds)
        projections[H] = {
            "fitting_rates": project_grid(projection_policies(a0, reach, emax, pairs_fit), env_fit, cals[H]),
            "held_out_rates": project_grid(projection_policies(a0, reach, emax, pairs_held), env_held, cals[H])}
        log(f"fold {H}: replay and projections done")

    # 6. final decision object (all five seasons; exposed fit)
    all_days = [days[(s, seed)] for s in SEASONS for seed in seeds]
    cuts_f = F.cutpoints(np.concatenate([d["p1"][d["opp"] & d["known"]] for d in all_days]))
    env_f, stats_f = F.fit_environment(all_days, cuts_f, late_days=LATE_DAYS)
    d_final = max(c.horizon for c in cals.values())
    reach_f = S.solve(env_f, horizon=d_final, objective="reach")
    emax_f = S.solve(env_f, horizon=d_final, objective="emax")
    obj_path = run_dir / "final_decision_object.npz"
    np.savez_compressed(obj_path, emax_policy=emax_f.policy, reach_policy=reach_f.policy, cuts=cuts_f,
                        d_final=d_final, late_days=LATE_DAYS, manifest_sha256=sha256((run_dir / "manifest.json").read_bytes()))
    obj_sha = sha256(obj_path.read_bytes())
    action_sha = {"emax": sha256(emax_f.policy.tobytes()), "reach": sha256(reach_f.policy.tobytes())}
    env_all, pairs_all = refined_environment(all_days, cuts_f, a0.base_bounds)
    final_proj = {s: project_grid(projection_policies(a0, reach_f, emax_f, pairs_all), env_all, cals[s]) for s in SEASONS}
    log("final decision object built and projected")

    # 7. aggregation and disposition
    def season_vals(arm, delta, metric):
        return {H: [float(np.mean(m[metric])) for m in per[(arm, delta, H)]] for H in SEASONS}

    def per_rep_contrast(a, b, delta, metric):
        """T_r: per replicate, seeds averaged within season, seasons equally weighted."""
        reps = per[(a, delta, SEASONS[0])][0][metric].size
        return np.array([np.mean([np.mean([ma[metric][r] - mb[metric][r]
                                           for ma, mb in zip(per[(a, delta, H)], per[(b, delta, H)])]) for H in SEASONS])
                         for r in range(reps)])

    arms_table = {arm: {str(delta): {metric: equal_season_mean(season_vals(arm, delta, metric))
                                     for metric in ("max", "resets", "reach20", "reach30", "reach40", "reach57")}
                        for delta in DELTAS} for arm in ("A0", "A1", "A2", "single", "double")}
    season_detail = {arm: {H: float(np.mean(season_vals(arm, 0.0, "max")[H])) for H in SEASONS} for arm in ("A0", "A1", "A2")}
    contrast = {}
    for name, b in (("objective", "A1"), ("switch", "A0")):
        seasons = [season_detail["A2"][H] - season_detail[b][H] for H in SEASONS]
        t10 = per_rep_contrast("A2", b, 0.10, "max")
        contrast[name] = {"d0": {"mean": float(np.mean(seasons)), "seasons": seasons, "range": [min(seasons), max(seasons)],
                                 "season_sd_descriptive": float(np.std(seasons, ddof=1))},
                          "d10": {"mean": float(t10.mean()), "mc_se": mc_se(t10)}}
    reach20 = {arm: {"d0": arms_table[arm]["0.0"]["reach20"], "d10": arms_table[arm]["0.1"]["reach20"]} for arm in ("A0", "A1", "A2")}
    reach20["mc_se_d10"] = {b: mc_se(per_rep_contrast("A2", b, 0.10, "reach20")) for b in ("A1", "A0")}
    sm = {"coverage_complete": all(not v for v in unknown.values()), "validation_ok": True, "projections_ok": True,
          "contrast": contrast, "reach20": reach20}
    disp = disposition(sm)
    results = {"schema": "c1_r4b_results_v1", "code": head, "run_dir": str(run_dir), "unknown_coverage": unknown,
               "arms": arms_table, "season_mean_max_d0": season_detail, "contrast": contrast, "reach20": reach20,
               "disposition": disp, "consequences_d0": consequences, "folds": folds_meta, "projections": projections,
               "final_object": {"path": obj_path.name, "sha256": obj_sha, "action_table_sha256": action_sha,
                                "d_final": d_final, "cuts": cuts_f.tolist(), "fit_stats": stats_f,
                                "projections_by_calendar": final_proj}}
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    log(f"done: {run_dir} disposition={disp['disposition']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
