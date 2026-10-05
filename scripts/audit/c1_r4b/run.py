"""C1 4b historical screen: orchestration (registration `docs/sota_audit/2026-10-04-prereg-c1-longest-streak-policy.md`).

The run proceeds in this order. Every check before step 6 is outcome-free or a stop; nothing corrected is inspected
before the preflight passes.
1. **Gates:** X-31; the owner rulings the run depends on (the 2023-10-02 coverage row must not be OPEN; the
   unretained profile-generator commit must have a recorded ruling); a clean tracked tree; the single-run rule.
2. **Outcome-free provenance manifest:**
   - the W0.1 manifest digest;
   - every profile's bytes and the profile recipe files;
   - the A0 digests and their pairing;
   - the pinned schedules and calendars;
   - the executing source hashes and `uv.lock`.
3. **Synthetic self-check:** the independent oracles in `oracle.py` against the executing solver, projection and
   replay.
4. **Verified bytes:** each profile is read once and its hash re-checked, then parsed from those bytes. July parity
   (Δ = 0) runs on a scratch root holding only those verified bytes, and a failure stops the run.
5. **Preflight:** every fold fit and the final all-season fit (cutpoints, environments, solves) runs before any replay.
6. **Corrected evaluation:**
   - folds replayed for every arm under the Δ grid with shared nested masks;
   - the fix ladder;
   - decision consequences;
   - the fixed-policy projections on the common refinement, for all arms;
   - the final decision object and its exposed-fit projections.
7. **Aggregation:** seeds within season, then seasons equally. The disposition flags come from the actual check
   results.

    python -m scripts.audit.c1_r4b.run --schedules ~/projects/bts/data/hetzner_results/c1/r3/schedules
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.audit.c1_r4b import data as D
from scripts.audit.c1_r4b import fit as F
from scripts.audit.c1_r4b import oracle as O
from scripts.audit.c1_r4b import project as P
from scripts.audit.c1_r4b import replay as R
from scripts.audit.c1_r4b import solvers as S

REPO = Path(__file__).resolve().parents[3]
X31_COMMIT: str | None = None          # the register commit that publishes X-31; set before any execution
SEASONS = (2021, 2022, 2023, 2024, 2025)
DELTAS = (0.0, 0.05, 0.10, 0.139)
REPS = 200                             # registered; the CLI cannot change it
C_GRID = (0.0, 0.02, 0.04)
H_GRID = (0.0, 0.02, 0.04)
PROJ_DELTAS = (0.0, 0.10)
LATE_DAYS = 30
A0_BASE_SHA = "66d154717ae51afb3343ee4bec8138c60bd1056e46a3de449043f4e9f76b93b4"
A0_TAIL_SHA = "dc5d0c9924431c11c43e720bf9e02903b709a157c0ebb699fdea43e7aaf949f4"
W0_MANIFEST_SHA = "d63fefffff0a8868de4d21c1aa680e352f9ee4badff158bd2a8a01c5e359e7c6"
SCHEDULE_PINS = {   # the frozen outcome-free schedule bytes (c1/r3/schedules, fetched 10/04 by the r3 acquisition)
    2021: "cd4106e88dbb5d2caf23a22f0a37f3d1fb325e74cfff684864790488e975b49c",
    2022: "f33de162f40b618d0fade6f421a04ae8debfcd871342c1c11911c67382317d12",
    2023: "25a20c7feb09202511f492c125beec134144b3c5e4c206f3121d677199339a38",
    2024: "578f565a5662f5a32d7e678afc1d5df9a2ba1b219b2da3f60dff34b99edd3195",
    2025: "16686b6a8135acf053cabf832071948a1d5dcbf723a2f76a90bd048ef5b378d9"}
CALENDAR_PINS = {   # docs/sota_audit/2026-10-04-c1-r4b-calendar-coverage.md: (opening, final, game days)
    2021: ("2021-04-01", "2021-10-03", 182), 2022: ("2022-04-07", "2022-10-05", 179),
    2023: ("2023-03-30", "2023-10-01", 182), 2024: ("2024-03-20", "2024-09-30", 185),   # 2023 after the no-play ruling
    2025: ("2025-03-18", "2025-09-28", 184)}
PROFILE_RECIPE = {
    "command": ("audit_driver.py --run-kind profiles --game-probability-mode estimated_pa --data-relay --boxes 12 "
                "--seeds 24 --test-seasons 2024,2025 --profile-seasons 2021,2022,2023,2024,2025 "
                "--no-log-pa-predictions"),
    "command_source": "docs/audit/2026-06-10-mdp-estpa-ab-methodology.md, 'Run that produced the profiles'",
    "mode": "estimated_pa", "generator_commit": None,
    "generator_commit_note": "not retained in the evidence (the run's audit_validation_split.json has no commit field)"}
RECIPE_FILES = ("audit_validation_split.json", "boxes.json")
SOURCE_FILES = ("scripts/audit/c1_r4b/solvers.py", "scripts/audit/c1_r4b/project.py", "scripts/audit/c1_r4b/data.py",
                "scripts/audit/c1_r4b/fit.py", "scripts/audit/c1_r4b/replay.py", "scripts/audit/c1_r4b/run.py",
                "scripts/audit/c1_r4b/oracle.py", "scripts/audit/c1_r3/acquire.py",
                "scripts/audit/dd_p_policy_value_sensitivity.py", "src/bts/simulate/mdp.py",
                "src/bts/simulate/tail_policy.py", "src/bts/simulate/quality_bins.py", "uv.lock")
COVERAGE_ROW = "C1-4b-2023-10-02"
EXCLUDED_CONTEST_DATES = {"2023-10-02": "register row C1-4b-2023-10-02: NO PLAY (resumed portion of the 9/28 suspended game)"}
GENERATOR_ROW = "C1-4b-generator-commit"
DATA = Path.home() / "projects" / "bts" / "data"
ARMS = ("A0", "A1", "A2", "single", "double")


class ProvenanceError(RuntimeError):
    pass


# ----------------------------------------------------------------------------------------------- gates
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


def owner_gates(register_text: str) -> list[str]:
    """The owner rulings this run depends on (code review r1 F7, F1)."""
    reasons = []
    row = next((l for l in register_text.splitlines() if l.startswith(f"| {COVERAGE_ROW} |")), None)
    if row is None or "**RULED" not in row:
        reasons.append(f"register row {COVERAGE_ROW} records no ruling: the 2023-10-02 coverage decision is pending")
    grow = next((l for l in register_text.splitlines() if l.startswith(f"| {GENERATOR_ROW} |")), None)
    if PROFILE_RECIPE["generator_commit"] is None and (grow is None or "**RULED" not in grow):
        reasons.append(f"the profile-generator commit is not retained and register row {GENERATOR_ROW} is not RULED")
    return reasons


def dirty_tree() -> list[str]:
    out = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
                         capture_output=True, text=True, check=True).stdout
    return [l for l in out.splitlines() if l.strip()]


def prior_runs(out_root: Path) -> list[str]:
    """The single authorized run: any earlier run directory that produced results or stopped needs an invalidation
    record (INVALIDATION_<run>.json in the runs root) before another run may start."""
    blocked = []
    if out_root.exists():
        for d in sorted(p for p in out_root.iterdir() if p.is_dir()):
            if ((d / "results.json").exists() or list(d.glob("STOPPED_*"))) \
                    and not (out_root / f"INVALIDATION_{d.name}.json").exists():
                blocked.append(d.name)
    return blocked


# ----------------------------------------------------------------------------------------------- pure logic
def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _manifest_entries(root: Path, manifest: Path) -> dict:
    want = {}
    for line in manifest.read_text().splitlines():
        if not line.strip():
            continue
        digest, rel = line.split(None, 1)
        want[rel.strip().removeprefix("./").removeprefix(f"{root.name}/")] = digest
    return want


def verify_profiles(root: Path, manifest: Path, files: list[Path]) -> dict:
    """Every consumed file's bytes must equal its line in the retained W0.1 manifest."""
    want, got = _manifest_entries(root, manifest), {}
    for f in files:
        rel = str(Path(f).relative_to(root))
        h = sha256(Path(f).read_bytes())
        if want.get(rel) != h:
            raise ProvenanceError(f"{rel}: sha256 {h[:12]} does not match the W0.1 manifest")
        got[rel] = h
    return got


def read_verified(root: Path, files: dict, digests: dict) -> dict:
    """Read each verified file exactly once, re-check its hash, and return the bytes every later reader parses."""
    out = {}
    for key, f in files.items():
        b = Path(f).read_bytes()
        if sha256(b) != digests[str(Path(f).relative_to(root))]:
            raise ProvenanceError(f"{f}: bytes changed after verification")
        out[key] = b
    return out


def mc_se(per_rep: np.ndarray) -> float:
    per_rep = np.asarray(per_rep, float)
    return float(np.std(per_rep, ddof=1) / np.sqrt(per_rep.size)) if per_rep.size > 1 else 0.0


def equal_season_mean(per_season_values: dict) -> float:
    return float(np.mean([np.mean(v) for v in per_season_values.values()]))


def coverage(days: dict, *, seasons, seeds) -> dict:
    """Unknown-coverage dates for every (season, seed): a date missing in ANY seed makes coverage incomplete."""
    by_season = {}
    for s in seasons:
        acc: dict = {}
        for seed in seeds:
            for d in days[(s, seed)]["unknown_dates"]:
                acc.setdefault(str(d), []).append(seed)
        by_season[str(s)] = {k: sorted(v) for k, v in sorted(acc.items())}
    return {"complete": all(not v for v in by_season.values()), "by_season": by_season}


def _ambiguous(stat: float, threshold: float, se: float) -> bool:
    """§4: a stochastic gate within two Monte Carlo SEs of its threshold is inconclusive; a sampled zero SE at exact
    equality is too (no mask-invariance proof is claimed)."""
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
    for k in ("objective", "switch"):
        c = con[k]["d10"]
        if _ambiguous(c["mean"], 0.0, c["mc_se"]):
            reasons.append(f"{k} contrast at Δ=0.10 within the MC band")
    for b in ("A1", "A0"):
        diff = r20["A2"]["d10"] - r20[b]["d10"]
        if _ambiguous(diff, -0.02, r20["mc_se_d10"][b]):
            reasons.append(f"reach-20 vs {b} at Δ=0.10 within the MC band")
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


# ----------------------------------------------------------------------------------------------- self-check
def self_check() -> dict:
    """The independent oracles against the executing code on fixed synthetic inputs (code review r1 F4/F7)."""
    errs = {}
    tiny = {"early": [(0.35, 0, True, 0.55, 0.30), (0.25, 0, False, 0.60, 0.0), (0.30, 1, True, 0.80, 0.62),
                      (0.10, None, False, 0.0, 0.0)],
            "late": [(0.5, 0, True, 0.50, 0.24), (0.3, 1, False, 0.75, 0.0), (0.2, 1, True, 0.70, 0.45)]}
    to_t = lambda xs: tuple(S.DayType(freq=f, q=q, partner=p, p_hit=h, p_both=b) for f, q, p, h, b in xs)
    env = S.Environment(n_bins=2, early=to_t(tiny["early"]), late=to_t(tiny["late"]))
    kw = dict(target=4, zone=(1, 1), late_days=2)
    for obj in ("emax", "reach"):
        sol = S.solve(env, horizon=4, objective=obj, target=4, saver_zone=(1, 1), late_days=2)
        pol = lambda s, m, d, sv, q, sol=sol: int(sol.policy[s, m, d, sv, q])
        worst = 0.0
        for s, m, d, sv in [(0, 0, 4, 1), (1, 2, 3, 1), (2, 2, 2, 0), (0, 3, 4, 0), (3, 3, 1, 1)]:
            opt = O.optimal(tiny, obj, s, m, d, sv, **kw)
            worst = max(worst, abs(sol.value[s, m, d, sv] - opt), abs(O.evaluate(tiny, pol, obj, s, m, d, sv, **kw) - opt))
        errs[f"solver_{obj}"] = worst
    fn = lambda s, m, d, sv, q: S.DOUBLE if (m < 2 and q == 1) else S.SINGLE
    table = lambda d, q: np.array([[[fn(s, m, d, sv, q) for sv in (0, 1)] for m in range(5)] for s in range(5)])
    days = [(i != 2, 6 - i) for i in range(6)]
    st = P.Stress(c=0.02, h=0.04, delta=0.1)
    got = P.project(table, env, days, target=4, saver_zone=(1, 1), late_days=2, stress=st, r_cap=2)["p_reach"]
    want = O.projection(fn, env, days, target=4, zone=(1, 1), late_days=2, stress=st, r_cap=2)
    errs["projection"] = abs(got - want)
    rng = np.random.default_rng(11)
    n = 40
    sd = {"opp": np.array([i % 13 != 5 for i in range(n)]), "known": np.array([i % 13 != 5 and i != 17 for i in range(n)]),
          "d_raw": np.arange(n, 0, -1), "p1": rng.uniform(0.6, 0.9, n), "hit1": rng.random(n) < 0.72,
          "partner": rng.random(n) < 0.9, "hit2": rng.random(n) < 0.7}
    prov = lambda s, m, d, sv, p1, partner: np.where((s >= 3) & (p1 > 0.75), S.DOUBLE, np.where(m > s + 4, S.SKIP, S.SINGLE))
    masks = rng.random((3, n)) > 0.3
    out = R.replay(sd, {"x": prov}, hit2_masks=masks & sd["hit2"][None, :])["x"]
    rerr = 0
    for r in range(3):
        ref = O.scalar_replay(sd, lambda s, m, d, sv, p1, partner: int(prov(np.array([s]), np.array([m]), d,
                                                                          np.array([sv]), p1, partner)[0]),
                              hit2=masks[r] & sd["hit2"])
        rerr += int(out["max"][r] != ref["max"]) + int(out["resets"][r] != ref["resets"])
    errs["replay_mismatches"] = rerr
    ok = all(v < 1e-12 for k, v in errs.items() if k != "replay_mismatches") and errs["replay_mismatches"] == 0
    return {"ok": bool(ok), "errors": {k: float(v) for k, v in errs.items()}}


# ----------------------------------------------------------------------------------------------- environments
def env_dict(env: S.Environment) -> dict:
    t = lambda xs: [{"freq": x.freq, "q": x.q, "partner": x.partner, "p_hit": x.p_hit, "p_both": x.p_both} for x in xs]
    return {"n_bins": env.n_bins, "early": t(env.early), "late": t(env.late)}


def refined_environment(season_days: list[dict], fold_cuts: np.ndarray, a0_bounds, *, late_days: int = LATE_DAYS):
    """The common refinement (§5): types by phase × A0 base bin × fold bin × partner availability, with primary and
    joint rates from the same stratum. q indexes `pairs` = [(a0_bin, fold_bin)]. The census keeps every cell,
    zero-frequency ones included (code review r1 F6); zero-weight cells carry no rates and are not projected."""
    nb0, nbf = len(a0_bounds) + 1, len(fold_cuts) + 1
    pairs = [(a, b) for a in range(nb0) for b in range(nbf)]
    phases, census = {}, {}
    for name, late in (("early", False), ("late", True)):
        rows = []
        for d in season_days:
            k = d["opp"] & d["known"] & ((d["d_raw"] <= late_days) if late else (d["d_raw"] > late_days))
            rows.append(pd.DataFrame({"p1": d["p1"][k], "hit1": d["hit1"][k], "partner": d["partner"][k],
                                      "hit2": d["hit2"][k]}))
        df = pd.concat(rows, ignore_index=True)
        df["q0"] = [R._bin_ge(p, a0_bounds) for p in df["p1"]]
        df["qf"] = F.classify(df["p1"].to_numpy(), fold_cuts)
        n, types, cells = len(df), [], {}
        for qi, (q0, qf) in enumerate(pairs):
            for partner in (True, False):
                g = df[(df["q0"] == q0) & (df["qf"] == qf) & (df["partner"] == partner)]
                c = int(len(g))
                cells[f"{q0}|{qf}|{'P' if partner else 'N'}"] = {
                    "n": c, "hit1": int(g["hit1"].sum()), "both": int((g["hit1"] & g["hit2"]).sum()) if partner else 0}
                if c == 0 or n == 0:
                    continue
                ph = float(g["hit1"].mean())
                pb = float((g["hit1"] & g["hit2"]).mean()) if partner else 0.0
                types.append(S.DayType(freq=c / n, q=qi, partner=partner, p_hit=ph, p_both=pb))
        phases[name] = tuple(types)
        census[name] = {"days": n, "cells": cells}
    return S.Environment(n_bins=len(pairs), early=phases["early"], late=phases["late"]), pairs, census


def projection_policies(a0: R.A0, reach: S.Solution, emax: S.Solution, pairs, target=S.TARGET):
    """(d, q) -> (T+1, T+1, 2) raw-action tables for every arm, each through its own classifier (pairs[q])."""
    T1 = target + 1
    Sg, Mg, SVg = np.meshgrid(np.arange(T1), np.arange(T1), np.arange(2), indexing="ij")
    const = lambda a: (lambda d, q: np.where(Sg >= target, 0, np.full((T1, T1, 2), a, np.int8)))

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
        return np.where(Sg >= target, 0, np.where(Sg + 2 * d_eff < target, tail, base))

    return {"A0": a0f, "A1": a1, "A2": a2, "single": const(S.SINGLE), "double": const(S.DOUBLE)}


def project_grid(policies: dict, env: S.Environment, cal: D.Calendar) -> list[dict]:
    days = [(opp, left) for _, opp, left in cal.days()]
    out = []
    for delta in PROJ_DELTAS:
        for c in C_GRID:
            for h in H_GRID:
                st = P.Stress(c=c, h=h, delta=delta)
                cell = {"delta": delta, "c": c, "h": h}
                for arm, pol in policies.items():
                    try:
                        r = P.project(pol, env, days, target=S.TARGET, saver_zone=S.SAVER_ZONE,
                                      late_days=LATE_DAYS, stress=st)
                        cell[arm] = {"p57": r["p_reach"], "p57_pp": 100 * r["p_reach"], "e_best": r["e_best"]}
                    except P.Unavailable as e:
                        cell[arm] = {"unavailable": str(e)}
                for b in ("A0", "A1"):
                    ok = "p57" in cell["A2"] and "p57" in cell[b]
                    cell[f"A2_minus_{b}_p57_pp"] = 100 * (cell["A2"]["p57"] - cell[b]["p57"]) if ok else "unavailable"
                out.append(cell)
    return out


def projection_summary(tables: dict) -> dict:
    """Per scenario: the equal-season (or equal-calendar) mean and range of each paired jackpot difference (pp),
    then the range of those means across scenarios. Unavailable cells are counted, never averaged as zero."""
    keys = sorted({(c["delta"], c["c"], c["h"]) for cells in tables.values() for c in cells})
    per, unavailable = [], 0
    for key in keys:
        row = {"delta": key[0], "c": key[1], "h": key[2]}
        for diff in ("A2_minus_A0_p57_pp", "A2_minus_A1_p57_pp"):
            vals = [c[diff] for cells in tables.values() for c in cells if (c["delta"], c["c"], c["h"]) == key]
            num = [v for v in vals if not isinstance(v, str)]
            unavailable += len(vals) - len(num)
            row[diff] = ({"mean": float(np.mean(num)), "min": float(min(num)), "max": float(max(num)), "n": len(num)}
                         if len(num) == len(vals) and num else "unavailable")
        per.append(row)
    span = {}
    for diff in ("A2_minus_A0_p57_pp", "A2_minus_A1_p57_pp"):
        means = [r[diff]["mean"] for r in per if isinstance(r[diff], dict)]
        span[diff] = [min(means), max(means)] if means else "unavailable"
    return {"by_scenario": per, "scenario_span_of_means_pp": span, "unavailable_cells": unavailable}


def tables_complete(tables: dict) -> bool:
    """Every required cell is present and either a finite number or an explicit unavailable record."""
    for cells in tables.values():
        if len(cells) != len(PROJ_DELTAS) * len(C_GRID) * len(H_GRID):
            return False
        for c in cells:
            for arm in ARMS:
                v = c.get(arm)
                if v is None or ("unavailable" not in v and not (np.isfinite(v["p57"]) and np.isfinite(v["e_best"]))):
                    return False
    return True


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
    if len(seeds) != 24 or len(out) != 24 * len(SEASONS) or {(s, k) for s in SEASONS for k in seeds} != set(out):
        raise ProvenanceError(f"expected the same 24 seeds in each of {len(SEASONS)} seasons; found {len(out)} files")
    return out


def _calendars(schedules: Path) -> tuple[dict, dict]:
    cals, shas = {}, {}
    for s in SEASONS:
        b = (schedules / f"sched_{s}.json").read_bytes()
        shas[s] = sha256(b)
        if shas[s] != SCHEDULE_PINS[s]:
            raise ProvenanceError(f"schedule {s}: sha256 {shas[s][:12]} is not the pinned outcome-free schedule")
        c = D.calendar_from_schedule(json.loads(b), s)
        c = D.exclude_dates(c, [x for x in EXCLUDED_CONTEST_DATES if int(x[:4]) == s])
        pin = CALENDAR_PINS[s]
        if (str(c.opening), str(c.final), sum(1 for _, o, _ in c.days() if o)) != pin:
            raise ProvenanceError(f"calendar {s} does not match its pin {pin}")
        cals[s] = c
    return cals, shas


def census(sd: dict, df: pd.DataFrame) -> dict:
    k = sd["opp"] & sd["known"]
    ranks = pd.Series(sd["partner_rank"][k & sd["partner"]]).value_counts().sort_index()
    return {"calendar_days": int(sd["opp"].size), "no_opportunity_days": int((~sd["opp"]).sum()),
            "opportunity_days": int(sd["opp"].sum()), "known_days": int(k.sum()),
            "unknown_days": len(sd["unknown_dates"]), "partner_days": int((k & sd["partner"]).sum()),
            "partnerless_days": int((k & ~sd["partner"]).sum()),
            "partner_rank_counts": {int(a): int(b) for a, b in ranks.items()},
            "profile_rows": int(len(df)), "excluded_rows": 0}


def _achieved(sd: dict, masks: np.ndarray) -> dict:
    k = sd["opp"] & sd["known"] & sd["partner"]
    legs = sd["hit2"] & k
    n_part, n_legs = int(k.sum()), int(legs.sum())
    removed = (legs[None, :] & ~masks).sum(axis=1)
    hk = legs & sd["hit1"]
    removed_h = (hk[None, :] & ~masks).sum(axis=1)
    return {"partner_days": n_part, "leg_hits": n_legs,
            "achieved_marginal_rate_reduction": float(removed.mean() / n_part) if n_part else None,
            "achieved_primary_hit_conditional_reduction": float(removed_h.mean() / max(1, int((k & sd["hit1"]).sum())))}


def _seed_metrics(res: dict) -> dict:
    return {"max": np.asarray(res["max"], float), "resets": np.asarray(res["resets"], float),
            "reach20": np.asarray(res["reach20"], float), "reach30": np.asarray(res["reach30"], float),
            "reach40": np.asarray(res["reach40"], float), "reach57": np.asarray(res["reach57"], float),
            **{f"act_{k}": np.asarray(v, float) for k, v in res["actions"].items()},
            **{f"skip_{k}": np.asarray(v, float) for k, v in res["skip_census"].items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", type=Path, default=DATA / "hetzner_results" / "mdp_estpa_run")
    ap.add_argument("--w0-manifest", type=Path, default=DATA / "hetzner_results" / "mdp_estpa_run.sha256")
    ap.add_argument("--schedules", type=Path, required=True)
    ap.add_argument("--a0-base", type=Path, default=DATA / "models" / "mdp_policy.npz")
    ap.add_argument("--a0-tail", type=Path, default=DATA / "models" / "mdp_tail_policy.npz")
    ap.add_argument("--out", type=Path, default=DATA / "hetzner_results" / "c1" / "r4b" / "runs")
    args = ap.parse_args(argv)
    log = lambda msg: print(f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}", file=sys.stderr, flush=True)  # noqa: E731

    # 1. gates
    head = x31_gate()
    gates = owner_gates((REPO / "docs/audit/2026-09-22-exposure-register.md").read_text())
    if gates:
        raise SystemExit("refusing: " + "; ".join(gates))
    dirty = dirty_tree()
    if dirty:
        raise SystemExit(f"refusing: tracked changes in the executing tree: {dirty[:5]}")
    blocked = prior_runs(args.out)
    if blocked:
        raise SystemExit(f"refusing: earlier runs without an invalidation record: {blocked}")
    run_dir = args.out / f"{head[:7]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    run_dir.mkdir(parents=True, exist_ok=False)

    def stop(name: str, text: str) -> int:
        (run_dir / f"STOPPED_{name}.txt").write_text(text)
        log(f"STOPPED ({name}): {text[:200]}")
        return 2

    # 2. outcome-free provenance
    if sha256(args.w0_manifest.read_bytes()) != W0_MANIFEST_SHA:
        raise ProvenanceError("the W0.1 manifest's own digest does not match the registered pin")
    files = _profiles(args.profiles)
    prof_sha = verify_profiles(args.profiles, args.w0_manifest, list(files.values()))
    recipe_sha = verify_profiles(args.profiles, args.w0_manifest, [args.profiles / n for n in RECIPE_FILES])
    a0, a0_meta = _load_a0(args.a0_base, args.a0_tail)
    cals, sched_sha = _calendars(args.schedules)
    sources = {f: sha256((REPO / f).read_bytes()) for f in SOURCE_FILES}
    manifest = {"code": head, "x31_commit": X31_COMMIT, "sources": sources, "w0_manifest_sha256": W0_MANIFEST_SHA,
                "profiles": prof_sha, "profile_recipe": {**PROFILE_RECIPE, "recipe_files": recipe_sha},
                "a0": a0_meta, "schedules": sched_sha,
                "excluded_contest_dates": EXCLUDED_CONTEST_DATES,
                "calendars": {s: {"opening": str(c.opening), "final": str(c.final), "exclusive_end": str(c.exclusive_end),
                                  "horizon": c.horizon, "no_opportunity": sorted(map(str, c.no_opportunity))}
                              for s, c in cals.items()},
                "grid": {"deltas": DELTAS, "reps": REPS, "c": C_GRID, "h": H_GRID, "proj_deltas": PROJ_DELTAS}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    log(f"manifest written: {len(prof_sha)} profiles verified against W0.1")

    # 3. synthetic self-check of the executing code
    check = self_check()
    (run_dir / "self_check.json").write_text(json.dumps(check, indent=1) + "\n")
    if not check["ok"]:
        return stop("self_check", json.dumps(check))

    # 4. verified bytes; parity on a root holding only them
    blobs = read_verified(args.profiles, files, prof_sha)
    proot = run_dir / "parity_inputs" / args.profiles.name
    for key, f in files.items():
        dest = proot / Path(f).relative_to(args.profiles)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blobs[key])
    from scripts.audit import dd_p_policy_value_sensitivity as ddp
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            ddp.main(["--root", str(proot), "--policy", str(args.a0_base), "--stage", "l2",
                      "--l2-deltas", "0", "--reps", "1", "--out", str(run_dir / "parity.json")])
    except AssertionError as e:
        return stop("parity", f"{e}\n{buf.getvalue()[-4000:]}")
    parity_ok = True
    shutil.rmtree(run_dir / "parity_inputs")
    log("parity: July Δ=0 anchors reproduced on the verified bytes")

    frames = {k: D.validate_profile(pd.read_parquet(io.BytesIO(b))) for k, b in blobs.items()}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    days = {(s, seed): D.season_days(frames[(s, seed)], cals[s]) for (s, seed) in frames}
    seeds = sorted({seed for _, seed in days})
    cov = coverage(days, seasons=SEASONS, seeds=seeds)
    census_tab = {f"{s}|{seed}": census(days[(s, seed)], frames[(s, seed)]) for (s, seed) in days}
    log(f"profiles decoded; unknown coverage: {cov['by_season']}")

    # 5. preflight: every fit before any replay
    try:
        folds = {}
        for H in SEASONS:
            fit_days = [days[(s, seed)] for s in SEASONS if s != H for seed in seeds]
            cuts = F.cutpoints(np.concatenate([d["p1"][d["opp"] & d["known"]] for d in fit_days]))
            env, stats = F.fit_environment(fit_days, cuts, late_days=LATE_DAYS)
            folds[H] = {"fit_days": fit_days, "cuts": cuts, "env": env, "stats": stats, "r_bar": F.r_bar(fit_days),
                        "reach": S.solve(env, horizon=cals[H].horizon, objective="reach"),
                        "emax": S.solve(env, horizon=cals[H].horizon, objective="emax")}
        all_days = [days[(s, seed)] for s in SEASONS for seed in seeds]
        cuts_f = F.cutpoints(np.concatenate([d["p1"][d["opp"] & d["known"]] for d in all_days]))
        env_f, stats_f = F.fit_environment(all_days, cuts_f, late_days=LATE_DAYS)
        d_final = max(c.horizon for c in cals.values())
        reach_f = S.solve(env_f, horizon=d_final, objective="reach")
        emax_f = S.solve(env_f, horizon=d_final, objective="emax")
    except ValueError as e:
        return stop("preflight", f"fitting preflight failed before any result: {e}")
    log("preflight: all fold fits and the final fit succeeded")

    # 6. corrected evaluation
    per, per_rep, achieved, consequences, ladder, projections, folds_out = {}, {}, [], {}, {}, {}, {}
    for H in SEASONS:
        fo = folds[H]
        arms = {"A0": a0, "A1": R.HybridArm(reach=fo["reach"], emax=fo["emax"], cuts=fo["cuts"]),
                "A2": R.Table(fo["emax"], fo["cuts"]), "single": R.const(S.SINGLE), "double": R.const(S.DOUBLE)}
        cons_h = {}
        for seed in seeds:
            sd = days[(H, seed)]
            for delta in DELTAS:
                masks = R.thin_masks(sd["hit2"] & sd["partner"], delta=delta, r_bar=fo["r_bar"], reps=REPS,
                                     identity=(H, seed))
                if delta > 0:
                    achieved.append({"season": H, "seed": seed, "delta": delta, **_achieved(sd, masks)})
                res = R.replay(sd, arms, hit2_masks=masks, compare=("A0", "A2") if delta == 0 else None)
                for arm in arms:
                    m = _seed_metrics(res[arm])
                    per.setdefault((arm, delta, H), []).append({k: float(v.mean()) for k, v in m.items()})
                    per_rep.setdefault((arm, delta, H), []).append(m)
                if delta == 0:
                    for k, v in res["_consequences"].items():
                        acc = cons_h.setdefault(k, {})
                        for kk, vv in v.items():
                            acc[kk] = acc.get(kk, 0) + vv
            # the cumulative fix ladder on fixed reference arms (Δ = 0)
            ones = np.ones((1, sd["opp"].size), bool) & (sd["hit2"] & sd["partner"])[None, :]
            refs = {"A0": R.A0BaseOnly(a0), "single": R.const(S.SINGLE), "double": R.const(S.DOUBLE)}
            rungs = {"0_july_semantics": dict(clock="row180", partnerless="plus2"),
                     "1_calendar_clock": dict(clock="calendar", partnerless="plus2"),
                     "2_legal_demotion": dict(clock="calendar", partnerless="legal"),
                     "3_m_state_plumbing": dict(clock="calendar", partnerless="legal")}
            for rung, opts in rungs.items():
                res = R.replay(sd, refs, hit2_masks=ones, **opts)
                for arm in refs:
                    ladder.setdefault((rung, arm, H), []).append(_seed_metrics(res[arm]))
            res = R.replay(sd, {"A0": a0}, hit2_masks=ones)
            ladder.setdefault(("4_a0_tail_routing", "A0", H), []).append(_seed_metrics(res["A0"]))
        consequences[H] = cons_h
        env_fit, pairs_fit, cen_fit = refined_environment(fo["fit_days"], fo["cuts"], a0.base_bounds)
        env_held, pairs_held, cen_held = refined_environment([days[(H, seed)] for seed in seeds], fo["cuts"], a0.base_bounds)
        projections[H] = {
            "fitting_rates": project_grid(projection_policies(a0, fo["reach"], fo["emax"], pairs_fit), env_fit, cals[H]),
            "held_out_rates": project_grid(projection_policies(a0, fo["reach"], fo["emax"], pairs_held), env_held, cals[H])}
        folds_out[H] = {"cuts": fo["cuts"].tolist(), "r_bar": fo["r_bar"], "horizon": cals[H].horizon,
                        "fit_counts": fo["stats"], "environment": env_dict(fo["env"]),
                        "environment_sha256": sha256(json.dumps(env_dict(fo["env"]), sort_keys=True).encode()),
                        "refined": {"fitting": {"pairs": pairs_fit, "census": cen_fit, "environment": env_dict(env_fit)},
                                    "held_out": {"pairs": pairs_held, "census": cen_held, "environment": env_dict(env_held)}}}
        log(f"fold {H}: replay, ladder and projections done")

    env_all, pairs_all, cen_all = refined_environment(all_days, cuts_f, a0.base_bounds)
    final_proj = {s: project_grid(projection_policies(a0, reach_f, emax_f, pairs_all), env_all, cals[s]) for s in SEASONS}
    final_env = {"fitted": env_dict(env_f), "fit_counts": stats_f, "cuts": cuts_f.tolist(),
                 "refined": {"pairs": pairs_all, "census": cen_all, "environment": env_dict(env_all)},
                 "recipe": {"objective": "own E[season best] capped at 57 (emax) with matched reach-57 A1",
                            "target": S.TARGET, "saver_zone": S.SAVER_ZONE, "late_days": LATE_DAYS,
                            "ties": {"emax": "stop rule, then play-first", "reach": "skip-first"},
                            "d_final": d_final, "calendars_sha256": sched_sha}}
    env_bytes = json.dumps(final_env, sort_keys=True, default=str).encode()
    (run_dir / "final_environment.json").write_bytes(env_bytes)
    obj_path = run_dir / "final_decision_object.npz"
    np.savez_compressed(obj_path, emax_policy=emax_f.policy, reach_policy=reach_f.policy, cuts=cuts_f,
                        d_final=d_final, late_days=LATE_DAYS, environment_sha256=sha256(env_bytes),
                        manifest_sha256=sha256((run_dir / "manifest.json").read_bytes()))
    log("final decision object built and projected")

    # 7. aggregation and disposition
    def season_mean(arm, delta, metric):
        return {H: [r[metric] for r in per[(arm, delta, H)]] for H in SEASONS}

    def per_rep_contrast(a, b, delta, metric):
        reps = per_rep[(a, delta, SEASONS[0])][0][metric].size
        return np.array([np.mean([np.mean([ma[metric][r] - mb[metric][r] for ma, mb in
                                           zip(per_rep[(a, delta, H)], per_rep[(b, delta, H)])]) for H in SEASONS])
                         for r in range(reps)])

    metrics = list(per[("A0", 0.0, SEASONS[0])][0].keys())
    arms_table = {arm: {str(delta): {m: equal_season_mean(season_mean(arm, delta, m)) for m in metrics}
                        for delta in DELTAS} for arm in ARMS}
    by_season = {arm: {str(delta): {m: {H: float(np.mean(season_mean(arm, delta, m)[H])) for H in SEASONS}
                                    for m in ("max", "reach20", "resets")} for delta in DELTAS} for arm in ARMS}
    reach57_counts = {arm: {str(delta): {
        "trajectories_reaching_57": int(sum(int(m["reach57"].sum()) for H in SEASONS for m in per_rep[(arm, delta, H)])),
        "trajectories": int(sum(m["reach57"].size for H in SEASONS for m in per_rep[(arm, delta, H)]))}
        for delta in DELTAS} for arm in ARMS}
    contrast = {}
    for name, a, b in (("objective", "A2", "A1"), ("switch", "A2", "A0"), ("adaptation", "A1", "A0")):
        seasons = [by_season[a]["0.0"]["max"][H] - by_season[b]["0.0"]["max"][H] for H in SEASONS]
        t10 = per_rep_contrast(a, b, 0.10, "max")
        contrast[name] = {"d0": {"mean": float(np.mean(seasons)), "seasons": seasons, "range": [min(seasons), max(seasons)],
                                 "season_sd_descriptive": float(np.std(seasons, ddof=1))},
                          "d10": {"mean": float(t10.mean()), "mc_se": mc_se(t10), "per_replicate": t10.tolist()}}
    reach20 = {arm: {"d0": arms_table[arm]["0.0"]["reach20"], "d10": arms_table[arm]["0.1"]["reach20"]} for arm in ("A0", "A1", "A2")}
    reach20["mc_se_d10"] = {b: mc_se(per_rep_contrast("A2", b, 0.10, "reach20")) for b in ("A1", "A0")}
    fix_ladder = {rung: {arm: {m: equal_season_mean({H: [float(x[m].mean()) for x in ladder[(rung, arm, H)]] for H in SEASONS})
                               for m in ("max", "reach20", "resets")}
                         for arm in {a for (r_, a, _) in ladder if r_ == rung}}
                  for rung in sorted({r_ for (r_, _, _) in ladder})}
    fold_proj_summary = {src: projection_summary({H: projections[H][src] for H in SEASONS})
                         for src in ("fitting_rates", "held_out_rates")}
    final_summary = projection_summary(final_proj)
    proj_tables = {f"{H}|{src}": projections[H][src] for H in SEASONS for src in ("fitting_rates", "held_out_rates")}
    proj_tables.update({f"final|{s}": final_proj[s] for s in SEASONS})
    validation_ok = bool(check["ok"] and parity_ok)        # provenance failures raise before this point
    projections_ok = tables_complete(proj_tables)
    sm = {"coverage_complete": cov["complete"], "validation_ok": validation_ok, "projections_ok": projections_ok,
          "contrast": contrast, "reach20": reach20}
    disp = disposition(sm)
    span = fold_proj_summary["held_out_rates"]["scenario_span_of_means_pp"]["A2_minus_A0_p57_pp"]
    headline = (f"The historical fitting-procedure screen changes mean season best by "
                f"{contrast['switch']['d0']['mean']:+.2f} streaks versus A0 [five-season range "
                f"{contrast['switch']['d0']['range'][0]:+.2f} to {contrast['switch']['d0']['range'][1]:+.2f}], conditional "
                f"on the retained participant surface and A0's prior exposure. Its jackpot-cost projections (held-out "
                f"rates, A2 minus A0) span {span} percentage points across the registered scenarios; the true jackpot "
                f"cost is unresolved. The separately identified final artifact has its own projection table. These "
                f"tables do not alone authorize a change to picks.")
    results = {"schema": "c1_r4b_results_v2", "code": head, "run_dir": str(run_dir),
               "qualification": ("Historical fitting-procedure screen on 2021-2025 estimated-PA profiles; A0 is a "
                                 "historically exposed comparator; every P(57) cell is a model projection, not a "
                                 "measured frequency; own E[best] is not a prize objective."),
               "headline": headline, "coverage": cov, "census": census_tab, "self_check": check,
               "validation_ok": validation_ok, "projections_ok": projections_ok,
               "arms": arms_table, "by_season": by_season, "reach57_counts": reach57_counts,
               "contrast": contrast, "reach20": reach20, "disposition": disp, "fix_ladder": fix_ladder,
               "achieved_haircuts": achieved, "consequences_d0": consequences, "folds": folds_out,
               "projections": projections, "projection_summary": fold_proj_summary,
               "final_object": {"path": obj_path.name, "sha256": sha256(obj_path.read_bytes()),
                                "environment_sha256": sha256(env_bytes),
                                "action_table_sha256": {"emax": sha256(emax_f.policy.tobytes()),
                                                        "reach": sha256(reach_f.policy.tobytes())},
                                "d_final": d_final, "projections_by_calendar": final_proj,
                                "projection_summary": final_summary},
               "per_seed": {f"{arm}|{delta}|{H}": per[(arm, delta, H)] for (arm, delta, H) in per}}
    (run_dir / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n")
    log(f"done: {run_dir} disposition={disp['disposition']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
