#!/usr/bin/env python
"""Versioned derivation of the skip-shadow breakeven BREAKEVEN_P ~= 0.744 (audit F10).

The constant in bts/skip_policy_shadow.py originally came from a Q-delta
diagnostic that lived only at /tmp/skip_qdelta.py on the box (referenced by
docs/audit/2026-06-20-skip-policy-shadow.md) — that file is gone. This script
re-derives the number from repo artifacts so the derivation is reproducible
and its inputs are fingerprinted.

WHAT IS DERIVED
    For a candidate with TRUE hit probability p at MDP state (streak s, days
    left d, saver sv), with V the optimal value function on the calibrated
    estimated-PA basis and EV[s',sv'] = E_q V[s',d-1,sv',q]:

        Q(skip)      = EV[s, sv]
        Q(single, p) = p * EV[s+1, sv] + (1-p) * EV[miss(s, sv)]
        miss(s, sv)  = (s, 0) if sv==1 and 10<=s<=15 else (0, sv)

        p*(s,d,sv)   = (EV[s,sv] - EV[miss]) / (EV[s+1,sv] - EV[miss])

    i.e. the candidate hit-prob at which taking the single equals skipping.
    Reported across the streak>=8 skip-rule states under BOTH objectives the
    project uses (P(reach 57) — the deployed solve — and E[season-best
    streak], evaluated at the m==s frontier), with robustness sweeps over
    bin count and saver availability. NOTE (Codex review L6): a D-horizon
    sweep is a NO-OP by construction — V at a fixed days-remaining d depends
    only on the recursion below d, so the d-grid IS the horizon sensitivity;
    D is fixed at 180. Streak grid runs 8..30 (review L2): the breakeven
    rises with streak, and shadow records above the 8-16 core band must be
    interpreted against the per-band numbers below, not the headline.

DATA
    Estimated-PA profiles (live-matched probability scale):
        --estpa-dir data/validation/estpa_profiles_2026-06-29   (box)
    Produced by: bts simulate backtest --game-probability-mode estimated_pa

RUN (on bts-hetzner, repo root; read-only):
    .venv/bin/python scripts/audit/skip_breakeven_derivation.py \
        --estpa-dir data/validation/estpa_profiles_2026-06-29 \
        --out data/validation/skip_breakeven_derivation.json

The solvers are the ones validated bit-for-bit against bts.simulate.mdp
(scripts/audit/skip_threshold_resolve.py, doc 2026-06-29 §1).
"""
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from skip_threshold_resolve import bin_arrays, solve_reach  # noqa: E402

from bts.simulate.quality_bins import compute_bins  # noqa: E402

TARGET = 57
STREAKS = range(8, 31)          # skip rule operating range + high-streak tail (review L2)
DAY_GRID = (30, 60, 90, 120, 150)


def load_profiles(estpa_dir: str) -> tuple[pd.DataFrame, list[dict]]:
    files = sorted(glob.glob(f"{estpa_dir}/backtest_*.parquet"))
    if not files:
        raise SystemExit(f"no backtest_*.parquet under {estpa_dir}")
    fingerprints = []
    frames = []
    for f in files:
        p = Path(f)
        fingerprints.append({
            "file": p.name,
            "size": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        })
        frames.append(pd.read_parquet(f))
    return pd.concat(frames, ignore_index=True), fingerprints


def solve_emax_snapshots(freq, ph, pb, snapshot_days, target=TARGET, D=180):
    """E[season-best streak] DP, retaining EV-by-quality-marginal snapshots at
    the requested days-remaining values. Adapted from skip_threshold_resolve.
    solve_emax (validated there); only the snapshot capture is new."""
    nq = len(freq)
    T = target
    Sg, Mg = np.meshgrid(np.arange(T + 1), np.arange(T + 1), indexing="ij")
    EVn = Mg.astype(float)[:, :, None].repeat(2, 2)      # d=0 terminal: E[max]=m
    s1 = np.minimum(Sg + 1, T)
    m1 = np.maximum(Mg, s1)
    s2 = np.minimum(Sg + 2, T)
    m2 = np.maximum(Mg, s2)
    Z = np.zeros_like(Sg)
    snaps = {}
    for d in range(1, D + 1):
        Vnew = np.empty_like(EVn)
        for sv in range(2):
            catch = (sv == 1) & (Sg >= 10) & (Sg <= 15)
            ev_miss = np.where(catch, EVn[Sg, Mg, 0], EVn[Z, Mg, sv])
            vsi = ph[None, None, :] * EVn[s1, m1, sv][:, :, None] \
                + (1 - ph[None, None, :]) * ev_miss[:, :, None]
            vdo = pb[None, None, :] * EVn[s2, m2, sv][:, :, None] \
                + (1 - pb[None, None, :]) * ev_miss[:, :, None]
            vsk = np.broadcast_to(EVn[Sg, Mg, sv][:, :, None], vsi.shape)
            st = np.stack([vsk, vsi, vdo], 0)
            b = st.argmax(0)
            Vnew[:, :, sv] = np.take_along_axis(st, b[None], 0)[0] @ freq
        if d in snapshot_days:
            snaps[d] = EVn.copy()   # EV entering day d (i.e. V at d-1, freq-marginalized)
        EVn = Vnew
    return snaps


def qdelta_breakevens_reach(V, freq, saver: int, target=TARGET):
    """p*(s,d,sv) grid for the P(reach 57) value tensor V[s,d,sv,q]."""
    out = []
    for d in DAY_GRID:
        EV = V[:, d - 1, :, :] @ freq          # EV[s', sv'] entering the next day
        for s in STREAKS:
            catch = (saver == 1) and (10 <= s <= 15)
            ev_miss = EV[s if catch else 0, 0 if catch else saver]
            ev_skip = EV[s, saver]
            ev_next = EV[min(s + 1, target), saver]
            denom = ev_next - ev_miss
            if denom <= 0:
                continue
            p_star = (ev_skip - ev_miss) / denom
            out.append({"s": int(s), "d": int(d), "sv": saver, "p_star": float(p_star)})
    return out


def qdelta_breakevens_emax(snaps, saver: int, target=TARGET):
    """p*(s,d,sv) at the m==s frontier for the E[max] snapshots."""
    out = []
    for d, EVn in snaps.items():
        for s in STREAKS:
            catch = (saver == 1) and (10 <= s <= 15)
            m = s                                   # current streak IS the season best
            s_next = min(s + 1, target)
            m_next = max(m, s_next)
            ev_miss = EVn[s, m, 0] if catch else EVn[0, m, saver]
            ev_skip = EVn[s, m, saver]
            ev_next = EVn[s_next, m_next, saver]
            denom = ev_next - ev_miss
            if denom <= 0:
                continue
            p_star = (ev_skip - ev_miss) / denom
            out.append({"s": int(s), "d": int(d), "sv": saver, "p_star": float(p_star)})
    return out


def summarize(rows):
    ps = sorted(r["p_star"] for r in rows)
    if not ps:
        return None
    return {
        "n_states": len(ps),
        "median": float(np.median(ps)),
        "min": ps[0],
        "max": ps[-1],
        "p25": float(np.percentile(ps, 25)),
        "p75": float(np.percentile(ps, 75)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estpa-dir", default="data/validation/estpa_profiles_2026-06-29")
    ap.add_argument("--out", default="data/validation/skip_breakeven_derivation.json")
    args = ap.parse_args()

    profiles, fingerprints = load_profiles(args.estpa_dir)
    print(f"profiles: {len(profiles)} rows from {len(fingerprints)} files", file=sys.stderr)

    report = {
        "schema": "bts_skip_breakeven_derivation_v1",
        "estpa_dir": args.estpa_dir,
        "input_files": fingerprints,
        "day_grid": list(DAY_GRID),
        "streak_grid": list(STREAKS),
        "sweeps": [],
    }

    for n_bins in (4, 5, 6):
        bins = compute_bins(profiles, n_bins=n_bins)
        freq, ph, pb, boundaries = bin_arrays(bins)
        V, _ = solve_reach(freq, ph, pb, D=180)
        for sv in (0, 1):
            rows = qdelta_breakevens_reach(V, freq, sv)
            report["sweeps"].append({
                "objective": "reach57", "n_bins": n_bins, "D": 180, "saver": sv,
                "boundaries": boundaries, "summary": summarize(rows),
                "summary_core_8_16": summarize([r for r in rows if r["s"] <= 16]),
                "summary_tail_17_30": summarize([r for r in rows if r["s"] >= 17]),
            })
        # E[max] is heavier — snapshot only at the default horizon
        snaps = solve_emax_snapshots(freq, ph, pb, snapshot_days=set(DAY_GRID), D=180)
        for sv in (0, 1):
            rows = qdelta_breakevens_emax(snaps, sv)
            report["sweeps"].append({
                "objective": "emax", "n_bins": n_bins, "D": 180, "saver": sv,
                "boundaries": boundaries, "summary": summarize(rows),
                "summary_core_8_16": summarize([r for r in rows if r["s"] <= 16]),
                "summary_tail_17_30": summarize([r for r in rows if r["s"] >= 17]),
            })
        print(f"n_bins={n_bins} done", file=sys.stderr)

    for obj in ("reach57", "emax"):
        for band, key in (("core_8_16", "summary_core_8_16"), ("tail_17_30", "summary_tail_17_30")):
            meds = [s[key]["median"] for s in report["sweeps"]
                    if s["objective"] == obj and s.get(key)]
            if meds:
                report[f"{obj}_{band}_median_of_medians"] = float(np.median(meds))
                report[f"{obj}_{band}_median_range"] = [float(min(meds)), float(max(meds))]
                print(f"{obj} [{band}]: median-of-medians p* = {np.median(meds):.4f} "
                      f"(medians span {min(meds):.4f}-{max(meds):.4f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
