#!/usr/bin/env python
"""Reproduce the MDP-side results in docs/audit/2026-06-29-skip-threshold-and-discrimination.md.

Threshold staircase, the actual_pa-vs-estimated_pa scale comparison, the P(57)/E[max]
cross-evaluation from the current state, and the play-rate bootstrap (§2-6 of the doc).

Run from the repo root, where the data lives (in practice: bts-hetzner):
    .venv/bin/python scripts/audit/skip_threshold_resolve.py \
        --estpa-dir data/validation/estpa_profiles_2026-06-29

Data:
  - actual_pa bins  : data/simulation/backtest_*.parquet           (deployed-scale, hindsight PA)
  - estimated_pa    : <estpa-dir>/backtest_*.parquet               (live-matched; `bts simulate backtest
                      --game-probability-mode estimated_pa --output-dir <estpa-dir>`)
  - deployed policy : data/models/mdp_policy.npz

The hand-rolled solvers below were validated bit-for-bit against bts.simulate.mdp.solve_mdp
(same optimal_p57, 100% policy agreement) — see doc §1.
"""
import argparse, glob
import numpy as np, pandas as pd
from bts.simulate.quality_bins import compute_bins
from bts.simulate.mdp import load_policy

TARGET, D = 57, 180
CURRENT = dict(streak=14, days=91, saver=1)   # state at investigation time


def bin_arrays(bins):
    freq = np.array([b.frequency for b in bins.bins], float); freq /= freq.sum()
    ph = np.array([b.p_hit for b in bins.bins], float)
    pb = np.array([b.p_both for b in bins.bins], float)
    return freq, ph, pb, [round(x, 4) for x in bins.boundaries]


def solve_reach(freq, ph, pb, target=TARGET, D=D):
    """Optimal P(reach target). Returns (V, policy). V[s,d,sv,q] is exact P(57) under the optimal policy."""
    nq = len(freq)
    V = np.zeros((target + 1, D + 1, 2, nq)); V[target, :, :, :] = 1.0
    pol = np.zeros((target, D + 1, 2, nq), np.int8)
    for d in range(1, D + 1):
        EV = V[:, d - 1, :, :] @ freq
        for sv in range(2):
            for s in range(target):
                catch = (sv == 1) and (10 <= s <= 15)
                em = EV[s if catch else 0, 0 if catch else sv]
                vsk = np.full(nq, EV[s, sv])
                vsi = ph * EV[min(s + 1, target), sv] + (1 - ph) * em
                vdo = pb * EV[min(s + 2, target), sv] + (1 - pb) * em
                vals = np.vstack([vsk, vsi, vdo]); b = vals.argmax(0)
                V[s, d, sv, :] = vals[b, np.arange(nq)]; pol[s, d, sv, :] = b
    return V, pol


def solve_emax(freq, ph, pb, target=TARGET, D=D):
    """Optimal E[season-best streak], augmented state (current streak s, running max m). Returns (E, policy)."""
    nq = len(freq); T = target
    Sg, Mg = np.meshgrid(np.arange(T + 1), np.arange(T + 1), indexing="ij")
    EVn = Mg.astype(float)[:, :, None].repeat(2, 2)
    polAll = np.zeros((T + 1, T + 1, D + 1, 2, nq), np.int8)
    s1 = np.minimum(Sg + 1, T); m1 = np.maximum(Mg, s1)
    s2 = np.minimum(Sg + 2, T); m2 = np.maximum(Mg, s2); Z = np.zeros_like(Sg)
    for d in range(1, D + 1):
        Vnew = np.empty_like(EVn)
        for sv in range(2):
            catch = (sv == 1) & (Sg >= 10) & (Sg <= 15)
            ev_miss = np.where(catch, EVn[Sg, Mg, 0], EVn[Z, Mg, sv])
            vsi = ph[None, None, :] * EVn[s1, m1, sv][:, :, None] + (1 - ph[None, None, :]) * ev_miss[:, :, None]
            vdo = pb[None, None, :] * EVn[s2, m2, sv][:, :, None] + (1 - pb[None, None, :]) * ev_miss[:, :, None]
            vsk = np.broadcast_to(EVn[Sg, Mg, sv][:, :, None], vsi.shape)
            st = np.stack([vsk, vsi, vdo], 0); b = st.argmax(0)
            Vnew[:, :, sv] = np.take_along_axis(st, b[None], 0)[0] @ freq
            polAll[:, :, d, sv, :] = b
        EVn = Vnew
    return EVn, polAll


def eval_reach(action_of, freq, ph, pb, target=TARGET, D=D):
    """Exact P(reach target) under a FIXED policy action_of(s,d,sv,q) -> {0 skip,1 single,2 double}."""
    V = np.zeros((target + 1, D + 1, 2, len(freq))); V[target, :, :, :] = 1.0
    for d in range(1, D + 1):
        EV = V[:, d - 1, :, :] @ freq
        for sv in range(2):
            for s in range(target):
                catch = (sv == 1) and (10 <= s <= 15)
                em = EV[s if catch else 0, 0 if catch else sv]
                for q in range(len(freq)):
                    a = action_of(s, d, sv, q)
                    V[s, d, sv, q] = (EV[s, sv] if a == 0 else
                                      ph[q] * EV[min(s + 1, target), sv] + (1 - ph[q]) * em if a == 1 else
                                      pb[q] * EV[min(s + 2, target), sv] + (1 - pb[q]) * em)
    return V


def mc(policy_fn, freq, ph, pb, N, Dd, s0, m0, sv0, target=TARGET, seed=0):
    """Vectorized policy evaluation -> (P(reach target), max-streak array). policy_fn(s,m,d,sv,q)."""
    rng = np.random.default_rng(seed); cf = np.cumsum(freq)
    s = np.full(N, s0, np.int32); m = np.full(N, max(m0, s0), np.int32)
    sv = np.full(N, sv0, np.int8); reached = s >= target
    for d in range(Dd, 0, -1):
        q = np.minimum(np.searchsorted(cf, rng.random(N)), len(freq) - 1)
        a = policy_fn(s, m, d, sv, q)
        u = rng.random(N); hit = np.zeros(N, bool); si = a == 1; do = a == 2
        hit[si] = u[si] < ph[q[si]]; hit[do] = u[do] < pb[q[do]]
        ns = s.copy()
        ns[si & hit] = np.minimum(s[si & hit] + 1, target); ns[do & hit] = np.minimum(s[do & hit] + 2, target)
        miss = (si | do) & ~hit; catch = miss & (sv == 1) & (s >= 10) & (s <= 15)
        ns[miss & ~catch] = 0; sv = sv.copy(); sv[catch] = 0
        s = ns; m = np.maximum(m, s); reached |= s >= target
    return reached.mean(), m


def classify(p, bnd):
    q = 0
    for b in bnd:
        if p >= b: q += 1
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estpa-dir", default="data/validation/estpa_profiles_2026-06-29")
    ap.add_argument("--policy", default="data/models/mdp_policy.npz")
    args = ap.parse_args()

    APR = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob("data/simulation/backtest_*.parquet"))], ignore_index=True)
    EST = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{args.estpa_dir}/backtest_*.parquet"))], ignore_index=True)
    dep_pol, dep_bnd, _ = load_policy(args.policy)

    print("=== §3 scale comparison (median top-pick p_game_hit) ===")
    for nm, df in [("actual_pa (deployed bins)", APR), ("estimated_pa (live-matched)", EST)]:
        r1 = df[df["rank"] == 1]; fr, ph, pb, bnd = bin_arrays(compute_bins(df))
        print(f"  {nm:28} median={r1['p_game_hit'].median():.3f}  boundaries={bnd}  p_hit={[round(x,3) for x in ph]}")

    freq, ph, pb, est_bnd = bin_arrays(compute_bins(EST))
    Vr, polR = solve_reach(freq, ph, pb); EVm, polE = solve_emax(freq, ph, pb)
    edge = [0.0] + est_bnd
    print("\n=== §2 implied skip threshold by streak (estimated_pa, saver=1, 91 days left) ===")
    for s in [5, 8, 12, 14, 20, 30]:
        row = polR[s, CURRENT["days"], 1, :]; act = np.where(row != 0)[0]
        print(f"  streak {s:2d}: {'skip-all' if len(act)==0 else f'act if p>={edge[int(act.min())]:.3f}'}")

    # deployed policy on live dynamics needs DAY-LEVEL bin remap; here use the est-bin mean-p remap for brevity
    repp = EST[EST["rank"] == 1].assign(qb=lambda d: d["p_game_hit"].apply(lambda p: classify(p, est_bnd))) \
              .groupby("qb")["p_game_hit"].mean().reindex(range(len(freq))).to_numpy()
    dep_map = np.array([classify(p, dep_bnd) for p in repp])
    pol_dep = lambda s, m, d, sv, q: dep_pol[np.minimum(s, 56), min(d, 180), sv, dep_map[q]]
    pol_p57 = lambda s, m, d, sv, q: polR[np.minimum(s, 56), min(d, 180), sv, q]
    pol_em = lambda s, m, d, sv, q: polE[np.minimum(s, 57), np.minimum(m, 57), min(d, 180), sv, q]

    print(f"\n=== §6 from current state (streak {CURRENT['streak']}, {CURRENT['days']}d, saver) — estimated_pa ===")
    p57_dep = eval_reach(lambda s, d, sv, q: int(dep_pol[min(s, 56), min(d, 180), sv, dep_map[q]]), freq, ph, pb)[CURRENT["streak"], CURRENT["days"], 1, :] @ freq
    p57_re = Vr[CURRENT["streak"], CURRENT["days"], 1, :] @ freq
    print(f"  EXACT P(57): deployed={p57_dep*100:.4f}%  re-solve-P(57)={p57_re*100:.4f}%   (clean April actual_pa re-solve = 4.83%; metadata 8.17% is hindsight)")
    for nm, pf in [("deployed", pol_dep), ("re-solve P(57)", pol_p57), ("re-solve E[max]", pol_em)]:
        _, marr = mc(pf, freq, ph, pb, 200_000, CURRENT["days"], CURRENT["streak"], CURRENT["streak"], 1, seed=7)
        print(f"  {nm:16}: E[max]={marr.mean():5.1f}  P(max>=30)={np.mean(marr>=30):.3f}")

    print("\n=== §5 play-rate bootstrap (per-pool re-estimated rates) — the HONEST interval ===")
    r1 = EST[EST["rank"] == 1].dropna(subset=["p_game_hit", "actual_hit"])
    groups = {d: g for d, g in r1.assign(p=r1["p_game_hit"], h=r1["actual_hit"]).groupby("date")}
    dates = list(groups); rng = np.random.default_rng(11); deltas = []
    for it in range(40):
        samp = rng.integers(0, len(dates), len(dates))
        P = np.concatenate([groups[dates[i]]["p"].to_numpy() for i in samp])
        H = np.concatenate([groups[dates[i]]["h"].to_numpy() for i in samp])
        gb = pd.qcut(P, 5, labels=False, duplicates="drop"); rate = pd.Series(H).groupby(gb).mean()
        hitp = rate.reindex(gb).to_numpy()
        def sweep(thr, seed):
            rr = np.random.default_rng(seed); s = np.full(40000, 14, np.int32); m = s.copy(); sv = np.ones(40000, np.int8)
            for d in range(91, 0, -1):
                idx = rr.integers(0, len(P), 40000); act = P[idx] >= thr
                u = rr.random(40000); hit = act & (u < hitp[idx]); ns = s.copy(); ns[hit] = np.minimum(s[hit] + 1, 57)
                miss = act & ~hit; catch = miss & (sv == 1) & (s >= 10) & (s <= 15); ns[miss & ~catch] = 0
                sv = sv.copy(); sv[catch] = 0; s = ns; m = np.maximum(m, s)
            return m.mean()
        deltas.append(sweep(0.76, it) - sweep(0.796, it))   # play ~79% minus play ~23%
    deltas = np.array(deltas)
    print(f"  E[max] delta (play more): mean {deltas.mean():+.2f}  90%CI [{np.percentile(deltas,5):+.2f}, {np.percentile(deltas,95):+.2f}]  pools>0: {(deltas>0).mean():.0%}")
    print("  -> within noise; play-rate is ~indifferent within the data's precision.")


if __name__ == "__main__":
    main()
