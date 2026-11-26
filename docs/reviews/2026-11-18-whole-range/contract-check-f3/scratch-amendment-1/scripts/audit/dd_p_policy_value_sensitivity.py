#!/usr/bin/env python3
"""Deployed-policy value contrast at streaks 0-2 under DD-leg p sensitivity.

Queued by docs/audit/2026-07-12-dd-leg-calibration.md (review r2#6): the 7/06
"strategy is a wash" replay (scripts/audit/confirm_mdp_policy_replay.py)
consumed the profiles' REALIZED DD-leg outcomes, which match stated p's to
~+1pp on the backtest basis. Live 2026 DD legs realize 0.595 vs 0.734 stated
(n=42, exact tail p=0.035). If the live shortfall is real rather than luck,
does the deployed always-double-at-low-streaks policy become value-negative
at streaks 0-2 — and at what leg-rate haircut does the optimal action flip?

Environment: the pooled estimated_pa profiles (data/hetzner_results/
mdp_estpa_run, 24 seeds x 5 seasons — the realistic basis; see the CLAUDE.md
PROFILE BASIS warning), paired under the production different-game DD rule.
Day types are the full (env-quintile x deployed-policy-bin) grid so the
deployed policy's own digitization (boundaries ~0.796+, mostly bin 0 on this
scale) and the environment's quintiles coexist in one type space.

Sensitivity parameter: an additive haircut delta on the per-type conditional
DD-leg rate — p_both' = max(0, p_both - delta * p_hit), i.e. the leg hits
delta pp less often than the profiles say, uniformly across types. Linear in
(p_hit, p_both), so shading commutes with type aggregation below the floor.
Delta is a SPECIFIED UNIFORM STRESS SCENARIO, not a direct operationalization
of the live estimate (which is a marginal shortfall on played DD legs and
cannot be localized by p2/bin/month at n=42). The conditional-vs-marginal and
pooled-vs-doubled-days denominator gaps are small on this data (deployed
doubled-day marginal leg rate 0.740 vs pooled 0.7465 — review 2026-07-13
r1#5), but a shortfall CONCENTRATED somewhere specific is out of scope.

Two layers:

L1 (analytic, exact backward induction; reach-K currencies K in {57,30,20}):
  - re-solved optimal policy on the shaded 5-quintile environment: does the
    optimal action at streaks 0-2 stop doubling, and at what delta?
  - fixed-policy values on the shaded joint environment: deployed vs
    no-DD-below-streak-3 vs always-single vs always-double vs the re-solved
    optimal (materiality gaps in currency units).
  - local leg-rate breakeven r* = (EV[s+1]-EV[0])/(EV[s+2]-EV[0]) under the
    deployed policy's own continuation values (valid outside the saver zone;
    at s<10 the miss branch is a common reset term that cancels).

L2 (empirical replay): the 7/06 comparator's different-game replay extended
  with stochastic leg thinning — each rep flips top2_hit=1 -> 0 with uniform
  q = delta / r_bar so the aggregate leg rate drops by exactly delta pp.
  Thinned outcomes are shared across policies (common random numbers).
  Metrics: mean max_streak, reach20/30, resets, paired per-trajectory wins.
  At delta=0 the replay must reproduce the 7/06 comparator numbers exactly
  (asserted, tol 0.005: mdp 17.73/31.7%/36.8, single 15.94/7.5%/41.4,
  double 17.85/40.8%/76.0).

Caveats (recorded up front):
  - delta is applied uniformly across types/days; the live evidence (n=42)
    cannot localize the shortfall beyond the DD slot.
  - P(57)-currency values on this basis are ~1e-4 (near-unwinnable; 7/06 doc)
    — K=20/30 are the decision-relevant currencies.
  - The 24 seeds re-use the same 908 season dates (~7.4x); profile-level SEs
    understate date-cluster noise (2026-07-12 doc r2#5). Per-season paired
    means are reported for sign-robustness.
  - estimated_pa conditions on realized participation; the environment
    inherits that (same doc, finding 3 caveat).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from bts.simulate.mdp import load_policy
from bts.simulate.quality_bins import QualityBins

SEASON_LENGTH = 180
LATE_PHASE_DAYS = 30
PROFILE_ROOT = "data/hetzner_results/mdp_estpa_run"
DEPLOYED_POLICY = Path("data/models/mdp_policy.npz")
DEPLOYED_STREAK_CAP = 56  # mirrors bts.simulate.mdp.lookup_action
THINNING_SEED = 20260713

# 7/06 comparator anchor (different-game pairing, delta=0) — reproduced run
# 2026-07-13; L2 asserts against these before any delta>0 cell is trusted.
# Tolerances = half the comparator's print precision (mean_max %.2f,
# reach20 %.1f%%, resets %.1f).
ANCHOR_DIFFGAME = {
    "mdp_deployed": {"mean_max": 17.73, "reach20": 0.3167, "resets": 36.8},
    "always_single": {"mean_max": 15.94, "reach20": 0.0750, "resets": 41.4},
    "always_double": {"mean_max": 17.85, "reach20": 0.4083, "resets": 76.0},
}
ANCHOR_TOL = {"mean_max": 0.005, "reach20": 0.0005, "resets": 0.05}


# --------------------------------------------------------------------- pairing


def pair_diffgame(df: pd.DataFrame) -> pd.DataFrame:
    """rank-1 + best rank>=2 in a DIFFERENT game_pk (production's DD rule).

    Semantics identical to confirm_mdp_policy_replay.pair_diffgame (verbatim
    fallback: no different-game candidate -> the leg outcome collapses onto
    rank-1's own outcome), extended to carry the leg's own p and the fallback
    flag. This is COMPARATOR PARITY, not production parity: production's
    decide_action guards executability and demotes a partnerless double to a
    single, whereas the fallback here lets a double advance +2 on rank-1's
    single outcome. It touches 0.46% of paired days; correcting it moved
    replay mean-max by <0.001 in review (2026-07-13 r1#6). Kept verbatim so
    the delta=0 anchor against the 7/06 comparator stays exact.
    """
    rows = []
    for date, g in df.sort_values("rank").groupby("date"):
        r1 = g[g["rank"] == 1]
        if r1.empty:
            continue
        r1 = r1.iloc[0]
        others = g[(g["rank"] >= 2) & (g["game_pk"] != r1["game_pk"])]
        if others.empty:
            top2, p2, fallback = int(r1["actual_hit"]), float(r1["p_game_hit"]), True
        else:
            leg = others.iloc[0]
            top2, p2, fallback = int(leg["actual_hit"]), float(leg["p_game_hit"]), False
        rows.append(
            {
                "date": date,
                "p1": float(r1["p_game_hit"]),
                "top1": int(r1["actual_hit"]),
                "top2": top2,
                "p2": p2,
                "same_game_fallback": fallback,
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------- environment


@dataclass(frozen=True)
class Env:
    """Day-type environment: frequencies and empirical transition p's.

    One entry per day type. p_both is the empirical joint P(top1 & top2)
    under the different-game pairing. env_bin / pol_bin give each type's
    quintile in the environment's boundaries and in the deployed policy's
    saved boundaries respectively.
    """

    freq: np.ndarray
    p_hit: np.ndarray
    p_both: np.ndarray
    env_bin: np.ndarray
    pol_bin: np.ndarray


def quintile_boundaries(values: pd.Series, n_bins: int = 5) -> list[float]:
    """Equal-frequency boundaries, mirroring compute_pooled_bins."""
    quantiles = [i / n_bins for i in range(1, n_bins)]
    return [float(pd.Series(values).quantile(q)) for q in quantiles]


def build_env(
    paired: pd.DataFrame,
    env_boundaries: list[float],
    pol_boundaries: list[float],
) -> Env:
    """Joint (env_bin x pol_bin) day-type environment on the FULL grid.

    Empty cells are retained at freq 0 (compute_bins_with_boundaries
    precedent) so early- and late-phase environments share one type space.
    np.digitize matches QualityBins.classify / lookup_action: a value equal
    to a boundary lands in the upper bin.
    """
    p1 = paired["p1"].to_numpy(dtype=float)
    top1 = paired["top1"].to_numpy(dtype=bool)
    top2 = paired["top2"].to_numpy(dtype=bool)
    eb = np.digitize(p1, env_boundaries)
    pb = np.digitize(p1, pol_boundaries)

    n_env = len(env_boundaries) + 1
    n_pol = len(pol_boundaries) + 1
    n_days = len(paired)
    freq, p_hit, p_both, env_bin, pol_bin = [], [], [], [], []
    for e in range(n_env):
        for p in range(n_pol):
            mask = (eb == e) & (pb == p)
            n = int(mask.sum())
            env_bin.append(e)
            pol_bin.append(p)
            if n == 0:
                freq.append(0.0)
                p_hit.append(0.0)
                p_both.append(0.0)
                continue
            freq.append(n / n_days)
            p_hit.append(float(top1[mask].mean()))
            p_both.append(float((top1[mask] & top2[mask]).mean()))
    return Env(
        freq=np.array(freq),
        p_hit=np.array(p_hit),
        p_both=np.array(p_both),
        env_bin=np.array(env_bin, dtype=np.int64),
        pol_bin=np.array(pol_bin, dtype=np.int64),
    )


def env_from_quality_bins(qb: QualityBins) -> Env:
    """Adapter for oracle comparisons against bts.simulate machinery."""
    t = len(qb.bins)
    return Env(
        freq=np.array([b.frequency for b in qb.bins]),
        p_hit=np.array([b.p_hit for b in qb.bins]),
        p_both=np.array([b.p_both for b in qb.bins]),
        env_bin=np.arange(t, dtype=np.int64),
        pol_bin=np.arange(t, dtype=np.int64),
    )


def aggregate_env_by_env_bin(env: Env) -> Env:
    """Collapse joint cells to env-quintile granularity (freq-weighted)."""
    out_bins = np.unique(env.env_bin)
    freq = np.zeros(len(out_bins))
    p_hit = np.zeros(len(out_bins))
    p_both = np.zeros(len(out_bins))
    for i, b in enumerate(out_bins):
        mask = env.env_bin == b
        f = env.freq[mask].sum()
        freq[i] = f
        if f > 0:
            p_hit[i] = float((env.freq[mask] * env.p_hit[mask]).sum() / f)
            p_both[i] = float((env.freq[mask] * env.p_both[mask]).sum() / f)
    return Env(freq=freq, p_hit=p_hit, p_both=p_both, env_bin=out_bins, pol_bin=out_bins.copy())


def shade_env(env: Env, delta: float) -> Env:
    """Haircut the conditional DD-leg rate by delta (primaries untouched).

    p_both' = p_hit * max(0, p_both/p_hit - delta) = max(0, p_both - delta*p_hit).
    Linear in (p_hit, p_both) below the floor, so it commutes with
    aggregate_env_by_env_bin.
    """
    return replace(env, p_both=np.clip(env.p_both - delta * env.p_hit, 0.0, None))


# ---------------------------------------------------------------------- solver


def freq_at(env_early: Env, env_late: Env | None, d: int, late_phase_days: int = LATE_PHASE_DAYS) -> np.ndarray:
    """Day-type frequencies for the phase at d days remaining."""
    if env_late is not None and d <= late_phase_days:
        return env_late.freq
    return env_early.freq


def solve_reach(
    env_early: Env,
    env_late: Env | None,
    K: int,
    season_length: int = SEASON_LENGTH,
    late_phase_days: int = LATE_PHASE_DAYS,
    action_fn=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Backward-induction reach-K value on the day-type environment.

    Semantics mirror bts.simulate.mdp.solve_mdp / transition_outcomes with
    target=K: success caps at K (absorbing, V=1); a miss with the saver at
    streak 10-15 holds the streak and consumes the saver; a miss otherwise
    resets the streak and keeps the saver.

    action_fn(s, d, saver) -> (T,) int actions fixes the policy (evaluation
    mode, returns policy=None); action_fn=None optimizes (argmax over
    skip/single/double, first-max tie-break like np.argmax / solve_mdp).
    """
    T = len(env_early.freq)
    if env_late is not None and len(env_late.freq) != T:
        raise ValueError("early/late environments must share one type space")
    V = np.zeros((K + 1, season_length + 1, 2, T))
    V[K, :, :, :] = 1.0
    policy = None if action_fn is not None else np.zeros((K, season_length + 1, 2, T), dtype=np.int8)
    idx = np.arange(T)

    for d in range(1, season_length + 1):
        env_now = env_late if (env_late is not None and d <= late_phase_days) else env_early
        nf = freq_at(env_early, env_late, d - 1, late_phase_days)
        p_hit = env_now.p_hit
        p_both = env_now.p_both
        for s in range(K):
            s1 = min(s + 1, K)
            s2 = min(s + 2, K)
            for saver in (0, 1):
                ev_stay = float(nf @ V[s, d - 1, saver, :])
                ev_s1 = float(nf @ V[s1, d - 1, saver, :])
                ev_s2 = float(nf @ V[s2, d - 1, saver, :])
                if saver and 10 <= s <= 15:
                    ev_miss = float(nf @ V[s, d - 1, 0, :])
                else:
                    ev_miss = float(nf @ V[0, d - 1, saver, :])
                q = np.empty((3, T))
                q[0] = ev_stay
                q[1] = p_hit * ev_s1 + (1.0 - p_hit) * ev_miss
                q[2] = p_both * ev_s2 + (1.0 - p_both) * ev_miss
                if action_fn is None:
                    a = q.argmax(axis=0)
                    policy[s, d, saver, :] = a
                else:
                    a = np.asarray(action_fn(s, d, saver), dtype=np.int64)
                V[s, d, saver, :] = q[a, idx]
    return V, policy


def start_value(
    V: np.ndarray,
    env_early: Env,
    env_late: Env | None,
    s: int,
    d: int,
    saver: int,
    late_phase_days: int = LATE_PHASE_DAYS,
) -> float:
    """E over day type of V at (s, d, saver) — solve_mdp's terminal convention."""
    return float(freq_at(env_early, env_late, d, late_phase_days) @ V[s, d, saver, :])


def table_action_fn(table: np.ndarray, bin_of_type: np.ndarray, streak_cap: int):
    """Action function reading a saved policy table through per-type bins."""
    bin_of_type = np.asarray(bin_of_type)
    d_cap = table.shape[1] - 1

    def fn(s: int, d: int, saver: int) -> np.ndarray:
        return table[min(s, streak_cap), min(d, d_cap), saver, :][bin_of_type]

    return fn


def const_action_fn(action: int, n_types: int):
    a = np.full(n_types, action, dtype=np.int8)

    def fn(s: int, d: int, saver: int) -> np.ndarray:
        return a

    return fn


def no_dd_low_action_fn(base_fn, max_streak: int = 2):
    """Deployed behavior except: no double-down at streaks <= max_streak."""

    def fn(s: int, d: int, saver: int) -> np.ndarray:
        a = np.array(base_fn(s, d, saver), dtype=np.int8, copy=True)
        if s <= max_streak:
            a[a == 2] = 1
        return a

    return fn


def leg_breakeven(
    V: np.ndarray,
    env_early: Env,
    env_late: Env | None,
    s: int,
    d: int,
    saver: int,
    K: int,
    late_phase_days: int = LATE_PHASE_DAYS,
) -> float:
    """Conditional leg rate r* at which Q(double) == Q(single).

    r* = (EV[s+1] - EV[0]) / (EV[s+2] - EV[0]) at d-1 under V's continuation.
    Valid only outside the saver-catch zone (there the miss branch differs
    between representations and the reset term no longer cancels).
    """
    if saver and 10 <= s <= 15:
        raise ValueError("leg_breakeven is undefined in the saver-catch zone (streak 10-15 with saver)")
    nf = freq_at(env_early, env_late, d - 1, late_phase_days)
    ev1 = float(nf @ V[min(s + 1, K), d - 1, saver, :])
    ev2 = float(nf @ V[min(s + 2, K), d - 1, saver, :])
    ev0 = float(nf @ V[0, d - 1, saver, :])
    denom = ev2 - ev0
    if denom <= 0:
        return float("inf")
    return (ev1 - ev0) / denom


# ---------------------------------------------------------------------- replay


def thinned_top2(top2: np.ndarray, q: float, n_reps: int, rng: np.random.Generator) -> np.ndarray:
    """(n_reps, n) leg outcomes with hits flipped off independently w.p. q."""
    top2 = np.asarray(top2, dtype=bool)
    if q <= 0:
        return np.broadcast_to(top2, (n_reps, top2.size)).copy()
    return top2[None, :] & (rng.random((n_reps, top2.size)) >= q)


def replay_provider_table(table: np.ndarray, bins_per_day: np.ndarray, streak_cap: int):
    bins_per_day = np.asarray(bins_per_day)
    d_cap = table.shape[1] - 1

    def provider(streak: np.ndarray, d: int, saver: np.ndarray, i: int) -> np.ndarray:
        return table[np.minimum(streak, streak_cap), min(d, d_cap), saver, bins_per_day[i]]

    return provider


def replay_provider_const(action: int):
    def provider(streak: np.ndarray, d: int, saver: np.ndarray, i: int) -> np.ndarray:
        return np.full(streak.shape, action, dtype=np.int8)

    return provider


def replay_provider_no_dd_low(base_provider, max_streak: int = 2):
    def provider(streak: np.ndarray, d: int, saver: np.ndarray, i: int) -> np.ndarray:
        a = np.array(base_provider(streak, d, saver, i), copy=True)
        a[(streak <= max_streak) & (a == 2)] = 1
        return a

    return provider


def replay_vectorized(
    top1: np.ndarray,
    thinned2: np.ndarray,
    provider,
    season_length: int = SEASON_LENGTH,
) -> dict[str, np.ndarray]:
    """Replay one season for all reps at once.

    Semantics are a vectorized port of confirm_mdp_policy_replay.replay_season
    (itself verbatim from scripts/mc_replay_ab.py): days_remaining = the fixed
    season_length minus row position (NOT the calendar clock production uses —
    the 179-185-date files lose the rows past position 180, 312 of 21,888
    paired rows; comparator parity again), streak >= 57 freezes the
    trajectory; the saver catches one miss at streak 10-15.
    """
    top1 = np.asarray(top1, dtype=bool)
    th2 = np.asarray(thinned2, dtype=bool)
    n_reps, n = th2.shape
    streak = np.zeros(n_reps, dtype=np.int64)
    saver = np.ones(n_reps, dtype=np.int64)
    maxs = np.zeros(n_reps, dtype=np.int64)
    resets = np.zeros(n_reps, dtype=np.int64)

    for i in range(n):
        d = season_length - i
        if d <= 0:
            break
        a = np.asarray(provider(streak, d, saver, i), dtype=np.int64)
        a = np.where(streak >= 57, 0, a)
        played = a > 0
        if not played.any():
            continue
        hit1 = bool(top1[i])
        success = played & (((a == 1) & hit1) | ((a == 2) & hit1 & th2[:, i]))
        miss = played & ~success
        catch = miss & (saver == 1) & (streak >= 10) & (streak <= 15)
        hard = miss & ~catch
        streak = streak + np.where(success, np.where(a == 2, 2, 1), 0)
        streak[hard] = 0
        saver[catch] = 0
        resets += hard
        np.maximum(maxs, streak, out=maxs)

    return {
        "max_streak": maxs,
        "resets": resets,
        "reach20": maxs >= 20,
        "reach30": maxs >= 30,
        "reach57": maxs >= 57,
    }


# ------------------------------------------------------------ run diagnostics


def hit_runs(outcomes: np.ndarray) -> list[int]:
    """Lengths of maximal consecutive-hit runs in a 0/1 outcome sequence."""
    runs, c = [], 0
    for x in np.asarray(outcomes, dtype=bool):
        if x:
            c += 1
        elif c:
            runs.append(c)
            c = 0
    if c:
        runs.append(c)
    return runs


RUN_TAIL_LS = (3, 5, 8, 10, 12, 15, 20)


def run_structure_diagnostics(
    paired_all: pd.DataFrame,
    window: int = 20,
    n_perms: int = 100,
    seed: int = THINNING_SEED,
) -> dict:
    """Measure serial run structure of realized rank-1 outcomes vs order nulls.

    Motivated 2026-07-13: the realized replay's reach-20 (7.5% for
    always-single) sits far below the iid day-type value (~27%) at the SAME
    daily hit rate. This quantifies the discrepancy with no replay machinery.

    Statistics per profile file: all-hit-`window` counts observed vs (a) a
    plug-in iid null at the file's own mean rate, (b) independence given the
    stated per-day p1, and (c) a within-file day-order PERMUTATION null
    (n_perms shuffles; preserves the file's exact hit count and length,
    destroys serial order) — (c) is the load-bearing null: it is immune to
    rate heterogeneity across files/seasons. The run-length survivor tail is
    reported against the same permutation null and against pooled geometric
    (heterogeneity makes the geometric comparison conservative: mixtures of
    geometrics are fatter-tailed than geometric at the pooled mean).

    Interpretation discipline (review 2026-07-13 r1#1): this measures
    long-window suppression of CONSECUTIVE-hit runs. It does not establish a
    monotone hazard (small-L ratios are noisy/non-monotone; the deep tail has
    few runs), a mechanism, or a uniform "iid inflates every policy" claim —
    the realized-vs-iid gap is policy-dependent (it grows with the
    consecutive-hit calendar-run length a policy needs). Effective sample is
    closer to the 5 seasons than the 120 files: the 24 seeds' rank-1
    sequences are heavily correlated within a season.
    """
    obs_w, iid_w, p1_w, perm_w = [], [], [], []
    all_runs: list[int] = []
    perm_run_counts = {L: 0.0 for L in RUN_TAIL_LS}
    perm_runs_total = 0.0
    lag1 = []
    total_hits = total_days = 0.0
    season_acc: dict[int, list[float]] = {}
    for fi, g in paired_all.groupby("file_idx"):
        # per-file rng stream (review r2: a single global stream makes every
        # later file's shuffles depend on the file set/order)
        rng = np.random.default_rng([seed, int(fi)])
        g = g.sort_values("date")
        t1 = g["top1"].to_numpy(dtype=float)
        p1 = g["p1"].to_numpy(dtype=float)
        n = len(t1)
        if n < window:
            continue
        total_hits += float(t1.sum())
        total_days += n
        lag1.append(float(np.corrcoef(t1[:-1], t1[1:])[0, 1]))
        w = np.lib.stride_tricks.sliding_window_view(t1, window)
        obs = float((w.min(axis=1) == 1).sum())
        obs_w.append(obs)
        exp_iid = (n - window + 1) * float(t1.mean()) ** window
        iid_w.append(exp_iid)
        wp = np.lib.stride_tricks.sliding_window_view(p1, window)
        p1_w.append(float(np.prod(wp, axis=1).sum()))
        all_runs.extend(hit_runs(t1))
        if "season" in g.columns:
            acc = season_acc.setdefault(int(g["season"].iloc[0]), [0.0, 0.0])
            acc[0] += obs
            acc[1] += exp_iid
        # day-order permutation null (rates/counts fixed, order destroyed)
        pw = 0.0
        for _ in range(n_perms):
            perm = rng.permutation(t1)
            wv = np.lib.stride_tricks.sliding_window_view(perm, window)
            pw += float((wv.min(axis=1) == 1).sum())
            pruns = hit_runs(perm)
            perm_runs_total += len(pruns) / n_perms
            for L in RUN_TAIL_LS:
                perm_run_counts[L] += sum(1 for r in pruns if r >= L) / n_perms
        perm_w.append(pw / n_perms)
    runs = np.array(all_runs)
    q_bar = float(total_hits / total_days)  # day-pooled (review r2: not file-mean)
    tail = {}
    for L in RUN_TAIL_LS:
        emp = float((runs >= L).mean())
        geo = q_bar ** (L - 1)  # survivor of Geometric(1-q) given run >= 1
        perm = perm_run_counts[L] / perm_runs_total if perm_runs_total else float("nan")
        tail[L] = {
            "empirical": emp,
            "geometric_pooled": geo,
            "permutation": perm,
            "ratio_vs_permutation": emp / perm if perm else float("nan"),
        }
    return {
        "window": window,
        "n_perms": n_perms,
        "n_runs": int(len(runs)),
        "pooled_top1_rate": q_bar,
        "lag1_autocorr_mean": float(np.mean(lag1)),
        "allhit_windows_per_file": {
            "observed": float(np.mean(obs_w)),
            "iid_at_file_mean": float(np.mean(iid_w)),
            "independent_given_p1": float(np.mean(p1_w)),
            "permutation": float(np.mean(perm_w)),
            "observed_over_iid": float(np.mean(obs_w) / np.mean(iid_w)),
            "observed_over_permutation": float(np.mean(obs_w) / np.mean(perm_w)),
        },
        "per_season_window_ratio": {
            s: {"observed": o, "iid_at_file_mean": e, "ratio": (o / e if e else float("nan"))}
            for s, (o, e) in sorted(season_acc.items())
        },
        "run_tail": tail,
    }


# ----------------------------------------------------------------- L1 driver


def split_phase(paired_all: pd.DataFrame, late_phase_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last late_phase_days distinct dates of each season are late-phase.

    Mirrors bts.simulate.pooled_policy.split_by_phase_pooled.
    """
    early_rows, late_rows = [], []
    for _season, group in paired_all.groupby("season"):
        dates = sorted(pd.Series(group["date"]).unique())
        if len(dates) <= late_phase_days:
            late_rows.append(group)
            continue
        cutoff = dates[-late_phase_days]
        late_rows.append(group[group["date"] >= cutoff])
        early_rows.append(group[group["date"] < cutoff])
    early = pd.concat(early_rows, ignore_index=True) if early_rows else paired_all.iloc[0:0]
    late = pd.concat(late_rows, ignore_index=True) if late_rows else paired_all.iloc[0:0]
    return early, late


def load_paired_pooled(root: str = PROFILE_ROOT) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(glob.glob(f"{root}/**/backtest_*.parquet", recursive=True))
    if not files:
        raise SystemExit(f"no profile parquets under {root}")
    frames = []
    for idx, f in enumerate(files):
        season = int(re.search(r"backtest_(\d{4})\.parquet", f).group(1))
        seed = int(re.search(r"simulation_seed(\d+)", f).group(1))
        paired = pair_diffgame(pd.read_parquet(f))
        paired["season"] = season
        paired["seed"] = seed
        paired["file_idx"] = idx
        frames.append(paired)
    return pd.concat(frames, ignore_index=True), files


def run_l1(
    paired_all: pd.DataFrame,
    table: np.ndarray,
    pol_boundaries: list[float],
    deltas: list[float],
    targets: list[int],
    d_report: list[int],
    season_length: int = SEASON_LENGTH,
    late_phase_days: int = LATE_PHASE_DAYS,
) -> dict:
    early_df, late_df = split_phase(paired_all, late_phase_days)
    env_bounds_early = quintile_boundaries(early_df["p1"])
    env_bounds_late = quintile_boundaries(late_df["p1"])
    joint_e = build_env(early_df, env_bounds_early, pol_boundaries)
    joint_l = build_env(late_df, env_bounds_late, pol_boundaries)
    assert np.array_equal(joint_e.pol_bin, joint_l.pol_bin)
    env5_e = aggregate_env_by_env_bin(joint_e)
    env5_l = aggregate_env_by_env_bin(joint_l)
    assert len(env5_e.freq) == 5 and len(env5_l.freq) == 5

    n_types = len(joint_e.freq)
    dep_fn = table_action_fn(table, joint_e.pol_bin, streak_cap=DEPLOYED_STREAK_CAP)
    agg_leg_rate0 = float((joint_e.freq * joint_e.p_both).sum() / (joint_e.freq * joint_e.p_hit).sum())

    out = {
        "env_boundaries_early": env_bounds_early,
        "env_boundaries_late": env_bounds_late,
        "pol_boundaries": pol_boundaries,
        "agg_conditional_leg_rate_early": agg_leg_rate0,
        "deltas": deltas,
        "d_report": d_report,
        "targets": {},
        "resolved_tables": {},  # delta -> (K=57) env5 policy table, for L2
    }

    for K in targets:
        rows = []
        for delta in deltas:
            jE, jL = shade_env(joint_e, delta), shade_env(joint_l, delta)
            e5E, e5L = shade_env(env5_e, delta), shade_env(env5_l, delta)

            V_opt, pol_opt = solve_reach(e5E, e5L, K, season_length, late_phase_days)
            V_dep, _ = solve_reach(jE, jL, K, season_length, late_phase_days, action_fn=dep_fn)
            V_nodd, _ = solve_reach(
                jE, jL, K, season_length, late_phase_days, action_fn=no_dd_low_action_fn(dep_fn)
            )
            V_sing, _ = solve_reach(
                jE, jL, K, season_length, late_phase_days, action_fn=const_action_fn(1, n_types)
            )
            V_ad, _ = solve_reach(
                jE, jL, K, season_length, late_phase_days, action_fn=const_action_fn(2, n_types)
            )
            if K == 57:
                out["resolved_tables"][delta] = pol_opt

            rec = {"delta": delta, "states": []}
            for s in (0, 1, 2):
                for d in d_report:
                    rec["states"].append(
                        {
                            "s": s,
                            "d": d,
                            "opt_actions_by_env_bin": [int(a) for a in pol_opt[s, d, 1, :]],
                            "v_opt": start_value(V_opt, e5E, e5L, s, d, 1, late_phase_days),
                            "v_deployed": start_value(V_dep, jE, jL, s, d, 1, late_phase_days),
                            "v_no_dd_low": start_value(V_nodd, jE, jL, s, d, 1, late_phase_days),
                            "v_always_single": start_value(V_sing, jE, jL, s, d, 1, late_phase_days),
                            "v_always_double": start_value(V_ad, jE, jL, s, d, 1, late_phase_days),
                            "leg_breakeven_deployed": leg_breakeven(
                                V_dep, jE, jL, s, d, 1, K, late_phase_days
                            ),
                        }
                    )
            rec["agg_conditional_leg_rate"] = max(0.0, agg_leg_rate0 - delta)
            rows.append(rec)
        out["targets"][K] = rows
    return out


# ----------------------------------------------------------------- L2 driver


def _season_arrays(paired_all: pd.DataFrame, env_bounds_early, env_bounds_late, pol_boundaries, late_phase_days):
    """Per-file day arrays for the replay, with phase-aware env binning."""
    arrays = []
    for (file_idx, season, seed), g in paired_all.groupby(["file_idx", "season", "seed"]):
        g = g.sort_values("date")
        dates = sorted(g["date"].unique())
        late_from = dates[-late_phase_days] if len(dates) > late_phase_days else dates[0]
        is_late = (g["date"] >= late_from).to_numpy()
        p1 = g["p1"].to_numpy(dtype=float)
        env_bin = np.where(
            is_late,
            np.digitize(p1, env_bounds_late),
            np.digitize(p1, env_bounds_early),
        )
        arrays.append(
            {
                "file_idx": int(file_idx),
                "season": int(season),
                "seed": int(seed),
                "top1": g["top1"].to_numpy(dtype=bool),
                "top2": g["top2"].to_numpy(dtype=bool),
                "pol_bin": np.digitize(p1, pol_boundaries),
                "env_bin": env_bin,
            }
        )
    return arrays


def run_l2(
    paired_all: pd.DataFrame,
    table: np.ndarray,
    pol_boundaries: list[float],
    env_bounds_early: list[float],
    env_bounds_late: list[float],
    resolved_tables: dict[float, np.ndarray],
    deltas: list[float],
    reps: int,
    season_length: int = SEASON_LENGTH,
    late_phase_days: int = LATE_PHASE_DAYS,
    assert_anchor: bool = True,
) -> dict:
    r_bar = float(paired_all["top2"].mean())
    arrays = _season_arrays(paired_all, env_bounds_early, env_bounds_late, pol_boundaries, late_phase_days)

    def providers_for(sa, delta):
        base = replay_provider_table(table, sa["pol_bin"], streak_cap=DEPLOYED_STREAK_CAP)
        provs = {
            "mdp_deployed": base,
            "always_single": replay_provider_const(1),
            "always_double": replay_provider_const(2),
            "deployed_no_dd_s0_2": replay_provider_no_dd_low(base, max_streak=2),
        }
        resolved = resolved_tables.get(delta)
        if resolved is not None:
            provs["resolved57_at_delta"] = replay_provider_table(
                resolved, sa["env_bin"], streak_cap=resolved.shape[0] - 1
            )
        return provs

    results = {"r_bar": r_bar, "reps": reps, "deltas": deltas, "cells": []}
    for d_idx, delta in enumerate(deltas):
        q = delta / r_bar
        n_reps = 1 if delta == 0 else reps
        per_policy: dict[str, dict[str, list[float]]] = {}
        realized_leg_rates = []
        for sa in arrays:
            rng = np.random.default_rng(THINNING_SEED + 1_000_003 * sa["file_idx"] + d_idx)
            th2 = thinned_top2(sa["top2"], q, n_reps, rng)
            realized_leg_rates.append(float(th2.mean()))
            for name, provider in providers_for(sa, delta).items():
                got = replay_vectorized(sa["top1"], th2, provider, season_length)
                rec = per_policy.setdefault(
                    name,
                    {"mean_max": [], "reach20": [], "reach30": [], "reach57": [], "resets": [], "season": []},
                )
                rec["mean_max"].append(float(got["max_streak"].mean()))
                rec["reach20"].append(float(got["reach20"].mean()))
                rec["reach30"].append(float(got["reach30"].mean()))
                rec["reach57"].append(float(got["reach57"].mean()))
                rec["resets"].append(float(got["resets"].mean()))
                rec["season"].append(sa["season"])

        cell = {"delta": delta, "q": q, "realized_leg_rate": float(np.mean(realized_leg_rates)), "policies": {}}
        for name, rec in per_policy.items():
            mm = np.array(rec["mean_max"])
            cell["policies"][name] = {
                "mean_max": float(mm.mean()),
                "mean_max_se_profiles": float(mm.std(ddof=1) / np.sqrt(len(mm))),
                "reach20": float(np.mean(rec["reach20"])),
                "reach30": float(np.mean(rec["reach30"])),
                "reach57": float(np.mean(rec["reach57"])),
                "resets": float(np.mean(rec["resets"])),
            }
        # paired contrasts vs deployed, per profile then per season
        dep = np.array(per_policy["mdp_deployed"]["mean_max"])
        seasons = np.array(per_policy["mdp_deployed"]["season"])
        for name, rec in per_policy.items():
            if name == "mdp_deployed":
                continue
            diff = np.array(rec["mean_max"]) - dep
            by_season = {
                int(s): float(diff[seasons == s].mean()) for s in np.unique(seasons)
            }
            cell["policies"][name]["mean_max_minus_deployed"] = float(diff.mean())
            cell["policies"][name]["mean_max_minus_deployed_se_profiles"] = float(
                diff.std(ddof=1) / np.sqrt(len(diff))
            )
            cell["policies"][name]["mean_max_minus_deployed_by_season"] = by_season
            r20 = np.array(rec["reach20"]) - np.array(per_policy["mdp_deployed"]["reach20"])
            cell["policies"][name]["reach20_minus_deployed"] = float(r20.mean())
        results["cells"].append(cell)

        if delta == 0 and assert_anchor:
            for name, want in ANCHOR_DIFFGAME.items():
                got = cell["policies"][name]
                for metric, expected in want.items():
                    actual = got[metric]
                    if abs(actual - expected) > ANCHOR_TOL[metric] + 1e-9:
                        raise AssertionError(
                            f"delta=0 anchor mismatch for {name}.{metric}: "
                            f"got {actual:.4f}, expected {expected:.4f} "
                            "(7/06 comparator, different-game pairing)"
                        )
            results["anchor_check"] = "passed"
    return results


# -------------------------------------------------------------------- reports


def _fmt_pct(x: float) -> str:
    return f"{100 * x:6.1f}%"


def print_l1_report(l1: dict) -> None:
    print("\n" + "=" * 78)
    print("L1 — exact value contrast on the shaded estimated_pa environment")
    print("=" * 78)
    print(f"  env quintile boundaries (early): {[round(b, 4) for b in l1['env_boundaries_early']]}")
    print(f"  deployed policy boundaries:      {[round(b, 4) for b in l1['pol_boundaries']]}")
    print(f"  aggregate conditional leg rate at delta=0: {l1['agg_conditional_leg_rate_early']:.4f}")
    for K, rows in l1["targets"].items():
        print(f"\n  ---- currency: P(reach {K}) ----")
        flip_summary = {}
        for rec in rows:
            for st in rec["states"]:
                key = (st["s"], st["d"])
                if key not in flip_summary and any(a != 2 for a in st["opt_actions_by_env_bin"]):
                    flip_summary[key] = (rec["delta"], st["opt_actions_by_env_bin"])
        for s in (0, 1, 2):
            for d in l1["d_report"]:
                got = flip_summary.get((s, d))
                if got is None:
                    print(f"    optimal at s={s}, d={d}: double in ALL env bins across the whole delta grid")
                else:
                    print(
                        f"    optimal at s={s}, d={d}: first non-double at delta={got[0]:.3f} "
                        f"(actions by env bin: {got[1]})"
                    )
        # value/materialty table at the reporting deltas
        for rec in rows:
            if round(rec["delta"], 3) not in (0.0, 0.05, 0.1, 0.139, 0.15, 0.2):
                continue
            print(f"\n    delta = {rec['delta']:.3f} (agg leg rate {rec['agg_conditional_leg_rate']:.3f})")
            print(
                f"      {'state':>10s}  {'V_deployed':>12s}  {'V_opt':>12s}  {'V_noDD<=2':>12s}"
                f"  {'V_single':>12s}  {'V_double':>12s}  {'leg r*':>8s}"
            )
            for st in rec["states"]:
                if st["d"] != 74:
                    continue
                print(
                    f"      s={st['s']}, d={st['d']:3d}  {st['v_deployed']:12.5g}  {st['v_opt']:12.5g}"
                    f"  {st['v_no_dd_low']:12.5g}  {st['v_always_single']:12.5g}"
                    f"  {st['v_always_double']:12.5g}  {st['leg_breakeven_deployed']:8.4f}"
                )


def print_l2_report(l2: dict) -> None:
    print("\n" + "=" * 78)
    print(f"L2 — thinned-leg replay on 120 estimated_pa profiles ({l2['reps']} reps, common random numbers)")
    print("=" * 78)
    print(f"  pooled leg rate r_bar = {l2['r_bar']:.4f}")
    for cell in l2["cells"]:
        print(
            f"\n  delta = {cell['delta']:.3f}  (thinning q = {cell['q']:.4f}, "
            f"realized leg rate {cell['realized_leg_rate']:.4f})"
        )
        print(
            f"    {'policy':22s}  {'mean_max':>8s}  {'reach20':>8s}  {'reach30':>8s}"
            f"  {'resets':>7s}  {'vs deployed':>12s}"
        )
        for name, p in cell["policies"].items():
            gap = p.get("mean_max_minus_deployed")
            gap_s = f"{gap:+12.2f}" if gap is not None else " " * 12
            print(
                f"    {name:22s}  {p['mean_max']:8.2f}  {_fmt_pct(p['reach20'])}  {_fmt_pct(p['reach30'])}"
                f"  {p['resets']:7.1f}  {gap_s}"
            )
    if "anchor_check" in l2:
        print(f"\n  delta=0 anchor vs 7/06 comparator: {l2['anchor_check']}")


# ------------------------------------------------------------------------ CLI


def _fingerprint(files: list[str]) -> dict:
    h = hashlib.sha256()
    per_file = []
    for f in files:
        fh = hashlib.sha256(Path(f).read_bytes()).hexdigest()
        per_file.append({"path": f, "sha256": fh})
        h.update(f.encode())
        h.update(fh.encode())
    return {"n_files": len(files), "combined_sha256": h.hexdigest(), "files": per_file}


def _git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=PROFILE_ROOT)
    ap.add_argument("--policy", default=str(DEPLOYED_POLICY))
    ap.add_argument("--out", default="docs/audit/2026-07-13-dd-p-policy-value-sensitivity.json")
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--l1-delta-max", type=float, default=0.20)
    ap.add_argument("--l1-delta-step", type=float, default=0.005)
    ap.add_argument(
        "--l2-deltas", default="0,0.05,0.10,0.139", help="comma-separated additive leg haircuts"
    )
    ap.add_argument("--targets", default="57,30,20")
    ap.add_argument("--d-report", default="180,120,74,40")
    ap.add_argument("--stage", choices=["l1", "l2", "all"], default="all")
    ap.add_argument("--no-anchor-assert", action="store_true")
    args = ap.parse_args(argv)

    table, pol_boundaries, season_length = load_policy(args.policy)
    policy_sha = hashlib.sha256(Path(args.policy).read_bytes()).hexdigest()
    print(f"deployed policy: {args.policy} sha256={policy_sha[:16]}… season_length={season_length}")

    paired_all, files = load_paired_pooled(args.root)
    n_fallback = int(paired_all["same_game_fallback"].sum())
    print(
        f"profiles: {len(files)} files, {len(paired_all)} paired days, "
        f"{n_fallback} same-game fallbacks ({n_fallback / len(paired_all):.2%})"
    )

    diag = run_structure_diagnostics(paired_all)
    aw = diag["allhit_windows_per_file"]
    print(
        f"run structure: all-hit 20-windows/file observed {aw['observed']:.3f} vs "
        f"iid {aw['iid_at_file_mean']:.3f} (x{aw['observed_over_iid']:.2f}) vs "
        f"day-order permutation {aw['permutation']:.3f} (x{aw['observed_over_permutation']:.2f}); "
        "run-tail vs permutation: "
        + ", ".join(
            f"P(>={L})x{v['ratio_vs_permutation']:.2f}" for L, v in diag["run_tail"].items() if L >= 10
        )
    )
    print(
        "  per-season 20-window observed/iid: "
        + ", ".join(f"{s}: x{v['ratio']:.2f}" for s, v in diag["per_season_window_ratio"].items())
        + f"; lag-1 autocorr {diag['lag1_autocorr_mean']:+.3f}"
    )

    l1_deltas = [round(x, 4) for x in np.arange(0.0, args.l1_delta_max + 1e-9, args.l1_delta_step)]
    l2_deltas = [float(x) for x in args.l2_deltas.split(",")]
    targets = [int(x) for x in args.targets.split(",")]
    d_report = [int(x) for x in args.d_report.split(",")]

    # L1 always runs (cheap, and L2's resolved-policy arm needs its tables);
    # --stage l2 just skips the L1 printout.
    missing = [d for d in l2_deltas if d not in l1_deltas]
    l1 = run_l1(
        paired_all,
        table,
        [float(b) for b in pol_boundaries],
        sorted(set(l1_deltas + missing)),
        targets,
        d_report,
        season_length,
    )
    if args.stage in ("l1", "all"):
        print_l1_report(l1)

    l2 = None
    if args.stage in ("l2", "all"):
        l2 = run_l2(
            paired_all,
            table,
            [float(b) for b in pol_boundaries],
            l1["env_boundaries_early"],
            l1["env_boundaries_late"],
            l1["resolved_tables"],
            l2_deltas,
            args.reps,
            season_length,
            assert_anchor=not args.no_anchor_assert,
        )
        print_l2_report(l2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "scripts/audit/dd_p_policy_value_sensitivity.py",
        "git_head": _git_head(),
        "policy_path": args.policy,
        "policy_sha256": policy_sha,
        "params": {
            "reps": args.reps,
            "l2_deltas": l2_deltas,
            "targets": targets,
            "d_report": d_report,
            "season_length": season_length,
            "late_phase_days": LATE_PHASE_DAYS,
            "thinning_seed": THINNING_SEED,
        },
        "inputs": _fingerprint(files),
        "run_structure_diagnostics": diag,
        "l1": {
            **{k: v for k, v in l1.items() if k != "resolved_tables"},
            "targets": {
                str(K): rows for K, rows in l1["targets"].items()
            },
        },
        "l2": l2,
    }
    out_path.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
