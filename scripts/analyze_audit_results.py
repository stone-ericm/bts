#!/usr/bin/env python3
"""Unified analysis of overnight audit results.

Sources:
1. Phase 1 (Hetzner) — 16 seeds × baseline scorecards at audit_phase1/
2. Partial Phase 2 (Vultr) — 3 seeds × 16-19 experiments at audit_vultr_partial/
3. Earlier 5-seed variance run at seed_variance/ (for cross-check)
"""
import json
import statistics
from pathlib import Path

ROOT = Path("/Users/stone/projects/bts/data/hetzner_results")


def get_delta(d, *path):
    x = d
    for p in path:
        if not isinstance(x, dict):
            return None
        x = x.get(p, {})
    if isinstance(x, dict):
        return x.get("delta")
    return x


def analyze_phase1():
    """Phase 1: 16 seed scorecards. Report distribution of baseline metrics."""
    print("=" * 80)
    print("PHASE 1: BASELINE DISTRIBUTION (16 seeds on Hetzner)")
    print("=" * 80)
    rows = []
    for sc_path in sorted((ROOT / "audit_phase1").rglob("seed*.json")):
        seed = int(sc_path.stem.replace("seed", ""))
        d = json.load(open(sc_path))
        rows.append({
            "seed": seed,
            "avg_p1": d["precision"]["1"],
            "p1_2024": d["precision_by_season"]["2024"]["1"],
            "p1_2025": d["precision_by_season"]["2025"]["1"],
            "mdp": d.get("p_57_mdp", 0),
            "mc": d["streak_metrics"]["p_57_monte_carlo"],
            "miss": d["miss_analysis"]["n_miss_days"],
        })

    rows.sort(key=lambda r: r["seed"])
    print(f"\n{'seed':>9} {'avg P@1':>10} {'2024 P@1':>10} {'2025 P@1':>10} "
          f"{'MDP P(57)':>11} {'MC P(57)':>10} {'miss_days':>10}")
    print("-" * 82)
    for r in rows:
        print(f"{r['seed']:>9} {r['avg_p1']*100:>9.3f}% {r['p1_2024']*100:>9.3f}% "
              f"{r['p1_2025']*100:>9.3f}% {r['mdp']*100:>10.3f}% {r['mc']*100:>9.3f}% "
              f"{r['miss']:>10}")

    def stats(key, scale=100, label=""):
        vals = [r[key] for r in rows]
        m = statistics.mean(vals)
        s = statistics.stdev(vals)
        mn = min(vals)
        mx = max(vals)
        print(f"  {label:20s} mean={m*scale:.3f}  std=±{s*scale:.3f}  "
              f"range={mn*scale:.3f} → {mx*scale:.3f}  n={len(vals)}")

    print("\nBaseline distribution statistics:")
    stats("avg_p1", label="avg P@1 (pp)")
    stats("p1_2024", label="2024 P@1 (pp)")
    stats("p1_2025", label="2025 P@1 (pp)")
    stats("mdp", label="MDP P(57) (pp)")
    stats("mc", label="MC P(57) (pp)")
    stats("miss", scale=1, label="miss_days")

    # Flag outliers
    mdp_vals = [r["mdp"] for r in rows]
    mdp_mean = statistics.mean(mdp_vals)
    mdp_std = statistics.stdev(mdp_vals)
    print(f"\nOutliers (>2σ from MDP mean):")
    for r in rows:
        z = (r["mdp"] - mdp_mean) / mdp_std
        if abs(z) > 2:
            print(f"  seed={r['seed']}: MDP={r['mdp']*100:.3f}% (z={z:+.2f})")

    return rows


def analyze_phase2_partial():
    """Partial Phase 2: 3 seeds × 16-19 experiments each."""
    print("\n" + "=" * 80)
    print("PHASE 2 (PARTIAL): MULTI-SEED EXPERIMENT DELTAS (3 seeds)")
    print("=" * 80)

    seeds_done = [1, 7, 42]
    seed_dirs = {
        1: ROOT / "audit_vultr_partial/vultr-1_seed1/phase1",
        7: ROOT / "audit_vultr_partial/vultr-2_seed7/phase1",
        42: ROOT / "audit_vultr_partial/vultr-3_seed42/phase1",
    }

    # Collect per-experiment deltas across seeds
    per_exp = {}  # {exp_name: {seed: {metric_deltas}}}
    for seed, seed_dir in seed_dirs.items():
        if not seed_dir.exists():
            continue
        for exp_dir in sorted(seed_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            diff_path = exp_dir / "diff.json"
            summary_path = exp_dir / "summary.txt"
            if not diff_path.exists():
                continue
            diff = json.load(open(diff_path))
            summary = summary_path.read_text().strip() if summary_path.exists() else "?"
            per_exp.setdefault(exp_dir.name, {})[seed] = {
                "passed": summary.startswith("PASS"),
                "p24": get_delta(diff, "p_at_1_by_season", "2024") or 0,
                "p25": get_delta(diff, "p_at_1_by_season", "2025") or 0,
                "pavg": get_delta(diff, "precision", "1"),
                "mdp": get_delta(diff, "p_57_mdp") or 0,
                "mc": get_delta(diff, "streak_metrics", "p_57_monte_carlo") or 0,
                "miss": get_delta(diff, "miss_analysis", "n_miss_days") or 0,
            }

    # Compute mean/std across seeds per experiment
    print(f"\n{'experiment':<24} {'#seeds':>7} {'pass@':>7} {'mean dAvgP@1':>14} "
          f"{'std':>8} {'mean dMDP':>12} {'std':>8}")
    print("-" * 90)
    rows = []
    for name, seed_data in sorted(per_exp.items()):
        n = len(seed_data)
        passes = sum(1 for s in seed_data.values() if s["passed"])
        pavg_list = [s["pavg"] if s["pavg"] is not None else (s["p24"] + s["p25"]) / 2
                     for s in seed_data.values()]
        mdp_list = [s["mdp"] for s in seed_data.values()]
        mean_pavg = statistics.mean(pavg_list)
        std_pavg = statistics.stdev(pavg_list) if n > 1 else 0
        mean_mdp = statistics.mean(mdp_list)
        std_mdp = statistics.stdev(mdp_list) if n > 1 else 0
        rows.append({
            "name": name, "n": n, "passes": passes,
            "mean_pavg": mean_pavg, "std_pavg": std_pavg,
            "mean_mdp": mean_mdp, "std_mdp": std_mdp,
        })

    rows.sort(key=lambda r: -r["mean_pavg"])
    for r in rows:
        print(f"{r['name']:<24} {r['n']:>7} {r['passes']:>2}/{r['n']:<4} "
              f"{r['mean_pavg']*100:>+12.3f}pp ±{r['std_pavg']*100:>5.3f} "
              f"{r['mean_mdp']*100:>+10.3f}pp ±{r['std_mdp']*100:>5.3f}")

    # Key finding: any experiments that are positive mean across 3 seeds?
    print("\nExperiments with mean dAvgP@1 > 0 (potential false-negative recoveries):")
    positives = [r for r in rows if r["mean_pavg"] > 0]
    if not positives:
        print("  (none)")
    else:
        for r in positives:
            strength = r["mean_pavg"] / r["std_pavg"] if r["std_pavg"] > 0 else float('inf')
            print(f"  {r['name']:<24} mean +{r['mean_pavg']*100:.3f}pp "
                  f"(±{r['std_pavg']*100:.3f}, n/σ={strength:.1f}, passed on {r['passes']}/{r['n']} seeds)")


if __name__ == "__main__":
    analyze_phase1()
    analyze_phase2_partial()
