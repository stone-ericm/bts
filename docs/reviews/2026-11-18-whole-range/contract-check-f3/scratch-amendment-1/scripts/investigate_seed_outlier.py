#!/usr/bin/env python3
"""Dig into why seed=6992150 produces MDP P(57)=9.20% when mean is 3.50%.

Compare its scorecard against the median seed and the seed=42 default.
Focus areas:
  1. Calibration array — are high-confidence bins systematically higher p_hit?
  2. Precision@K curve — is rank-1 accuracy elevated?
  3. Per-season breakdown — is one season lucky and the other normal?
  4. streak_metrics — is longest_replay and mean_max_streak elevated?
  5. miss_analysis — is rank-2 recovery rate on miss days exceptional?
"""
import json
import statistics
from pathlib import Path

ROOT = Path("/Users/stone/projects/bts/data/hetzner_results/audit_phase1")


def load_scorecard(seed: int) -> dict:
    for sc_path in ROOT.rglob(f"seed{seed}.json"):
        return json.load(open(sc_path))
    raise FileNotFoundError(f"seed={seed} not found")


def summarize(seed: int, label: str) -> dict:
    d = load_scorecard(seed)
    print(f"\n=== {label} (seed={seed}) ===")
    print(f"  MDP P(57):             {d['p_57_mdp']*100:6.3f}%")
    print(f"  MC P(57):              {d['streak_metrics']['p_57_monte_carlo']*100:6.3f}%")
    print(f"  Exact P(57):           {d['p_57_exact']*100:6.3f}%")
    print(f"  avg P@1:               {d['precision']['1']*100:6.3f}%")
    print(f"  2024 P@1:              {d['precision_by_season']['2024']['1']*100:6.3f}%")
    print(f"  2025 P@1:              {d['precision_by_season']['2025']['1']*100:6.3f}%")
    print(f"  mean_max_streak (MC):  {d['streak_metrics']['mean_max_streak']:.2f}")
    print(f"  median_max_streak:     {d['streak_metrics']['median_max_streak']}")
    print(f"  p90_max_streak:        {d['streak_metrics']['p90_max_streak']}")
    print(f"  p99_max_streak:        {d['streak_metrics']['p99_max_streak']}")
    print(f"  longest_replay_streak: {d['streak_metrics']['longest_replay_streak']}")
    print(f"  n_miss_days:           {d['miss_analysis']['n_miss_days']}")
    print(f"  rank_2_recovery:       {d['miss_analysis']['rank_2_hit_rate_on_miss']*100:6.3f}%")
    print(f"  mean_p_hit_on_miss:    {d['miss_analysis']['mean_p_hit_on_miss']*100:6.3f}%")
    print(f"  mean_p_hit_on_hit:     {d['miss_analysis']['mean_p_hit_on_hit']*100:6.3f}%")
    print(f"  calibration (sorted by bin, (conf, p_hit)):")
    for row in d["calibration"]:
        conf, p_hit, n = row
        print(f"    conf={conf:.4f}  p_hit={p_hit:.4f}  n={n}")
    return d


def compare_to_mean():
    """Load all 16 scorecards and see where seed=6992150 deviates."""
    all_data = {}
    for sc_path in sorted(ROOT.rglob("seed*.json")):
        seed = int(sc_path.stem.replace("seed", ""))
        all_data[seed] = json.load(open(sc_path))

    print(f"\n{'='*72}")
    print("CROSS-SEED COMPARISON — calibration top bin (highest confidence)")
    print(f"{'='*72}")
    print(f"{'seed':>9} {'top_conf':>10} {'top_p_hit':>11} {'p90_stk':>9} {'longest':>9} {'rank2_rec':>10} {'mdp':>8}")
    rows = []
    for seed, d in sorted(all_data.items()):
        top_bin = d["calibration"][-1]  # last bin = highest confidence
        rows.append({
            "seed": seed,
            "top_conf": top_bin[0],
            "top_p_hit": top_bin[1],
            "p90_max": d["streak_metrics"]["p90_max_streak"],
            "longest": d["streak_metrics"]["longest_replay_streak"],
            "rank2": d["miss_analysis"]["rank_2_hit_rate_on_miss"],
            "mdp": d["p_57_mdp"],
        })
    rows.sort(key=lambda r: -r["mdp"])
    for r in rows:
        flag = " <-- OUTLIER" if r["seed"] == 6992150 else (" <-- seed=42" if r["seed"] == 42 else "")
        print(f"{r['seed']:>9} {r['top_conf']*100:>8.2f}% {r['top_p_hit']*100:>9.2f}% "
              f"{r['p90_max']:>8} {r['longest']:>8} {r['rank2']*100:>8.2f}% "
              f"{r['mdp']*100:>7.3f}%{flag}")

    # Correlations: what best predicts MDP P(57)?
    print("\nCorrelations with MDP P(57) across 16 seeds:")
    mdps = [r["mdp"] for r in rows]
    for key in ["top_conf", "top_p_hit", "p90_max", "longest", "rank2"]:
        vals = [r[key] for r in rows]
        n = len(vals)
        mean_v = statistics.mean(vals)
        mean_m = statistics.mean(mdps)
        num = sum((vals[i] - mean_v) * (mdps[i] - mean_m) for i in range(n))
        den = (sum((v - mean_v) ** 2 for v in vals) ** 0.5) * (sum((m - mean_m) ** 2 for m in mdps) ** 0.5)
        r_val = num / den if den else 0
        print(f"  {key:15s}  r = {r_val:+.3f}")


def main():
    summarize(6992150, "OUTLIER")
    summarize(42, "historical baseline")
    # Pick a median seed by MDP (approximately) for comparison
    all_data = {int(p.stem.replace("seed", "")): json.load(open(p)) for p in ROOT.rglob("seed*.json")}
    sorted_by_mdp = sorted(all_data.items(), key=lambda kv: kv[1]["p_57_mdp"])
    median_seed = sorted_by_mdp[len(sorted_by_mdp) // 2][0]
    summarize(median_seed, "median (by MDP)")
    compare_to_mean()


if __name__ == "__main__":
    main()
