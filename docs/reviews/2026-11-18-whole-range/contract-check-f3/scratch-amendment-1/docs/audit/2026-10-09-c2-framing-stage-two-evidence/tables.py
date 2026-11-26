"""The stage-two results note's tables, rebuilt from the hash-matched run copies, and checked against the box aggregate.

Usage (from the repo root): PYTHONPATH=. uv run --offline python -B <this file> <root with all ten seed_<s>/<run>> <aggregate.box.json>

Read-only over the copies. P@1 and hit counts come from the retained profiles (each day's rank-1 `actual_hit`), not
from the scorecards; the per-seed rule and the §5 quantities are recomputed here from the retained diffs with this
file's own arithmetic, then compared with the box aggregate. Prints the tables and every cross-check.
"""
import json
import math
import statistics
import sys
from pathlib import Path

import pandas as pd

SEEDS = (2273360, 260991262, 1746737973, 2048, 3629294338, 1277948386, 3219332220, 2207587974, 3170105529, 2675988121)
SEASONS = ("2024", "2025")
root, agg = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
assert agg["seeds"] == list(SEEDS), agg["seeds"]
dirs = {}
for s in SEEDS:
    found = sorted((root / f"seed_{s}").iterdir())
    assert len(found) == 1, (s, found)
    dirs[s] = found[0]

checks = []


def check(name, ok):
    checks.append((name, bool(ok)))


def rank1(d, v, season):
    p = pd.read_parquet(d / f"profiles_{v}_{season}.parquet")
    r1 = p[p["rank"] == 1]
    assert r1["date"].is_unique and (r1["season"].astype(str) == season).all()
    return int(r1["actual_hit"].sum()), len(r1)


def rule(diff):
    """The per-seed rule, written out: both seasons' P@1 up; else no drop beyond 0.3pp, mean_max_streak delta >= 0 and
    exact P(57) strictly up."""
    deltas = [diff["p_at_1_by_season"][s]["delta"] for s in SEASONS]
    if all(x > 0 for x in deltas):
        return True, "both seasons up"
    streak = diff["streak_metrics"]["mean_max_streak"]["delta"]
    p57 = diff["p_57_exact"]["delta"]
    if all(x >= -0.003 for x in deltas):
        if streak >= 0 and p57 > 0:
            return True, "fallback"
        return False, f"fallback fails (streak {streak:+.2f}, exact P(57) delta {p57!r})"
    return False, "a season drops more than 0.3pp"


print("## P@1 by seed (rank-1 hits / days from the retained profiles)\n")
print("| # | Seed | Season | Baseline | A | A − baseline | B | B − baseline |")
print("|---|---|---|---|---|---|---|---|")
per = {"A": [], "B": []}
for i, s in enumerate(SEEDS, 1):
    d = dirs[s]
    diffs = {v: json.loads((d / f"diff_{v}.json").read_text()) for v in ("A", "B")}
    for season in SEASONS:
        hb, n = rank1(d, "baseline", season)
        row = [f"| {i} | {s} | {season} | {100*hb/n:.1f}% ({hb}/{n}) |"]
        for v in ("A", "B"):
            hv, nv = rank1(d, v, season)
            assert nv == n
            dd = diffs[v]["p_at_1_by_season"][season]
            check(f"seed {s} {v} {season}: profile P@1 equals the diff's", math.isclose(hv / n, dd["variant"], abs_tol=1e-12)
                  and math.isclose(hb / n, dd["baseline"], abs_tol=1e-12) and math.isclose((hv - hb) / n, dd["delta"], abs_tol=1e-12))
            row.append(f" {100*hv/n:.1f}% ({hv}) | **{100*(hv-hb)/n:+.2f}pp** ({hv-hb:+d}) |")
        print("".join(row))
    for v in ("A", "B"):
        passed, why = rule(diffs[v])
        per[v].append({"seed": s, "deltas": {x: diffs[v]["p_at_1_by_season"][x]["delta"] for x in SEASONS},
                       "passed": passed, "why": why,
                       "streak": diffs[v]["streak_metrics"]["mean_max_streak"]["delta"],
                       "p57": diffs[v]["p_57_exact"]["delta"]})

print("\n## §5 quantities (recomputed here) and the box aggregate\n")
print("| | A | B |\n|---|---|---|")
rows = {}
for v in ("A", "B"):
    ps = per[v]
    mean_by = {x: statistics.fmean(p["deltas"][x] for p in ps) for x in SEASONS}
    d_ = [statistics.fmean(p["deltas"][x] for x in SEASONS) for p in ps]
    m, sd = statistics.fmean(d_), statistics.stdev(d_)
    t = m / (sd / math.sqrt(len(d_)))
    passes = sum(p["passed"] for p in ps)
    pos = all(x > 0 for x in mean_by.values()) and m >= 0.003 and t >= 1.5 and passes >= 6
    neg = m <= 0 and passes < 6
    disp = "positive" if pos else ("negative" if neg else "inconclusive")
    a = agg["variants"][v]
    check(f"{v}: disposition equals the box's", disp == a["disposition"])
    check(f"{v}: passes equal the box's", passes == a["per_seed_passes"])
    check(f"{v}: per-seed verdicts equal the box's", [p["passed"] for p in ps] == [x["passed"] for x in a["per_seed"]])
    check(f"{v}: m and t equal the box's", math.isclose(m, a["seed_level_mean"], abs_tol=1e-12) and math.isclose(t, a["t"], rel_tol=1e-9))
    check(f"{v}: season means equal the box's", all(math.isclose(mean_by[x], a["mean_p_at_1_delta"][x], abs_tol=1e-12) for x in SEASONS))
    check(f"{v}: seed-level d equal the box's", all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(d_, a["seed_level"])))
    rows[v] = dict(mean_by=mean_by, d=d_, m=m, sd=sd, t=t, passes=passes, disp=disp)
print(f"| Mean 2024 delta | {100*rows['A']['mean_by']['2024']:+.2f}pp | {100*rows['B']['mean_by']['2024']:+.2f}pp |")
print(f"| Mean 2025 delta | {100*rows['A']['mean_by']['2025']:+.2f}pp | {100*rows['B']['mean_by']['2025']:+.2f}pp |")
print("| Seed-level d (seeds 1–10) | " + " | ".join(", ".join(f"{100*x:+.2f}" for x in rows[v]["d"]) + "pp" for v in ("A", "B")) + " |")
print(f"| m (mean of d) | {100*rows['A']['m']:+.2f}pp | {100*rows['B']['m']:+.2f}pp |")
print(f"| sd of d | {100*rows['A']['sd']:.2f}pp | {100*rows['B']['sd']:.2f}pp |")
print(f"| t = m / (sd/√10) | {rows['A']['t']:.2f} | {rows['B']['t']:.2f} |")
print(f"| Per-seed rule passed | {rows['A']['passes']} of 10 | {rows['B']['passes']} of 10 |")
print(f"| **Disposition** | **{rows['A']['disp']}** | **{rows['B']['disp']}** |")

print("\n## Per-seed rule, seed by seed\n")
for v in ("A", "B"):
    for i, p in enumerate(per[v], 1):
        print(f"- {v} seed {i} ({p['seed']}): {'PASS' if p['passed'] else 'fail'} — {p['why']}; "
              f"2024 {100*p['deltas']['2024']:+.2f}pp, 2025 {100*p['deltas']['2025']:+.2f}pp")

print("\n## Secondary metrics (from the retained diffs)\n")
print("| # | Seed | A: mean_max_streak Δ | A: exact P(57) Δ | B: mean_max_streak Δ | B: exact P(57) Δ |\n|---|---|---|---|---|---|")
for i, (a, b) in enumerate(zip(per["A"], per["B"]), 1):
    print(f"| {i} | {a['seed']} | {a['streak']:+.2f} | {a['p57']:.3g} | {b['streak']:+.2f} | {b['p57']:.3g} |")

print("\n## Cost (results.json total_cpu_s and units.json)\n")
tot = 0.0
for i, s in enumerate(SEEDS, 1):
    res = json.loads((dirs[s] / "results.json").read_text())
    units = json.loads((dirs[s] / "units.json").read_text())
    us = sum(u["cpu_s"] for u in units)
    lab = sum(u["labels"]["changed"] + u["labels"]["void_dropped"] for u in units)
    tot += res["total_cpu_s"]
    print(f"- seed {i} ({s}): total_cpu_s {res['total_cpu_s']:.1f} = {res['total_cpu_s']/3600:.4f} h; units sum {us:.1f}; label changes + void rows {lab}")
check("aggregate total_cpu_h equals the results' sum", math.isclose(tot / 3600, agg["total_cpu_h"], rel_tol=1e-12))
print(f"- ten seeds: {tot:.1f} s = {tot/3600:.4f} h (box aggregate total_cpu_h {agg['total_cpu_h']:.4f})")

print("\n## Cross-checks\n")
for name, ok in checks:
    if not ok:
        print("FAILED:", name)
print(f"{sum(ok for _, ok in checks)} of {len(checks)} cross-checks pass")
sys.exit(0 if all(ok for _, ok in checks) else 1)
