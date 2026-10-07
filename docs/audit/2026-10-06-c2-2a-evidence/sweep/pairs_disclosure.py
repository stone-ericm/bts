"""Pair coverage of a sweep run (manager's row C2-2a-sweep-scope): which (point, scenario) pairs were faulted.

A point is a (line or call, file, line, call-site file, call-site line). A pair is a point together with a scenario that
reaches it. The sweep faults each point in the first scenario that reaches it, and with `--all-pairs-in` also in every
listed scenario that reaches it. This prints, per scenario class, the pairs reached, faulted and not faulted; and,
among the pairs not faulted, how many belong to designed-fault-only points (reached by no plain scenario, so faulted
only on top of the first designed fault that reaches them) and how many to points that every plain scenario reaching
them already faulted.

    python pairs_disclosure.py SWEEP.jsonl             # the run must have been made with --all-pairs-in (its header
                                                       # records every scenario's reached points)
"""
import json
import sys
from collections import Counter

rows = [json.loads(l) for l in open(sys.argv[1])]
head, results = rows[0], [r for r in rows if "point" in r]
reached = {n: [tuple(p) for p in pts] for n, pts in head["reached"].items()}
excluded = {tuple(c[:5]) for c in head["computation"]}
plain_set = set(head["all_pairs_in"])
names = head["scenarios"]
cls = {n: "plain" if n in plain_set else ("designed fault" if n.startswith("fault_") else "genuine failure")
       for n in names}
faulted = {(r["scenario"], tuple(r["point"])) for r in results}
in_plain = {p for n in names if cls[n] == "plain" for p in reached[n]}
table = Counter()
for n in names:
    for p in reached[n]:
        if p in excluded:
            table[cls[n], "computation (genuine-failure scenario instead)"] += 1
            continue
        table[cls[n], "reached"] += 1
        if (n, p) in faulted:
            table[cls[n], "faulted"] += 1
        elif p in in_plain:
            table[cls[n], "not faulted: point faulted in every plain scenario reaching it"] += 1
        else:
            table[cls[n], "not faulted: designed-fault-only point (faulted once, on its first designed fault)"] += 1
failed = [r for r in results if not r["ok"]]
print(f"run at {head['head'][:7]}: {len(names)} scenarios, all pairs in {len(plain_set)}; {len(results)} faulted runs, "
      f"{len(failed)} failed; {len(head['computation'])} computation points excluded")
for c in ("plain", "designed fault", "genuine failure"):
    k = sorted(set(n for n in names if cls[n] == c))
    print(f"\n## {c} ({len(k)} scenarios)")
    for key in ("reached", "faulted", "not faulted: point faulted in every plain scenario reaching it",
                "not faulted: designed-fault-only point (faulted once, on its first designed fault)",
                "computation (genuine-failure scenario instead)"):
        print(f"  {key}: {table[c, key]}")
only = sorted({p for n in names if cls[n] != "plain" for p in reached[n]} - in_plain - excluded)
print(f"\n## designed-fault-only points: {len(only)} (each faulted once, in the first scenario reaching it)")
first = {tuple(r["point"]): r["scenario"] for r in reversed(results)}
for p in only:
    print(f"  {p[0]:4} {p[1]}:{p[2]} <- {p[3]}:{p[4]}  faulted in {first.get(p, 'NOT FAULTED')}")
sys.exit(1 if failed or any(p not in first for p in only) else 0)
