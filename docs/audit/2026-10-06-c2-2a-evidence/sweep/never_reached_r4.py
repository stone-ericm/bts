"""Classify the final sweep's never-reached lines: executed by the unit tests (line coverage), an exception handler
whose innermost try body was fault-injected by the sweep with every injected run matching the golden, or other.

    python never_reached_r4.py SWEEP.jsonl COVERAGE.json      # from the repo root; prints the classification

COVERAGE.json is `coverage json` of the c2_2a unit tests under COVERAGE_CORE=sysmon (the sys.settrace-based tests
would otherwise switch a settrace-based coverage tracer off part-way)."""
import ast, json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
summ = rows[-1]; pts = [r for r in rows if "point" in r]
injected = {}
for r in pts:
    injected.setdefault((r["point"][1], r["point"][2]), []).append(r["ok"])
cov = json.load(open(sys.argv[2]))["files"]
paths = {"calibrate.py": "src/bts/model/calibrate.py", "predict.py": "src/bts/model/predict.py",
         "orchestrator.py": "src/bts/orchestrator.py", "serving_witness.py": "src/bts/serving_witness.py",
         "slate.py": "src/bts/slate.py"}
trees = {f: ast.parse(open(p).read()) for f, p in paths.items()}
def innermost(f, n):
    best = None
    for node in ast.walk(trees[f]):
        if isinstance(node, ast.Try):
            for h in node.handlers:
                if h.lineno <= n <= h.end_lineno and (best is None or h.end_lineno - h.lineno < best[1].end_lineno - best[1].lineno):
                    best = (node, h)
    return best
out = {"executed by the unit tests": [], "handler; its try body fault-injected, all runs equal the golden": [],
       "handler; try body not injected": [], "other": []}
for loc in summ["never_reached"]:
    f, n = loc.split(":"); n = int(n)
    if n in set(cov[paths[f]]["executed_lines"]):
        out["executed by the unit tests"].append(loc); continue
    b = innermost(f, n)
    if b is None:
        out["other"].append(loc); continue
    body = [l for s in b[0].body for l in range(s.lineno, s.end_lineno + 1)]
    hit = {l: injected[(f, l)] for l in body if (f, l) in injected}
    if hit and all(all(v) for v in hit.values()):
        out["handler; its try body fault-injected, all runs equal the golden"].append(f"{loc} <- {sorted(hit)} ({sum(map(len, hit.values()))} runs)")
    else:
        out["handler; try body not injected"].append(f"{loc} <- body {body}")
print("sweep points", len(pts), "failed", sum(not r["ok"] for r in pts), "| never reached by the sweep's plain and designed-fault runs:", len(summ["never_reached"]))
for k, v in out.items():
    print(f"{k}: {len(v)}")
for k, v in out.items():
    print("\n##", k)
    for x in v:
        print(" ", x)
