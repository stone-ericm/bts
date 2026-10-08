"""Diagnostic: recompute each run's scorecards from its retained profiles on this machine and list every field that
differs from the stored (box-written) scorecard. Read-only over the copies."""
import json, sys
from pathlib import Path
import pandas as pd
from scripts.audit.c2_framing import screen as S
from bts.validate.scorecard import compute_full_scorecard

root = Path(sys.argv[1])

def walk(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            yield from walk(a.get(k, "<missing>"), b.get(k, "<missing>"), f"{path}.{k}")
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            yield from walk(x, y, f"{path}[{i}]")
    elif a != b:
        yield path, a, b

for d in sorted(root.glob("seed_*/*")):
    for v in ("baseline", "A", "B"):
        parts = [pd.read_parquet(d / f"profiles_{v}_{s}.parquet") for s in S.TEST_SEASONS]
        card = json.loads(S._canon(compute_full_scorecard(pd.concat(parts, ignore_index=True), **S.SCORING)))
        stored = json.loads((d / f"scorecard_{v}.json").read_text())
        diffs = [x for x in walk(stored, card) if x[0] != ".timestamp"]
        print(d.parent.name, v, "differences:", len(diffs))
        for p, a, b in diffs[:12]:
            print("   ", p, "stored", repr(a), "| mac", repr(b), "| rel", f"{abs(a-b)/abs(a):.2e}" if isinstance(a,float) and isinstance(b,float) and a else "")
