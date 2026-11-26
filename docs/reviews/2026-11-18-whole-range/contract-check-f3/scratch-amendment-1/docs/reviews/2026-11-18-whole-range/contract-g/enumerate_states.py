"""Enumerate, from the shipped reach-57 artifact alone, every state where the table skips a quality bin
while a lower bin plays (change set G). Usage: python enumerate_states.py <repo root> <out.csv>.

One row per affected (streak, days_left, saver) state; each has exactly one affected bin (asserted).
Columns: streak, days_left, saver, bin, table_action, from_bin, guarded_action, row (the five table actions).
"""
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

root, out = Path(sys.argv[1]), Path(sys.argv[2])
raw = (root / "data/models/mdp_policy.npz").read_bytes()
sha = hashlib.sha256(raw).hexdigest()
assert sha.startswith("66d15471"), sha
import io
with np.load(io.BytesIO(raw)) as data:
    table = data["policy_table"]
ACTIONS = ("skip", "single", "double")
S, D, V, Q = table.shape
assert (S, D, V, Q) == (58, 181, 2, 5)
rows, states = [], 0
for s in range(S):
    for d in range(D):
        for v in range(V):
            states += 1
            row = [int(x) for x in table[s, d, v]]
            hit = [q for q in range(Q) if row[q] == 0 and any(row[j] != 0 for j in range(q))]
            if not hit:
                continue
            assert len(hit) == 1, (s, d, v, row)
            q = hit[0]
            j = max(j for j in range(q) if row[j] != 0)
            rows.append([s, d, v, q, "skip", j, ACTIONS[row[j]], "".join(ACTIONS[a][0] for a in row)])
with out.open("w", newline="") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["streak", "days_left", "saver", "bin", "table_action", "from_bin", "guarded_action", "row"])
    w.writerows(rows)
from collections import Counter
print("artifact sha256", sha)
print("states", states, "affected", len(rows))
print("by (bin, from_bin, guarded_action):", sorted(Counter((r[3], r[5], r[6]) for r in rows).items()))
print("streak range", min(r[0] for r in rows), max(r[0] for r in rows), "streak 57 rows", sum(r[0] == 57 for r in rows),
      "days 0 rows", sum(r[1] == 0 for r in rows))
print("by saver", sorted(Counter(r[2] for r in rows).items()))
print("late (days<=30)", sum(r[1] <= 30 for r in rows), "early", sum(r[1] > 30 for r in rows))
print("csv sha256", hashlib.sha256(out.read_bytes()).hexdigest())
