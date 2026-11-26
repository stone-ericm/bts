"""Capture both implementations' independent short grid runs and compare corresponding cells."""
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sources = {"base": ROOT / "src", "f1": HERE / "scratch" / "src"}
groups = ("qualify", "match_right", "match_wrong", "no_pick", "mismatch", "present_unverified")
counts = {}
for group in groups:
    outputs = {}
    for label, source in sources.items():
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (HERE, source, ROOT))),
                   PYTHONDONTWRITEBYTECODE="1", TZ="America/New_York")
        output = HERE / f"grid-{group}-{label}.jsonl"
        with output.open("w") as handle:
            subprocess.run(["nice", "-n", "19", "timeout", "10s", sys.executable, str(HERE / "grid.py"), group],
                           stdout=handle, env=env, cwd=ROOT, timeout=12, check=True)
        outputs[label] = output
    if group == "qualify":
        counts[group] = {k: [v["verifier"][1], v["qualification"], v["exit"]]
                         for k, v in json.loads(outputs["f1"].read_text()).items()}
        continue
    differences = Counter()
    pairs = []
    row_no = 0
    for left, right in zip(*(p.open() for p in outputs.values()), strict=True):
        row_no += 1
        a, b = json.loads(left), json.loads(right)
        for field in ("stdout", "stderr", "dm", "attempted_dm"):
            for instance in (a, b):
                if isinstance(instance[field], str):
                    instance[field] = instance[field].replace("'" + str(HERE), "'REVIEW")
        keys = ("category", "delivery", "prior", "transport", "band")
        assert all(a[k] == b[k] for k in keys)
        for item in (a, b):
            if item["marker"]:
                item["marker"].pop("receipt", None)
            for key in ("dm", "attempted_dm"):
                item[key] = [text.split(" --picks-dir ")[0] for text in item[key]]
            item["stderr"] = (item["stderr"].split(" -> ")[0].split(str(HERE))[0]
                              if item["exception"] not in (None, "SystemExit") else item["stderr"])
        if a != b:
            differences[(a["exit"], b["exit"], a["receipt_marker"], b["receipt_marker"],
                         len(a["dm"]), len(b["dm"]))] += 1
            pairs.append({"base": a, "f1": b})
    counts[group] = {"cells": row_no, "changed": len(pairs),
                     "transitions": [{"signature": k, "count": v} for k, v in differences.items()]}
    (HERE / f"grid-{group}-changes.json").write_text(json.dumps(pairs, indent=2))
print(json.dumps(counts, indent=2))
