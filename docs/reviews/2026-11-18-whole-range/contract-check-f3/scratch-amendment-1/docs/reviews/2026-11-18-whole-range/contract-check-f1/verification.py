"""Compare receipt fields, enumerate assertions/errors and verify the factors of the full grid."""
from collections import Counter
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def read_pair(stem):
    return [json.loads((HERE / f"{stem}-{label}.json").read_text()) for label in ("base", "f1")]


base, scratch = read_pair("fixture")
receipt_diff = {}
for label in base:
    diffs = []
    for a, b in zip(base[label]["runs"], scratch[label]["runs"], strict=True):
        aa, bb = a["receipts"][-1], b["receipts"][-1]
        keys = sorted(k for k in aa.keys() | bb.keys() if aa.get(k) != bb.get(k) and k != "attempt_id")
        diffs.append(keys)
    receipt_diff[label] = diffs

b, f = read_pair("edges")
factor_counts = Counter()
for base, scratch in zip(b["all_qualifications"], f["all_qualifications"], strict=True):
    assert base["qualify"] == scratch["qualify"] and base["verifier"] == scratch["verifier"]
    which = "changed" if base != scratch else "same"
    assert (which == "changed") == ("wrong_game" in base["qualify"] and base["verifier"] == "match")
    factor_counts[(base["verifier"], which)] += 1

audit = (HERE / "pytest" / "test_audit_driver.py.log").read_text()
audit_base = (HERE / "pytest" / "test_audit_driver.base.log").read_text()
summaries = lambda data: re.findall(r"^(?:FAILED|ERROR) tests/.*$", data, re.M)
assert summaries(audit) == summaries(audit_base)
field = (HERE / "pytest" / "scripts__test_field_products_run.py.log").read_text()
test_findings = ["Scratch failing assertion from source relocation (base file 20 passed):", *summaries(field), "",
                 "Missing GNU rsync 3.x blocked these same tests on both trees before their assertions:",
                 *summaries(audit)]
(HERE / "test-findings.txt").write_text("\n".join(test_findings) + "\n")

groups = json.loads((HERE / "grid-summary.json").read_text())
counts = {name: info["cells"] for name, info in groups.items() if "cells" in info}
full_size = 72 * sum(count for name, count in counts.items() if name != "match_wrong")
print(json.dumps({"fixture_receipt_differences_except_attempt_id": receipt_diff,
                  "qualification_factors": [{"reason": k[0], "comparison": k[1], "count": v}
                                            for k, v in factor_counts.items()],
                  "ledger_cases_per_build": sum(counts.values()), "composed_grid_cases_per_build": full_size,
                  "audit_identical_blocked_nodes": len(summaries(audit)), "search_files": 63}, indent=2))
