"""Rerun the operator rehearsal and select final refusal DMs by their opening."""

import json
import sys
from pathlib import Path

import probe_operator


if __name__ == "__main__":
    probe_operator.late_start(False)
    probe_operator.late_start(True)
    probe_operator.late_start(False, last_only=True)
    probe_operator.guarded_delivery()
    results = json.loads(Path(sys.argv[1]).read_text())
    for row in results.values():
        row["wire"] = [body for body in row["wire"] if "- [contest_state] Morning check" not in body]
    probe_operator.overnight(results)
