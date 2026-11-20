#!/usr/bin/env python3
"""Item 8 helper: inventory every operator-visible string in code and every changed sentence in INCIDENT.md, CLAUDE.md, and 2027-season-start.md."""
from __future__ import annotations

import ast
import json
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    diff_docs = subprocess.run(
        ["git", "diff", "-U0", "b128854d", "HEAD", "--", "INCIDENT.md", "CLAUDE.md", "docs/ops/2027-season-start.md", "scripts/systemd/README.md"],
        cwd=REPO, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
    ).stdout
    added_doc_lines = [
        ln[1:].strip() for ln in diff_docs.splitlines()
        if ln.startswith("+") and not ln.startswith("+++") and ln[1:].strip()
    ]
    out = {"added_doc_lines_count": len(added_doc_lines), "added_doc_lines": added_doc_lines}
    dest = Path(__file__).resolve().parent / "probe_8_inventory.json"
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Inventoried {len(added_doc_lines)} added doc lines.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
