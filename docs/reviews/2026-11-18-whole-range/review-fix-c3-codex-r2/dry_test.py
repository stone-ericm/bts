"""Prove dry comparison reports the first divergence without mutating input."""

from pathlib import Path
import json
import subprocess
import sys

from compare_probes import HERE, scratch
from probe_support import dump, pick_data


with scratch("dry-test") as folder:
    root = Path(folder)
    dump(root / "2027-01-01.json", pick_data("2027-01-01", "hit"))
    dump(root / "2027-01-02.json", pick_data("2027-01-02", chosen_on={
        "schema": "bts_pick_chosen_on_v1", "required": False, "kind": "model", "streak": 1,
        "saver_available": True, "dated": None}))
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    result = subprocess.run([sys.executable, str(HERE / "dry_compare.py"), str(root), "--today", "2027-12-31"],
                            text=True, capture_output=True, timeout=100)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    print("exit:", result.returncode)
    observed = json.loads(result.stdout)
    assert result.returncode == 2 and observed["first_difference"]["as_of"] == "2027-01-01"
    assert observed["first_block"]["619abda7"]["as_of"] == "2027-01-03"
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
