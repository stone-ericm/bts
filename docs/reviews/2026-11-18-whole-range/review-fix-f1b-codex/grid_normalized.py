"""Correct the grid comparison for a marker that references an unsealed receipt."""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

import grid


def compare():
    with tempfile.TemporaryDirectory(dir=grid.HERE) as tmp:
        root = Path(tmp)
        archive = subprocess.check_output(["git", "archive", "1e69d0f0", "src"], cwd=grid.ROOT, timeout=4)
        subprocess.run(["tar", "xf", "-", "-C", tmp], input=archive, check=True, timeout=4)
        result = {}
        for label, src in (("base", root / "src"), ("candidate", grid.ROOT / "src")):
            path = root / f"{label}.json"
            subprocess.run([sys.executable, __file__, str(path), tmp], cwd=grid.ROOT, check=True, timeout=38,
                           env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (src, grid.ROOT)))})
            result[label] = json.loads(path.read_text())
    normalized = 0
    amendment = 0
    for key, base in result["base"].items():
        candidate = result["candidate"][key]
        markers = [row["marker"] for row in (base, candidate) if row["marker"] and not row["receipts"]]
        for marker in markers:
            if marker.get("receipt"):
                marker["receipt"] = "unsealed-receipt-reference"
        if base == candidate:
            normalized += 1
            continue
        state, capture, prior, *_ = key.split("|")
        assert state == "match" and capture == "broken" and prior != "final", (key, base, candidate)
        assert base["marker"]["status"] == candidate["marker"]["status"] == "confirmed"
        assert base["marker"]["reason"] == "match" and candidate["marker"]["reason"] == "match_game_unchecked"
        assert base["exit"] == candidate["exit"] == 0
        assert "game check unavailable (BadGzipFile); batter match only" in candidate["stderr"]
        assert not (base["receipts"] or candidate["receipts"]) and base["morning"] == candidate["morning"]
        amendment += 1
    print(f"{normalized + amendment} pairs; {normalized} identical, {amendment} Amendment 3 changes; no other differences")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        compare()
    else:
        grid.execute(Path(sys.argv[1]), Path(sys.argv[2]))
