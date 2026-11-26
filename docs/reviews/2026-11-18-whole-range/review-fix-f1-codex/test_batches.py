"""Run one bounded group from the brief's test-file search."""
from pathlib import Path
import subprocess
import sys

import bts
import pytest

PATTERN = (r"check_pick_entered|check-pick-entered|_check_pick_entered|pick_entry_status|"
           r"entry_receipt|EntryReceipt|qualify\(|load_units|target_rows|pick_entry_check|_selection")


def batches():
    found = subprocess.run(["rg", "-l", PATTERN, "tests", "scripts"], check=True,
                           capture_output=True, text=True, timeout=4).stdout.splitlines()
    tests = sorted(path for path in found if path.startswith("tests/") and path.endswith(".py"))
    parts = {"audit": [], "f1": [], "scripts": [], "experiments": [], "other": []}
    for path in tests:
        if path.endswith("/test_audit_driver.py"):
            parts["audit"].append(path)
        elif path.endswith("/test_wr_f1_wrong_game.py"):
            parts["f1"].append(path)
        elif path.startswith("tests/scripts/"):
            parts["scripts"].append(path)
        elif path.startswith(("tests/experiment/", "tests/leaderboard/")):
            parts["experiments"].append(path)
        else:
            parts["other"].append(path)
    result = {key: value for key, value in parts.items() if key in ("audit", "f1", "experiments")}
    for key, size in (("scripts", 6), ("other", 7)):
        for index in range(0, len(parts[key]), size):
            result[f"{key}{index // size + 1}"] = parts[key][index:index + size]
    return result


if __name__ == "__main__":
    print(f"bts.__file__ = {bts.__file__}", flush=True)
    groups = batches()
    if sys.argv[1] == "list":
        for name, paths in groups.items():
            print(name, len(paths), *paths, sep="\n")
        print("total", sum(map(len, groups.values())))
    else:
        paths = groups[sys.argv[1]]
        assert all(Path(path).is_file() for path in paths) and paths
        print(*paths, sep="\n", flush=True)
        raise SystemExit(pytest.main(["-p", "no:cacheprovider", "-q", "--tb=short", *paths, *sys.argv[2:]]))
