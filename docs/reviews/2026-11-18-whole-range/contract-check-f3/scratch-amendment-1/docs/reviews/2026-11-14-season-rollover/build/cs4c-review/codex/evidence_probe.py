"""Check source preservation and make the two surviving mutants fail."""

import ast
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from review_harness import MUTANTS, ROOT, SCRATCH


def functions(source):
    return {node.name: node for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef)}


def audit():
    base = SCRATCH / "base"
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "1c6cbc77", "HEAD", "--", "src", "scripts"],
        cwd=ROOT, text=True,
    ).splitlines()
    assert set(changed) == {
        "src/bts/contest_state.py", "src/bts/picks.py", "src/bts/scheduler.py",
        "scripts/live_forward_capture_once.py", "scripts/live_forward_provenance_inventory.py",
    }
    unexpected = set()
    for name in changed:
        before = functions((base / name).read_text())
        after = functions((ROOT / name).read_text())
        modified = []
        for key in before:
            assert key in after, (name, key)
            if ast.dump(before[key]) != ast.dump(after[key]):
                modified.append(key)
        print(f"{name}: existing functions changed {modified}; added {sorted(after.keys() - before)}")
        unexpected.update(modified)
        if name == "src/bts/contest_state.py":
            loader = after["load_decision_streak_state"]
            results = [node for node in ast.walk(loader) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == "DecisionStreakState"]
            last = [call for call in results if call.keywords[-1].arg == "contest_recorded_at"]
            assert len(last) == 1
            assert ast.unparse(last[0].keywords.pop().value) == "contest.recorded_at"
            assert ast.dump(loader) == ast.dump(before["load_decision_streak_state"])
    assert unexpected == {
        "load_decision_streak_state", "load_shadow_pick", "parse_daily_pick",
        "_deliver_and_lock_pick", "_decision_snapshot",
    }
    print("All other production files and all other pre-existing functions match the base.")


def gaps():
    for name, action in (("accept-age-bool", "wire"), ("implicit-warning-date", "notifications")):
        relative, mutate, _ = MUTANTS[name]
        with tempfile.TemporaryDirectory(dir=SCRATCH, prefix="gap-") as folder:
            source = Path(folder) / relative
            source.parent.mkdir(parents=True)
            # The package search path can fall through to the branch for every other module.
            initializer = ROOT / "src" / "bts" / "__init__.py"
            (source.parent / "__init__.py").write_text(
                initializer.read_text() + "\nfrom pkgutil import extend_path\n"
                "__path__ = extend_path(__path__, __name__)\n",
            )
            source.write_text(mutate((ROOT / "src" / relative).read_text()))
            env = os.environ | {
                "PYTHONPATH": os.pathsep.join(map(str, (SCRATCH, folder, ROOT / "src", ROOT))),
            }
            child = subprocess.run(
                [sys.executable, str(SCRATCH / "edge_probe.py"), action],
                cwd=ROOT, env=env, capture_output=True, text=True,
            )
            assert child.returncode == 1 and "AssertionError" in child.stderr, child
            assert "OFFLINE GUARD BLOCKED" not in child.stderr
            if name == "accept-age-bool":
                assert 'loaded.automatic_reading == want' in child.stderr
                assert "'age_hours': True" in child.stderr
            else:
                assert "alert.send_dm.call_count == 1" in child.stderr
            print(f"{name} exposed by independent {action} probe:")
            print("\n".join(child.stderr.splitlines()[-4:]))


if __name__ == "__main__":
    audit()
    gaps()
