"""Check that the named assertions detect three independent source changes."""
from pathlib import Path
import os
import runpy
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from replay_support import HERE, SCRATCH, World, source
import state_sequences
import support


def manual(root):
    w = World(root)
    w.commit_at("10:50", at="09:40")
    w.cron("10:00")
    # A base T-30 reminder must be a reminder, with the already spent initial tier preserved.
    r = w.invoke("10:20", manual=True)
    assert len(r["sent"]) == 1 and r["sent"][0].startswith("⏰"), r


def execute(root):
    candidate = SCRATCH / "candidate"
    cases = (
        ("gate", "entry_receipt.py", "range(0, 14 * 60, 15)", "range(15, 14 * 60, 15)"),
        ("ledger", "cli.py", "    if prior.get(\"date\") != today:\n",
         "    if not cron:\n        prior = {}\n    if prior.get(\"date\") != today:\n"),
        ("precedence", "health/pick_entry.py", "    if marker.get(\"date\") != today.isoformat():\n"
         "        marker = read_early_marker(marker_path.with_name(EARLY_MARKER))",
         "    early = read_early_marker(marker_path.with_name(EARLY_MARKER))\n"
         "    if early.get(\"date\") == today.isoformat():\n        marker = early"),
    )
    for name, path, old, new in cases:
        mutated = root / name
        shutil.copytree(candidate / "src", mutated)
        target = mutated / "bts" / path
        contents = target.read_text()
        assert contents.count(old) == 1
        target.write_text(contents.replace(old, new))
        script = HERE / ("mutations.py" if name == "ledger" else "contract_rows.py")
        for control, tree in ((True, candidate / "src"), (False, mutated)):
            command = [sys.executable, str(script), name]
            env = os.environ | {"PYTHONPATH": f"{tree}:{candidate}"}
            result = subprocess.run(command, env=env, cwd=candidate, text=True, capture_output=True, timeout=4)
            print(name, "control" if control else "changed source", result.returncode, result.stdout, result.stderr)
            assert result.returncode == (0 if control else 1)
            if not control:
                assert "AssertionError" in result.stderr and str(mutated) in env["PYTHONPATH"]


if __name__ == "__main__":
    with TemporaryDirectory(dir=SCRATCH) as temp:
        if sys.argv[1:] == ["ledger"]:
            from bts.contest_fetch import pick_entry_status
            print("source", sys.modules[pick_entry_status.__module__].__file__)
            manual(Path(temp))
        elif sys.argv[1:] == ["archive"]:
            from replay_support import ARCHIVE
            with patch.object(support, "source", source), patch.object(state_sequences, "source", source):
                sys.argv[:] = ["surviving_mutations.py", "base"]
                runpy.run_path(str(ARCHIVE / "review-fix-f5-codex-r2/surviving_mutations.py"), run_name="__main__")
        else:
            execute(Path(temp))
