"""Replay of review r8's R8-1 witness against the revision-8 runner (from git, dcdbb71) and the revision-9 runner.
Run from the repository root: UV_CACHE_DIR=/tmp/uv-cache uv run python -B docs/audit/2026-10-06-c2-framing-evidence/r8_witness_replay.py
The witness: a pytest_sessionfinish wrapper registered through pluggy's base class during the first test, which
unregisters itself BEFORE yielding and aborts after it. The bodies genuinely fail under the mutant."""
import importlib.util, subprocess, sys, tempfile
from pathlib import Path
sys.path[:0] = [".", "src"]
from scripts.audit.c2_framing import mutant_runner as R9
import tests.scripts.c2_framing.test_screen as T
old = Path(tempfile.mkdtemp(prefix="runner-r8-")) / "runner_r8.py"
old.write_text(subprocess.run(["git", "show", "dcdbb71:docs/audit/2026-10-06-c2-framing-evidence/mutant_runner.py"],
                              capture_output=True, text=True, check=True).stdout)
spec = importlib.util.spec_from_file_location("runner_r8", old)
R8 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R8)
body = T._late("pluggy.PluginManager.register(pm, Late(pm, True, True), 'gone-wrapper')")
for label, R in (("revision 8", R8), ("revision 9", R9)):
    tmp = Path(tempfile.mkdtemp(prefix="r8-witness-"))
    _, root = T._scratch_mutant(tmp, ["test_scratch.py"], body)
    if R is R9:
        print(f"{label} gate:", R.scope_problems(root, ["test_scratch.py"]))
    intended = R.collect(root, ["test_scratch.py"])
    (root / "target.py").write_text("value = 2\n")                     # the mutant, applied by hand
    rc, _, bundle = R.run_mutant(root, ["test_scratch.py"])
    print(f"{label}" + (" run anyway:" if R is R9 else ":"), R.classify(rc, intended, bundle)[0], "| late_plugins", bundle.get("late_plugins"),
          "| not_outermost", bundle.get("not_outermost"), "| plugin_changes", bundle.get("plugin_changes"))
