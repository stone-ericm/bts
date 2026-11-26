"""Replay of review r9's three findings against the revision-9 runner (from git, 537e37b) and the revision-10 runner.
Run from the repository root: UV_CACHE_DIR=/tmp/uv-cache uv run python -B docs/audit/2026-10-06-c2-framing-evidence/r9_witness_replay.py
Each witness is benign: the code that should never run writes a canary file, so "canary written" means code outside the
boundary executed. The suites stay inside the reviewed vocabulary, so the revision-9 gate admits all three.
- R9-1: a package __init__.py ABOVE the root (pytest walked the root's ancestors under the empty configuration file).
- R9-2: an allowed import spelled numpy resolves to an uncommitted local numpy.py.
- R9-3: a pre-existing numpy.py symlink names a file outside every scanned tree; a child the test starts writes that
  file during the run, and the test then imports numpy."""
import contextlib, importlib.util, io, json, subprocess, sys, tempfile
from pathlib import Path
sys.path[:0] = [".", "src"]
from scripts.audit.c2_framing import mutant_runner as R10
import tests.scripts.c2_framing.test_screen as T

old = Path(tempfile.mkdtemp(prefix="runner-r9-")) / "runner_r9.py"
old.write_text(subprocess.run(["git", "show", "537e37b:scripts/audit/c2_framing/mutant_runner.py"],
                              capture_output=True, text=True, check=True).stdout)
spec = importlib.util.spec_from_file_location("runner_r9", old)
R9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R9)


def ancestor(tmp, canary):
    (tmp / "__init__.py").write_text(f"import pathlib\npathlib.Path({str(canary)!r}).write_text('ran')\n")
    return T._scratch_mutant(tmp, ["test_scratch.py"], T._two_read_failing())


def look_alike(tmp, canary):
    spec, root = T._scratch_mutant(tmp, ["test_scratch.py"], T._numpy_suite)
    (root / "numpy.py").write_text(T._look_alike(canary))
    return spec, root


def linked(tmp, canary):
    outside = tmp / "outside"
    outside.mkdir()
    source = T._look_alike(canary)
    spec, root = T._scratch_mutant(tmp, ["test_scratch.py"], lambda r: T._READ + (
        "import subprocess\n"
        "def test_first():\n"
        f"    subprocess.run(['sh', '-c', 'printf %s \"$0\" > {outside / 'numpy.py'}', {source!r}], check=True)\n"
        "    import numpy as np\n"
        f"    assert np.isclose(1, 1) and val() == {T._ONE}\n"))
    subprocess.run(["ln", "-s", str(outside / "numpy.py"), str(root / "numpy.py")], check=True)
    T._commit_all(root)                                    # the link is committed; what it names never is
    return spec, root


for finding, build in (("R9-1", ancestor), ("R9-2", look_alike), ("R9-3", linked)):
    for label, R in (("revision 9", R9), ("revision 10", R10)):
        tmp = Path(tempfile.mkdtemp(prefix=f"r9-witness-{finding}-")).resolve()
        canary = tmp / "canary"
        spec, root = build(tmp, canary)
        if R is R10:
            print(f"{finding} {label} gate:", R.scope_problems(root, ["test_scratch.py"]), "| boundary:",
                  R.boundary_problems(root, ["test_scratch.py"]))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = R.main(str(spec), root=root)
        print(f"{finding} {label}:", out.getvalue().splitlines()[0], f"| main {rc} | canary written: {canary.exists()}")
        if R is R10 and finding != "R9-1":
            # past the gate and the boundary check: the run's own audit hook
            canary.unlink(missing_ok=True)
            mutated = (root / "target.py").read_text().replace("value = 1", "value = 2").encode()
            (root / "target.py").write_bytes(mutated)
            try:
                rc, _, bundle = R.run_mutant(root, ["test_scratch.py"], {(root / "target.py").resolve(): mutated})
            finally:
                (root / "target.py").write_text("value = 1\n")
            print(f"{finding} {label} run anyway:", R.classify(rc, ["test_scratch.py::test_first"], bundle)[0],
                  "| boundary", json.dumps(bundle["boundary"]).replace(str(tmp), "<tmp>"), f"| canary written: {canary.exists()}")
