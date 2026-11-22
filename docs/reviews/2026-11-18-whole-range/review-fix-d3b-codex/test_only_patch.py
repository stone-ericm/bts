"""Run the old and amended B8a assertions in an isolated source copy."""
import ast
import os
from pathlib import Path
from shutil import copy2
from subprocess import run
from tempfile import TemporaryDirectory

from review_box import HERE, ROOT

TEST = Path("tests/test_wr_b_contract.py")
DOC = Path("docs/ops/2027-season-start.md")
NAME = "test_b8a_requires_successful_fetch_automatic_readiness_and_saver_read"


def check(path, expected):
    file = path / TEST
    source = ast.parse(file.read_text(), filename=str(file))
    source.body = [node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == NAME]
    assert len(source.body) == 1
    import re
    scope = {"re": re, "Path": Path, "__file__": str(file)}
    exec(compile(source, str(file), "exec"), scope)
    try:
        scope[NAME]()
    except AssertionError:
        observed = False
    else:
        observed = True
    assert observed == expected


with TemporaryDirectory(prefix="b8a-test-patch-", dir=HERE) as folder:
    scratch = Path(folder)
    for filename in (TEST, DOC):
        (scratch / filename).parent.mkdir(parents=True)
        copy2(ROOT / filename, scratch / filename)
    check(scratch, False)
    print("Original assertion fails on the current B8a row.")
    patch = HERE / "4733a298-test-only.patch"
    isolated = os.environ | {"GIT_CEILING_DIRECTORIES": str(HERE)}
    run(["git", "apply", "--no-index", "--check", str(patch)], cwd=scratch, env=isolated, check=True, timeout=5)
    run(["git", "apply", "--no-index", str(patch)], cwd=scratch, env=isolated, check=True, timeout=5)
    assert "saver-state[^\x60]*" in (scratch / TEST).read_text()
    check(scratch, True)
    print("Patched assertion passes on the unchanged B8a row.")
    row = (scratch / DOC).read_text()
    controls = [
        ("automatic path alone: accepted", "automatic path alone: refused", False),
        ("Initialize only if it reports `uninitialized`", "Initialize regardless of its status", False),
        ("read `bts saver-state --show", "read `bts saver-state-typo --show", True),
    ]
    for original, replacement, expected in controls:
        assert original in row
        (scratch / DOC).write_text(row.replace(original, replacement))
        check(scratch, expected)
        print(f"{replacement!r} -> {'accepted' if expected else 'rejected'}")
