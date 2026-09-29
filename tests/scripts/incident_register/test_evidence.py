"""The evidence harness against a throwaway git repo (no network, no box data)."""
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audit.incident_register.evidence import (
    NodeResult,
    classify,
    current_defence,
    harness_changes,
    historical_replay,
    use_src,
)


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


TEST_AT_FIX = '''from bts.mod import grade

def test_symptom():
    assert grade() == "void"

def test_new_api():
    from bts.newmod import helper
    assert helper() == 1

def test_unchanged():
    assert 1 == 1
'''


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@example.com")
    git(root, "config", "user.name", "t")
    write(root, "src/bts/__init__.py", "")
    write(root, "src/bts/mod.py", 'def grade():\n    return "miss"\n')
    write(root, "tests/test_mod.py", "def test_unchanged():\n    assert 1 == 1\n")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "parent")
    write(root, "src/bts/mod.py", 'def grade():\n    return "void"\n')
    write(root, "src/bts/newmod.py", "def helper():\n    return 1\n")
    write(root, "tests/test_mod.py", TEST_AT_FIX)
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "fix")
    return root


@pytest.fixture
def worktree(repo, tmp_path, monkeypatch):
    wt = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "--detach", str(wt), "HEAD")
    monkeypatch.setenv("PYTHONPATH", str(wt / "src"))
    return wt


PY = [sys.executable]


def test_use_src_removes_files_the_ref_lacks(worktree):
    assert (worktree / "src/bts/newmod.py").exists()
    use_src(worktree, "HEAD^")
    assert not (worktree / "src/bts/newmod.py").exists()
    assert 'return "miss"' in (worktree / "src/bts/mod.py").read_text()
    use_src(worktree, "HEAD")
    assert (worktree / "src/bts/newmod.py").exists()


def test_historical_replay_classifies_nodes(repo, worktree, tmp_path):
    fix = git(repo, "rev-parse", "HEAD")
    rep = historical_replay(repo, worktree, "demo", fix, ["tests/test_mod.py"], tmp_path / "out", PY)
    classes = {k.rsplit("::", 1)[-1]: v for k, v in rep.classes.items()}
    assert classes == {"test_symptom": "symptom_candidate",
                       "test_new_api": "new_api",          # passes only if newmod.py leaked into red
                       "test_unchanged": "passes_at_parent"}
    assert rep.red["imported_bts"] == str(worktree / "src" / "bts" / "__init__.py")
    assert rep.label_kind == "semantic_regression_replay"
    assert (tmp_path / "out" / "replay.json").exists()


def test_harness_audit_flags_conftest_changes(repo):
    parent = git(repo, "rev-parse", "HEAD")
    write(repo, "tests/conftest.py", "import pytest\n")
    write(repo, "tests/data/fixture.json", "{}")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "fixture change")
    kinds = {c["path"]: c["kind"] for c in harness_changes(repo, parent, "HEAD")}
    assert kinds == {"tests/conftest.py": "conftest", "tests/data/fixture.json": "test_data"}


def test_classify_rules():
    ok = NodeResult("passed", "call")
    assert classify(NodeResult("failed", "call", "AssertionError"), ok) == "symptom_candidate"
    assert classify(NodeResult("failed", "call", "TypeError"), ok) == "new_api"
    assert classify(NodeResult("error", "setup", "ValueError"), ok) == "setup_error"
    assert classify(NodeResult("error", "collection", "ImportError"), ok) == "collection_error"
    assert classify(NodeResult("skipped", ""), ok) == "skipped"
    assert classify(ok, ok) == "passes_at_parent"
    assert classify(None, ok) == "absent_at_parent"
    assert classify(NodeResult("failed", "call", "AssertionError"), NodeResult("failed", "call")) \
        == "green_not_passing"


def _mutant_patch(repo: Path, tmp_path: Path, touch_tests: bool = False) -> Path:
    scratch = tmp_path / "scratch"
    git(repo, "worktree", "add", "-q", "--detach", str(scratch), "HEAD")
    write(scratch, "src/bts/mod.py",
          'from bts._w15_witness import hook\n\ndef grade():\n    hook("branch")\n    return "miss"\n')
    if touch_tests:
        write(scratch, "tests/test_mod.py", "def test_symptom():\n    pass\n")
    patch = tmp_path / "mutant.patch"
    patch.write_text(git(scratch, "diff") + "\n")
    return patch


def test_current_defence_red_then_restored(repo, worktree, tmp_path):
    patch = _mutant_patch(repo, tmp_path)
    res = current_defence(worktree, "HEAD", "demo", patch, ["tests/test_mod.py::test_symptom"],
                          tmp_path / "def", PY)
    node = next(iter(res["mutant"]["nodes"]))
    assert res["green_before"]["exit"] == 0 and res["green_after"]["exit"] == 0
    assert res["mutant"]["nodes"][node]["outcome"] == "failed"
    assert res["mutant"]["nodes"][node]["phase"] == "call"
    assert res["tests_unchanged_after_mutant"] is True
    assert res["witness_records"] >= 1
    assert not (worktree / "src/bts/_w15_witness.py").exists()


def test_current_defence_refuses_a_patch_touching_tests(repo, worktree, tmp_path):
    patch = _mutant_patch(repo, tmp_path, touch_tests=True)
    with pytest.raises(RuntimeError, match="may not touch tests"):
        current_defence(worktree, "HEAD", "demo", patch, ["tests/test_mod.py"], tmp_path / "d", PY)
