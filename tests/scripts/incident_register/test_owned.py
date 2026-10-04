"""Owned evidence worktrees (Codex phase-1 r2 #1, #2): every refusal happens before any change."""
import os
from pathlib import Path

import pytest

from scripts.audit.incident_register import owned
from tests.scripts.incident_register.synth import git, make_repo, write

MOD = "def grade():\n    return 'void'\n"
TEST = "from bts.mod import grade\n\ndef test_grade():\n    assert grade() == 'void'\n"


@pytest.fixture
def repo(tmp_path):
    return make_repo(tmp_path / "repo", {"src/bts/mod.py": MOD, "tests/test_mod.py": TEST})


@pytest.fixture
def wt(repo, tmp_path):
    path = Path(os.path.realpath(tmp_path)) / "wt"
    owned.create(repo, "HEAD", path)
    return path


def test_create_marks_a_detached_linked_worktree(wt, repo):
    rec = owned.assert_owned(wt)
    assert rec["path"] == str(wt)
    assert git(wt, "rev-parse", "HEAD") == git(repo, "rev-parse", "HEAD")
    assert not (wt / owned.OWNER_FILE).exists()          # the record is outside the working tree


def test_reset_restores_tracked_files_and_removes_strays(wt):
    write(wt, "src/bts/mod.py", "def grade():\n    return 'miss'\n")
    write(wt, "stray.txt", "x")
    write(wt, "src/bts/__pycache__/junk.pyc", "x")
    owned.reset(wt, "HEAD")
    assert (wt / "src/bts/mod.py").read_text() == MOD
    assert not (wt / "stray.txt").exists() and not (wt / "src/bts/__pycache__").exists()


def test_primary_checkout_is_refused_untouched(repo):
    """r2 #1 measured: reset_worktree(primary, "HEAD") discarded unsaved work."""
    write(repo, "keep_me.txt", "sentinel")
    write(repo, "src/bts/mod.py", "# unsaved edit\n")
    with pytest.raises(owned.OwnershipError, match="primary checkout"):
        owned.reset(repo, "HEAD")
    assert (repo / "keep_me.txt").read_text() == "sentinel"
    assert (repo / "src/bts/mod.py").read_text() == "# unsaved edit\n"


def test_subdirectory_is_refused(wt):
    write(wt, "src/bts/keep.txt", "sentinel")
    with pytest.raises(owned.OwnershipError, match="root"):
        owned.reset(wt / "src", "HEAD")
    assert (wt / "src/bts/keep.txt").exists()


def test_symlinked_root_is_refused(wt, tmp_path):
    link = Path(os.path.realpath(tmp_path)) / "link"
    link.symlink_to(wt)
    write(wt, "keep.txt", "sentinel")
    with pytest.raises(owned.OwnershipError, match="canonical"):
        owned.reset(link, "HEAD")
    assert (wt / "keep.txt").exists()


def test_unowned_linked_worktree_is_refused(repo, tmp_path):
    other = Path(os.path.realpath(tmp_path)) / "other"
    git(repo, "worktree", "add", "-q", "--detach", str(other), "HEAD")
    write(other, "keep.txt", "sentinel")
    with pytest.raises(owned.OwnershipError, match="ownership record"):
        owned.reset(other, "HEAD")
    assert (other / "keep.txt").exists()


def test_attached_branch_worktree_is_refused(wt):
    git(wt, "switch", "-q", "-c", "somebranch")
    write(wt, "keep.txt", "sentinel")
    with pytest.raises(owned.OwnershipError, match="detached"):
        owned.reset(wt, "HEAD")
    assert (wt / "keep.txt").exists()


@pytest.mark.parametrize("rel, why", [
    ("../outside.txt", "normalized"),
    ("/etc/hosts", "normalized"),
    ("src/bts/../../tests/test_mod.py", "normalized"),
    ("tests/test_mod.py", "src/bts"),
    ("conftest.py", "src/bts"),
    ("pytest.ini", "src/bts"),
    ("src/bts/new_file.py", "tracked"),
    ("src/bts/mod.txt", "src/bts"),
])
def test_mutation_paths_outside_the_hard_rule_are_refused(wt, rel, why):
    """r2 #1 (traversal) and #2 (tests/config frozen even if a spec allows them)."""
    (Path(os.path.realpath(wt)).parent / "outside.txt").write_text("untouched")
    with pytest.raises(owned.PathRefused, match=why):
        owned.apply_edits(wt, [[rel, "a", "b"]], allowed={rel})
    assert (Path(os.path.realpath(wt)).parent / "outside.txt").read_text() == "untouched"
    assert git(wt, "status", "--porcelain") == ""


def test_symlinked_component_is_refused(wt, tmp_path):
    target = Path(os.path.realpath(tmp_path)) / "elsewhere"
    target.mkdir()
    (target / "x.py").write_text("a\n")
    (wt / "src/bts/sub").symlink_to(target)
    with pytest.raises(owned.PathRefused):
        owned.apply_edits(wt, [["src/bts/sub/x.py", "a", "b"]], allowed={"src/bts/sub/x.py"})
    assert (target / "x.py").read_text() == "a\n"


def test_edits_are_validated_before_any_write(wt):
    edits = [["src/bts/mod.py", "'void'", "'miss'"], ["tests/test_mod.py", "'void'", "'miss'"]]
    with pytest.raises(owned.PathRefused):
        owned.apply_edits(wt, edits, allowed={"src/bts/mod.py", "tests/test_mod.py"})
    assert (wt / "src/bts/mod.py").read_text() == MOD


def test_edit_old_text_must_be_unique(wt):
    with pytest.raises(owned.PathRefused, match="exactly once"):
        owned.apply_edits(wt, [["src/bts/mod.py", "r", "R"]], allowed={"src/bts/mod.py"})


def test_valid_edit_applies_and_reports_touched(wt):
    touched = owned.apply_edits(wt, [["src/bts/mod.py", "'void'", "'miss'"]], allowed={"src/bts/mod.py"})
    assert touched == ["src/bts/mod.py"]
    assert "'miss'" in (wt / "src/bts/mod.py").read_text()


def test_manifest_hashes_working_bytes_and_lists_untracked(wt):
    m0 = owned.manifest(wt)
    write(wt, "tests/test_mod.py", TEST + "# changed working bytes, index untouched\n")
    write(wt, "new_untracked.txt", "x")
    m1 = owned.manifest(wt)
    assert m1["files"]["tests/test_mod.py"] != m0["files"]["tests/test_mod.py"]
    assert list(m1["untracked"]) == ["new_untracked.txt"] and m0["untracked"] == {}
    assert m1["untracked"]["new_untracked.txt"] == owned._sha(b"x")        # frozen by content, not only name
    assert owned.manifest(wt, exclude={"tests/test_mod.py"})["files"].keys() == m0["files"].keys() - {"tests/test_mod.py"}


def test_destroy_removes_only_an_owned_worktree(wt, repo):
    with pytest.raises(owned.OwnershipError):
        owned.destroy(repo)
    owned.destroy(wt)
    assert not wt.exists()


def test_tracked_symlink_into_the_frozen_harness_is_refused(tmp_path):
    """A tracked src/bts/*.py symlink resolving INSIDE the worktree (to a test file) passes the
    containment check; only the symlink rule stops an edit writing through it."""
    repo = make_repo(tmp_path / "r2", {"src/bts/mod.py": MOD, "tests/test_mod.py": TEST})
    (repo / "src/bts/alias.py").symlink_to("../../tests/test_mod.py")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "alias")
    path = Path(os.path.realpath(tmp_path)) / "wt2"
    owned.create(repo, "HEAD", path)
    with pytest.raises(owned.PathRefused, match="symlink"):
        owned.apply_edits(path, [["src/bts/alias.py", "'void'", "'miss'"]], allowed={"src/bts/alias.py"})
    assert (path / "tests/test_mod.py").read_text() == TEST


def test_manifest_sees_a_retargeted_symlink(tmp_path):
    repo = make_repo(tmp_path / "r3", {"a.txt": "same\n", "b.txt": "same\n"})
    (repo / "link").symlink_to("a.txt")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "link")
    m0 = owned.manifest(repo)
    (repo / "link").unlink()
    (repo / "link").symlink_to("b.txt")          # identical bytes, different target
    assert owned.manifest(repo)["files"]["link"] != m0["files"]["link"]


def _fake_venv(root: Path, outside: Path) -> Path:
    """A worktree-like dir whose .venv holds one installed module, a .pyc, and a .pth to ``outside``."""
    site = root / ".venv" / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    (site / "pkg.py").write_text("VALUE = 1\n")
    (site / "pkg.cpython-312.pyc").write_bytes(b"\x00bytecode")
    outside.mkdir()
    (outside / "helper.py").write_text("X = 1\n")
    (site / "extra.pth").write_text(f"{outside}\n")
    return site


def test_venv_fingerprint_hashes_file_contents_not_just_names(tmp_path):
    """Sweep W3: an installed file whose bytes change under the same name (same size here) must change
    the fingerprint — bytecode included, since a cached .pyc is what runs (Codex phase-1 r4 #4)."""
    root = tmp_path / "wt"
    site = _fake_venv(root, tmp_path / "outside")
    f0 = owned.venv_fingerprint(root)
    (site / "pkg.py").write_text("VALUE = 2\n")
    f1 = owned.venv_fingerprint(root)
    assert f1 != f0
    (site / "pkg.cpython-312.pyc").write_bytes(b"\x01bytecode")
    assert owned.venv_fingerprint(root) != f1


def test_a_same_size_rewrite_with_its_mtime_put_back_still_changes_the_fingerprint(tmp_path):
    """Sweep W9: digests are memoized per process on the file's identity. After a same-size rewrite,
    os.utime can put the mtime back; only the ctime still moves, so it must be part of the memo key
    (the test above changes the mtime, which misses the memo on its own)."""
    root = tmp_path / "wt"
    site = _fake_venv(root, tmp_path / "outside")
    f = site / "pkg.py"
    f0 = owned.venv_fingerprint(root)
    before = f.stat()
    f.write_text("VALUE = 2\n")                                 # same size, same inode
    os.utime(f, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = f.stat()
    assert (after.st_ino, after.st_size, after.st_mtime_ns) == (before.st_ino, before.st_size, before.st_mtime_ns)
    assert after.st_ctime_ns != before.st_ctime_ns
    assert owned.venv_fingerprint(root) != f0


def test_venv_fingerprint_hashes_pth_trees_outside_the_worktree(tmp_path):
    """Sweep W4: a directory a .pth file adds from OUTSIDE the worktree is part of the environment."""
    root = tmp_path / "wt"
    _fake_venv(root, tmp_path / "outside")
    f0 = owned.venv_fingerprint(root)
    (tmp_path / "outside" / "helper.py").write_text("X = 2\n")
    assert owned.venv_fingerprint(root) != f0
