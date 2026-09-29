"""Harness-owned evidence worktrees (Codex phase-1 r2 #1, #2).

Every destructive operation — reset, clean, source swap, mutation edit, removal — runs only inside
a detached linked worktree that ``create`` made and marked. The ownership record lives in that
worktree's private git directory (``.git/worktrees/<name>/w15-evidence-owner.json``), outside the
working tree: ``git clean`` never touches it and nothing under test sees it. All checks run BEFORE
anything is modified, so a refusal leaves the target exactly as it was.

Mutation targets obey a hard rule that no spec can widen: existing, tracked ``src/bts/**.py``
files, named by a normalized relative path with no ``..``, no absolute prefix and no symlinked
component. Tests, conftest, configuration, lock files, scripts and the observer are frozen whatever
a spec's allowlist says.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
from pathlib import Path

OWNER_FILE = "w15-evidence-owner.json"
MUTABLE_PREFIX = "src/bts/"


class OwnershipError(RuntimeError):
    """The target is not a harness-owned evidence worktree (nothing was changed)."""


class PathRefused(RuntimeError):
    """A mutation path or edit is outside the hard rule (nothing was changed)."""


def _git(cwd, *args, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def _out(cwd, *args) -> str:
    return _git(cwd, *args).stdout.strip()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create(repo, ref: str, path) -> dict:
    """``git worktree add --detach path ref`` and write the ownership record."""
    path = Path(path)
    if not path.is_absolute() or Path(os.path.realpath(path.parent)) / path.name != path:
        raise OwnershipError(f"{path}: target must be a canonical absolute path")
    if path.exists():
        raise OwnershipError(f"{path}: target already exists")
    _git(repo, "worktree", "add", "-q", "--detach", str(path), ref)
    gitdir = Path(_out(path, "rev-parse", "--absolute-git-dir"))
    common = _out(path, "rev-parse", "--path-format=absolute", "--git-common-dir")
    record = {"path": str(path), "common_dir": os.path.realpath(common),
              "created_at_ref": _out(path, "rev-parse", "HEAD"), "nonce": secrets.token_hex(8)}
    (gitdir / OWNER_FILE).write_text(json.dumps(record, sort_keys=True))
    return record


def assert_owned(path) -> dict:
    """The exact root of a detached linked worktree carrying a matching ownership record."""
    p = Path(path)
    if not p.is_absolute() or os.path.realpath(p) != str(p):
        raise OwnershipError(f"{path}: not a canonical absolute path (symlinked or relative root)")
    top = _git(p, "rev-parse", "--show-toplevel", check=False)
    if top.returncode != 0:
        raise OwnershipError(f"{p}: not inside a git worktree")
    if top.stdout.strip() != str(p):
        raise OwnershipError(f"{p}: not the worktree root ({top.stdout.strip()})")
    gitdir = _out(p, "rev-parse", "--absolute-git-dir")
    common = _out(p, "rev-parse", "--path-format=absolute", "--git-common-dir")
    if os.path.realpath(gitdir) == os.path.realpath(common):
        raise OwnershipError(f"{p}: primary checkout, not a linked evidence worktree")
    if _git(p, "symbolic-ref", "-q", "HEAD", check=False).returncode == 0:
        raise OwnershipError(f"{p}: HEAD is attached to a branch; evidence worktrees are detached")
    owner = Path(gitdir) / OWNER_FILE
    if not owner.is_file():
        raise OwnershipError(f"{p}: no ownership record (not created by this harness)")
    record = json.loads(owner.read_text())
    if record.get("path") != str(p) or record.get("common_dir") != os.path.realpath(common):
        raise OwnershipError(f"{p}: ownership record mismatch")
    return record


def reset(path, ref: str) -> None:
    """Hard reset + clean (``.venv`` kept) of an owned worktree, verified clean afterwards."""
    assert_owned(path)
    _git(path, "reset", "-q", "--hard", ref)
    _git(path, "clean", "-qffdx", "-e", ".venv")
    left = [ln for ln in _out(path, "status", "--porcelain", "--ignored").splitlines() if ln != "!! .venv/"]
    if left:
        raise OwnershipError(f"{path}: reset left residue: {left[:5]}")


def destroy(path) -> None:
    record = assert_owned(path)
    _git(Path(record["common_dir"]).parent, "worktree", "remove", "--force", str(path))


def swap_src(path, ref: str) -> str:
    """Make ``src/`` exactly ``ref:src`` (no extra files) in an owned worktree; returns the tree id."""
    assert_owned(path)
    shutil.rmtree(Path(path) / "src", ignore_errors=True)
    _git(path, "checkout", ref, "--", "src/")
    same = _git(path, "diff", "--quiet", ref, "--", "src", check=False).returncode == 0
    extra = _out(path, "ls-files", "--others", "--exclude-standard", "src")
    if not same or extra:
        raise OwnershipError(f"src/ is not exactly {ref}: extra={extra!r}")
    return _out(path, "rev-parse", f"{ref}:src")


def check_mutation_path(root, rel) -> Path:
    """Refuse anything but an existing, tracked, symlink-free ``src/bts/**.py`` under ``root``."""
    root = Path(root)
    if (not isinstance(rel, str) or not rel or rel.startswith("/") or "\\" in rel
            or any(part in ("", ".", "..") for part in rel.split("/")) or os.path.normpath(rel) != rel):
        raise PathRefused(f"{rel!r}: not a normalized relative path")
    if not (rel.startswith(MUTABLE_PREFIX) and rel.endswith(".py")):
        raise PathRefused(f"{rel!r}: mutations are limited to existing tracked src/bts/*.py files")
    cur = root
    for part in rel.split("/"):
        cur = cur / part
        if cur.is_symlink():
            raise PathRefused(f"{rel!r}: symlinked path component {part!r}")
    full = root / rel
    if os.path.commonpath([os.path.realpath(full), os.path.realpath(root)]) != os.path.realpath(root):
        raise PathRefused(f"{rel!r}: resolves outside the worktree")
    if not full.is_file() or _git(root, "ls-files", "--error-unmatch", "--", rel, check=False).returncode != 0:
        raise PathRefused(f"{rel!r}: is not an existing tracked file")
    return full


def apply_edits(root, edits, allowed) -> list[str]:
    """Validate every edit first (hard rule, allowlist, unique ``old``), then write."""
    assert_owned(root)
    allowed = set(allowed)
    staged: dict[str, str] = {}
    for edit in edits:
        if not (isinstance(edit, (list, tuple)) and len(edit) == 3):
            raise PathRefused(f"malformed edit {edit!r}")
        rel, old, new = edit
        full = check_mutation_path(root, rel)
        if rel not in allowed:
            raise PathRefused(f"{rel!r}: not in this spec's allowed paths")
        text = staged.get(rel, full.read_text())
        if not old or text.count(old) != 1:
            raise PathRefused(f"{rel!r}: edit text must occur exactly once (found {text.count(old) if old else 0})")
        staged[rel] = text.replace(old, new, 1)
    for rel, text in staged.items():
        (Path(root) / rel).write_text(text)
    return sorted(staged)


def manifest(root, exclude=()) -> dict:
    """sha256 of the WORKING bytes of every tracked file (minus ``exclude``) + untracked files."""
    root = Path(root)
    exclude = set(exclude)
    files = {}
    for rel in _git(root, "ls-files", "-z").stdout.split("\0"):
        if not rel or rel in exclude:
            continue
        p = root / rel
        if p.is_symlink():
            files[rel] = _sha(("symlink:" + os.readlink(p)).encode())
        elif p.is_file():
            files[rel] = _sha(p.read_bytes())
        else:
            files[rel] = "missing"
    untracked = sorted(r for r in _git(root, "ls-files", "--others", "--exclude-standard", "-z").stdout.split("\0") if r)
    return {"files": files, "untracked": untracked}


def manifest_digest(m: dict) -> str:
    return _sha(json.dumps(m, sort_keys=True).encode())


def venv_fingerprint(root) -> str:
    """pyvenv.cfg + every .pth + every installed distribution's RECORD in ``root/.venv``."""
    env = Path(root) / ".venv"
    if not env.is_dir():
        return "absent"
    parts = [("cfg", _sha((env / "pyvenv.cfg").read_bytes()) if (env / "pyvenv.cfg").exists() else "none")]
    for site in sorted(env.glob("lib/python3*/site-packages")):
        for p in sorted(site.iterdir()):
            if p.suffix == ".pth":
                parts.append(("pth", p.name, _sha(p.read_bytes())))
            elif p.name.endswith(".dist-info"):
                rec = p / "RECORD"
                parts.append(("dist", p.name, _sha(rec.read_bytes()) if rec.exists() else "none"))
    return _sha(json.dumps(parts).encode())
