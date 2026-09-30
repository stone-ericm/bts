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


class ClosureRefused(RuntimeError):
    """The execution environment has a shape whose effects no hash here covers (Codex phase-1 r5 #4)."""


# An executable ``.pth`` line runs at every interpreter start. It is allowed only in the one shape this
# fingerprint covers: ``import <module>`` of a module in the same site-packages (its bytes are in the
# tree hash) whose reviewed content is listed here. Anything else is refused: the code such a line runs
# can put roots outside every hash on sys.path (Codex phase-1 r5 #4).
REVIEWED_PTH_IMPORTS = {
    "_virtualenv": {
        "cfb3db86aaa53bb62b5ff764970bec2d71c9228590a0ebec57f6ec926cc0bf1a":
            "uv's venv hook (reviewed 2026-09-30): installs a meta-path finder that patches distutils/setuptools "
            "config parsing; imports only the stdlib and adds no sys.path entry",
    },
}


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


def _file_state(p: Path) -> str:
    """A file's working bytes. A symlink is frozen by its spelling AND the bytes it resolves to (a
    directory target by its whole tree, cycles marked), so a harness file reached through a link
    cannot change unseen (Codex phase-1 r5 #4)."""
    if p.is_symlink():
        h = hashlib.sha256(("symlink:" + os.readlink(p)).encode())
        target = os.path.realpath(p)
        if os.path.isdir(target):
            _tree_hash(h, Path(target))
        elif os.path.isfile(target):
            _hash_file(h, target)
        else:
            h.update(b"dangling")
        return h.hexdigest()
    if p.is_file():
        return _sha(p.read_bytes())
    return "missing"


def manifest(root, exclude=()) -> dict:
    """sha256 of the WORKING bytes of every tracked file (minus ``exclude``) and of every untracked,
    non-ignored file: an untracked input is frozen by content, not only by name (Codex phase-1 r4 #4)."""
    root = Path(root)
    exclude = set(exclude)
    files = {rel: _file_state(root / rel) for rel in _git(root, "ls-files", "-z").stdout.split("\0")
             if rel and rel not in exclude}
    untracked = {rel: _file_state(root / rel)
                 for rel in sorted(_git(root, "ls-files", "--others", "--exclude-standard", "-z").stdout.split("\0")) if rel}
    return {"files": files, "untracked": untracked}


def manifest_digest(m: dict) -> str:
    return _sha(json.dumps(m, sort_keys=True).encode())


_DIGESTS: dict[tuple, bytes] = {}


def _hash_file(h, path: str) -> None:
    """Feed the file's content digest into ``h``. Digests are memoized per process on the file's full
    identity — device, inode, size, mtime AND ctime (ns): ``os.utime`` can put an mtime back, but any
    write or utime moves the ctime, so a same-size, same-mtime rewrite still misses the memo and is
    re-read (the adversarial case of Codex phase-1 r4 #4)."""
    st = os.stat(path)
    key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    digest = _DIGESTS.get(key)
    if digest is None:
        d = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                d.update(chunk)
        digest = _DIGESTS[key] = d.digest()
    h.update(digest)


def _tree_hash(h, base: Path, visited: set | None = None) -> None:
    """Hash every file under ``base`` as Python would execute it (Codex phase-1 r4 #4):
    * bytecode INCLUDED — ``-B``/``PYTHONDONTWRITEBYTECODE`` stop writes, not reads, and a cached .pyc
      whose source stamp still matches is what runs;
    * a symlink contributes its spelling AND the bytes it resolves to; a symlinked directory is walked
      at its real path; each real directory is walked once (cycles and repeats are marked, not followed)."""
    visited = set() if visited is None else visited
    real = os.path.realpath(base)
    if real in visited:
        h.update(b"revisit:" + real.encode())
        return
    visited.add(real)
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames.sort()
        for d in [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]:
            full = os.path.join(dirpath, d)             # os.walk lists a directory symlink but never enters it
            h.update(os.path.relpath(full, base).encode() + b"\0dirlink:" + os.readlink(full).encode())
            _tree_hash(h, Path(os.path.realpath(full)), visited)
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            h.update(os.path.relpath(p, base).encode() + b"\0")
            if os.path.islink(p):
                h.update(b"link:" + os.readlink(p).encode())
                target = os.path.realpath(p)
                if os.path.isdir(target):
                    _tree_hash(h, Path(target), visited)
                elif os.path.isfile(target):
                    _hash_file(h, target)
                else:
                    h.update(b"dangling")
            elif os.path.isfile(p):
                _hash_file(h, p)


def venv_fingerprint(root) -> str:
    """Content hash of the worktree's execution environment (Codex phase-1 r3 #4): every file of
    ``root/.venv`` (bytecode included, see ``_tree_hash``) and every directory a ``.pth`` path line adds
    from OUTSIDE the worktree. Directories inside the worktree (the editable ``src``) are the manifest's
    job. Raises ``ClosureRefused`` for an executable ``.pth`` line that is not a reviewed import hook."""
    root = Path(root)
    env = root / ".venv"
    if not env.is_dir():
        return "absent"
    wt = os.path.realpath(root) + os.sep
    h = hashlib.sha256()
    _tree_hash(h, env)
    for pth in sorted(env.glob("lib/python3*/site-packages/*.pth")):
        for line in pth.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("import ", "import\t")):       # site.addpackage exec()s these lines
                name = line[len("import"):].strip()
                hook = pth.parent / f"{name}.py"
                if not (name.isidentifier() and hook.is_file()
                        and _sha(hook.read_bytes()) in REVIEWED_PTH_IMPORTS.get(name, {})):
                    raise ClosureRefused(f"{pth.name}: executable line {line[:80]!r} is not a reviewed import hook")
                continue                                 # the hook module's bytes are in the tree hash
            # site.addpackage semantics: a relative line is relative to the .pth file's own directory,
            # never to the reviewer's working directory (Codex phase-1 r4 #4)
            target = os.path.realpath(os.path.join(os.path.dirname(pth), line))
            if os.path.isdir(target) and not target.startswith(wt):
                h.update(b"pth:" + target.encode())
                _tree_hash(h, Path(target))
    return h.hexdigest()
