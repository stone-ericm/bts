"""Reproduce a local main left behind by a successful remote merge."""
import os
import tempfile
from pathlib import Path

from fixtures import HERE, fixture

shell = fixture("probe_ops").shell
env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1",
           GIT_AUTHOR_NAME="Reviewer", GIT_AUTHOR_EMAIL="review@offline.invalid",
           GIT_COMMITTER_NAME="Reviewer", GIT_COMMITTER_EMAIL="review@offline.invalid",
           GIT_ALLOW_PROTOCOL="file")


def git(root, *words):
    return shell(["git", *words], root, env)


with tempfile.TemporaryDirectory(dir=HERE) as name:
    root = Path(name)
    bare = root / "origin.git"
    git(root, "init", "--bare", "-q", "-b", "main", str(bare))
    git(root, "clone", "-q", str(bare), "fixer")
    fixer = root / "fixer"
    git(fixer, "commit", "--allow-empty", "-q", "-m", "base")
    old = git(fixer, "rev-parse", "HEAD")
    git(fixer, "push", "-q", "origin", "main", "main:deploy")
    git(fixer, "switch", "-q", "-c", "fix/review")
    git(fixer, "commit", "--allow-empty", "-q", "-m", "approved fix")
    approved = git(fixer, "rev-parse", "HEAD")
    git(fixer, "push", "-q", "origin", "fix/review")
    git(root, "clone", "-q", str(bare), "merger")
    merger = root / "merger"
    git(merger, "merge", "--ff-only", "origin/fix/review")
    git(merger, "push", "origin", "main")
    git(fixer, "push", "origin", "main:deploy")
    assert git(bare, "rev-parse", "main") == approved != old
    assert git(bare, "rev-parse", "deploy") == old == git(fixer, "rev-parse", "main")
    print("After literal post-merge command: origin/main", approved, "origin/deploy and local main", old)
    git(fixer, "switch", "main")
    git(fixer, "pull", "--ff-only", "origin", "main")
    assert git(fixer, "rev-parse", "main") == approved
    git(fixer, "push", "origin", "main:deploy")
    assert git(bare, "rev-parse", "deploy") == approved
    print("With a fast-forward and approved-SHA check: origin/deploy", approved)
