"""Synthetic repositories for the evidence-tooling tests.

Each synthetic repo has its own ``pytest.ini`` (so an ancestor ``pyproject.toml`` is never the
rootdir config — Codex phase-1 r2 #9), a ``.gitignore`` for ``.venv/`` and caches, a tiny
``src/bts`` package and tests. ``make_venv`` gives an owned worktree its own interpreter
environment: a stdlib ``venv`` whose ``.pth`` exposes this process's site-packages (pytest) and the
worktree's ``src``.
"""
from __future__ import annotations

import subprocess
import sys
import sysconfig
import venv
from pathlib import Path

PYTEST_INI = "[pytest]\naddopts = -p no:cacheprovider\n"
GITIGNORE = ".venv/\n__pycache__/\n*.pyc\n.pytest_cache/\n"


def git(cwd, *args) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def write(root: Path, rel: str, text: str) -> None:
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def make_repo(root: Path, files: dict[str, str]) -> Path:
    root = Path(root)
    root.mkdir(parents=True)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "t@t")
    git(root, "config", "user.name", "t")
    base = {"pytest.ini": PYTEST_INI, ".gitignore": GITIGNORE, "src/bts/__init__.py": "",
            "tests/__init__.py": ""}
    for rel, text in {**base, **files}.items():
        write(root, rel, text)
    git(root, "add", "-A")
    git(root, "commit", "-qm", "base")
    return Path(root.resolve())


def commit(root: Path, files: dict[str, str], message: str) -> str:
    for rel, text in files.items():
        write(root, rel, text)
    git(root, "add", "-A")
    git(root, "commit", "-qm", message)
    return git(root, "rev-parse", "HEAD")


def make_venv(worktree: Path) -> Path:
    env_dir = Path(worktree) / ".venv"
    venv.EnvBuilder(with_pip=False, symlinks=True).create(env_dir)
    # stdlib venv site-packages path for this interpreter version
    site = next((env_dir / "lib").glob("python3*/site-packages"))
    (site / "w15_test.pth").write_text(f"{sysconfig.get_paths()['purelib']}\n{Path(worktree) / 'src'}\n")
    return env_dir / "bin" / "python"


def base_python() -> str:
    return getattr(sys, "_base_executable", sys.executable)


# ---------------------------------------------------------------------------- a defended project
TRANSPORT = '''def send(recipient, text):
    return "id-1"
'''

MOD = '''from bts import transport


def grade():
    return "void"


def deliver(ready=True, late=False):
    if late:
        transport.send("eric", "BTS health CRITICAL: late")
    if ready:
        transport.send("eric", "pick: Turner")
    return "done"


def run(ready=True):
    result = deliver(ready)
    if result == "defer":
        transport.send("eric", "pick: Turner (late)")
    return result


async def adeliver(ready=True):
    import asyncio
    await asyncio.sleep(0)
    if ready:
        transport.send("eric", "pick: async")
'''

TESTS = '''import asyncio
from unittest.mock import patch

from bts import mod


def _texts(send):
    return [c.args[1] for c in send.call_args_list]


def test_grade():
    assert mod.grade() == "void"  # ASSERT-GRADE


def test_deliver_sends_one_pick():
    with patch("bts.transport.send") as send:
        mod.run(True)
    assert _texts(send) == ["pick: Turner"]  # ASSERT-SEND


def test_run_sends_once_and_returns_done():
    with patch("bts.transport.send") as send:
        result = mod.run(True)
    assert len(send.call_args_list) == 1  # ASSERT-COUNT
    assert result == "done"  # ASSERT-RESULT


def test_async_delivery():
    with patch("bts.transport.send") as send:
        asyncio.run(mod.adeliver(True))
    assert _texts(send) == ["pick: async"]  # ASSERT-ASYNC


def test_unrelated():
    assert 1 + 1 == 2
'''

DM = {"name": "dm", "binding": "bts.transport:send", "value": ["args[1]", "kw:text"],
      "classify": [["alert", "^BTS health"], ["pick", "^pick"]]}


def defended_project(tmp_path, extra: dict | None = None):
    """(repo, owned worktree with its own venv) for the defended synthetic project."""
    import os

    from scripts.audit.incident_register import owned

    files = {"src/bts/transport.py": TRANSPORT, "src/bts/mod.py": MOD, "tests/test_mod.py": TESTS}
    files.update(extra or {})
    repo = make_repo(Path(os.path.realpath(tmp_path)) / "repo", files)
    wt = Path(os.path.realpath(tmp_path)) / "wt"
    owned.create(repo, "HEAD", wt)
    make_venv(wt)
    return repo, wt


def spec(**over) -> dict:
    """An absence-kind spec: the mutant stops deliver() sending the pick."""
    base = {
        "label": "synthetic", "baseline": "HEAD", "tests": ["tests/test_mod.py", "-q"],
        "allowed_paths": ["src/bts/mod.py"],
        "mutation_edits": [["src/bts/mod.py",
                            '    if ready:\n        transport.send("eric", "pick: Turner")',
                            '    if not ready:\n        transport.send("eric", "pick: Turner")']],
        "branch": {"path": "src/bts/mod.py", "text": "if not ready:"},
        "entry": {"path": "src/bts/mod.py", "qualname": "deliver"},
        "boundaries": [DM],
        "symptom": {"kind": "absence", "boundary": "dm", "category": "pick"},
        "killing": [{"node": "tests/test_mod.py::test_deliver_sends_one_pick",
                     "assertion": {"path": "tests/test_mod.py", "text": "# ASSERT-SEND"}}],
    }
    base.update(over)
    return base
