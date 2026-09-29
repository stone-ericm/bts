"""current_defence_v2 against a throwaway git repo (Codex phase-1 r1 #2, #3, #4)."""
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audit.incident_register.evidence import current_defence_v2, reset_worktree

H = "from bts._w15_witness import hook as _w15"


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


MOD = '''def grade():
    return "void"


def deliver(send):
    if grade() == "void":
        return None
    return send("pick")


def deliver_hit(send):
    ready = True
    if ready:
        send("pick")
    return "done"
'''
TESTS = '''from bts.mod import deliver, grade

def test_symptom():
    assert grade() == "void"

def test_sends_nothing_for_void():
    sent = []
    deliver(sent.append)
    assert sent == []

def test_unrelated():
    assert 1 == 1

def test_hit_is_sent():
    from bts.mod import deliver_hit
    sent = []
    deliver_hit(sent.append)
    assert sent == ["pick"]
'''


@pytest.fixture
def wt(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.email", "t@example.com")
    git(repo, "config", "user.name", "t")
    write(repo, "src/bts/__init__.py", "")
    write(repo, "src/bts/mod.py", MOD)
    write(repo, "tests/test_mod.py", TESTS)
    write(repo, "uv.lock", "# lock\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "baseline")
    wt = tmp_path / "wt"
    git(repo, "worktree", "add", "-q", "--detach", str(wt), "HEAD")
    monkeypatch.setenv("PYTHONPATH", str(wt / "src"))
    return wt


def _spec(**over):
    spec = {
        "label": "demo", "baseline": "HEAD", "allowed_paths": ["src/bts/mod.py"],
        "witness_edits": [
            ["src/bts/mod.py", 'def grade():\n    return "void"\n',
             f'def grade():\n    {H}\n    _w15("entry")\n    _w15("branch")\n    _w15("boundary")\n    return "void"\n'],
        ],
        "mutation_edits": [["src/bts/mod.py", '    return "void"\n\n\ndef deliver', '    return "miss"\n\n\ndef deliver']],
        "tests": ["tests/test_mod.py"], "killing_node": "tests/test_mod.py::test_symptom",
        "entry": "grade", "entry_file": "src/bts/mod.py", "branch_file": "src/bts/mod.py", "kind": "event",
    }
    spec.update(over)
    return spec


PY = [sys.executable]


def test_accepted_event_certificate_and_clean_restore(wt, tmp_path):
    res = current_defence_v2(wt, _spec(), tmp_path / "out", PY)
    assert res["verdict"] == "accepted", res["reasons"]
    assert res["kills"] == ["tests/test_mod.py::test_sends_nothing_for_void", "tests/test_mod.py::test_symptom"]
    assert res["killing_call"]["exc_type"] == "AssertionError"
    assert git(wt, "status", "--porcelain", "--untracked-files=all") == ""


def test_conftest_only_mutation_is_refused(wt, tmp_path):
    """Codex phase-1 r1 #2: a root conftest that swaps the function must never certify."""
    spec = _spec(mutation_edits=[["tests/test_mod.py", "def test_unrelated", "def test_unrelated_x"]])
    with pytest.raises(RuntimeError, match="outside its allowlist"):
        current_defence_v2(wt, spec, tmp_path / "out", PY)
    assert git(wt, "status", "--porcelain", "--untracked-files=all") == ""


def test_edit_to_a_production_file_outside_the_allowlist_is_refused(wt, tmp_path):
    spec = _spec(mutation_edits=[["src/bts/__init__.py", "", "X = 1\n"]])
    with pytest.raises(RuntimeError, match="outside its allowlist"):
        current_defence_v2(wt, spec, tmp_path / "out", PY)


def test_stray_files_present_before_a_run_do_not_survive_it(wt, tmp_path):
    write(wt, "conftest.py", "raise SystemExit('a stray conftest must never run')\n")
    res = current_defence_v2(wt, _spec(), tmp_path / "out", PY)
    assert res["verdict"] == "accepted", res["reasons"]
    assert not (wt / "conftest.py").exists()


def test_hooks_that_change_behaviour_are_rejected(wt, tmp_path):
    spec = _spec(witness_edits=[["src/bts/mod.py", '    return "void"\n\n\ndef deliver',
                                 f'    {H}\n    _w15("entry")\n    return "void!"\n\n\ndef deliver']],
                 mutation_edits=[])
    res = current_defence_v2(wt, spec, tmp_path / "out", PY)
    assert res["verdict"] == "rejected"
    assert any("not observational" in r for r in res["reasons"])


def test_event_certificate_for_an_extra_send(wt, tmp_path):
    spec = _spec(
        witness_edits=[["src/bts/mod.py", 'def deliver(send):\n    if grade() == "void":\n        return None\n    return send("pick")\n',
                        f'def deliver(send):\n    {H}\n    _w15("entry")\n    _w15("branch")\n    if grade() == "void":\n        _w15("done")\n        return None\n    _w15("boundary")\n    return send("pick")\n']],
        # mutant: a void grade no longer suppresses the send -> the "nothing sent" test now sees a send
        mutation_edits=[["src/bts/mod.py", '    if grade() == "void":\n        _w15("done")', '    if False:\n        _w15("done")']],
        killing_node="tests/test_mod.py::test_sends_nothing_for_void", entry="deliver", kind="event")
    res = current_defence_v2(wt, spec, tmp_path / "out", PY)
    assert res["verdict"] == "accepted", res["reasons"]


def test_reset_worktree_removes_untracked_and_ignored(wt):
    write(wt, "src/bts/stray.py", "x = 1\n")
    write(wt, "conftest.py", "import os\n")
    reset_worktree(wt, "HEAD")
    assert not (wt / "src/bts/stray.py").exists() and not (wt / "conftest.py").exists()


def test_absence_certificate_with_positive_control(wt, tmp_path):
    """Missing event: the witness-only run shows the send (positive control); the mutant's run
    reaches the decision and completes with NO send under the same invocation."""
    spec = _spec(
        witness_edits=[["src/bts/mod.py",
                        'def deliver_hit(send):\n    ready = True\n    if ready:\n        send("pick")\n    return "done"\n',
                        f'def deliver_hit(send):\n    {H}\n    _w15("entry")\n    ready = True\n    _w15("branch")\n'
                        f'    if ready:\n        _w15("boundary")\n        send("pick")\n    _w15("done")\n    return "done"\n']],
        mutation_edits=[["src/bts/mod.py", '    ready = True\n', '    ready = False\n']],
        killing_node="tests/test_mod.py::test_hit_is_sent", entry="deliver_hit", kind="absence")
    res = current_defence_v2(wt, spec, tmp_path / "out", PY)
    assert res["verdict"] == "accepted", res["reasons"]
    assert res["positive_control"]["ok"] and res["witness"]["ok"]


def test_absence_without_positive_control_is_rejected(wt, tmp_path):
    """If the baseline never produces the event, a 'missing event' cannot be certified."""
    spec = _spec(
        witness_edits=[["src/bts/mod.py",
                        'def deliver_hit(send):\n    ready = True\n    if ready:\n        send("pick")\n    return "done"\n',
                        f'def deliver_hit(send):\n    {H}\n    _w15("entry")\n    ready = True\n    _w15("branch")\n'
                        f'    if ready:\n        send("pick")\n    _w15("done")\n    return "done"\n']],
        mutation_edits=[["src/bts/mod.py", '    ready = True\n', '    ready = False\n']],
        killing_node="tests/test_mod.py::test_hit_is_sent", entry="deliver_hit", kind="absence")
    res = current_defence_v2(wt, spec, tmp_path / "out", PY)
    assert res["verdict"] == "rejected"
    assert any("positive control" in r for r in res["reasons"])
