"""Codex phase-1 r5 counterexamples, each pinned in the direction the design requires.

Adapted from the reviewer's retained probes (``test_review_r5.py``). Each of them measured a FALSE
acceptance at 274ceeb. A certificate that cannot establish its coverage must be refused or
unavailable (design §9.3 as amended), never turned into an absence, classification or closure claim.
"""
import os
import subprocess

import pytest

from scripts.audit.incident_register import acceptance, owned
from tests.scripts.incident_register.synth import defended_project


# --- finding 4: the execution closure --------------------------------------------------------------

def _site(wt):
    return next((wt / ".venv/lib").glob("python*/site-packages"))


def test_an_unreviewed_executable_pth_line_is_refused(tmp_path):
    """r5 #4.1 measured: an executable .pth line put an external root on sys.path; the imported module
    changed 1 -> 2 while the fingerprint stayed identical. Such a line is now refused, not hashed as
    text."""
    _, wt = defended_project(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (_site(wt) / "extra.pth").write_text(f"import sys; sys.path.insert(0, {str(external)!r})\n")
    (external / "local_dep.py").write_text("VALUE=1\n")
    cmd = [str(wt / ".venv/bin/python"), "-B", "-c", "import local_dep;print(local_dep.VALUE)"]
    assert subprocess.check_output(cmd, cwd=wt, text=True).strip() == "1"     # the hook really runs
    with pytest.raises(owned.ClosureRefused, match="extra.pth"):
        owned.venv_fingerprint(wt)


def test_only_a_reviewed_import_hook_is_allowed(tmp_path, monkeypatch):
    """The one executable shape allowed: ``import <module>`` of a module in the same site-packages
    (so its bytes are in the tree hash) whose reviewed content is listed, like uv's _virtualenv hook."""
    _, wt = defended_project(tmp_path)
    (_site(wt) / "hook.pth").write_text("import hook_mod\n")
    (_site(wt) / "hook_mod.py").write_text("X = 1\n")
    with pytest.raises(owned.ClosureRefused, match="not a reviewed import hook"):
        owned.venv_fingerprint(wt)
    monkeypatch.setitem(owned.REVIEWED_PTH_IMPORTS, "hook_mod", {owned._sha(b"X = 1\n"): "test hook"})
    owned.venv_fingerprint(wt)                                         # the reviewed content is allowed
    (_site(wt) / "hook_mod.py").write_text("X = 2\n")
    with pytest.raises(owned.ClosureRefused, match="not a reviewed import hook"):
        owned.venv_fingerprint(wt)                                     # changed content is not reviewed


def test_an_untracked_symlinks_target_bytes_are_in_the_manifest(tmp_path):
    """r5 #4.2: the manifest hashed a symlink's spelling only, so its target could change unseen."""
    _, wt = defended_project(tmp_path)
    external = tmp_path / "external.py"
    external.write_text("VALUE=1\n")
    (wt / "tests/helper.py").symlink_to(external)
    before = owned.manifest(wt)["untracked"]["tests/helper.py"]
    external.write_text("VALUE=2\n")
    assert owned.manifest(wt)["untracked"]["tests/helper.py"] != before


def test_a_pair_whose_symlinked_helper_changes_is_rejected(tmp_path):
    """r5 #4.2 measured: the marked run rewrote the helper's target through the link, the --runxfail run
    imported the new value, and the pair was accepted with no drift reason."""
    from tests.scripts.incident_register.test_expected_failure import GOOD, REGISTRY, setup_project
    _, wt = setup_project(tmp_path, GOOD)
    target = tmp_path / "external-helper.py"
    target.write_text("VALUE=1\n")
    (wt / "tests/helper.py").symlink_to(target)
    (wt / "tests/test_incident.py").write_text(
        GOOD + "\nfrom pathlib import Path\nfrom tests import helper\n\n\ndef test_changed_helper():\n"
        "    expected = 2 if 'W15_ORACLE_OUT' in os.environ else 1\n    assert helper.VALUE == expected\n"
        "    if expected == 1:\n        Path(helper.__file__).write_text('VALUE=2\\n')\n")
    res = acceptance.run_pair(wt, REGISTRY, ["tests/test_incident.py", "-q"], tmp_path / "out")
    assert target.read_text() == "VALUE=2\n"                            # the probe really changed it
    assert res["verdict"] == "rejected"
    assert any("after marked: untracked files changed: ['tests/helper.py']" in r for r in res["reasons"]), res["reasons"]
