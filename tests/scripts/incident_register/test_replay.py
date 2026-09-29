"""historical_replay with an acceptance object (Codex phase-1 r2 #8; r1 #6)."""
import json

import pytest

from scripts.audit.incident_register.replay import audit_problems, harness_changes, historical_replay
from tests.scripts.incident_register.synth import commit, defended_project, git

FIXED_MOD_OLD = 'def grade():\n    return "void"'
TEST_GRADE = "tests/test_grade.py::test_pass_is_void"
GRADE_TEST = 'from bts import mod\n\n\ndef test_pass_is_void():\n    assert mod.grade() == "void"  # ASSERT-VOID\n'


@pytest.fixture
def history(tmp_path):
    """repo: base (grade → 'miss', the defect) → fix commit (grade → 'void' + the test)."""
    repo, wt = defended_project(tmp_path, extra={"src/bts/mod.py": _mod('"miss"')})
    fix = commit(repo, {"src/bts/mod.py": _mod('"void"'), "tests/test_grade.py": GRADE_TEST}, "fix: grade Pass as void")
    return repo, wt, fix


def _mod(value):
    from tests.scripts.incident_register.synth import MOD
    return MOD.replace(FIXED_MOD_OLD, f"def grade():\n    return {value}")


def rspec(fix, **over):
    base = {"label": "synthetic-replay", "fix_set": [fix], "tests": ["tests/test_grade.py", "-q"],
            "symptom_nodes": [{"node": TEST_GRADE, "assertion": {"path": "tests/test_grade.py", "text": "# ASSERT-VOID"}}],
            "audit": {"tests/test_grade.py": {"decision": "neutral", "reason": "new test file carrying the contract"}},
            "deployed_ref": {"sha": "unknown", "basis": "unknown"}}
    base.update(over)
    return base


def test_replay_accepted_with_audit_retained(history, tmp_path):
    repo, wt, fix = history
    res = historical_replay(repo, wt, rspec(fix), tmp_path / "out")
    assert res["verdict"] == "accepted", res["reasons"]
    assert res["label_kind"] == "semantic_regression_replay" and res["closure"] == "src_swap"
    assert res["classes"][TEST_GRADE] == "symptom_candidate"
    saved = json.loads((tmp_path / "out" / "acceptance.json").read_text())
    assert saved["spec"]["audit"]["tests/test_grade.py"]["reason"] == "new test file carrying the contract"
    assert git(wt, "status", "--porcelain") == ""


def test_decision_without_reason_is_unaudited(history, tmp_path):
    repo, wt, fix = history
    res = historical_replay(repo, wt, rspec(fix, audit={"tests/test_grade.py": {"decision": "neutral"}}),
                            tmp_path / "out")
    assert res["verdict"] == "rejected" and res["label_kind"] == "semantic_regression_replay_unaudited"


def test_unaudited_change_is_rejected(history, tmp_path):
    repo, wt, fix = history
    res = historical_replay(repo, wt, rspec(fix, audit={}), tmp_path / "out")
    assert res["verdict"] == "rejected" and "unaudited harness change: tests/test_grade.py" in res["reasons"]


def test_new_api_failure_is_not_a_symptom(tmp_path):
    repo, wt = defended_project(tmp_path)
    fix = commit(repo, {"src/bts/mod.py": _mod('"void"') + "\n\ndef grade_v2():\n    return 'void'\n",
                        "tests/test_grade.py": 'from bts import mod\n\n\ndef test_pass_is_void():\n'
                                               '    assert mod.grade_v2() == "void"  # ASSERT-VOID\n'}, "fix")
    res = historical_replay(repo, wt, rspec(fix), tmp_path / "out")
    assert res["verdict"] == "rejected"
    assert res["classes"][TEST_GRADE] == "new_api"
    assert any("AttributeError" in r for r in res["reasons"]), res["reasons"]


def test_collection_error_at_the_parent_is_rejected(tmp_path):
    repo, wt = defended_project(tmp_path)
    fix = commit(repo, {"src/bts/newmod.py": "def f():\n    return 1\n",
                        "tests/test_grade.py": "from bts.newmod import f\n\n\ndef test_pass_is_void():\n"
                                               "    assert f() == 1  # ASSERT-VOID\n"}, "fix")
    res = historical_replay(repo, wt, rspec(fix), tmp_path / "out")
    assert res["verdict"] == "rejected" and any("collection errors" in r for r in res["reasons"])


def test_harness_changes_keep_both_rename_paths(tmp_path):
    repo, wt = defended_project(tmp_path)
    base = git(repo, "rev-parse", "HEAD")
    git(repo, "mv", "tests/test_mod.py", "tests/test_renamed.py")
    git(repo, "commit", "-qm", "rename")
    changes = harness_changes(repo, base, "HEAD")
    assert changes[0]["from"] == "tests/test_mod.py" and changes[0]["path"] == "tests/test_renamed.py"
    assert audit_problems(changes, {"tests/test_renamed.py": {"decision": "neutral", "reason": "rename"}}) == [
        "unaudited harness change: tests/test_mod.py"]


def test_adapter_needs_evidence():
    changes = [{"status": "M", "path": "tests/conftest.py"}]
    assert audit_problems(changes, {"tests/conftest.py": {"decision": "adapter", "reason": "clock"}}) == [
        "tests/conftest.py: adapter decision without adapter evidence"]


def test_symptom_at_an_undeclared_assertion_is_rejected(tmp_path):
    """The parent fails, but at the FIRST assertion, not the declared observable-contract one."""
    repo, wt = defended_project(tmp_path, extra={"src/bts/mod.py": _mod('"miss"')})
    two = ('from bts import mod\n\n\ndef test_pass_is_void():\n    assert mod.grade() != "miss"  # ASSERT-DIAG\n'
           '    assert mod.grade() == "void"  # ASSERT-VOID\n')
    fix = commit(repo, {"src/bts/mod.py": _mod('"void"'), "tests/test_grade.py": two}, "fix")
    res = historical_replay(repo, wt, rspec(fix), tmp_path / "out")
    assert res["verdict"] == "rejected"
    assert any("not " in r and "ASSERT" not in r and "raised at" in r for r in res["reasons"]), res["reasons"]
