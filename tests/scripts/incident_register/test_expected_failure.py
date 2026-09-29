"""Expected-failure acceptance from a marked + --runxfail pair (Codex phase-1 r2 #3, #4; r1 #5)."""
import pytest

from scripts.audit.incident_register import acceptance, owned, runner
from tests.scripts.incident_register.synth import MOD, commit, defended_project, write

NODE = "tests/test_incident.py::test_pass_is_void"
HEADER = '''import pytest
from bts import mod


class IncidentExpectedFailure(Exception):
    pass


class GradedAsMiss(IncidentExpectedFailure):
    pass


def _oracle(actual, required, bad, exc, what):
    if actual == required:
        return
    if actual == bad:
        raise exc(f"{what}: got the declared bad value {actual!r}; required {required!r}")
    raise AssertionError(f"{what}: {actual!r} is neither {required!r} nor {bad!r}")

'''
GOOD = HEADER + '''
@pytest.mark.xfail(strict=True, raises=GradedAsMiss, reason="L01 synthetic")
def test_pass_is_void():
    _oracle(mod.grade(), "void", "miss", GradedAsMiss, "pass")


def test_control():
    assert mod.deliver(False) == "done"
'''
REGISTRY = [{"node": NODE, "exception": "tests.test_incident.GradedAsMiss",
             "oracle": {"path": "tests/test_incident.py", "qualname": "_oracle"},
             "entry": {"path": "src/bts/mod.py", "qualname": "grade"}, "bad": "'miss'", "required": "'void'"}]


def setup_project(tmp_path, test_text, extra=None):
    files = {"src/bts/mod.py": MOD.replace('def grade():\n    return "void"', 'def grade():\n    return "miss"'),
             "tests/test_incident.py": test_text}
    files.update(extra or {})
    return defended_project(tmp_path, extra=files)


def pair(wt, tmp_path, registry=REGISTRY, between=None):
    cfg = acceptance.observe_config(wt, registry)
    marked = runner.run(wt, ["tests/test_incident.py", "-q"], tmp_path / "out", "marked", observe=cfg)
    if between:
        between()
    unmarked = runner.run(wt, ["tests/test_incident.py", "-q", "--runxfail"], tmp_path / "out", "unmarked", observe=cfg)
    return acceptance.accept(marked, unmarked, worktree=wt, registry=registry)


def test_genuine_reproduction_is_accepted(tmp_path):
    repo, wt = setup_project(tmp_path, GOOD)
    got = pair(wt, tmp_path)
    assert got == {"_session": [], NODE: []}


def test_foreign_oracle_module_is_rejected(tmp_path):
    """r2 #4 measured: a `foreign_test_h.py::_oracle` satisfied a request for `test_h.py`, with no
    production call and no bad-value check."""
    foreign = "class GradedAsMiss(Exception):\n    pass\n\n\ndef _oracle():\n    raise GradedAsMiss('no production')\n"
    text = ('import pytest\nfrom tests.foreign_test_incident import GradedAsMiss, _oracle\n\n\n'
            '@pytest.mark.xfail(strict=True, raises=GradedAsMiss, reason="wrong module")\n'
            'def test_pass_is_void():\n    _oracle()\n')
    repo, wt = setup_project(tmp_path, text, extra={"tests/foreign_test_incident.py": foreign})
    got = pair(wt, tmp_path)
    reasons = " | ".join(got[NODE])
    assert "raised tests.foreign_test_incident.GradedAsMiss" in reasons
    assert "raised in" in reasons and "never invoked" in reasons and "bad/required" in reasons


def test_imperative_xfail_is_rejected(tmp_path):
    text = HEADER + ('\n@pytest.mark.xfail(strict=True, raises=GradedAsMiss, reason="x")\n'
                     'def test_pass_is_void():\n    mod.grade()\n    pytest.xfail("imperative")\n')
    repo, wt = setup_project(tmp_path, text)
    got = pair(wt, tmp_path)
    assert "marked run: imperative pytest.xfail()" in got[NODE], got


def test_non_strict_marker_is_rejected(tmp_path):
    repo, wt = setup_project(tmp_path, GOOD.replace("strict=True, ", ""))
    assert "marked run: marker is not strict" in pair(wt, tmp_path)[NODE]


def test_same_named_exception_from_another_module_is_rejected(tmp_path):
    registry = [dict(REGISTRY[0], exception="tests.elsewhere.GradedAsMiss")]
    repo, wt = setup_project(tmp_path, GOOD)
    got = pair(wt, tmp_path, registry=registry)
    assert any("marker raises" in r for r in got[NODE]) and any("--runxfail: raised" in r for r in got[NODE])


def test_oracle_fed_a_literal_has_no_production_invocation(tmp_path):
    repo, wt = setup_project(tmp_path, GOOD.replace('_oracle(mod.grade(), "void"', '_oracle("miss", "void"'))
    assert any("never invoked" in r for r in pair(wt, tmp_path)[NODE])


def test_message_without_the_declared_values_is_rejected(tmp_path):
    repo, wt = setup_project(tmp_path, GOOD.replace("got the declared bad value", "got"))
    assert "--runxfail: the message lacks the declared bad/required values" in pair(wt, tmp_path)[NODE]


def test_teardown_error_rejects_the_session(tmp_path):
    conftest = ("import pytest\n\n@pytest.fixture(autouse=True)\ndef boom():\n    yield\n"
                "    raise ValueError('teardown')\n")
    repo, wt = setup_project(tmp_path, GOOD, extra={"tests/conftest.py": conftest})
    assert pair(wt, tmp_path)["_session"]


def test_changed_test_bytes_between_the_runs_are_rejected(tmp_path):
    repo, wt = setup_project(tmp_path, GOOD)
    got = pair(wt, tmp_path, between=lambda: write(wt, "tests/test_incident.py", GOOD + "\n# edited\n"))
    assert "the two runs collected different test-file bytes" in got["_session"]
