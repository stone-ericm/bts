"""Expected-failure acceptance from a marked + --runxfail pair (Codex phase-1 r1 #5; r2 #3, #4; r3 #4, #8)."""
import json

import pytest

from scripts.audit.incident_register import acceptance, runner
from tests.scripts.incident_register.synth import MOD, defended_project, write

NODE = "tests/test_incident.py::test_pass_is_void"
HEADER = '''import json
import os

import pytest
from bts import mod


class IncidentExpectedFailure(Exception):
    pass


class GradedAsMiss(IncidentExpectedFailure):
    pass


def _oracle(actual, required, bad, exc, what):
    path = os.environ.get("W15_ORACLE_OUT")
    if path:
        node = os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]
        with open(path, "a") as fh:
            fh.write(json.dumps({"node": node, "what": what, "actual": actual, "required": required, "bad": bad}) + "\\n")
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
             "entry": {"path": "src/bts/mod.py", "qualname": "grade"}, "bad": "'miss'", "required": "'void'",
             "bad_json": "miss", "required_json": "void",
             "connection": {"kind": "return", "path": "src/bts/mod.py", "qualname": "grade",
                            "review": "the oracle's actual is grade()'s return on the same line"}}]
MISS = MOD.replace('def grade():\n    return "void"', 'def grade():\n    return "miss"')


def setup_project(tmp_path, test_text, extra=None, mod=MISS):
    files = {"src/bts/mod.py": mod, "tests/test_incident.py": test_text}
    files.update(extra or {})
    return defended_project(tmp_path, extra=files)


def pair(wt, tmp_path, registry=REGISTRY, between=None):
    """accept() over a marked + --runxfail pair (the closure freeze is run_pair's job, tested below)."""
    cfg = acceptance.observe_config(wt, registry)
    out = tmp_path / "out"
    marked = runner.run(wt, ["tests/test_incident.py", "-q"], out, "marked", observe=cfg)
    if between:
        between()
    oracle = out / "oracle.jsonl"
    unmarked = runner.run(wt, ["tests/test_incident.py", "-q", "--runxfail"], out, "unmarked", observe=cfg,
                          env_extra={"W15_ORACLE_OUT": str(oracle)})
    return acceptance.accept(marked, unmarked, worktree=wt, registry=registry,
                             oracle_records=acceptance.load_oracle_records(oracle))


def test_genuine_reproduction_is_accepted_as_a_reviewed_value_match(tmp_path):
    repo, wt = setup_project(tmp_path, GOOD)
    got = pair(wt, tmp_path)
    assert got["_session"] == [] and got[NODE] == []
    assert got["_connections"] == {NODE: "value_match"}


def test_a_value_match_without_a_recorded_review_is_unmatched(tmp_path):
    """Codex phase-1 r4 #3: the tool matches values, it does not see which call the oracle consumed."""
    repo, wt = setup_project(tmp_path, GOOD)
    reg = [dict(REGISTRY[0], connection={k: v for k, v in REGISTRY[0]["connection"].items() if k != "review"})]
    got = pair(wt, tmp_path, registry=reg)
    assert got["_connections"][NODE] == "unmatched"
    assert "return connection without a recorded fixture review" in got[NODE]


def test_run_pair_accepts_one_frozen_closure(tmp_path):
    repo, wt = setup_project(tmp_path, GOOD)
    res = acceptance.run_pair(wt, REGISTRY, ["tests/test_incident.py", "-q"], tmp_path / "out")
    assert res["verdict"] == "accepted", res["reasons"]
    assert res["connections"] == {NODE: "value_match"} and len(res["registry_sha256"]) == 64
    assert res["passed_nodes"] == ["tests/test_incident.py::test_control"]       # what a record may cite


def test_a_control_skipped_under_runxfail_rejects_the_pair(tmp_path):
    """A record's controls must have passed in BOTH runs. ``passed_nodes`` lists the marked run's passes of
    an accepted pair only; that is sound because the gate refuses every other --runxfail outcome for an
    unregistered node, so a control the --runxfail run skips rejects the pair."""
    text = GOOD.replace("def test_control():\n", "import os\n\n\n@pytest.mark.skipif(\"W15_ORACLE_OUT\" in os.environ, "
                        "reason=\"x\")\ndef test_control():\n")
    repo, wt = setup_project(tmp_path, text)
    res = acceptance.run_pair(wt, REGISTRY, ["tests/test_incident.py", "-q"], tmp_path / "out")
    assert res["verdict"] == "rejected"
    assert "--runxfail: tests/test_incident.py::test_control: skipped" in res["reasons"]
    assert res["passed_nodes"] == []


def test_a_literal_oracle_value_disconnected_from_production_is_rejected(tmp_path):
    """r3 #4 measured: production returned the REQUIRED value; the oracle was fed a literal bad value."""
    text = GOOD.replace('_oracle(mod.grade(), "void"', 'mod.grade()\n    _oracle("miss", "void"')
    repo, wt = setup_project(tmp_path, text, mod=MOD)          # production correct: grade() == "void"
    got = pair(wt, tmp_path)
    assert got["_connections"][NODE] == "unmatched"
    assert any("returned 'void', the oracle saw 'miss'" in r for r in got[NODE]), got[NODE]


def test_a_helper_changed_inside_the_pair_is_rejected(tmp_path):
    """r3 #4 measured: a tracked helper changed between the runs; the pair was accepted."""
    helper = "def choose(x):\n    return x\n"
    text = GOOD.replace("from bts import mod\n", "from bts import mod\nfrom tests import inputs\n").replace(
        '_oracle(mod.grade(), "void"', '_oracle(inputs.choose(mod.grade()), "void"').replace(
        "def test_control():\n", "def test_control():\n    import pathlib\n"
        "    p = pathlib.Path(inputs.__file__)\n    p.write_text(\"def choose(x):\\n    return 'miss'\\n\")\n")
    repo, wt = setup_project(tmp_path, text, extra={"tests/inputs.py": helper})
    res = acceptance.run_pair(wt, REGISTRY, ["tests/test_incident.py", "-q"], tmp_path / "out")
    assert res["verdict"] == "rejected"
    assert any("after marked: frozen files changed" in r for r in res["reasons"]), res["reasons"]
    assert res["passed_nodes"] == []                  # a rejected pair vouches for no control


def test_derived_connection_needs_a_recorded_review(tmp_path):
    repo, wt = setup_project(tmp_path, GOOD)
    reg = [dict(REGISTRY[0], connection={"kind": "derived", "review": ""})]
    got = pair(wt, tmp_path, registry=reg)
    assert got["_connections"][NODE] == "unmatched"
    reg = [dict(REGISTRY[0], connection={"kind": "derived", "review": "fixture reviewed (r3 answer 4)"})]
    assert pair(wt, tmp_path, registry=reg)["_connections"][NODE] == "exception_shape"


def test_missing_oracle_record_is_rejected(tmp_path):
    repo, wt = setup_project(tmp_path, GOOD)
    cfg = acceptance.observe_config(wt, REGISTRY)
    out = tmp_path / "out"
    marked = runner.run(wt, ["tests/test_incident.py", "-q"], out, "marked", observe=cfg)
    unmarked = runner.run(wt, ["tests/test_incident.py", "-q", "--runxfail"], out, "unmarked", observe=cfg)
    got = acceptance.accept(marked, unmarked, worktree=wt, registry=REGISTRY, oracle_records={})
    assert "--runxfail: no structured oracle record" in got[NODE]


def test_foreign_oracle_module_is_rejected(tmp_path):
    """r2 #4 measured: a `foreign_test_h.py::_oracle` satisfied a request for `test_h.py`, with no
    production call and no bad-value check."""
    foreign = "class GradedAsMiss(Exception):\n    pass\n\n\ndef _oracle():\n    raise GradedAsMiss('no production')\n"
    text = ('import pytest\nfrom tests.foreign_test_incident import GradedAsMiss, _oracle\n\n\n'
            '@pytest.mark.xfail(strict=True, raises=GradedAsMiss, reason="wrong module")\n'
            'def test_pass_is_void():\n    _oracle()\n')
    repo, wt = setup_project(tmp_path, text, extra={"tests/foreign_test_incident.py": foreign})
    reasons = " | ".join(pair(wt, tmp_path)[NODE])
    assert "raised tests.foreign_test_incident.GradedAsMiss" in reasons
    assert "raised in" in reasons and "never invoked" in reasons and "bad/required" in reasons


def test_imperative_xfail_is_rejected(tmp_path):
    text = HEADER + ('\n@pytest.mark.xfail(strict=True, raises=GradedAsMiss, reason="x")\n'
                     'def test_pass_is_void():\n    mod.grade()\n    pytest.xfail("imperative")\n')
    repo, wt = setup_project(tmp_path, text)
    assert "marked run: imperative pytest.xfail()" in pair(wt, tmp_path)[NODE]


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


def test_driver_writes_a_rejected_artifact_on_failure(tmp_path):
    from scripts.audit.incident_register import run_expected_failures as drv
    repo, wt = setup_project(tmp_path, GOOD)
    res = drv.main(str(wt), "HEAD", str(tmp_path / "drv"))          # the synthetic repo has no registry file
    saved = json.loads((tmp_path / "drv" / "expected_failures_acceptance.json").read_text())
    assert res["verdict"] == saved["verdict"] == "rejected" and saved["accepted_nodes"] == []
    assert saved.get("passed_nodes") == []
    assert any("refused" in r for r in saved["reasons"])
