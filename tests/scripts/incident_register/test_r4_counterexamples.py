"""Codex phase-1 r4 counterexamples, each pinned in the direction the design requires.

Adapted from the reviewer's retained probes. Each of them measured a FALSE acceptance at 74b04e8. A
certificate that cannot establish its coverage must be refused or unavailable (design §9.3 as amended),
never turned into an absence or connection claim.
"""
import copy

import pytest

from scripts.audit.incident_register import acceptance, defence, observer
from tests.scripts.incident_register.synth import DM, defended_project
from tests.scripts.incident_register.test_expected_failure import HEADER, NODE, REGISTRY, setup_project


def one_spec(old, new, *, assertion="# ASSERT-RESULT", kind="absence", category="pick", boundary=DM):
    return dict(label="r4", baseline="HEAD", tests=["tests/test_probe.py", "-q"],
                allowed_paths=["src/bts/mod.py"], mutation_edits=[["src/bts/mod.py", old, new]],
                branch=dict(path="src/bts/mod.py", text="# BRANCH"),
                entry=dict(path="src/bts/mod.py", qualname="deliver"), boundaries=[boundary],
                symptom=dict(kind=kind, boundary="dm", category=category),
                killing=[dict(node="tests/test_probe.py::test_probe",
                              assertion=dict(path="tests/test_probe.py", text=assertion))])


# --- finding 1: the certificate's event set --------------------------------------------------------

def test_an_initially_unsupported_c_boundary_makes_absence_unavailable(tmp_path):
    """r4 #1.1: the gap was recorded before obs_start and so fell outside the certified interval."""
    prod = "from bts import transport\ndef deliver():\n    transport.send('pick: Turner')  # BRANCH\n    return 'done'\n"
    transport = "sent=[]\ndef send(text):\n    sent.append(text)\n"
    test = ("from bts import mod,transport\ndef test_probe():\n    result=mod.deliver()\n"
            "    assert transport.sent==['pick: Turner']\n    assert result=='done'  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "src/bts/transport.py": transport,
                                        "tests/test_probe.py": test})
    spec = one_spec("return 'done'", "return 'defer'", boundary=dict(DM, value=["args[0]"]))
    spec["mutation_edits"].append(["src/bts/transport.py", "def send(text):\n    sent.append(text)", "send=sent.append"])
    spec["allowed_paths"].append("src/bts/transport.py")
    res = defence.current_defence(wt, spec, tmp_path / "out")
    assert res["verdict"] == "rejected"
    assert any("coverage gap" in r for r in res["reasons"]), res["reasons"]


def test_a_temporary_rebinding_called_from_c_is_seen(tmp_path):
    """r4 #1.2: production swapped the binding, sent via map(), and restored it."""
    prod = "from bts import transport\ndef deliver():\n    transport.send('eric','pick: Turner')  # BRANCH\n    return 'done'\n"
    transport = ("sent=[]\ndef send(recipient,text):\n    sent.append((recipient,text))\n"
                 "def replacement(recipient,text):\n    sent.append((recipient,text))\n")
    test = ("from bts import mod,transport\ndef test_probe():\n    result=mod.deliver()\n"
            "    assert transport.sent==[('eric','pick: Turner')]\n    assert result=='done'  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "src/bts/transport.py": transport,
                                        "tests/test_probe.py": test})
    spec = one_spec("    transport.send('eric','pick: Turner')  # BRANCH\n    return 'done'",
                    "    saved=transport.send\n    transport.send=transport.replacement  # BRANCH\n"
                    "    list(map(transport.send,['eric'],['pick: Turner']))\n    transport.send=saved\n    return 'defer'")
    res = defence.current_defence(wt, spec, tmp_path / "out")
    assert res["verdict"] == "rejected"
    cert = res["certificates"]["tests/test_probe.py::test_probe"]
    assert any("qualifying 'dm' call(s) occurred" in r for r in cert["reasons"]), cert["reasons"]


def test_a_worker_alive_before_the_call_phase_makes_absence_unavailable(tmp_path):
    """r4 #1.3: a fixture-started worker did the declared work after the observed invocation returned."""
    prod = ("from bts import transport\nimport threading\nrelease=threading.Event()\nqueued=False\n"
            "def later():\n    release.wait()\n    if queued:\n        transport.send('eric','pick: Turner')\n"
            "def deliver():\n    transport.send('eric','pick: Turner')  # BRANCH\n    return 'done'\n")
    test = ("import pytest,threading\nfrom unittest.mock import patch\nfrom bts import mod\n"
            "@pytest.fixture(autouse=True)\ndef worker():\n    with patch('bts.transport.send') as send:\n"
            "        t=threading.Thread(target=mod.later)\n        t.start()\n        yield\n"
            "        mod.release.set()\n        t.join()\n        assert send.call_count==1\n"
            "def test_probe():\n    result=mod.deliver()\n    assert result=='done'  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "tests/test_probe.py": test})
    spec = one_spec("    transport.send('eric','pick: Turner')  # BRANCH\n    return 'done'",
                    "    global queued\n    queued=True  # BRANCH\n    return 'defer'")
    res = defence.current_defence(wt, spec, tmp_path / "out")
    assert res["verdict"] == "rejected"
    assert any("already running when the observed interval began" in r for r in res["reasons"]), res["reasons"]


def test_a_call_on_another_receiver_is_not_the_declared_boundary(tmp_path):
    """r4 #1.4: two instances share the method's code; the event belonged to the wrong receiver."""
    transport = ("class Transport:\n    def __init__(self):\n        self.sent=[]\n    def send(self, text):\n"
                 "        self.sent.append(text)\nreal=Transport()\nother=Transport()\nsend=real.send\n")
    prod = "from bts import transport\ndef deliver():\n    transport.send('pick: Turner')  # BRANCH\n    return 'done'\n"
    test = ("from bts import mod, transport\ndef test_probe():\n    mod.deliver()\n"
            "    assert transport.real.sent == ['pick: Turner']  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/transport.py": transport, "src/bts/mod.py": prod,
                                        "tests/test_probe.py": test})
    spec = one_spec("transport.send('pick: Turner')", "transport.other.send('pick: Turner')", kind="event",
                    boundary=dict(DM, value=["args[1]"]))
    res = defence.current_defence(wt, spec, tmp_path / "out")
    assert res["verdict"] == "rejected"
    cert = res["certificates"]["tests/test_probe.py::test_probe"]
    assert any("no 'pick' event" in r for r in cert["reasons"]), cert["reasons"]


# --- finding 2: observation runs no application code and never passes a partial value as whole ------

def test_serializing_a_dict_runs_no_key_equality():
    calls = []

    class Key:
        def __hash__(self):
            return hash("x")

        def __eq__(self, other):
            calls.append(other)
            return False

    value = {Key(): 1, "x": 2}
    calls.clear()
    got = observer._safe(value)
    assert calls == []
    assert got["map"] == {"x": 2} and got.get("incomplete") is True        # the non-string key is omitted


@pytest.mark.parametrize("value", [[0] * 51, "y" * 501, {str(i): i for i in range(51)}])
def test_a_truncated_value_never_compares_equal(value):
    safe = observer._safe(value)
    assert safe.get("incomplete") is True
    assert acceptance.unwrap(safe) is acceptance.UNAVAILABLE


def test_a_truncated_return_is_not_a_match(tmp_path):
    """r4 #2: production returned the correct 51 elements; the observer kept 50 and matched a wrong oracle."""
    required, bad = [0] * 51, [0] * 50
    test = HEADER + ("\n@pytest.mark.xfail(strict=True,raises=GradedAsMiss,reason='probe')\ndef test_pass_is_void():\n"
                     "    mod.grade()\n    _oracle([0]*50,[0]*51,[0]*50,GradedAsMiss,'pass')\n")
    _, wt = setup_project(tmp_path, test, mod="def grade():\n    return [0]*51\n")
    reg = copy.deepcopy(REGISTRY)
    reg[0].update(bad=repr(bad), required=repr(required), bad_json=bad, required_json=required)
    res = acceptance.run_pair(wt, reg, ["tests/test_incident.py", "-q"], tmp_path / "out")
    assert res["verdict"] == "rejected" and res["connections"][NODE] == "unmatched", res


# --- finding 3: a value match is only a value match -------------------------------------------------

def test_a_readback_value_match_needs_a_recorded_fixture_review(tmp_path):
    """r4 #3: a loader read of an UNRELATED selection matched the oracle's value. The tool cannot see
    which selection the oracle consumed, so it reports value_match only with a recorded review."""
    prod = "def grade():\n    return 'void'\ndef read(which):\n    return {'correct':'void','unrelated':'miss'}[which]\n"
    test = HEADER + ("\n@pytest.mark.xfail(strict=True,raises=GradedAsMiss,reason='probe')\ndef test_pass_is_void():\n"
                     "    mod.grade()\n    _oracle([mod.read('unrelated')],['void'],['miss'],GradedAsMiss,'pass')\n")
    _, wt = setup_project(tmp_path, test, mod=prod)
    reg = copy.deepcopy(REGISTRY)
    reg[0].update(bad="['miss']", required="['void']", bad_json=["miss"], required_json=["void"],
                  connection={"kind": "reads", "components": [{"path": "src/bts/mod.py", "qualname": "read"}]})
    res = acceptance.run_pair(wt, reg, ["tests/test_incident.py", "-q"], tmp_path / "out")
    assert res["verdict"] == "rejected" and res["connections"][NODE] == "unmatched"
    assert any("recorded fixture review" in r for r in res["reasons"] + res["per_node"][NODE]), res


def test_a_node_that_fails_differently_when_observed_is_rejected(tmp_path):
    """r4 #2 (conformance): pass/fail agreement is not enough; under the mutant the killing node fails at
    the declared assertion only while observed and with another exception while unobserved."""
    from tests.scripts.incident_register.test_defence import run as defence_run
    extra = {"tests/test_diff.py": (
        "import sys\nfrom unittest.mock import patch\nfrom bts import mod\n\n\ndef test_diff():\n"
        "    with patch('bts.transport.send') as send:\n        mod.run(True)\n"
        "    if not send.call_args_list:\n        if sys.monitoring.get_tool(4) is not None:\n"
        "            assert send.call_args_list  # ASSERT-DIFF\n        raise RuntimeError('unobserved')\n")}
    project = defended_project(tmp_path, extra=extra)
    res = defence_run(project, tmp_path, tests=["tests/test_mod.py", "tests/test_diff.py", "-q"],
                      killing=[{"node": "tests/test_diff.py::test_diff",
                                "assertion": {"path": "tests/test_diff.py", "text": "# ASSERT-DIFF"}}])
    assert res["verdict"] == "rejected"
    assert any("tests/test_diff.py::test_diff failed differently observed" in r for r in res["reasons"]), res["reasons"]


# --- finding 4: the environment closure covers what Python actually executes -------------------------

def _dep_value(wt):
    import subprocess
    return subprocess.check_output([str(wt / ".venv/bin/python"), "-B", "-c", "import local_dep;print(local_dep.VALUE)"],
                                   cwd=wt, text=True).strip()


@pytest.mark.parametrize("mode", ["symlink", "relative_pth"])
def test_external_code_the_environment_imports_is_fingerprinted(tmp_path, mode):
    """r4 #4: a site-packages symlink to an external file, and a RELATIVE .pth line (Python resolves it
    against the .pth file's directory), both ran changed code under an unchanged fingerprint."""
    import os
    from scripts.audit.incident_register import owned
    _, wt = defended_project(tmp_path)
    site = next((wt / ".venv/lib").glob("python*/site-packages"))
    external = tmp_path / "external"
    external.mkdir()
    mod = external / "local_dep.py"
    mod.write_text("VALUE=1\n")
    if mode == "symlink":
        (site / "local_dep.py").symlink_to(mod)
    else:
        (site / "extra.pth").write_text(os.path.relpath(external, site) + "\n")
    before = owned.venv_fingerprint(wt)
    assert _dep_value(wt) == "1"
    mod.write_text("VALUE=2\n")
    assert _dep_value(wt) == "2"
    assert owned.venv_fingerprint(wt) != before


def test_dependency_bytecode_is_fingerprinted(tmp_path):
    """r4 #4: -B stops bytecode WRITES, not reads; a replaced cached .pyc with the same source ran new code."""
    import os
    import py_compile
    from scripts.audit.incident_register import owned
    _, wt = defended_project(tmp_path)
    site = next((wt / ".venv/lib").glob("python*/site-packages"))
    p = site / "local_dep.py"
    p.write_text("VALUE=1\n")
    os.utime(p, (1700000000, 1700000000))
    py_compile.compile(str(p), doraise=True)
    before = owned.venv_fingerprint(wt)
    assert _dep_value(wt) == "1"
    p.write_text("VALUE=2\n")
    os.utime(p, (1700000000, 1700000000))
    py_compile.compile(str(p), doraise=True)
    p.write_text("VALUE=1\n")
    os.utime(p, (1700000000, 1700000000))
    assert _dep_value(wt) == "2"                          # the stale bytecode is what runs
    assert owned.venv_fingerprint(wt) != before


def test_the_manifest_hashes_untracked_file_contents(tmp_path):
    """r4 #4: the manifest stored untracked NAMES only, so an untracked input could change unseen."""
    from scripts.audit.incident_register import owned
    from tests.scripts.incident_register.synth import write
    _, wt = defended_project(tmp_path)
    write(wt, "tests/helper_data.txt", "one\n")
    m0, v0 = owned.manifest(wt), owned.venv_fingerprint(wt)
    write(wt, "tests/helper_data.txt", "two\n")
    m1 = owned.manifest(wt)
    assert set(m0["untracked"]) == set(m1["untracked"]) and m0 != m1
    # the real venv fingerprint, so the only reason left is the content change (sweep D13: a dummy
    # fingerprint made _drift return "the venv changed" whatever the untracked comparison did)
    assert defence._drift(wt, m0["files"], m0["untracked"], v0, (), "x") == [
        "x: untracked files changed: ['tests/helper_data.txt']"]


def test_zipped_read_backs_of_different_cardinalities_do_not_match():
    """zip() would silently drop the unmatched read and compare a prefix."""
    from types import SimpleNamespace
    wt, node = "/wt", "tests/test_incident.py::test_pass_is_void"
    f = "/wt/src/bts/mod.py"
    stack = lambda q, fr: [[q, f, 1, fr]]                                           # noqa: E731
    ev = [{"kind": "obs_start", "seq": 1, "node": node, "thread": 1},
          {"kind": "entry", "seq": 2, "node": node, "file": f, "qualname": "grade", "frame": 100, "thread": 1,
           "stack": stack("grade", 100)},
          {"kind": "entry_exit", "seq": 3, "node": node, "file": f, "qualname": "grade", "frame": 100, "thread": 1,
           "how": "return"},
          {"kind": "return", "seq": 4, "node": node, "file": f, "qualname": "a", "value": 1, "frame": 200, "thread": 1,
           "stack": stack("a", 200)},
          {"kind": "return", "seq": 5, "node": node, "file": f, "qualname": "a", "value": 2, "frame": 201, "thread": 1,
           "stack": stack("a", 201)},
          {"kind": "return", "seq": 6, "node": node, "file": f, "qualname": "b", "value": True, "frame": 202, "thread": 1,
           "stack": stack("b", 202)},
          {"kind": "obs_end", "seq": 7, "node": node, "thread": 1, "outstanding_threads": 0, "preexisting_threads_alive": 0}]
    reg = {"entry": {"path": "src/bts/mod.py", "qualname": "grade"},
           "connection": {"kind": "reads", "shape": "zip", "review": "reviewed",
                          "components": [{"path": "src/bts/mod.py", "qualname": "a", "all": True},
                                         {"path": "src/bts/mod.py", "qualname": "b", "all": True}]}}
    status, why = acceptance.connection(SimpleNamespace(events=ev), node, reg, [[1, True]], wt)
    assert status == "unmatched" and "different cardinalities" in why[0], (status, why)


def test_replay_sees_an_untracked_file_change_in_content():
    """r4 #4: replay compared untracked NAMES only."""
    import tempfile
    from pathlib import Path
    from scripts.audit.incident_register import owned, replay
    from tests.scripts.incident_register.synth import write
    _, wt = defended_project(Path(tempfile.mkdtemp()))
    write(wt, "tests/helper_data.txt", "one\n")
    m0, v0 = owned.manifest(wt), owned.venv_fingerprint(wt)
    write(wt, "tests/helper_data.txt", "two\n")
    assert any("untracked files changed" in r for r in replay._drift(wt, m0, v0, "after green", src_too=True))
