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


# --- finding 1: boundary discovery -----------------------------------------------------------------

from scripts.audit.incident_register import defence, observer  # noqa: E402
from tests.scripts.incident_register.synth import DM  # noqa: E402
from tests.scripts.incident_register.test_r4_counterexamples import one_spec  # noqa: E402

TRANSPORT = ("sent=[]\ndef send(recipient,text):\n    sent.append((recipient,text))\n"
             "def replacement(recipient,text):\n    sent.append((recipient,text))\n")
PROBE = ("from bts import mod,transport\ndef test_probe():\n    result=mod.deliver()\n"
         "    assert transport.sent==[('eric','pick: Turner')]\n    assert result=='done'  # ASSERT-RESULT\n")
NODE = "tests/test_probe.py::test_probe"
HELPER_PROD = ("from bts import transport\nfrom tests.helper import invoke\ndef deliver():\n"
               "    invoke(transport,False)  # BRANCH\n    return 'done'\n")
HELPER_MUTANT = ("invoke(transport,False)  # BRANCH\n    return 'done'", "invoke(transport,True)  # BRANCH\n    return 'defer'")


def _defend(tmp_path, files, old, new, boundaries=None):
    _, wt = defended_project(tmp_path, {"src/bts/transport.py": TRANSPORT, "tests/test_probe.py": PROBE, **files})
    spec = one_spec(old, new)
    if boundaries:
        spec["boundaries"] = boundaries
    return defence.current_defence(wt, spec, tmp_path / "out")


def _refused(res, reason):
    cert = res["certificates"][NODE]
    assert res["verdict"] == "rejected" and not cert["ok"], res
    assert any(reason in r for r in cert["reasons"]), cert["reasons"]
    return cert


def test_a_rebinding_inside_a_frozen_helper_is_seen(tmp_path):
    """r5 #1.1 measured: a test helper rebound transport.send, called it through map() and restored it.
    Discovery at production calls came too late, and the absence was accepted with zero calls."""
    helper = ("def invoke(transport,rebind):\n    saved=transport.send\n    if rebind:\n"
              "        transport.send=transport.replacement\n    list(map(transport.send,['eric'],['pick: Turner']))\n"
              "    transport.send=saved\n")
    res = _defend(tmp_path, {"src/bts/mod.py": HELPER_PROD, "tests/helper.py": helper}, *HELPER_MUTANT)
    assert _refused(res, "1 qualifying 'dm' call(s) occurred")["linked"]["calls_observed"] == 1


def test_a_callable_bound_to_two_boundaries_is_attributed_to_neither(tmp_path):
    """r5 #1.2 measured: the mutant bound send_b to send_a's function, which was already registered for
    the other boundary; the one send was recorded only under that name, and send_b's absence was
    accepted."""
    transport = ("sent=[]\ndef send_a(recipient,text):\n    sent.append((recipient,text))\n"
                 "def send_b(recipient,text):\n    sent.append((recipient,text))\n")
    prod = "from bts import transport\ndef deliver():\n    transport.send_b('eric','pick: Turner')  # BRANCH\n    return 'done'\n"
    res = _defend(tmp_path, {"src/bts/mod.py": prod, "src/bts/transport.py": transport},
                  "    transport.send_b('eric','pick: Turner')  # BRANCH\n    return 'done'",
                  "    saved=transport.send_b\n    transport.send_b=transport.send_a  # BRANCH\n"
                  "    transport.send_b('eric','pick: Turner')\n    transport.send_b=saved\n    return 'defer'",
                  boundaries=[dict(DM, name="other", binding="bts.transport:send_a"),
                              dict(DM, binding="bts.transport:send_b")])
    _refused(res, "'dm' call(s) without identity")


def test_a_callable_captured_before_a_restore_is_still_a_boundary_call(tmp_path):
    """Beyond the r5 probes: the replacement is captured, the binding restored, and only then called.
    It is no longer the current value, so the call cannot be attributed, but it is not absent."""
    helper = ("def invoke(transport,rebind):\n    if rebind:\n        saved=transport.send\n"
              "        transport.send=transport.replacement\n        f=transport.send\n        transport.send=saved\n"
              "        f('eric','pick: Turner')\n    else:\n        transport.send('eric','pick: Turner')\n")
    res = _defend(tmp_path, {"src/bts/mod.py": HELPER_PROD, "tests/helper.py": helper}, *HELPER_MUTANT)
    _refused(res, "'dm' call(s) without identity")


def test_a_code_object_swapped_in_place_is_seen(tmp_path):
    """Beyond the r5 probes: the bound function stays the same object, but its ``__code__`` is replaced
    for one call (a function watcher registers the new code)."""
    helper = ("def invoke(transport,rebind):\n    if rebind:\n        code=transport.send.__code__\n"
              "        transport.send.__code__=transport.replacement.__code__\n        transport.send('eric','pick: Turner')\n"
              "        transport.send.__code__=code\n    else:\n        transport.send('eric','pick: Turner')\n")
    res = _defend(tmp_path, {"src/bts/mod.py": HELPER_PROD, "tests/helper.py": helper}, *HELPER_MUTANT)
    _refused(res, "1 qualifying 'dm' call(s) occurred")


def test_a_binding_through_an_instance_namespace_is_a_coverage_gap(tmp_path):
    """An instance's ``__dict__`` can be replaced wholesale, which no watcher sees: such a path is
    unsupported, so the absence is unavailable instead of certified."""
    transport = TRANSPORT + "class Holder:\n    pass\nholder=Holder()\nholder.send=send\n"
    prod = ("from bts import transport\ndef deliver():\n    transport.holder.send('eric','pick: Turner')  # BRANCH\n"
            "    return 'done'\n")
    res = _defend(tmp_path, {"src/bts/mod.py": prod, "src/bts/transport.py": transport},
                  "    transport.holder.send('eric','pick: Turner')  # BRANCH\n    return 'done'",
                  "    pass  # BRANCH\n    return 'defer'",
                  boundaries=[dict(DM, binding="bts.transport:holder.send")])
    _refused(res, "coverage gap (unsupported namespace")


def test_a_replaced_mock_call_is_a_coverage_gap():
    """A mock boundary is recognised by mock_call_code; replacing its class's ``__call__`` would bypass
    that code, so the store is watched and recorded as a gap."""
    import sys
    import types
    from unittest import mock
    mod = types.ModuleType("w15_r5_mockmod")
    mod.send = mock.MagicMock()
    sys.modules["w15_r5_mockmod"] = mod
    mon = observer._Monitor("n", {"boundaries": [dict(DM, binding="w15_r5_mockmod:send")]}, "")
    try:
        mon.start()
        type(mod.send).__call__ = lambda self, *a, **k: None
    finally:
        mon.stop()
        del sys.modules["w15_r5_mockmod"]
    assert any(e["kind"] == "boundary_gap" and e["reason"] == "a mock boundary's __call__ was replaced"
               for e in mon.events), mon.events


# --- finding 3: resolution runs no application code ------------------------------------------------

def test_resolving_a_binding_runs_no_application_attribute_code():
    """r5 #3 measured: _resolve called getattr(obj, '__dict__'), which ran an overridden
    __getattribute__. Namespaces are now read by C descriptors, and only modules and classes."""
    import sys
    import types
    called = []

    class Holder:
        def __getattribute__(self, name):
            called.append(name)
            return object.__getattribute__(self, name)

    class LoudModule(types.ModuleType):
        def __getattribute__(self, name):
            called.append(name)
            return types.ModuleType.__getattribute__(self, name)

    class DictPropertyModule(types.ModuleType):
        @property
        def __dict__(self):
            called.append("__dict__ property")
            return {}

    holder = Holder()
    holder.send = len
    loud = LoudModule("w15_r5_loud")
    loud.send, loud.holder = len, holder
    odd = DictPropertyModule("w15_r5_odd")
    sys.modules.update(w15_r5_loud=loud, w15_r5_odd=odd)
    try:
        assert observer._resolve("w15_r5_loud:send") is len                  # raw module namespace
        assert observer._resolve("w15_r5_loud:holder.send") is None          # an instance: unsupported
        assert observer._resolve("w15_r5_odd:anything") is None              # __dict__ redefined: refused
        assert called == []
    finally:
        del sys.modules["w15_r5_loud"], sys.modules["w15_r5_odd"]


# --- finding 2: nested completeness ----------------------------------------------------------------

NESTED = [{"tag": "pick", "payload": [0] * 51},                     # a truncated nested sequence
          {"tag": "pick", "payload": [__import__("threading").Lock()]},   # a nested unsupported object
          {"tag": "pick", "payload": [[[0]]]}]                      # past the depth limit


@pytest.mark.parametrize("value", NESTED, ids=["sequence", "unsupported", "depth"])
def test_a_nested_incomplete_identity_is_unavailable(value):
    """r5 #2 measured: only the top level's flag was checked, so the value classified as 'pick'."""
    ident = observer._Monitor("n", {}, "")._identity({"value": ["args[0]"], "classify": [["pick", "pick"]]}, [value], {})
    assert acceptance.unwrap(ident["value"]) is acceptance.UNAVAILABLE
    assert ident["category"] == "unavailable"


@pytest.mark.parametrize("payload", ["[0]*51", "[__import__('threading').Lock()]", "[[[0]]]"],
                         ids=["sequence", "unsupported", "depth"])
def test_a_nested_incomplete_return_is_not_certified(tmp_path, payload):
    """r5 #2 measured through the real defence runner: the mutant's incompletely serialized return was
    classified 'bad' and certified."""
    prod = "def deliver():\n    return {'tag':'bad','payload':1}  # BRANCH\n"
    test = "from bts import mod\ndef test_probe():\n    result=mod.deliver()\n    assert result['payload']==1  # ASSERT-RESULT\n"
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "tests/test_probe.py": test})
    spec = one_spec("'payload':1}", f"'payload':{payload}}}", kind="return", category="bad")
    spec["boundaries"] = []
    spec["returns"] = [{"path": "src/bts/mod.py", "qualname": "deliver", "classify": [["bad", "bad"]]}]
    spec["symptom"] = {"kind": "return", "path": "src/bts/mod.py", "qualname": "deliver", "category": "bad",
                       "classify": [["bad", "bad"]]}
    res = defence.current_defence(wt, spec, tmp_path / "out")
    assert res["verdict"] == "rejected", res
    assert any("no 'bad' return after the branch" in r for r in res["certificates"][NODE]["reasons"])


# --- finding 6: chronology coherence ---------------------------------------------------------------

def _chronology(tmp_path, edit):
    from scripts.audit.incident_register.records import validate
    from tests.scripts.incident_register.test_records import base, ef_root
    r = base()
    edit(r["occurrences"][0])
    return validate([r], evidence_root=ef_root(tmp_path / "evidence"))


def test_a_condition_observed_after_its_restoration_is_refused(tmp_path):
    """r5 #6 measured: restored at 7/16 12:00, then 'observed still deviating' at 7/17 09:00, with zero
    errors. A recurrence is its own occurrence; it is not witnessed after this one ended."""
    def edit(o):
        o["restored_verification"] = {"at": "2026-07-16T12:00:00-04:00", "evidence": ["ev1"]}
        o["observed"] = [{"at": "2026-07-17T09:00:00-04:00", "evidence": ["ev1"]}]
    assert any("observed after its restoration" in e for e in _chronology(tmp_path, edit))


def test_an_alert_confirmed_before_its_attempt_is_refused(tmp_path):
    """r5 #6 measured: an alert confirmed at 18:00 but first attempted at 19:00, with zero errors."""
    def edit(o):
        o["alert"]["attempted"] = {"at": "2026-07-16T19:00:00-04:00", "evidence": ["ev1"]}
        o["alert"]["confirmed"] = {"at": "2026-07-16T18:00:00-04:00", "evidence": ["ev1"]}
    assert any("the alert confirmation precedes the alert attempt" in e for e in _chronology(tmp_path, edit))


@pytest.mark.parametrize("field, value, message", [
    ("first_detectable", {"at": "2026-07-15T12:00:00-04:00", "evidence": ["ev1"]}, "the first detectable time precedes the onset"),
    ("observed", [{"at": "2026-07-15T12:00:00-04:00", "evidence": ["ev1"]}], "the observed time precedes the onset"),
])
def test_other_certainly_reversed_endpoints_are_refused(tmp_path, field, value, message):
    """Nothing about a condition can be detectable or observed before its onset (onset: 7/16)."""
    def edit(o):
        o[field] = value
    assert any(message in e for e in _chronology(tmp_path, edit))


def test_overlapping_or_unknown_endpoints_are_left_as_they_are(tmp_path):
    """Only a CERTAIN reversal is refused: a same-day observation and a restoration later that day pass."""
    def edit(o):
        o["restored_verification"] = {"at": "2026-07-16T12:00:00-04:00", "evidence": ["ev1"]}
        o["observed"] = [{"at": "2026-07-16", "evidence": ["ev1"]}]
    assert not [e for e in _chronology(tmp_path, edit) if "chronology" in e]


# --- finding 7: deploy observation precision -------------------------------------------------------

def _run(run_id, text, created="2026-07-01T00:00:00Z"):
    from scripts.audit.incident_register import deploy_runs
    return {"run_id": run_id, "created_at": created, "log": "retained", **deploy_runs.extract(text)}


def test_same_run_fractional_times_bound_a_non_empty_install_interval(monkeypatch):
    """r5 #7 measured: extraction dropped the fractions, and first_live returned the empty (t, t]."""
    from scripts.audit.incident_register import deploy_runs
    text = ("job\tstep\t2026-07-01T00:00:00.100Z Pre-deploy SHA: aaaaaaa\n"
            "job\tstep\t2026-07-01T00:00:00.900Z Deployed bbbbbbb\n")
    monkeypatch.setattr(deploy_runs, "_is_ancestor", lambda fix, sha, repo: sha == "bbbbbbb")
    got = deploy_runs.first_live([_run(1, text)], "bbbbbbb", repo=".")
    # the deployed line proves the install by the END of its millisecond, never earlier
    assert (got["not_live_before"], got["live_by"]) == ("2026-07-01T00:00:00.100000Z", "2026-07-01T00:00:00.901000Z")


def test_a_second_precision_deploy_line_bounds_live_by_at_the_end_of_its_second(monkeypatch):
    """A whole-second stamp truncates: the install happened by the end of that second, so the same
    second for both lines still gives a non-empty interval."""
    from scripts.audit.incident_register import deploy_runs
    text = "job\tstep\t2026-07-01T00:00:00Z Pre-deploy SHA: aaaaaaa\njob\tstep\t2026-07-01T00:00:00Z Deployed bbbbbbb\n"
    monkeypatch.setattr(deploy_runs, "_is_ancestor", lambda fix, sha, repo: sha == "bbbbbbb")
    got = deploy_runs.first_live([_run(1, text)], "bbbbbbb", repo=".")
    assert (got["not_live_before"], got["live_by"]) == ("2026-07-01T00:00:00Z", "2026-07-01T00:00:01Z")


def test_disagreeing_observations_that_precision_cannot_order_are_refused():
    """Generalizes r4 #9's equal-time refusal: '00:00:00Z' may be any instant in that second, so it
    cannot be ordered against '00:00:00.500Z' from another run when the two disagree."""
    from scripts.audit.incident_register import deploy_runs
    a = _run(1, "job\tstep\t2026-07-01T00:00:00Z Deployed aaaaaaa\n")
    b = _run(2, "job\tstep\t2026-07-01T00:00:00.500Z Deployed bbbbbbb\n")
    with pytest.raises(ValueError, match="cannot be ordered"):
        deploy_runs.observations([a, b])
