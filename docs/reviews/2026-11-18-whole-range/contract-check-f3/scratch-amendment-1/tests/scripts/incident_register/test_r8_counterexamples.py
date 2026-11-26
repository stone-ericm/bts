"""Codex phase-1 r8 counterexamples, each pinned in the direction the narrowed model requires.

Adapted from the reviewer's retained probes (``r8-probes/test_r8_review.py``). Each measured an ACCEPTED
positive certificate at 31ef140 that the model forbids: a former callable left current after a namespace
clear (#1), the reserved ``unavailable`` category and an incomplete return value accepted as witnesses (#2),
and application ``__format__`` run by the serializer inside a pure-labelled observation (#3). The cloned
return (#4) stays accepted: return identity is source-file and qualname identity (``certify.COVERAGE``).
"""
import pytest

from scripts.audit.incident_register import certify as certify_mod
from scripts.audit.incident_register import defence, observer, runner
from scripts.audit.incident_register.certify import certify
from tests.scripts.incident_register.synth import DM, defended_project
from tests.scripts.incident_register.test_certify import ENTRY, F, IN, N, branch, call, entry, ev, stack, window
from tests.scripts.incident_register.test_r4_counterexamples import ALERT, one_spec
from tests.scripts.incident_register.test_r6_counterexamples import NODE, TRANSPORT, _defend

RESULT_TEST = "from bts import mod\ndef test_probe():\n    result=mod.deliver()\n    assert result=='done'  # ASSERT-RESULT\n"


def _boundaries(tmp_path, out="out"):
    return [e for e in runner.load(tmp_path / out / "mutant.events.jsonl") if e["kind"] == "boundary"]


# --- finding 1: a namespace clear invalidates the current binding --------------------------------

# the sender keeps its list in a default argument, so the send really happens while the binding is absent
CLEARED_TRANSPORT = "sent=[]\ndef send(recipient,text,_sent=sent):\n    _sent.append((recipient,text))\n"


def test_a_namespace_clear_leaves_no_former_callable_current(tmp_path):
    """r8 #1: dict.clear() on the watched module recorded a gap but left the captured former function
    'current', so its call was attributed and certified."""
    helper = ("def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n        return\n"
              "    namespace=transport.__dict__\n    saved=dict(namespace)\n    callback=transport.send\n"
              "    namespace.clear()\n    try:\n        assert 'send' not in namespace\n"
              "        list(map(callback,['eric'],[ALERT]))\n    finally:\n        namespace.update(saved)\n")
    res = _defend(tmp_path, helper, transport=CLEARED_TRANSPORT)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    alerts = [e for e in _boundaries(tmp_path) if e["identity"].get("value") == ALERT or "held earlier" in
              str(e["identity"].get("reason"))]
    assert alerts and all(e["identity"]["category"] == "unavailable" for e in alerts), alerts
    assert any("held earlier" in e["identity"]["reason"] for e in alerts), alerts


def test_a_per_key_store_after_a_clear_makes_the_binding_current_again(tmp_path):
    """Control: once the cleared binding is stored again key by key, the value it then holds is current,
    and its call witnesses the event."""
    helper = ("def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n        return\n"
              "    namespace=transport.__dict__\n    saved=dict(namespace)\n    namespace.clear()\n"
              "    for key,value in saved.items():\n        namespace[key]=value\n"
              "    transport.send('eric',ALERT)\n")
    res = _defend(tmp_path, helper, transport=CLEARED_TRANSPORT)
    assert res["verdict"] == "accepted" and res["certificates"][NODE]["ok"], res


# --- finding 2: no witness is unattributed, incomplete or of the reserved category -----------------

def test_the_reserved_category_is_never_a_witness():
    unattributed = ev("boundary", 3, name="dm", caller=["deliver", F, 7], stack=stack(IN),
                      identity={"value": None, "category": "unavailable", "reason": "the callable is bound to several boundaries"})
    got = certify(window(entry(1, 10), branch(2, IN), unattributed), node=N, kind="event", entry=ENTRY,
                  bad={"boundary": "dm", "category": "unavailable"})
    assert not got["ok"] and any("reserved" in r for r in got["reasons"]), got
    ret = ev("return", 3, file=F, qualname="deliver", frame=10, value={"str": "x" * 500, "length": 501, "incomplete": True},
             category="unavailable", stack=stack(IN))
    got = certify(window(entry(1, 10), branch(2, IN), ret), node=N, kind="return", entry=ENTRY,
                  bad={"file": F, "qualname": "deliver", "category": "unavailable"})
    assert not got["ok"] and any("reserved" in r for r in got["reasons"]), got


def test_an_attribution_failure_never_witnesses_whatever_its_category():
    forged = call(3, "alert", IN)
    forged["identity"]["reason"] = "a callable the binding held earlier in the interval, not its current value"
    got = certify(window(entry(1, 10), branch(2, IN), forged), node=N, kind="event", entry=ENTRY,
                  bad={"boundary": "dm", "category": "alert"})
    assert not got["ok"], got


def test_an_incomplete_event_identity_never_witnesses():
    partial = call(3, "alert", IN)
    partial["identity"]["value"] = {"seq": [{"str": "x" * 500, "length": 501, "incomplete": True}]}
    got = certify(window(entry(1, 10), branch(2, IN), partial), node=N, kind="event", entry=ENTRY,
                  bad={"boundary": "dm", "category": "alert"})
    assert not got["ok"], got


@pytest.mark.parametrize("value", [
    {"str": "x" * 500, "length": 501, "incomplete": True},
    {"seq": ["a", {"str": "x" * 500, "length": 501, "incomplete": True}]},
    {"map": {"k": {"unavailable": "depth", "incomplete": True}}},
    {"type": "m.R", "unavailable": "no plain __dict__", "incomplete": True},
], ids=["truncated", "nested-seq", "nested-map", "no-dict"])
def test_an_incomplete_return_value_never_witnesses(value):
    """r8 #2: the value matched by direct equality, completeness unchecked."""
    ret = ev("return", 3, file=F, qualname="deliver", frame=10, value=value, category="bad", stack=stack(IN))
    for bad in ({"file": F, "qualname": "deliver", "value": value}, {"file": F, "qualname": "deliver", "category": "bad"}):
        got = certify(window(entry(1, 10), branch(2, IN), ret), node=N, kind="return", entry=ENTRY, bad=bad)
        assert not got["ok"], (bad, got)


def test_a_complete_nested_return_value_still_witnesses():
    value = {"seq": ["a", {"map": {"k": "v"}}]}
    ret = ev("return", 3, file=F, qualname="deliver", frame=10, value=value, category="bad", stack=stack(IN))
    got = certify(window(entry(1, 10), branch(2, IN), ret), node=N, kind="return", entry=ENTRY,
                  bad={"file": F, "qualname": "deliver", "value": value})
    assert got["ok"], got


@pytest.mark.parametrize("where", ["symptom", "boundary", "return"])
def test_a_spec_cannot_request_the_reserved_category(where):
    spec = one_spec("x", "y")
    if where == "symptom":
        spec["symptom"]["category"] = "unavailable"
    elif where == "boundary":
        spec["boundaries"] = [dict(DM, classify=[["unavailable", "^BTS"], ["pick", "^pick"]])]
    else:
        spec["returns"] = [dict(path="src/bts/mod.py", qualname="deliver", classify=[["unavailable", "x"]])]
    with pytest.raises(defence.SpecError, match="reserved"):
        defence._check_spec(spec)


def test_a_full_run_requesting_the_unavailable_category_is_refused(tmp_path):
    """r8 #2, first probe: one function bound to two boundaries (both calls correctly unattributed), with
    the symptom asking for the category those identities carry."""
    prod = "from bts import transport\ndef deliver():\n    return 'done'  # BRANCH\n"
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "src/bts/transport.py": TRANSPORT + "other=send\n",
                                        "tests/test_probe.py": RESULT_TEST})
    spec = one_spec("return 'done'  # BRANCH", f"transport.send('eric',{ALERT!r})  # BRANCH\n    return 'defer'",
                    category="unavailable")
    spec["boundaries"] = [DM, dict(DM, name="other", binding="bts.transport:other")]
    res = defence.current_defence(wt, spec, tmp_path / "out")
    assert res["verdict"] == "rejected" and not res["stages"], res
    assert any("reserved" in r for r in res["reasons"]), res["reasons"]


def test_a_full_run_with_an_incomplete_return_value_is_refused(tmp_path):
    """r8 #2, second probe: a 501-character return, requested by its truncated serialization."""
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": "def deliver():\n    return 'done'  # BRANCH\n",
                                        "tests/test_probe.py": RESULT_TEST})
    value = {"str": "x" * 500, "length": 501, "incomplete": True}
    spec = one_spec("return 'done'  # BRANCH", "return 'x'*501  # BRANCH", kind="return")
    spec.update(boundaries=[], returns=[dict(path="src/bts/mod.py", qualname="deliver")],
                symptom=dict(kind="return", path="src/bts/mod.py", qualname="deliver", value=value))
    res = defence.current_defence(wt, spec, tmp_path / "out")
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    ret = next(e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "return")
    assert ret["category"] == "unavailable" and ret["value"] == value


# --- finding 3: class metadata is never formatted inside observation ------------------------------

def _formatting_name(hits):
    class Name:
        def __format__(self, spec):
            hits.append("format")
            return "app.module"

        def __str__(self):
            hits.append("str")
            return "app.module"

        def __repr__(self):
            hits.append("repr")
            return "app.module"
    return Name()


def test_class_metadata_that_is_not_a_string_runs_no_application_code():
    hits = []
    name = _formatting_name(hits)

    class Result:
        __module__ = name

        def __init__(self):
            self.status = "done"

    class Boom(Exception):
        __module__ = name

    assert observer.type_name(Result) == "<unnamed>"
    safe = observer._safe(Result())
    assert safe.get("incomplete") and not observer._complete(safe), safe
    assert observer._raised(Boom) == {"raised": "<unnamed>", "incomplete": True}
    mon = observer._Monitor("n", {}, "")
    mon._err("probe", Boom("x"))
    assert mon.events[-1]["error_type"] == "<unnamed>"
    assert hits == [], hits
    assert observer.type_name(ValueError) == "builtins.ValueError"
    assert observer._raised(ValueError) == {"raised": "builtins.ValueError"}


def test_a_full_run_never_formats_application_metadata(tmp_path):
    """r8 #3: the return's class carried a module-name object with an application __format__; the observed
    run called it once (the unobserved twin never did) while both purity endpoints read clean."""
    prod = ("formats=0\nclass ModuleName:\n    def __format__(self,spec):\n        global formats\n"
            "        formats+=1\n        return 'model.module'\nclass Result:\n    __module__=ModuleName()\n"
            "    def __init__(self,status):\n        self.status=status\ndef deliver():\n"
            "    return Result('done')  # BRANCH\n")
    test = ("from bts import mod\ndef test_probe():\n    result=mod.deliver()\n"
            "    print('application_formats',mod.formats)\n    assert result.status=='done'  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "tests/test_probe.py": test})
    spec = one_spec("Result('done')  # BRANCH", "Result('defer')  # BRANCH", kind="return")
    spec.update(boundaries=[], returns=[dict(path="src/bts/mod.py", qualname="deliver", classify=[["bad", "defer"]])],
                symptom=dict(kind="return", path="src/bts/mod.py", qualname="deliver", category="bad"))
    res = defence.current_defence(wt, spec, tmp_path / "out")
    assert "application_formats 0" in (tmp_path / "out/mutant.stdout.txt").read_text()
    assert "application_formats 0" in (tmp_path / "out/mutant_unobserved.stdout.txt").read_text()
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    ret = next(e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "return")
    assert ret["category"] == "unavailable", ret


# --- finding 4: return identity is source-file and qualname identity -------------------------------

def test_a_cloned_return_function_is_certified_by_source_identity_only(tmp_path):
    """r8 #4 (SHOULD, characterization): a FunctionType clone of the declared function, with other globals,
    returns the bad value; the certificate is accepted because return identity is the source file and
    qualname, not the function object (COVERAGE says so; a claim needing the object reads unavailable)."""
    prod = ("calls=[]\nvalue='done'\nalternate='done'\ndef grade():\n    calls.append('grade')\n"
            "    return value  # BRANCH\ndef deliver():\n    return clone()\n")
    test = ("import types\nfrom bts import mod\nother=[]\n"
            "mod.clone=types.FunctionType(mod.grade.__code__,{'calls':other,'value':'done','alternate':'defer'})\n"
            "def test_probe():\n    result=mod.deliver()\n    assert mod.calls==[]\n"
            "    assert other==['grade']\n    assert mod.grade()=='done'\n"
            "    assert result=='done'  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "tests/test_probe.py": test})
    spec = one_spec("return value  # BRANCH", "return alternate  # BRANCH", kind="return")
    spec.update(boundaries=[], returns=[dict(path="src/bts/mod.py", qualname="grade", classify=[["bad", "^defer$"]])],
                symptom=dict(kind="return", path="src/bts/mod.py", qualname="grade", category="bad"))
    res = defence.current_defence(wt, spec, tmp_path / "out")
    assert res["verdict"] == "accepted", res["reasons"]
    values = [e["value"] for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "return"]
    assert values == ["defer", "done"], values
    assert "not Python function-object" in certify_mod.COVERAGE["return_identity"]
