"""Phase-1 r10 counterexamples: the rest of plan ruling 11's class.

Found in my own review while Codex r10 part 1 ran: hashing a code object hashes its ``co_name`` and its
``co_consts`` (CPython 3.12 ``code_hash``), and comparing two code objects compares them. ``co_name`` may be a
``str`` subclass and ``co_consts`` may hold any object (``code.replace``), so every observer map KEYED BY
CODE OBJECT ran application ``__hash__`` / ``__eq__`` at function starts inside observation: the
boundary-code map (every started code was looked up in it) and the set of instrumented production code.
Those maps are now keyed by the code object's id, with the object itself held in the value.

Codex phase-1 r10 part 1 (BLOCK) measured the same code-object finding independently (#1, return runs), and two
more: an exact-key store re-resolved a binding through a namespace that still held a key that is not an exact
str (#2), and unreadable import records were omitted instead of recorded unavailable, so the gate never
refused them (#3). Its probes (``r8-probes/r10/test_r10_review.py``) are turned to the required direction.
"""
import json

import pytest

from scripts.audit.incident_register import defence, observer, runner
from tests.scripts.incident_register.synth import DM, defended_project
from tests.scripts.incident_register.test_r4_counterexamples import one_spec
from tests.scripts.incident_register.test_r6_counterexamples import NODE, _defend
from tests.scripts.incident_register.test_r9_counterexamples import _boundaries, _counting_key, _counts


def _crafted_code(hits):
    """A code object whose ``co_name`` and one of whose constants record every hash or comparison."""
    Key = _counting_key(hits, "inner", equal=True)

    class Const:
        def __hash__(self):
            hits.append("const-hash")
            return 1

        def __eq__(self, other):
            hits.append("const-eq")
            return self is other

    def inner():
        return "done"
    return inner.__code__.replace(co_name=Key("inner"), co_consts=inner.__code__.co_consts + (Const(),))


@pytest.mark.parametrize("where", ["outside-production", "production", "boundary"])
def test_a_crafted_code_object_is_never_hashed_or_compared(where):
    hits = []
    code = _crafted_code(hits)
    if where == "production":
        code = code.replace(co_filename="/wt/src/bts/x.py")
    mon = observer._Monitor("n", {}, "/wt/src")
    hits.clear()
    if where == "boundary":
        spec = {"name": "dm"}
        mon._add_code(code, spec, None)
        mon._add_code(code, spec, None)                       # the second registration is a no-op
        mon._add_code(code.replace(co_firstlineno=1), spec, None)   # an equal-looking other code object
        assert len(mon.boundary_codes) == 2
    else:
        first = mon._on_start(code, 0)
        mon._on_start(code, 0)                                # production: already instrumented
        mon._on_start(code.replace(co_firstlineno=1), 0)
        if where == "outside-production":
            assert first is observer.sys.monitoring.DISABLE
        else:
            assert first is None and len(mon.instrumented) == 2
    assert hits == [], hits
    assert not [e for e in mon.events if e["kind"] == "observer_error"], mon.events


CRAFTED_TRANSPORT = (
    "import sys\nsent=[]\nobserved=0\n"
    "def _count():\n    global observed\n    f=sys._getframe(1)\n    while f is not None:\n"
    "        if '/w15obs_' in f.f_code.co_filename:\n            observed+=1\n            return\n        f=f.f_back\n"
    "class Key(str):\n    def __hash__(self):\n        _count()\n        return str.__hash__(self)\n"
    "    def __eq__(self,other):\n        _count()\n        return str.__eq__(self,other)\n"
    "class Const:\n    def __hash__(self):\n        _count()\n        return 1\n"
    "    def __eq__(self,other):\n        _count()\n        return self is other\n"
    "def _send(recipient,text):\n    sent.append((recipient,text))\n"
    "send=type(_send)(_send.__code__.replace(co_name=Key('send'),co_consts=_send.__code__.co_consts+(Const(),)),"
    "globals(),'send')\n"
    "def replacement(recipient,text):\n    sent.append((recipient,text))\n")


def test_a_full_run_never_hashes_crafted_code(tmp_path):
    """The boundary is a function whose code carries a str-subclass co_name and an application constant;
    the helper also starts such a function outside production. Only hashes and comparisons made with an
    observer frame on the stack are counted. Names are not identity, so the send is still witnessed."""
    helper = ("def _noop():\n    return None\n"
              "def invoke(transport,rebind):\n"
              "    noop=type(_noop)(_noop.__code__.replace(co_name=transport.Key('noop'),"
              "co_consts=_noop.__code__.co_consts+(transport.Const(),)),{})\n"
              "    noop()\n"
              "    transport.send('eric',ALERT if rebind else 'pick: Turner')\n"
              "    print('observer_calls',transport.observed)\n")
    res = _defend(tmp_path, helper, transport=CRAFTED_TRANSPORT)
    observed, plain = _counts(tmp_path)
    assert "observer_calls 0" in observed and "observer_calls 0" in plain, (observed, plain)
    assert res["verdict"] == "accepted" and res["certificates"][NODE]["ok"], res


def _return_spec():
    spec = one_spec("return 'done'  # BRANCH", "return 'defer'  # BRANCH", kind="return")
    spec.update(boundaries=[], returns=[dict(path="src/bts/mod.py", qualname="deliver", classify=[["bad", "defer"]])],
                symptom=dict(kind="return", path="src/bts/mod.py", qualname="deliver", category="bad"))
    return spec


@pytest.mark.parametrize("field", ["co_name", "co_consts"])
def test_a_return_run_never_hashes_a_crafted_code_object(tmp_path, field):
    """Codex r10 #1: both accepted at 0550cda with 3 application hash calls observed, 0 plain."""
    prod = ("calls=0\nclass Key(str):\n    def __hash__(self):\n        global calls\n        calls+=1\n"
            "        return str.__hash__(self)\n    def __eq__(self,other):\n        global calls\n        calls+=1\n"
            "        return str.__eq__(self,other)\ndef _inner():\n    return 'ordinary'\n"
            "crafted=type(_inner)(_inner.__code__.replace("
            + ("co_name=Key('inner')" if field == "co_name" else "co_consts=_inner.__code__.co_consts+(Key('extra'),)")
            + "),{})\ndef deliver():\n    crafted()\n    return 'done'  # BRANCH\n")
    test = ("from bts import mod\ndef test_probe():\n    mod.calls=0\n    result=mod.deliver()\n"
            "    print('application_calls',mod.calls)\n    assert result=='done'  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "tests/test_probe.py": test})
    res = defence.current_defence(wt, _return_spec(), tmp_path / "out")
    observed, plain = _counts(tmp_path)
    assert "application_calls 0" in observed and "application_calls 0" in plain, (observed, plain)
    assert res["verdict"] == "accepted" and res["certificates"][NODE]["ok"], res
    assert not [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "observer_error"]


# --- Codex r10 #2: an exact-key store does not make a still-unsupported namespace supported ----------

REVIVE_TRANSPORT = ("sent=[]\ndef send(recipient,text,_sent=sent):\n    _sent.append((recipient,text))\n"
                    "class Key(str):\n    pass\nclass API:\n    send=send\n")


def _revive(tmp_path, where, remove_first):
    ns = "transport.API" if where == "class" else "transport"
    raw = ("gc.get_referents(type.__dict__['__dict__'].__get__(transport.API,type))[0]" if where == "class"
           else "transport.__dict__")
    restore = ("        del namespace[key]\n        namespace['send']=callback\n" if remove_first else
               "        namespace['send']=callback\n")
    helper = (f"import gc\ndef invoke(transport,rebind):\n    if not rebind:\n        {ns}.send('eric','pick: Turner')\n"
              f"        return\n    callback={ns}.send\n    namespace={raw}\n    key=transport.Key('extra')\n"
              f"    namespace[key]=1\n    try:\n        namespace['send']=lambda *_args: None\n{restore}"
              f"        list(map(callback,['eric'],[ALERT]))\n    finally:\n        namespace.pop(key,None)\n")
    binding = dict(DM, binding="bts.transport:API.send") if where == "class" else DM
    return _defend(tmp_path, helper, transport=REVIVE_TRANSPORT, boundaries=[binding])


@pytest.mark.parametrize("where", ["module", "class"])
def test_an_exact_store_never_revives_a_binding_while_its_namespace_holds_a_non_string_key(tmp_path, where):
    """Codex r10 #2: both accepted at 0550cda (an attributed alert) while the namespace still held Key('extra')."""
    res = _revive(tmp_path, where, remove_first=False)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    bs = _boundaries(tmp_path)
    assert bs and not any(e["identity"]["category"] == "alert" for e in bs), bs
    assert any(e["identity"]["category"] == "unavailable" for e in bs), bs


@pytest.mark.parametrize("where", ["module", "class"])
def test_an_exact_store_after_the_non_string_key_is_removed_revives_the_binding(tmp_path, where):
    """Control: once the key that is not an exact str is gone, a key-by-key restore makes the binding current."""
    res = _revive(tmp_path, where, remove_first=True)
    assert res["verdict"] == "accepted" and res["certificates"][NODE]["ok"], res


# --- Codex r10 #3: an unreadable import record is recorded unavailable (refused), never omitted -------

UNREADABLE_IMPORTS = {
    "file-subclass": "class File(str):\n    pass\n__file__=File(__file__)\n",
    "file-pathlike": ("calls=0\nclass File:\n    def __init__(self,path):\n        self.path=path\n"
                      "    def __fspath__(self):\n        global calls\n        calls+=1\n        return self.path\n"
                      "__file__=File(__file__)\n"),
    "no-file": "import sys,types\nsys.modules['bts.virtual']=types.ModuleType('bts.virtual')\n",
    "object-key": ("import importlib,sys,types\nclass ModuleKey:\n    def __hash__(self):\n        return hash('bts.hidden')\n"
                   "    def __eq__(self,other):\n        return other=='bts.hidden'\n"
                   "module=types.ModuleType('bts.hidden')\nmodule.__file__='/outside-the-reviewed-source.py'\n"
                   "sys.modules[ModuleKey()]=module\nassert importlib.import_module('bts.hidden') is module\n"),
    "subclass-key": ("import importlib,sys,types\nclass ModuleKey(str):\n    def __hash__(self):\n        return hash('bts.hidden')\n"
                     "    def __eq__(self,other):\n        return other=='bts.hidden'\n"
                     "module=types.ModuleType('bts.hidden')\nmodule.__file__='/outside-the-reviewed-source.py'\n"
                     "sys.modules[ModuleKey('zzz')]=module\nassert importlib.import_module('bts.hidden') is module\n"),
}


@pytest.mark.parametrize("shape", sorted(UNREADABLE_IMPORTS))
def test_an_unreadable_import_record_is_refused_not_omitted(tmp_path, shape):
    """Codex r10 #3: file-subclass, file-pathlike and object-key were accepted at 0550cda, the module simply
    missing from the imports record; no-file and subclass-key (a str subclass whose text is not a bts name but
    which hashed lookup resolves as one) are the same class."""
    prod = UNREADABLE_IMPORTS[shape] + "def deliver():\n    return 'done'  # BRANCH\n"
    test = "from bts import mod\ndef test_probe():\n    assert mod.deliver()=='done'  # ASSERT-RESULT\n"
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "tests/test_probe.py": test})
    res = defence.current_defence(wt, _return_spec(), tmp_path / "out")
    assert res["verdict"] == "rejected", res
    assert "import provenance unavailable" in json.dumps(res), res
    imports = next(e for e in runner.load(tmp_path / "out/green.events.jsonl") if e["kind"] == "imports")
    assert "unavailable" in imports or any("unavailable" in m for m in imports["modules"].values()), imports
