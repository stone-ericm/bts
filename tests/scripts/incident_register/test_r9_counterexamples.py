"""Codex phase-1 r9 counterexamples, and the class they belong to (plan ruling 11).

r9 measured ACCEPTED certificates whose observation ran application code: a str-subclass key colliding with
``__dict__`` / ``__module__`` in a class namespace (#1) or with ``text`` in a Mock call's keyword dict (#2)
made the observer's hashed lookups run that key's ``__eq__``; session import records did the same after the
interval (#3). Ruling 11 fixes the class, not the instances: inside observation the observer never hashes,
compares or dispatches on an application object unless its type is exactly a trusted builtin. Dicts it reads
are walked item by item, and a key that is not an exact str makes that read unsupported (never skipped,
since skipping could miss the entry real lookup finds). The same rule covers code-object names, which
crafted code can make str subclasses, and watched stores made through such a key.
"""
import sys
import types
from unittest.mock import Mock

import pytest

from scripts.audit.incident_register import defence, observer, runner
from tests.scripts.incident_register.synth import DM, defended_project, write
from tests.scripts.incident_register.test_r4_counterexamples import ALERT, one_spec
from tests.scripts.incident_register.test_r6_counterexamples import NODE, _defend


def _counting_key(hits, collide_with, equal=False):
    """A str subclass hashing like ``collide_with``; its __hash__ and __eq__ record each call."""
    class Key(str):
        def __hash__(self):
            hits.append("hash")
            return str.__hash__(collide_with)

        def __eq__(self, other):
            hits.append("eq")
            return str.__eq__(self, other) if equal else False
    return Key


def _boundaries(tmp_path):
    return [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "boundary"]


def _counts(tmp_path):
    observed = (tmp_path / "out/mutant.stdout.txt").read_text()
    plain = (tmp_path / "out/mutant_unobserved.stdout.txt").read_text()
    return observed, plain


# --- r9 #1: class namespaces are read without hashed lookups --------------------------------------

@pytest.mark.parametrize("collide", ["__dict__", "__module__"])
def test_a_class_namespace_with_a_non_string_key_runs_no_application_code(collide):
    hits = []
    Key = _counting_key(hits, collide)
    Result = type("Result", (), {Key("extra"): 1})
    value = Result()
    value.status = "defer"
    Boom = type("Boom", (Exception,), {Key("extra"): 1})
    hits.clear()
    safe = observer._safe(value)
    assert not observer._complete(safe), safe
    assert observer.type_name(Boom) == "<unnamed>"
    assert observer._raised(Boom) == {"raised": "<unnamed>", "incomplete": True}
    mon = observer._Monitor("n", {}, "")
    mon._err("probe", Boom("x"))
    assert mon.events[-1]["error_type"] == "<unnamed>"
    assert hits == [], hits


def test_an_unsupported_class_namespace_never_falls_through_to_a_base():
    """A clean class whose MIDDLE base has an unsupported namespace: real lookup of ``__dict__`` would compare
    the middle class's same-hash key before reaching the base's descriptor, so the search ends there."""
    hits = []
    Key = _counting_key(hits, "__dict__")
    Base = type("Base", (), {})
    Middle = type("Middle", (Base,), {Key("extra"): 1})
    Result = type("Result", (Middle,), {})
    value = Result()
    value.status = "defer"
    hits.clear()
    assert not observer._complete(observer._safe(value))
    assert hits == [], hits

def test_ordinary_classes_still_read_completely():
    class Result:
        def __init__(self):
            self.status = "defer"
    assert observer._safe(Result()) == {"fields": {"status": "defer"}, "type": f"{__name__}.{Result.__qualname__}"}
    assert observer.type_name(ValueError) == "builtins.ValueError"


@pytest.mark.parametrize("collide", ["__dict__", "__module__"])
def test_a_full_run_never_compares_class_namespace_keys(tmp_path, collide):
    """r9 #1: both runs were accepted while the observer ran the key's __eq__ (3 and 1 extra calls)."""
    prod = ("comparisons=0\nclass Key(str):\n    def __hash__(self):\n        return str.__hash__(" + repr(collide) + ")\n"
            "    def __eq__(self,other):\n        global comparisons\n        comparisons+=1\n        return False\n"
            "Result=type('Result',(),{Key('extra'):1})\ndef deliver():\n    result=Result()\n"
            "    result.status='done'  # BRANCH\n    return result\n")
    test = ("from bts import mod\ndef test_probe():\n    mod.comparisons=0\n    result=mod.deliver()\n"
            "    print('application_comparisons',mod.comparisons)\n    assert result.status=='done'  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "tests/test_probe.py": test})
    spec = one_spec("result.status='done'  # BRANCH", "result.status='defer'  # BRANCH", kind="return")
    spec.update(boundaries=[], returns=[dict(path="src/bts/mod.py", qualname="deliver", classify=[["bad", "defer"]])],
                symptom=dict(kind="return", path="src/bts/mod.py", qualname="deliver", category="bad"))
    res = defence.current_defence(wt, spec, tmp_path / "out")
    observed, plain = _counts(tmp_path)
    assert "application_comparisons 0" in observed and "application_comparisons 0" in plain, (observed, plain)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    ret = next(e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "return")
    assert ret["category"] == "unavailable", ret


# --- r9 #2: keyword identities are read by iteration over exact-str keys ---------------------------

@pytest.mark.parametrize("equal", [False, True], ids=["other-key", "equal-key"])
def test_a_keyword_dict_with_a_non_string_key_is_unavailable_without_application_code(equal):
    hits = []
    Key = _counting_key(hits, "text", equal=equal)
    kwargs = {Key("text" if equal else "extra"): "hidden"}
    hits.clear()
    assert observer._access(["kw:text"], [], kwargs) == (False, None)
    assert hits == [], hits
    assert observer._access(["kw:text"], [], {"text": "x"}) == (True, "x")


KEY_TRANSPORT = ("import sys\nfrom unittest.mock import Mock\nsent=[]\ncomparisons=0\nclass Key(str):\n"
                 "    def __hash__(self):\n        return str.__hash__('text')\n"
                 "    def __eq__(self,other):\n        global comparisons\n        f=sys._getframe(1)\n"
                 "        while f is not None:\n            if '/w15obs_' in f.f_code.co_filename:\n"
                 "                comparisons+=1\n                break\n            f=f.f_back\n"
                 "        return str.__eq__(self,other)\n"
                 "def effect(**kwargs):\n    sent.append(('eric',kwargs['text']))\nsend=Mock(side_effect=effect)\n")


def test_a_full_run_never_compares_keyword_keys(tmp_path):
    """r9 #2: accepted, with the key's __eq__ run twice more under observation than without. Only comparisons
    made with an observer frame on the stack are counted (application-side ones vary with hash seeds)."""
    helper = ("def invoke(transport,rebind):\n    transport.comparisons=0\n"
              "    transport.send(**{transport.Key('text'):ALERT if rebind else 'pick: Turner'})\n"
              "    print('observer_comparisons',transport.comparisons)\n")
    res = _defend(tmp_path, helper, transport=KEY_TRANSPORT, boundaries=[dict(DM, value=["kw:text"])])
    observed, plain = _counts(tmp_path)
    assert "observer_comparisons 0" in observed and "observer_comparisons 0" in plain, (observed, plain)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    assert _boundaries(tmp_path) and all(e["identity"]["category"] == "unavailable" for e in _boundaries(tmp_path))


def test_a_full_run_with_exact_keyword_keys_still_witnesses(tmp_path):
    helper = ("def invoke(transport,rebind):\n"
              "    transport.send(text=ALERT if rebind else 'pick: Turner')\n")
    res = _defend(tmp_path, helper, transport=KEY_TRANSPORT, boundaries=[dict(DM, value=["kw:text"])])
    assert res["verdict"] == "accepted" and res["certificates"][NODE]["ok"], res


# --- the class: code-object names, watched stores, effective __call__, binding paths ---------------

def test_code_object_names_that_are_not_strings_run_no_application_code():
    hits = []
    Key = _counting_key(hits, "/x.py", equal=True)
    mon = observer._Monitor("n", {}, "/wt/src")
    hits.clear()
    assert mon._real(Key("/x.py")) == observer._NO_PATH
    assert observer._name(Key("q")) == "<unnamed>" and observer._name("q") == "q"
    assert hits == [], hits


def test_a_full_run_never_hashes_crafted_code_names(tmp_path):
    """Crafted code (code.replace) may carry str-subclass co_filename / co_qualname, which every observed
    function start used as a dict key and a path."""
    prod = ("calls=0\nclass Key(str):\n    def __hash__(self):\n        global calls\n        calls+=1\n"
            "        return str.__hash__(self)\n    def __eq__(self,other):\n        global calls\n        calls+=1\n"
            "        return str.__eq__(self,other)\ndef _inner():\n    return 'done'\n"
            "crafted=type(_inner)(_inner.__code__.replace(co_filename=Key('/elsewhere.py'),co_qualname=Key('inner')),{})\n"
            "named=type(_inner)(_inner.__code__.replace(co_qualname=Key('inner')),{})\n"
            "def deliver():\n    crafted()\n    named()\n    return 'done'  # BRANCH\n")
    test = ("from bts import mod\ndef test_probe():\n    mod.calls=0\n    result=mod.deliver()\n"
            "    print('application_calls',mod.calls)\n    assert result=='done'  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "tests/test_probe.py": test})
    spec = one_spec("return 'done'  # BRANCH", "return 'defer'  # BRANCH", kind="return")
    spec.update(boundaries=[], returns=[dict(path="src/bts/mod.py", qualname="deliver", classify=[["bad", "defer"]])],
                symptom=dict(kind="return", path="src/bts/mod.py", qualname="deliver", category="bad"))
    res = defence.current_defence(wt, spec, tmp_path / "out")
    observed, plain = _counts(tmp_path)
    assert "application_calls 0" in observed and "application_calls 0" in plain, (observed, plain)
    # Codex phase-1 r10 false greens: the result was discarded, and only the non-production filename was
    # crafted, so O60 (the qualname guard, reached only for production code) was never exercised here;
    # ``named`` keeps its exact production filename. Names are not identity: the return is still witnessed
    assert res["verdict"] == "accepted" and res["certificates"][NODE]["ok"], res


def test_a_store_through_a_non_string_key_leaves_nothing_current(tmp_path):
    """A str-subclass key equal to 'send' updates the binding under the original key; the watcher saw a
    store whose key was not an exact str and ignored it, so the captured former callable stayed current."""
    transport = ("sent=[]\ndef send(recipient,text,_sent=sent):\n    _sent.append((recipient,text))\n"
                 "def replacement(recipient,text):\n    pass\nclass Key(str):\n    __hash__=str.__hash__\n"
                 "    __eq__=str.__eq__\n")
    helper = ("def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n        return\n"
              "    callback=transport.send\n    transport.__dict__[transport.Key('send')]=transport.replacement\n"
              "    list(map(callback,['eric'],[ALERT]))\n")
    res = _defend(tmp_path, helper, transport=transport)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    gaps = [e["reason"] for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "boundary_gap"]
    assert any("not an exact str" in g for g in gaps), gaps


@pytest.mark.parametrize("where", ["module", "class"])
def test_a_binding_path_namespace_with_a_non_string_key_is_unsupported(tmp_path, where):
    """A module namespace with such a key is refused at every stage's gate (its import provenance is
    unavailable); a class namespace on the binding path reaches certification with a gap, unattributed."""
    transport = "sent=[]\ndef send(recipient,text):\n    sent.append((recipient,text))\nclass Key(str):\n    pass\n"
    if where == "module":
        transport += "globals()[Key('extra')]=1\n"
        binding, sender = DM, "transport.send"
    else:
        transport += "API=type('API',(),{'send':send,Key('extra'):1})\n"
        binding, sender = dict(DM, binding="bts.transport:API.send"), "transport.API.send"
    helper = f"def invoke(transport,rebind):\n    {sender}('eric',ALERT if rebind else 'pick: Turner')\n"
    res = _defend(tmp_path, helper, transport=transport, boundaries=[binding])
    assert res["verdict"] == "rejected", res
    if where == "module":
        assert any("bts.transport: import provenance unavailable" in r for r in res["reasons"]), res["reasons"]
        return
    assert not res["certificates"][NODE]["ok"], res
    gaps = [e["reason"] for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "boundary_gap"]
    assert any("not an exact str" in g for g in gaps), gaps


def test_a_non_string_key_equal_to_call_makes_a_mock_unverified():
    """Real lookup finds a str-subclass '__call__' key that iteration over exact-str keys would skip."""
    class Key(str):
        __hash__ = str.__hash__
        __eq__ = str.__eq__
    Sub = type("Sub", (Mock,), {Key("__call__"): lambda self, *a, **k: None})
    mon = observer._Monitor("n", {}, "")
    assert mon._effective_call(Sub) is None
    assert mon._effective_call(Mock) is mon.mock_call_fn


@pytest.mark.parametrize("level,restore", [
    ("module", "clone"), ("module", "keys"), ("root", "clone"), ("root", "keys"),
    ("class", "clone"), ("class", "keys"), ("parent", "old-child-store"),
])
def test_wholesale_invalidation_at_every_chain_level(tmp_path, level, restore):
    """r8 #1 at every level of the binding path (Codex r9's probes, kept as regressions): a clone into the
    cleared namespace, or a store into a detached child, leaves the captured callable unattributed; a
    key-by-key restore makes the binding current again."""
    transport = "sent=[]\ndef send(recipient,text,_sent=sent):\n    _sent.append((recipient,text))\n"
    if level in ("class", "parent"):
        transport += "class API:\n    send=send\n"
    binding = dict(DM, binding="bts.transport:API.send") if level in ("class", "parent") else DM
    sender = "transport.API.send" if level in ("class", "parent") else "transport.send"
    ns = {"module": "transport.__dict__", "root": "sys.modules",
          "class": 'gc.get_referents(type.__dict__["__dict__"].__get__(transport.API,type))[0]',
          "parent": "transport.__dict__"}[level]
    body = (f"import sys,gc\ndef invoke(transport,rebind):\n    if not rebind:\n        {sender}('eric','pick: Turner')\n"
            f"        return\n    callback={sender}\n    namespace={ns}\n    saved=dict(namespace)\n")
    if level == "parent":
        body += "    child=transport.API\n"
    body += "    namespace.clear()\n    try:\n"
    if restore == "keys":
        body += "        for key,value in saved.items():\n            namespace[key]=value\n"
    elif restore == "clone":
        body += "        namespace.update(saved)\n"
    else:
        body += "        child.send=callback\n"
    body += "        list(map(callback,['eric'],[ALERT]))\n    finally:\n        namespace.update(saved)\n"
    res = _defend(tmp_path, body, transport=transport, boundaries=[binding])
    assert res["verdict"] == ("accepted" if restore == "keys" else "rejected"), res
    calls = _boundaries(tmp_path)
    assert calls
    if restore == "keys":
        assert any(e["identity"]["category"] == "alert" for e in calls)
    else:
        assert all(e["identity"]["category"] == "unavailable" for e in calls)


# --- r9 #3: session import records --------------------------------------------------------------

def _session_record(monkeypatch):
    out = []
    monkeypatch.setattr(observer, "_write", out.append)
    monkeypatch.setattr(observer, "src_digest", lambda root: None)
    observer.pytest_sessionfinish(None, 0)
    return next(r for r in out if r["kind"] == "imports")


def test_session_imports_of_a_module_subclass_run_no_application_dispatch(monkeypatch):
    hits = []

    class Custom(types.ModuleType):
        def __getattribute__(self, name):
            hits.append(name)
            return super().__getattribute__(name)
    name = "bts.r9_session_probe"
    monkeypatch.setitem(sys.modules, name, Custom(name))
    rec = _session_record(monkeypatch)
    assert hits == [], hits
    # the exact reason: since Codex phase-1 r10 #3 an unreadable __file__ is unavailable too, so presence alone
    # no longer shows which guard refused the record (O62)
    assert rec["modules"].get(name) == {"unavailable": "not an exact module"}, rec["modules"].get(name)


def test_session_imports_of_an_exact_module_with_a_non_string_key_compare_nothing(monkeypatch):
    hits = []
    Key = _counting_key(hits, "__file__")
    name = "bts.r9_exact_session_probe"
    mod = types.ModuleType(name)
    mod.__dict__[Key("extra")] = 1
    monkeypatch.setitem(sys.modules, name, mod)
    hits.clear()
    rec = _session_record(monkeypatch)
    assert hits == [], hits
    assert rec["modules"].get(name) == {"unavailable": observer._NON_STR_KEY}, rec["modules"].get(name)   # (O63)


def test_a_run_whose_import_provenance_is_unavailable_is_rejected(tmp_path):
    _, wt = defended_project(tmp_path)
    write(wt, "tests/test_weird.py", "import sys, types\nclass Custom(types.ModuleType):\n    pass\n"
          "sys.modules['bts.weird'] = Custom('bts.weird')\ndef test_weird():\n    pass\n")
    r = runner.run(wt, ["tests", "-q"], tmp_path / "out", "green")
    why = runner.gate(r, worktree=wt, mode="green")
    assert any("bts.weird" in w and "provenance unavailable" in w for w in why), why


def test_session_imports_under_a_non_string_sys_modules_key_dispatch_nothing(monkeypatch):
    hits = []

    class Key(str):
        def __hash__(self):
            hits.append("hash")
            return str.__hash__(self)

        def __eq__(self, other):
            hits.append("eq")
            return str.__eq__(self, other)

        def __str__(self):
            hits.append("str")
            return "bts.elsewhere"
    monkeypatch.setitem(sys.modules, Key("bts.r9_key_probe"), types.ModuleType("bts.r9_key_probe"))
    hits.clear()
    rec = _session_record(monkeypatch)
    assert hits == [], hits
    # since Codex phase-1 r10 #3 any key that is not an exact str makes the whole record unavailable: hashed
    # lookup may resolve it as any module name, whatever its text
    assert rec["modules"] == {} and rec.get("unavailable") == "a sys.modules key that is not an exact str", rec


def test_a_sys_namespace_with_a_non_string_key_makes_import_provenance_unavailable(tmp_path):
    _, wt = defended_project(tmp_path)
    write(wt, "tests/test_weird.py", "import sys\nclass Key(str):\n    pass\nsys.__dict__[Key('extra')] = 1\n"
          "def test_weird():\n    pass\n")
    r = runner.run(wt, ["tests", "-q"], tmp_path / "out", "green")
    why = runner.gate(r, worktree=wt, mode="green")
    assert any(w.startswith("import provenance unavailable: unsupported namespace") for w in why), why
