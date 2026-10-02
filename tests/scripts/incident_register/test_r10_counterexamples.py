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

Part 2 (BLOCK) measured a concurrent case of #2: another thread's invalidation landed while a re-resolution
was reading, and the re-resolution then published over it. Every binding-state transition and every
attribution read now holds one lock, which each watched store's callback takes before its change; a call is
attributed only if, at the call, the binding still resolves to it through watched, supported namespaces
(``_current_now``). Its probes (``r8-probes/r10/part2/``) are turned to the required direction too.
"""
import json
import sys
import types

import pytest

from scripts.audit.incident_register import defence, observer, runner
from tests.scripts.incident_register.synth import DM, defended_project
from tests.scripts.incident_register.test_r4_counterexamples import ALERT, one_spec
from tests.scripts.incident_register.test_r6_counterexamples import MUTANT, NODE, PROD, TEST, _defend
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
    # the re-resolution's own refusal (O67), not only the call-time check (which would reject this run too)
    gaps = [e["reason"] for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "boundary_gap"]
    assert observer._NON_STR_KEY in gaps, gaps
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


# --- Codex r10 part 2 #1: a concurrent invalidation is never overwritten; attribution reads the binding at the call

RACE_HELPER = """import threading,time,gc
def invoke(transport,rebind):
    if not rebind:
        transport.API.send('eric','pick: Turner')
        return
    new=type('Next',(),dict.fromkeys(('padding'+str(i) for i in range(500000)),0))
    new.send=transport.send
    parent_namespace=transport.__dict__
    key=transport.Key('extra')
    stored=threading.Event()
    assigning=threading.Event()
    ready=threading.Event()
    place=[]
    def change():
        ready.set()
        assert assigning.wait(10)
        time.sleep(0)
        place.append('began during store' if not stored.is_set() else 'began after store')
        parent_namespace[key]=1
        place.append('stored')
    worker=threading.Thread(target=change)
    worker.start()
    assert ready.wait(10)
    try:
        assigning.set()
        transport.API=new
        stored.set()
        worker.join(timeout=12)
        assert not worker.is_alive() and place[-1:]==['stored'],place
        print('change_place',place)
        list(map(new.send,['eric'],[ALERT]))
    finally:
        worker.join(timeout=12)
        parent_namespace.pop(key,None)
"""


def test_a_concurrent_invalidation_is_never_overwritten(tmp_path):
    """Codex r10 part 2 #1: accepted at 0dad851 (an attributed alert after the worker's bad-key gap). The worker
    adds a key that is not an exact str to the parent module, racing the main thread's store of a large class; the
    key stays until after the call. Whatever the schedule, the run must be rejected (Codex r11 #3: the printed
    label is not a commit witness, so it is not asserted; the worker's completed store is)."""
    res = _defend(tmp_path, RACE_HELPER, transport=REVIVE_TRANSPORT,
                  boundaries=[dict(DM, binding="bts.transport:API.send")])
    observed, _plain = _counts(tmp_path)
    assert "'stored']" in observed, observed
    assert not [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "observer_error"]
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    bs = _boundaries(tmp_path)
    assert bs and not any(e["identity"]["category"] == "alert" for e in bs), bs


def test_attribution_reads_the_binding_at_the_call(monkeypatch):
    """The bookkept current value counts only while the binding resolves to it NOW, through watched namespaces
    that are all supported. No watcher is installed here, so each change below leaves the bookkeeping stale on
    purpose: only the call-time read (O70, O71) can see it."""
    def send(recipient, text):
        pass

    def other(recipient, text):
        pass
    mod = types.ModuleType("bts.r10_now_probe")
    mod.send = send
    monkeypatch.setitem(sys.modules, "bts.r10_now_probe", mod)
    mon = observer._Monitor("n", {"boundaries": [dict(name="dm", binding="bts.r10_now_probe:send")]}, "")
    spec = mon.boundaries[0]
    mon._track(spec)
    assert mon.state["dm"]["current"] is send and mon._current_now(spec) is send
    key = _counting_key([], "zzz")("extra")
    mod.__dict__[key] = 1
    assert mon._current_now(spec) is None                     # a key that is not an exact str (O70)
    del mod.__dict__[key]
    assert mon._current_now(spec) is send
    mod.send = other
    assert mon._current_now(spec) is None                     # the binding holds another value (O70)
    mod.send = send
    assert mon._current_now(spec) is send
    del mon.watched[id(mod.__dict__)]
    assert mon._current_now(spec) is None                     # a namespace the observer does not watch (O71)


@pytest.mark.parametrize("level", [0, 1])
def test_a_bad_key_at_a_root_level_is_not_revived(tmp_path, level):
    """Codex r10 part 2 (controls at sys's own namespace and sys.modules): an exact store at the level that
    holds a key that is not an exact str leaves the binding unavailable."""
    setup = ("namespace=sys.__dict__\n    exact='modules'\n    value=sys.modules" if level == 0 else
             "namespace=sys.modules\n    exact='bts.transport'\n    value=transport")
    helper = ("import sys\nclass Key(str):\n    pass\ndef invoke(transport,rebind):\n    if not rebind:\n"
              "        transport.send('eric','pick: Turner')\n        return\n    callback=transport.send\n"
              f"    {setup}\n    key=Key('extra-root-key')\n    namespace[key]=1\n    try:\n"
              "        namespace[exact]=value\n        list(map(callback,['eric'],[ALERT]))\n    finally:\n"
              "        namespace.pop(key,None)\n")
    res = _defend(tmp_path, helper)
    assert res["verdict"] == "rejected", res
    bs = _boundaries(tmp_path)
    assert bs and all(e["identity"]["category"] == "unavailable" for e in bs), bs


def test_a_non_string_file_is_never_resolved():
    """Codex r10 part 2 #2: the full PathLike run checks refusal, not purity; this counts __fspath__ (O72)."""
    hits = []

    class File:
        def __fspath__(self):
            hits.append("fspath")
            return "/unused"
    m = types.ModuleType("bts.probe")
    m.__file__ = File()
    assert observer._import_record(m) == {"unavailable": "no __file__ that is an exact str"}
    assert hits == [], hits


def test_composite_serialization_runs_no_member_code():
    """Codex r10 part 2 (call-phase audit): exact tuples, lists and dicts are walked without hashing or
    formatting their members; frozensets, slices and ranges are not read through."""
    hits = []

    class Member:
        def __hash__(self):
            hits.append("hash")
            return 1

        def __eq__(self, other):
            hits.append("eq")
            return self is other

        def __repr__(self):
            hits.append("repr")
            return "member"

        def __format__(self, spec):
            hits.append("format")
            return "member"
    value = Member()
    forms = [(value,), [value], {"x": value}, frozenset([value]), slice(value, value, value), range(4)]
    hits.clear()                                              # building the frozenset hashed the member
    for form in forms:
        observer._safe(form)
    assert hits == [], hits


# --- Codex r11 #1 (plan ruling 12): while another thread is alive the observer reads no application object ------

def test_alone_reads_the_interpreters_thread_states():
    assert observer._alone()
    import threading
    gate = threading.Event()
    worker = threading.Thread(target=gate.wait)
    worker.start()
    try:
        assert not observer._alone()                             # O73
    finally:
        gate.set()
        worker.join()
    assert observer._alone()


def _concurrent_boundaries(tmp_path):
    return [e for e in _boundaries(tmp_path) if e["identity"].get("reason") == observer._CONCURRENT]


THREAD_AROUND = ("import threading\ndef invoke(transport,rebind):\n    if not rebind:\n"
                 "        transport.send('eric','pick: Turner')\n        return\n"
                 "    gate=threading.Event()\n    worker=threading.Thread(target=gate.wait)\n    worker.start()\n"
                 "    try:\n        transport.send('eric',ALERT)\n    finally:\n        gate.set()\n        worker.join()\n")
MOCK_TRANSPORT = ("from unittest.mock import Mock\nsent=[]\ndef effect(recipient,text):\n    sent.append((recipient,text))\n"
                  "send=Mock(side_effect=effect)\n")


@pytest.mark.parametrize("kind", ["function", "mock"])
def test_a_boundary_call_while_another_thread_is_alive_reads_nothing(tmp_path, kind):
    """The call is recorded with an unavailable identity: neither its arguments nor the binding is read (O74, O75)."""
    res = _defend(tmp_path, THREAD_AROUND, **({"transport": MOCK_TRANSPORT} if kind == "mock" else {}))
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    bs = _boundaries(tmp_path)
    assert bs and _concurrent_boundaries(tmp_path) and not any(e["identity"]["category"] == "alert" for e in bs), bs


def test_a_return_while_another_thread_is_alive_is_not_read(tmp_path):
    """The bad return is real, but it is observed while a worker is alive, so its value is not read (O76)."""
    prod = ("import threading\ngate=threading.Event()\nworkers=[]\ndef deliver():\n"
            "    w=threading.Thread(target=gate.wait)\n    w.start()\n    workers.append(w)\n"
            "    return 'done'  # BRANCH\n")
    test = ("from bts import mod\ndef test_probe():\n    result=mod.deliver()\n    mod.gate.set()\n"
            "    [w.join() for w in mod.workers]\n    assert result=='done'  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "tests/test_probe.py": test})
    res = defence.current_defence(wt, _return_spec(), tmp_path / "out")
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    ret = next(e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "return")
    assert ret["category"] == "unavailable" and ret["value"].get("unavailable") == observer._CONCURRENT, ret


def test_a_store_while_another_thread_is_alive_cuts_the_binding(tmp_path):
    """A rebind and restore made while a worker is alive is not re-resolved (that would read application objects):
    the binding is cut, so the later call, made alone, is not attributed (O77)."""
    helper = ("import threading\ndef invoke(transport,rebind):\n    if not rebind:\n"
              "        transport.send('eric','pick: Turner')\n        return\n"
              "    gate=threading.Event()\n    worker=threading.Thread(target=gate.wait)\n    worker.start()\n"
              "    try:\n        original=transport.send\n        transport.send=transport.replacement\n"
              "        transport.send=original\n    finally:\n        gate.set()\n        worker.join()\n"
              "    transport.send('eric',ALERT)\n")
    res = _defend(tmp_path, helper)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    gaps = [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "boundary_gap"]
    assert any(g["reason"] == observer._CONCURRENT and g.get("where") == "store" for g in gaps), gaps


def test_a_thread_alive_through_the_call_phase_leaves_the_binding_untracked(tmp_path):
    """A daemon thread started at the test module's import is alive when the call phase starts and when it ends:
    the binding is not resolved at the start, the end check reads nothing, and no call is attributed (O78, O79)."""
    test = ("import threading\nfrom bts import mod,transport\n_gate=threading.Event()\n"
            "threading.Thread(target=_gate.wait,daemon=True).start()\ndef test_probe():\n    result=mod.deliver()\n"
            "    assert transport.sent==[('eric','pick: Turner')]\n    assert result=='done'  # ASSERT-RESULT\n")
    res = _defend(tmp_path, "def invoke(transport,rebind):\n    transport.send('eric',ALERT if rebind else 'pick: Turner')\n",
                  test=test)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    gaps = [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "boundary_gap"]
    wheres = {g.get("where") for g in gaps if g["reason"] == observer._CONCURRENT}
    assert {"start", "end"} <= wheres, gaps


FINALIZER_TRANSPORT = ("import sys,threading\nsent=[]\nobserved=0\nreleased=0\napplication_lock=threading.Lock()\n"
                       "cycle_timeouts=0\ndef send(recipient,text):\n    sent.append((recipient,text))\n"
                       "class Ref:\n    __slots__=('code',)\n    def __init__(self,code):\n        self.code=code\n"
                       "    def __del__(self):\n        global observed,released,cycle_timeouts\n        released+=1\n"
                       "        f=sys._getframe(1)\n        while f is not None:\n"
                       "            if '/w15obs_' in f.f_code.co_filename:\n                observed+=1\n"
                       "                if not application_lock.acquire(timeout=.25):\n                    cycle_timeouts+=1\n"
                       "                else:\n                    application_lock.release()\n                return\n"
                       "            f=f.f_back\n")
FINALIZER_HELPER = ("import threading,time,sys\ndef invoke(transport,rebind):\n    if not rebind:\n"
                    "        transport.send('eric','pick: Turner')\n        return\n"
                    "    padding=[(transport.send.__code__,i) for i in range(500000)]\n"
                    "    holder=[transport.Ref(transport.send.__code__)]\n    main=threading.get_ident()\n"
                    "    finished=threading.Event()\n    begin=threading.Event()\n    ready=threading.Event()\n"
                    "    def release():\n        ready.set()\n        assert begin.wait(10)\n"
                    "        deadline=time.monotonic()+10\n        while time.monotonic()<deadline:\n"
                    "            f=sys._current_frames().get(main)\n            while f is not None:\n"
                    "                if (f.f_code.co_name=='<genexpr>' and '/w15obs_' in f.f_code.co_filename and f.f_back is not None\n"
                    "                        and f.f_back.f_code.co_name=='_code_shared'):\n"
                    "                    with transport.application_lock:\n                        holder.clear()\n"
                    "                        transport.worker_store=1\n                    return\n"
                    "                f=f.f_back\n            if finished.is_set():\n                holder.clear()\n"
                    "                return\n            time.sleep(0)\n        holder.clear()\n"
                    "    worker=threading.Thread(target=release)\n    worker.start()\n    assert ready.wait(10)\n"
                    "    try:\n        begin.set()\n        transport.send('eric',ALERT)\n        finished.set()\n"
                    "        worker.join(timeout=12)\n        assert not worker.is_alive()\n"
                    "        print('finalizer',transport.released,'observer_calls',transport.observed,"
                    "'cycle_timeouts',transport.cycle_timeouts)\n    finally:\n        worker.join(timeout=12)\n"
                    "        holder.clear()\n")


def test_no_application_finalizer_runs_inside_observation(tmp_path):
    """Codex r11 #1: accepted at 308f51a while an application __del__ ran inside the observer's shared-code census
    (released by a worker while the census held the last reference), and under the lock it waited on an
    application lock a worker held. Now nothing is read while the worker is alive: no finalizer runs with an
    observer frame on the stack, nothing waits, and the call is not attributed."""
    res = _defend(tmp_path, FINALIZER_HELPER, transport=FINALIZER_TRANSPORT)
    observed, plain = _counts(tmp_path)
    assert "observer_calls 0 cycle_timeouts 0" in observed and "observer_calls 0 cycle_timeouts 0" in plain, (observed, plain)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    assert _concurrent_boundaries(tmp_path), _boundaries(tmp_path)


def test_alone_never_reports_alone_while_another_thread_exists():
    """Codex r12: _alone read the list head, then the head's successor, in two calls, and reported "alone" while
    the driver thread was alive (at 78d6d38). Between the calls the GIL can pass to a short-lived child thread
    that may finish meanwhile: following another thread's sampled state was unsafe, with the exact cause
    (the unlink/free sequence) inferred, not measured (Codex phase-1 r13). The read now touches only this
    thread's own state and the interpreter's head pointer."""
    import threading
    import time
    stop, ready = threading.Event(), threading.Event()

    def churn():
        ready.set()
        while not stop.is_set():
            child = threading.Thread(target=lambda: None)
            child.start()
            child.join()
    driver = threading.Thread(target=churn)
    driver.start()
    assert ready.wait(5)
    wrong, checks, deadline = 0, 0, time.monotonic() + 3
    try:
        while time.monotonic() < deadline:
            checks += 1
            wrong += observer._alone()
    finally:
        stop.set()
        driver.join(timeout=5)
    assert not driver.is_alive() and checks > 1000 and wrong == 0, (wrong, checks)


def test_alone_is_the_head_with_no_successor(monkeypatch):
    """Deterministic: alone exactly when this thread's state heads the list and has no successor, read twice
    (new thread states are inserted at the head); an unreadable state is not alone (O80, O81, O82)."""
    me, other = 0x1000, 0x2000
    monkeypatch.setattr(observer, "_TS_GET", lambda: me)
    monkeypatch.setattr(observer, "_INTERP_GET", lambda: 0x10)
    for head, nxt, want in [(me, 0, True), (other, 0, False), (me, other, False)]:
        monkeypatch.setattr(observer, "_THREAD_HEAD", lambda _i, h=head: h)
        monkeypatch.setattr(observer, "_THREAD_NEXT", lambda _t, n=nxt: n)
        assert observer._alone() is want, (head, nxt)
    heads = iter([me, other])                                    # a thread state inserted between the two reads
    monkeypatch.setattr(observer, "_THREAD_HEAD", lambda _i: next(heads))
    monkeypatch.setattr(observer, "_THREAD_NEXT", lambda _t: 0)
    assert observer._alone() is False

    def boom(*_a):
        raise OSError("unreadable")
    monkeypatch.setattr(observer, "_THREAD_HEAD", boom)
    assert observer._alone() is False


# --- Codex r12 #1: no frame-locals synchronization; the frame's own slots are read --------------------------

CACHED_TRANSPORT = ("import sys,threading\nsent=[]\nreleased=0\nobserved=0\ncycle_timeouts=0\n"
                    "application_lock=threading.Lock()\nclass Ref:\n    def __del__(self):\n"
                    "        global released,observed,cycle_timeouts\n        released+=1\n        frame=sys._getframe(1)\n"
                    "        while frame is not None:\n            if '/w15obs_' in frame.f_code.co_filename:\n"
                    "                observed+=1\n                if application_lock.acquire(timeout=0.02):\n"
                    "                    application_lock.release()\n                else:\n                    cycle_timeouts+=1\n"
                    "                return\n            frame=frame.f_back\n"
                    "def make():\n    held=Ref()\n    def send(recipient,text):\n        sent.append((recipient,text))\n"
                    "        yield held\n    def clear():\n        nonlocal held\n        held=None\n    return send,clear\n"
                    "send,clear=make()\n")


def _cached_helper(worker, lock):
    start = ("    gate=threading.Event()\n    worker=threading.Thread(target=gate.wait)\n    worker.start()\n" if worker else "")
    stop = ("    gate.set()\n    worker.join(timeout=5)\n    assert not worker.is_alive()\n" if worker else "")
    step = ("    with transport.application_lock:\n        next(gen)\n" if lock else "    next(gen)\n")
    return ("import inspect,sys,threading\ndef invoke(transport,rebind):\n" + start +
            "    gen=transport.send('eric',ALERT if rebind else 'pick: Turner')\n    inspect.getgeneratorlocals(gen)\n"
            "    transport.clear()\n" + step + "    gen.close()\n    del gen\n" + stop +
            "    print('finalizer',transport.released,'observer_calls',transport.observed,'cycle_timeouts',transport.cycle_timeouts)\n")


@pytest.mark.parametrize("lock", [False, True], ids=["no-lock", "app-lock"])
@pytest.mark.parametrize("worker", [False, True], ids=["alone", "worker"])
def test_a_cached_generator_locals_dict_is_never_refreshed(tmp_path, worker, lock):
    """Codex r12 #1: accepted at 78d6d38 with ONE thread. inspect had cached the generator's locals; the closure
    cell was then cleared, and the observer's frame.f_locals refreshed the cached dict, releasing the last
    reference to Ref, whose __del__ ran inside _on_start (and, holding an application lock, timed out on it).
    Arguments now come from the frame's own slots: no finalizer runs with an observer frame on the stack. Alone,
    the generator boundary is still witnessed; with a worker alive the call is refused (ruling 12)."""
    res = _defend(tmp_path, _cached_helper(worker, lock), transport=CACHED_TRANSPORT)
    observed, plain = _counts(tmp_path)
    assert "observer_calls 0 cycle_timeouts 0" in observed and "observer_calls 0 cycle_timeouts 0" in plain, (observed, plain)
    assert not [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "observer_error"]
    if worker:
        assert res["verdict"] == "rejected" and _concurrent_boundaries(tmp_path), res
    else:
        assert res["verdict"] == "accepted" and res["certificates"][NODE]["ok"], res


def test_fast_locals_read_the_frames_own_slots():
    """The reader returns the objects in a live frame's slots (arguments, *args, **kwargs, a cell argument's
    contents), refuses a cell when cells are not allowed, an unbound slot, an index out of range and a frame/code
    mismatch, and never creates the frame's locals dict (O84, O85)."""
    import ctypes
    a, b = object(), object()

    def probe(x, *rest, k=None, **kw):
        frame, code = sys._getframe(), probe.__code__
        iframe = ctypes.c_void_p.from_address(id(frame) + observer._PY_FRAME_F_FRAME).value
        got = [observer._fast_local(frame, code, i, True) for i in range(4)]
        mismatched = observer._fast_local(frame, test_fast_locals_read_the_frames_own_slots.__code__, 0, True)
        out_of_range = observer._fast_local(frame, code, code.co_nlocals, True)
        unbound = observer._fast_local(frame, code, code.co_varnames.index("later"), True)
        locals_dict = ctypes.c_void_p.from_address(iframe + 5 * ctypes.sizeof(ctypes.c_void_p)).value
        later = 1
        return got, mismatched, out_of_range, unbound, locals_dict, later
    got, mismatched, out_of_range, unbound, locals_dict, _ = probe(a, 1, k=b, z=2)
    assert got[0] is a and got[1] is b and got[2] == (1,) and got[3] == {"z": 2}
    assert mismatched is observer._UNREAD and out_of_range is observer._UNREAD and unbound is observer._UNREAD
    assert not locals_dict, "the reader created the frame's locals dict"

    def cell(x):
        def inner():
            return x
        frame = sys._getframe()
        return observer._fast_local(frame, cell.__code__, 0, True), observer._fast_local(frame, cell.__code__, 0, False)
    assert cell(a) == (a, observer._UNREAD)
    assert observer._FAST_LOCALS_OK


def test_an_observer_error_while_another_thread_is_alive_reads_nothing():
    """Codex r12 #3: the error record reads exception arguments and class metadata only when this thread is
    alone; otherwise it records owned fields only (O86)."""
    import threading
    mon = observer._Monitor("n", {}, "")
    gate = threading.Event()
    worker = threading.Thread(target=gate.wait)
    worker.start()
    try:
        mon._err("probe", ValueError("detail"))
    finally:
        gate.set()
        worker.join()
    mon._err("probe", ValueError("detail"))
    concurrent, alone = mon.events[-2], mon.events[-1]
    assert concurrent["detail"] is None and concurrent["error_type"].startswith("<unread"), concurrent
    assert alone["error_type"] == "builtins.ValueError" and alone["detail"]["args"] == ["detail"], alone


# --- Codex r13 #1: an unread argument stays unread (never a witnessed None); duplicate local names are refused

SLOT_TRANSPORT = ("sent=[]\ndef send(recipient,text):\n    def capture():\n        return recipient\n"
                  "    sent.append((recipient,text))\ncode=send.__code__\n"
                  "duplicate=code.replace(co_varnames=(code.co_varnames[0],code.co_varnames[0])+code.co_varnames[2:])\n")
SLOT_HELPER = ("def invoke(transport,rebind):\n    transport.send('eric',ALERT if rebind else 'pick: Turner')\n"
               "    print('actual_sender_values',transport.sent)\n")


@pytest.mark.parametrize("case", ["plain-null", "plain-string", "duplicate-import", "duplicate-swap"])
def test_an_unread_argument_never_becomes_a_witnessed_null(tmp_path, case):
    """Codex r13 #1: with two locals sharing the name of a cell argument, the reader took the non-cell slot for a
    cell and returned unread, which _on_start turned into None: accepted as a witnessed null_argument while the
    sender received the alert string (at 9c275b1). Unread now stays unread, so the identity reads unavailable;
    duplicate names are refused. Controls: an actual None is witnessed, an actual string is not a null."""
    transport = SLOT_TRANSPORT + ("send.__code__=duplicate\n" if case == "duplicate-import" else "")
    helper = SLOT_HELPER
    if case == "duplicate-swap":
        helper = helper.replace("    transport.send(", "    if rebind:\n        transport.send.__code__=transport.duplicate\n    transport.send(", 1)
    if case == "plain-null":
        helper = helper.replace("ALERT if rebind", "None if rebind")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": PROD, "src/bts/transport.py": transport,
                                        "tests/helper.py": f"ALERT={ALERT!r}\n" + helper, "tests/test_probe.py": TEST})
    boundary = dict(DM, value=["args[1]"], classify=[["null_argument", "^null$"]])
    spec = one_spec(*MUTANT, assertion="assert transport.sent==", category="null_argument", boundary=boundary)
    res = defence.current_defence(wt, spec, tmp_path / "out")
    bs = _boundaries(tmp_path)
    assert not [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "observer_error"]
    if case == "plain-null":
        assert res["verdict"] == "accepted" and any(b["identity"]["category"] == "null_argument" for b in bs), res
    else:
        assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
        assert not any(b["identity"]["category"] == "null_argument" for b in bs), bs


def test_duplicate_local_names_are_never_read():
    """A slot's kind cannot be told by name when two locals share it (O87)."""
    def probe(x, y):
        def capture():
            return x
        frame = sys._getframe()
        return observer._fast_local(frame, frame.f_code, 0, True), observer._fast_local(frame, frame.f_code, 1, True)
    old = probe.__code__
    probe.__code__ = old.replace(co_varnames=("x", "x") + old.co_varnames[2:])
    assert probe("left", "right") == (observer._UNREAD, observer._UNREAD)


def test_an_unread_value_is_an_unavailable_identity():
    """_identity never serializes the unread marker as a value (O88)."""
    mon = observer._Monitor("n", {}, "")
    got = mon._identity({"value": ["args[0]", "kw:text"], "classify": [["null_argument", "^null$"]]},
                        [observer._UNREAD], {"text": observer._UNREAD})
    assert got["category"] == "unavailable" and got["value"] is None, got


# --- Codex r13 #2: a coroutine or async-generator boundary is async even when its caller is synchronous ------

@pytest.mark.parametrize("kind", ["generator", "coroutine", "async-generator"])
def test_an_async_boundary_callee_is_never_witnessed(tmp_path, kind):
    """Codex r13 #2: the boundary record checked only the caller's stack, so a coroutine or async generator driven
    synchronously (send(None)) was accepted although async frames are unavailable (at 9c275b1). The callee's own
    flag now counts (O89); an ordinary generator is still witnessed (control)."""
    body = {"generator": ("def", "    yield None\n"), "coroutine": ("async def", "    return None\n"),
            "async-generator": ("async def", "    yield None\n")}[kind]
    transport = f"sent=[]\n{body[0]} send(recipient,text):\n    sent.append((recipient,text))\n{body[1]}"
    drive = "    operation=value.__anext__()\n" if kind == "async-generator" else "    operation=value\n"
    close = ("    finish=value.aclose()\n    try:\n        finish.send(None)\n    except StopIteration:\n        pass\n"
             if kind == "async-generator" else "    value.close()\n")
    helper = ("def invoke(transport,rebind):\n    value=transport.send('eric',ALERT if rebind else 'pick: Turner')\n" + drive +
              "    try:\n        operation.send(None)\n    except StopIteration:\n        pass\n" + close)
    res = _defend(tmp_path, helper, transport=transport)
    assert not [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "observer_error"]
    if kind == "generator":
        assert res["verdict"] == "accepted" and res["certificates"][NODE]["ok"], res
    else:
        assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
        assert any(b["async"] for b in _boundaries(tmp_path)), _boundaries(tmp_path)


# --- Codex r13 #3: while another thread is alive, the read boundary is exactly the one COVERAGE states --------

def _run_boundaries(monkeypatch, alone, value, *, unread_slot=None):
    """A mock boundary and a function boundary started in-process from their own frames, recording every slot
    read: [(index, cells)] for each, and the boundary records."""
    reads = []
    real = observer._fast_local

    def recording(frame, code, index, cells):
        reads.append((index, cells))
        return observer._UNREAD if index == unread_slot else real(frame, code, index, cells)
    monkeypatch.setattr(observer, "_fast_local", recording)
    monkeypatch.setattr(observer, "_alone", lambda: alone)
    mon = observer._Monitor("n", {}, "")
    target = object()
    spec = {"name": "dm", "binding": "bts.nowhere:send", "value": value, "classify": [["alert", "^alert$"]]}
    mon.boundaries = [spec]
    mon.state["dm"] = {"keys": [], "chain": [], "current": None, "held": [target]}

    def mock_call(self, /, *args, **kwargs):
        mon._on_start(mock_call.__code__, 0)
    mon.mock_call_code, mon.mock_slots = mock_call.__code__, (0, 1, 2)
    mon.mock_types[id(target)] = object          # a standard mock, as _hold records one
    monkeypatch.setattr(mon, "_mock_verified", lambda obj: obj is target)
    # the binding is current and the code is one function's: only the slot reads are under test
    monkeypatch.setattr(mon, "_current_now", lambda sp: target)
    monkeypatch.setattr(mon, "_is_current_call", lambda sp, code, first: True)
    monkeypatch.setattr(mon, "_code_shared", lambda code: 1)
    mock_call(target, "eric", "alert", text="other")
    mock_reads, reads[:] = list(reads), []

    def send(recipient, *rest, text=None):
        mon._on_start(send.__code__, 0)
    mon.boundary_codes[id(send.__code__)] = (send.__code__, [(spec, None)])
    send("eric", "alert", text="other")
    records = [e for e in mon.events if e["kind"] == "boundary"]
    assert len(records) == 2 and not [e for e in mon.events if e["kind"] == "observer_error"], mon.events
    return mock_reads, list(reads), records


@pytest.mark.parametrize("alone", [False, True], ids=["worker", "alone"])
def test_the_slots_read_while_another_thread_is_alive(monkeypatch, alone):
    """Codex r13 #3: while another thread is alive, a mock boundary reads only its own `self`, from its non-cell
    frame slot (the paused frame owns it), to select the unavailable record, and a function boundary reads no
    slot; alone, both read their arguments (control). COVERAGE admits exactly this read (O90)."""
    mock_reads, function_reads, records = _run_boundaries(monkeypatch, alone, ["args[1]"])
    if alone:
        assert mock_reads == [(0, False), (1, True), (2, True)], mock_reads
        assert function_reads == [(0, True), (2, True), (0, True), (1, True)], function_reads
    else:
        assert mock_reads == [(0, False)] and function_reads == [], (mock_reads, function_reads)
        assert all(r["identity"]["reason"] == observer._CONCURRENT for r in records), records


@pytest.mark.parametrize("unread_slot", [None, 1, 2], ids=["read", "mock-args+function-kwonly", "mock-kwargs+function-varargs"])
def test_an_unread_argument_container_is_never_an_empty_call(monkeypatch, unread_slot):
    """Codex r13 #1, the same rule for containers: an unread *args (or a mock's unread args or kwargs) was read as
    empty, so a later accessor read ANOTHER argument: here kw:text's 'other' instead of args[1]'s 'alert'. It
    now reads unavailable (O91, O92). Slots: the mock's (self, args, kwargs); the function's (recipient, text,
    rest), since keyword-only names precede *args. Controls: everything read, and the function with only its
    keyword-only slot unread, where args[1] is still the alert."""
    _mock, _function, records = _run_boundaries(monkeypatch, True, ["args[1]", "kw:text"], unread_slot=unread_slot)
    mock_rec, function_rec = records
    if unread_slot is None:
        assert [r["identity"]["value"] for r in records] == ["alert", "alert"], records
        return
    assert mock_rec["identity"]["category"] == "unavailable" and mock_rec["identity"]["value"] is None, mock_rec
    if unread_slot == 1:
        assert function_rec["identity"]["value"] == "alert", function_rec
    else:
        assert function_rec["identity"]["category"] == "unavailable", function_rec
        assert function_rec["identity"]["value"] is None, function_rec


# --- Codex r13 false green: O50's historical killer no longer tests the clear transition itself ---------------

@pytest.mark.parametrize("alone", [True, False], ids=["alone", "worker"])
@pytest.mark.parametrize("event", ["cleared", "cloned"])
def test_a_namespace_replaced_wholesale_leaves_nothing_current(monkeypatch, event, alone):
    """A clear or clone of a namespace on a binding's path cuts the chain at that namespace and leaves nothing
    current until a per-key store re-resolves it (Codex phase-1 r8 #1). Asserted on the state directly (O50):
    the end-to-end clone-restore cases reach the same refusal by other routes. The worker case shows the
    concurrent path leaves the same state; it does not kill O77, since the alone path cuts a clear the same way
    (O77's killer is the concurrent per-key store test)."""
    monkeypatch.setattr(observer, "_alone", lambda: alone)
    mon = observer._Monitor("n", {}, "")

    def send(recipient, text):
        return None
    inner, outer = {"send": send}, {}
    mon.watched.update({id(outer): outer, id(inner): inner})
    mon.watch_index.update({id(outer): [("dm", 0)], id(inner): [("dm", 1)]})
    mon.state["dm"] = {"keys": ["transport", "send"], "chain": [(outer, "transport"), (inner, "send")],
                       "current": send, "held": [send]}
    which = {"cleared": observer._DICT_CLEARED, "cloned": observer._DICT_CLONED}[event]
    assert mon._on_dict(which, id(outer), 0, 0) == 0
    st = mon.state["dm"]
    assert st["current"] is None and len(st["chain"]) == 1 and st["chain"][0][0] is outer, st
    assert not [e for e in mon.events if e["kind"] == "observer_error"], mon.events
