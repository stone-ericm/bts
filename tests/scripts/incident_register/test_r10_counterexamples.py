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
from tests.scripts.incident_register.test_r6_counterexamples import MUTANT, NODE, PROD, TEST, TRANSPORT, _defend
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
        first = mon._on_start(code, observer._first_resume(code))
        mon._on_start(code, observer._first_resume(code))                                # production: already instrumented
        mon._on_start(code.replace(co_firstlineno=1), observer._first_resume(code))
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
    """_identity never serializes the unread marker as a value (O88), reached positionally and, separately, by
    keyword (Codex r14 false green: the positional case alone never reached the keyword accessor)."""
    mon = observer._Monitor("n", {}, "")
    spec = {"value": ["args[0]", "kw:text"], "classify": [["null_argument", "^null$"]]}
    for args, kwargs in (([observer._UNREAD], {"text": observer._UNREAD}), ([], {"text": observer._UNREAD})):
        got = mon._identity(spec, args, kwargs)
        assert got["category"] == "unavailable" and got["value"] is None, (args, kwargs, got)


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

def _locals_dict(frame):
    """The frame's cached locals dict pointer (NULL until something reads frame.f_locals)."""
    import ctypes
    iframe = ctypes.c_void_p.from_address(id(frame) + observer._PY_FRAME_F_FRAME).value
    return ctypes.c_void_p.from_address(iframe + 5 * ctypes.sizeof(ctypes.c_void_p)).value


def _run_boundaries(monkeypatch, alone, value, *, unread_slot=None, cached=False):
    """A mock boundary and a function boundary started in-process from their own frames, recording every slot
    read: [(index, cells)] for each, and the boundary records. Neither frame's locals dict may exist after the
    observer ran: a read outside _fast_local, through frame.f_locals, creates it (Codex phase-1 r14 #2). With
    ``cached``, the application has already created a now-stale dict, and the observer must not refresh it
    (the r12 finalizer shape)."""
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
        marker = "before"
        stale = sys._getframe().f_locals if cached else None        # the application's own read
        marker = "after"                                            # noqa: F841 - only the frame's slot changes
        mon._on_start(mock_call.__code__, observer._first_resume(mock_call.__code__))
        if cached:
            assert stale["marker"] == "before", "the observer refreshed the mock frame's locals dict"
        else:
            assert not _locals_dict(sys._getframe()), "the observer created the mock frame's locals dict"
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
        marker = "before"
        stale = sys._getframe().f_locals if cached else None
        marker = "after"                                            # noqa: F841
        mon._on_start(send.__code__, observer._first_resume(send.__code__))
        if cached:
            assert stale["marker"] == "before", "the observer refreshed the function frame's locals dict"
        else:
            assert not _locals_dict(sys._getframe()), "the observer created the function frame's locals dict"
    mon.boundary_codes[id(send.__code__)] = (send.__code__, [(spec, None)])
    send("eric", "alert", text="other")
    records = [e for e in mon.events if e["kind"] == "boundary"]
    assert len(records) == 2 and not [e for e in mon.events if e["kind"] == "observer_error"], mon.events
    return mock_reads, list(reads), records


@pytest.mark.parametrize("cached", [False, True], ids=["no-dict", "stale-dict"])
@pytest.mark.parametrize("alone", [False, True], ids=["worker", "alone"])
def test_the_slots_read_while_another_thread_is_alive(monkeypatch, alone, cached):
    """Codex r13 #3: while another thread is alive, a mock boundary calls _fast_local only for its own `self`,
    from its non-cell frame slot (the paused frame owns it), to select the unavailable record, and a function
    boundary calls it for no slot; alone, both read their arguments (control). This pins the _fast_local
    invocation sequence (O90); reads outside that helper are checked separately: the locals dict is never
    created, nor refreshed when the application already holds a stale one (O96, O97; Codex r14 #2). `worker`
    stubs _alone to False; no thread is started."""
    mock_reads, function_reads, records = _run_boundaries(monkeypatch, alone, ["args[1]"], cached=cached)
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


# --- Codex phase-1 r14: with UNIQUE names, a relabel still moves a cell name onto another slot ----------------

RELABEL_TRANSPORT = ("sent=[]\ndef send(recipient,text):\n    def capture():\n        return recipient\n"
                     "    sent.append((recipient,text))\ncode=send.__code__\n"
                     "relabelled=code.replace(co_varnames=('other',code.co_varnames[0])+code.co_varnames[2:])\n")


RELABEL_CASES = {  # case: (code relabelled at import, swapped in the call phase, the argument)
    "plain": (False, False, "ALERT"), "relabelled-import": (True, False, "types.CellType(ALERT)"),
    "relabelled-swap": (False, True, "types.CellType(ALERT)"), "relabelled-string": (True, False, "ALERT"),
    "normal-cell-argument": (False, False, "types.CellType(ALERT)")}


@pytest.mark.parametrize("case", list(RELABEL_CASES))
def test_a_relabelled_cell_name_never_unwraps_a_cell_argument(tmp_path, case):
    """Codex r14 #1: CodeType.replace gave the cell argument's name to the text slot (names unique), so the
    reader took that slot for a cell and unwrapped the application's own CellType argument: its contents, the
    alert, were accepted as the argument, with the code installed at import or swapped in the call phase (at
    e927a75). A slot's kind now comes from the prologue's MAKE_CELL, and the reader refuses code whose names
    disagree with it (O93). Controls: the alert passed plainly is witnessed; the relabelled code with a plain
    string is unread; normal code given a cell argument reads the cell object, which is incomplete."""
    at_import, swap, value = RELABEL_CASES[case]
    transport = RELABEL_TRANSPORT + ("send.__code__=relabelled\n" if at_import else "")
    helper = ("import types\ndef invoke(transport,rebind):\n"
              f"    value={value} if rebind else 'pick: Turner'\n"
              + ("    if rebind:\n        transport.send.__code__=transport.relabelled\n" if swap else "")
              + "    transport.send('eric',value)\n    assert transport.sent[-1][1] is value\n")
    res = _defend(tmp_path, helper, transport=transport)
    assert not [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "observer_error"]
    if case == "plain":
        assert res["verdict"] == "accepted" and res["certificates"][NODE]["ok"], res
    else:
        assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
        assert not any(b["identity"]["category"] == "alert" for b in _boundaries(tmp_path)), _boundaries(tmp_path)


def test_slot_kinds_come_from_the_prologue():
    """The prologue's MAKE_CELL slots decide which slot holds a cell (O93, O94, O95): a relabelled code object
    is never read; an application's own CellType argument in a non-cell slot is that object, never unwrapped; a
    prologue holding any other instruction reads None; a cell past slot 255 is found through EXTENDED_ARG."""
    import opcode
    import types

    def probe(x, y):
        def capture():
            return x
        frame = sys._getframe()
        return observer._fast_local(frame, frame.f_code, 0, True), observer._fast_local(frame, frame.f_code, 1, True)
    cell = types.CellType("right")
    assert probe("left", cell) == ("left", cell) and probe("left", cell)[1] is cell
    original = probe.__code__
    probe.__code__ = original.replace(co_varnames=("other", "x") + original.co_varnames[2:])
    assert probe("left", cell) == (observer._UNREAD, observer._UNREAD)
    assert observer._prologue_cells(original) == {0}
    # another instruction ahead of the real prologue, which keeps its MAKE_CELL and RESUME
    crafted = original.replace(co_code=bytes([opcode.opmap["NOP"], 0, opcode.opmap["LOAD_CONST"], 0]) + original.co_code)
    assert observer._prologue_cells(crafted) is None
    names = [f"a{i}" for i in range(300)]
    ns = {"OBS": observer, "sys": sys}
    exec(f"def wide({', '.join(names)}):\n    def g():\n        return a299\n    frame = sys._getframe()\n"
         "    return OBS._fast_local(frame, frame.f_code, 299, True)\n", ns)
    assert observer._prologue_cells(ns["wide"].__code__) == {299} and ns["wide"](*range(300)) == 299


# --- own review during r14 (cause of Codex r14's observed-only segfault): a Python function watcher clobbers an
# exception pending in C. CPython calls function watchers for EVERY function's creation and destruction; a
# lambda freed after a failed C call fires one with the error set, the ctypes trampoline returns a result with
# an exception set, and CPython replaces the application's exception with SystemError (Cython code then built
# a traceback with no exception set and crashed: pyarrow's ParquetWriter).

EXCEPTION_HELPER = ("def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n"
                    "        return\n    try:\n        sorted([1,'a'],key=lambda x: x)\n    except TypeError:\n"
                    "        return\n    except SystemError:\n        transport.send('eric',ALERT)\n")


def test_observation_never_changes_an_application_exception(tmp_path):
    """The alert is sent only on the SystemError path, which the plain run never takes; the mutant's frozen test
    fails at the same assertion in both twins. At e927a75 the observed run took that path and sent the alert:
    observation changed the application. The certificate was refused only because _on_func recorded the
    converted SystemError as an observer error. Without a function watcher the observed run raises TypeError
    as the plain run does, and nothing is sent or witnessed."""
    res = _defend(tmp_path, EXCEPTION_HELPER)
    observed = (tmp_path / "out/mutant.stdout.txt").read_text()
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    assert not any(b["identity"]["category"] == "alert" for b in _boundaries(tmp_path)), _boundaries(tmp_path)
    assert "SystemError" not in observed, observed


def test_a_running_observer_leaves_a_pending_c_exception_alone():
    """In-process: with the observer started, a C call that fails while its lambda argument is freed still
    raises its own TypeError (at e927a75: SystemError, 'error return without exception set')."""
    mon = observer._Monitor("n", {"boundaries": [dict(DM, binding="os:getcwd")]}, "")
    mon.start()
    try:
        with pytest.raises(TypeError):
            sorted([1, "a"], key=lambda x: x)
    finally:
        mon.stop()


# --- own review during r14: the function watcher's replacement registers a swapped code at its first start ----

@pytest.mark.parametrize("alone", [True, False], ids=["alone", "worker"])
def test_a_swapped_code_is_registered_at_its_first_start_only_while_alone(monkeypatch, alone):
    """A held function's __code__ replaced in place is registered at the new code's first start (O98), and only
    while this thread is alone: with another thread alive the held function is not read and the start is not
    registered, which can only miss the call (O99). `worker` stubs _alone; no thread is started."""
    monkeypatch.setattr(observer, "_alone", lambda: alone)
    mon = observer._Monitor("n", {}, "")
    spec = {"name": "dm", "binding": "bts.nowhere:send", "value": ["args[1]"], "classify": [["alert", "^alert$"]]}
    mon.boundaries = [spec]
    mon.state["dm"] = {"keys": [], "chain": [], "current": None, "held": []}

    def send(recipient, text):
        return None

    def replacement(recipient, text):
        return None
    mon.functions[id(send)] = (send, [(spec, None)])
    send.__code__ = replacement.__code__
    mon._on_start(replacement.__code__, observer._first_resume(replacement.__code__))
    registered = observer._code_key(replacement.__code__) in mon.boundary_codes
    records = [e for e in mon.events if e["kind"] == "boundary"]
    assert not [e for e in mon.events if e["kind"] == "observer_error"], mon.events
    if alone:
        assert registered and len(records) == 1, (registered, records)
    else:
        assert not registered and not records, (registered, records)


def test_the_observer_source_never_reads_frame_locals():
    """Supplementary source guard (Codex r14 #2): no f_locals attribute and no locals-materialising helper
    anywhere in the observer's code; docstrings and comments are not code."""
    import ast
    from pathlib import Path
    banned = {"f_locals", "getargvalues", "getgeneratorlocals", "getcoroutinelocals", "getasyncgenlocals",
              "PyFrame_FastToLocals", "PyFrame_FastToLocalsWithError", "PyFrame_LocalsToFast", "PyFrame_GetLocals",
              "locals", "vars"}
    hits = []
    for node in ast.walk(ast.parse(Path(observer.__file__).read_text())):
        name = (node.attr if isinstance(node, ast.Attribute) else node.id if isinstance(node, ast.Name)
                else node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None)
        if name in banned:
            hits.append((node.lineno, name))
    assert not hits, hits


# --- Codex r15 probe: a later RESUME carrying the start argument fires PY_START again, mid-call ---------------

LATER_TRANSPORT = ("import dis,types\nsent=[]\ndef send(recipient,text):\n    def capture():\n        return text\n"
                   "    sent.append((recipient,text))\n    yield None\n    text='BTS health CRITICAL: x'\n    yield None\n"
                   "code=send.__code__\nraw=bytearray(code.co_code)\n"
                   "last=[i.offset for i in dis.get_instructions(code) if i.opname=='RESUME'][-1]\n"
                   "raw[last+1]=0\nlater_start=code.replace(co_code=bytes(raw))\n")
LATER_DIRECT = ("c=later_start\nsend.__code__=types.CodeType(c.co_argcount,c.co_posonlyargcount,c.co_kwonlyargcount,"
                "c.co_nlocals,c.co_stacksize,c.co_flags,c.co_code,c.co_consts,c.co_names,c.co_varnames,c.co_filename,"
                "c.co_name,c.co_qualname,c.co_firstlineno,c.co_linetable,c.co_exceptiontable,c.co_freevars,"
                "c.co_cellvars)\n")


@pytest.mark.parametrize("case", ["ordinary", "later-import", "later-swap", "direct-constructor"])
def test_a_later_resume_is_never_a_calls_start(tmp_path, case):
    """Codex r15: the generator's later RESUME was given the start argument, so PY_START fired again after the
    body had rebound `text` to the alert, and the observer read that as the call's argument. A start is now
    only a PY_START at the code's first RESUME (O100). The argument actually sent is a CellType, so no case may
    be witnessed; the ordinary generator is the control for that."""
    transport = LATER_TRANSPORT + {"later-import": "send.__code__=later_start\n",
                                   "direct-constructor": LATER_DIRECT}.get(case, "")
    helper = ("import types\ndef invoke(transport,rebind):\n"
              + ("    if rebind:\n        transport.send.__code__=transport.later_start\n" if case == "later-swap" else "")
              + "    value=types.CellType(ALERT) if rebind else 'pick: Turner'\n    operation=transport.send('eric',value)\n"
              "    for _ in range(3):\n        try:\n            next(operation)\n        except StopIteration:\n            break\n"
              "    assert transport.sent[-1][1] is value\n")
    res = _defend(tmp_path, helper, transport=transport)
    assert not [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "observer_error"]
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    assert not any(b["identity"]["category"] == "alert" for b in _boundaries(tmp_path)), _boundaries(tmp_path)


def test_a_start_is_read_only_at_the_first_resume(monkeypatch):
    """In-process: a PY_START reported at any offset but the code's first RESUME reads no slot and records
    nothing (O100); at the first RESUME the same boundary is read (control); a first RESUME without the entry
    argument makes no start of that code readable (O103)."""
    monkeypatch.setattr(observer, "_alone", lambda: True)
    reads = []
    real = observer._fast_local
    monkeypatch.setattr(observer, "_fast_local", lambda f, c, i, cells: reads.append(i) or real(f, c, i, cells))
    mon = observer._Monitor("n", {}, "")
    spec = {"name": "dm", "binding": "bts.nowhere:send", "value": ["args[1]"], "classify": [["alert", "^alert$"]]}
    mon.boundaries = [spec]
    mon.state["dm"] = {"keys": [], "chain": [], "current": None, "held": []}
    starts = []

    def send(recipient, text):
        mon._on_start(send.__code__, starts[-1])
    mon.boundary_codes[observer._code_key(send.__code__)] = (send.__code__, [(spec, None)])
    first = observer._first_resume(send.__code__)
    starts.append(first + 2)
    send("eric", "alert")
    assert reads == [] and not [e for e in mon.events if e["kind"] == "boundary"], (reads, mon.events)
    starts.append(first)
    send("eric", "alert")
    assert reads and len([e for e in mon.events if e["kind"] == "boundary"]) == 1, (reads, mon.events)
    # the first RESUME must carry the entry argument, or no start of that code is read (O103)
    raw = bytearray(send.__code__.co_code)
    raw[first + 1] = 1
    assert observer._first_resume(send.__code__.replace(co_code=bytes(raw))) is None


def test_raw_memory_reads_stay_in_their_two_audited_functions():
    """Supplementary source guard (Codex r15 probe: a raw read of an argument slot, outside _fast_local and before
    the mock gate, created no locals dict and so passed the read-set checks). Raw memory access (`from_address`)
    and the frame-layout offsets appear only in _fast_local; an address is turned into an object
    (`ctypes.cast(..., py_object)`) only in _fast_local and in _on_dict's alone-only path (O101)."""
    import ast
    from pathlib import Path
    allowed = {"from_address": {"_fast_local"}, "_PY_FRAME_F_FRAME": {"_fast_local"},
               "_IFRAME_LOCALSPLUS": {"_fast_local"}, "cast": {"_fast_local", "_on_dict"}}
    tree = ast.parse(Path(observer.__file__).read_text())
    hits = []

    def visit(node, function):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else function
            name = (child.attr if isinstance(child, ast.Attribute) else child.id if isinstance(child, ast.Name) else None)
            if name in allowed and inner is not None and inner not in allowed[name]:
                hits.append((child.lineno, name, inner))
            visit(child, inner)
    visit(tree, None)
    assert not hits, hits


# --- Codex r16 probe: code that jumps back to its own entry RESUME re-fires PY_START at the entry offset -------

REVISIT_TRANSPORT = ("import dis,opcode,types\nsent=[]\nentered=False\ndef send(recipient,text):\n"
                     "    def capture():\n        return text\n    global entered\n    if entered:\n        return\n"
                     "    entered=True\n    sent.append((recipient,text))\n    yield None\n"
                     "    text='BTS health CRITICAL: x'\n    yield None\n"
                     "code=send.__code__\nraw=bytearray(code.co_code)\n"
                     "resumes=[i.offset for i in dis.get_instructions(code) if i.opname=='RESUME']\n"
                     "first,last=resumes[0],resumes[-1]\n"
                     "raw[last:last+4]=bytes([opcode.opmap['POP_TOP'],0,opcode.opmap['JUMP_BACKWARD'],(last+4-first)//2])\n"
                     "revisit=code.replace(co_code=bytes(raw))\n")


@pytest.mark.parametrize("case", ["ordinary", "revisit-import", "revisit-swap", "direct-constructor", "noncell-revisit"])
def test_a_jump_back_to_the_entry_is_never_a_second_start(tmp_path, case):
    """Codex r16: after the body rebinds `text`, a jump back to the entry RESUME fires PY_START at the entry offset
    itself, which the offset check alone accepted. Code whose jumps or handlers lead to its entry is never read
    (O104, O105). The argument sent is a CellType, so nothing may be witnessed; ordinary code is the control."""
    transport = REVISIT_TRANSPORT
    if case == "noncell-revisit":
        transport = transport.replace("    def capture():\n        return text\n", "")
    transport += {"revisit-import": "send.__code__=revisit\n", "noncell-revisit": "send.__code__=revisit\n",
                  "direct-constructor": LATER_DIRECT.replace("later_start", "revisit")}.get(case, "")
    helper = ("import types\ndef invoke(transport,rebind):\n"
              + ("    if rebind:\n        transport.send.__code__=transport.revisit\n" if case == "revisit-swap" else "")
              + "    value=types.CellType(ALERT) if rebind else 'pick: Turner'\n    operation=transport.send('eric',value)\n"
              "    for _ in range(3):\n        try:\n            next(operation)\n        except StopIteration:\n            break\n"
              "    assert transport.sent[-1][1] is value\n")
    res = _defend(tmp_path, helper, transport=transport)
    assert not [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "observer_error"]
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    assert not any(b["identity"]["category"] == "alert" for b in _boundaries(tmp_path)), _boundaries(tmp_path)


def test_a_plain_function_that_jumps_back_to_its_entry_is_never_read(tmp_path):
    """Codex r16: the same shape without a generator: the final return becomes a jump back to the entry RESUME."""
    transport = ("import dis,opcode,types\nsent=[]\nentered=False\ndef send(recipient,text):\n    global entered\n"
                 "    if entered:\n        return\n    entered=True\n    sent.append((recipient,text))\n"
                 "    text='BTS health CRITICAL: x'\ncode=send.__code__\nraw=bytearray(code.co_code)\n"
                 "first=next(i.offset for i in dis.get_instructions(code) if i.opname=='RESUME')\n"
                 "last=list(dis.get_instructions(code))[-1].offset\n"
                 "raw[last:last+2]=bytes([opcode.opmap['JUMP_BACKWARD'],(last+2-first)//2])\n"
                 "send.__code__=code.replace(co_code=bytes(raw))\n")
    helper = ("import types\ndef invoke(transport,rebind):\n    value=types.CellType(ALERT) if rebind else 'pick: Turner'\n"
              "    transport.send('eric',value)\n    assert transport.sent[-1][1] is value\n")
    res = _defend(tmp_path, helper, transport=transport)
    assert not [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "observer_error"]
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    assert not any(b["identity"]["category"] == "alert" for b in _boundaries(tmp_path)), _boundaries(tmp_path)


def test_an_entry_that_a_jump_or_handler_targets_is_unreadable():
    """Unit: compiled code has a readable entry (control); a jump back to the entry (O104) or an exception handler
    targeting it (O105) makes the entry None, so no start of that code is read."""
    import dis
    import opcode

    def f(x):
        def capture():
            return x
        return x
    code = f.__code__
    first = observer._first_resume(code)
    assert first is not None
    raw = bytearray(code.co_code)
    last = list(dis.get_instructions(code))[-1].offset
    raw[last:last + 2] = bytes([opcode.opmap["JUMP_BACKWARD"], (last + 2 - first) // 2])
    assert observer._first_resume(code.replace(co_code=bytes(raw))) is None
    # one handler entry (start marked by bit 7): covering the instruction after the entry, handled at the entry
    table = bytes([0x80 | (first // 2 + 1), 1, first // 2, 0])
    assert observer._first_resume(code.replace(co_exceptiontable=table)) is None


def test_a_frame_started_before_observation_never_becomes_a_witness(tmp_path):
    """Codex r16: a generator created and advanced before the call phase (so its body already rebound `text`) is
    resumed during it and jumps back to its entry RESUME: the entry start fired inside the interval with the
    changed local. Code whose jumps lead to its entry is never read (O104)."""
    transport = (REVISIT_TRANSPORT + "send.__code__=revisit\noriginal=types.CellType('initial cell')\n"
                 "saved=send('eric',original)\nnext(saved)\nnext(saved)\n")
    helper = ("def invoke(transport,rebind):\n    if rebind:\n        try:\n            next(transport.saved)\n"
              "        except StopIteration:\n            pass\n    assert transport.sent[-1][1] is transport.original\n")
    test = ("from bts import mod,transport\ndef test_probe():\n    result=mod.deliver()\n"
            "    assert transport.sent==[('eric',transport.original)] and result=='done'\n")
    res = _defend(tmp_path, helper, transport=transport, test=test)
    assert not [e for e in runner.load(tmp_path / "out/mutant.events.jsonl") if e["kind"] == "observer_error"]
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    assert not any(b["identity"]["category"] == "alert" for b in _boundaries(tmp_path)), _boundaries(tmp_path)


# --- Codex r16 part 2 probes: application tracing, f_lineno jumps, and code lifetimes -------------------------

TRACE_TRANSPORT = ("sent=[]\nentered=False\ndef send(recipient,text):\n    global entered\n    if entered:\n        return\n"
                   "    entered=True\n    sent.append((recipient,text))\n    text='BTS health CRITICAL: x'\n    return None\n")


def _trace_helper(mode, value="types.CellType(ALERT)"):
    """A helper whose sys.settrace function either only watches (`none`) or, at the boundary's last line, sets
    f_lineno back to its def line, re-running the entry RESUME after `text` was rebound (`jump`)."""
    return ("import sys,types\ndef invoke(transport,rebind):\n"
            f"    value={value} if rebind else 'pick: Turner'\n    jumped=[]\n    target=transport.send.__code__\n"
            "    def trace(frame,event,arg):\n        if frame.f_code is target:\n"
            f"            if {mode!r}=='jump' and event=='line' and frame.f_lineno==target.co_firstlineno+7 and not jumped:\n"
            "                frame.f_lineno=target.co_firstlineno\n                jumped.append(True)\n"
            "            return trace\n        return None\n    sys.settrace(trace)\n    try:\n"
            "        transport.send('eric',value)\n    finally:\n        sys.settrace(None)\n"
            "    assert transport.sent[-1][1] is value\n    print('jumped',jumped)\n")


@pytest.mark.parametrize("mode", ["none", "jump"])
def test_an_application_tracer_during_the_call_phase_refuses_the_interval(tmp_path, mode):
    """Codex r16: a tracer setting f_lineno to the def line re-ran the entry RESUME, and the changed local was
    certified (`jump`, at 6b63b61); and the tracer receives 'call' events for the observer's dict-watcher callback,
    so application code ran inside the observer (`none`). An application trace, profile or monitoring function
    installed during the call phase now refuses the interval (O107, O108); a refusal, never a witness."""
    res = _defend(tmp_path, _trace_helper(mode), transport=TRACE_TRANSPORT)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    assert any("install(s) during the call phase" in r for r in res["certificates"][NODE].get("why", []) + res["reasons"]), res


def test_a_tracer_with_a_real_alert_is_refused_not_accepted(tmp_path):
    """The same tracer around a real alert: the alert is genuine, but the interval ran application code inside
    the observer, so it is refused (a missed certificate, never an impure one)."""
    res = _defend(tmp_path, _trace_helper("none", value="ALERT"), transport=TRACE_TRANSPORT)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res


MONITORING_HELPER = ("import sys\ndef invoke(transport,rebind):\n    if rebind:\n        m=sys.monitoring\n"
                     "        tool=next(i for i in range(6) if m.get_tool(i) is None)   # the observer holds its own id\n"
                     "        m.use_tool_id(tool,'application')\n        m.register_callback(tool,m.events.PY_START,lambda c,o:None)\n"
                     "        m.set_events(tool,m.events.PY_START)\n        try:\n            transport.send('eric',ALERT)\n"
                     "        finally:\n            m.set_events(tool,0)\n            m.register_callback(tool,m.events.PY_START,None)\n"
                     "            m.free_tool_id(tool)\n        return\n    transport.send('eric','pick: Turner')\n")


def test_an_application_monitoring_tool_refuses_the_interval(tmp_path):
    """An application sys.monitoring tool can set f_lineno from its own callbacks too; one registered during the
    call phase refuses the interval (O106 keeps the observer's own registrations out of that count)."""
    res = _defend(tmp_path, MONITORING_HELPER)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res
    assert any("during the call phase" in r for r in res["certificates"][NODE].get("why", []) + res["reasons"]), res


def test_a_tracer_active_before_the_call_phase_refuses_the_interval(tmp_path):
    """A trace function installed at import (before the call phase) and still active at its start: obs_start
    records it, and the interval is refused."""
    transport = TRANSPORT + "import sys\ndef _watch(frame,event,arg):\n    return None\nsys.settrace(_watch)\n"
    res = _defend(tmp_path, "def invoke(transport,rebind):\n    transport.send('eric',ALERT if rebind else 'pick: Turner')\n",
                  transport=transport)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res


CONSTANT_HELPER = ("import types\nreleased=[]\nclass Constant:\n    def __del__(self):\n        released.append('released')\n"
                   "def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n        return\n"
                   "    constant=Constant()\n    template=compile('lambda: None','<qa-code-constant>','eval').co_consts[0]\n"
                   "    code=template.replace(co_consts=template.co_consts+(constant,))\n    fn=types.FunctionType(code,{})\n"
                   "    del constant\n    fn()\n    del fn,code\n    print('constant_released',bool(released))\n"
                   "    transport.send('eric',None if released else ALERT)\n")


def test_observation_never_keeps_an_application_code_object_alive(tmp_path):
    """Codex r16: the entry cache held every started code object, so a code constant's finalizer did not run
    when observed (it did unobserved) and the application sent the alert only when observed. The cache holds
    weak references (O109)."""
    res = _defend(tmp_path, CONSTANT_HELPER)
    observed = (tmp_path / "out/mutant.stdout.txt").read_text()
    plain = (tmp_path / "out/mutant_unobserved.stdout.txt").read_text()
    assert "constant_released True" in observed and "constant_released True" in plain, (observed, plain)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res


def test_a_malformed_exception_table_makes_the_entry_unreadable():
    """Codex r16 probe: a table that ends mid-entry was read as having no handler (O108); an entry without its start
    marker is not read either (O109). Control: a well-formed table whose handler is after the entry."""
    def f(x):
        try:
            return x
        except ValueError:
            return None
    code = f.__code__
    assert observer._first_resume(code) == 0
    assert observer._first_resume(code.replace(co_exceptiontable=bytes([0x80, 1, 0x40]))) is None
    assert observer._first_resume(code.replace(co_exceptiontable=bytes([0x01, 1, 2, 0]))) is None


def test_purity_reports_a_trace_function_the_census_could_not_count(monkeypatch):
    """A trace function installed before the trusted bootstrap (by site initialisation or a reviewed .pth hook)
    is invisible to the census; _purity still reports it from sys.gettrace, so the interval is refused (O106,
    which C31's install count otherwise masks)."""
    import sys as _sys
    monkeypatch.setattr(observer, "_AUDIT_CENSUS", {"hooks_added": 0, "tracers_installed": 0, "own": []})
    mon = observer._Monitor("n", {}, "")
    assert mon._purity()["tracing"] is False
    previous = _sys.gettrace()
    _sys.settrace(lambda frame, event, arg: None)
    try:
        traced = mon._purity()
    finally:
        _sys.settrace(previous)
    assert traced["tracing"] is True and traced["tracers_installed"] == 0, traced


# --- Codex r17: lifetime and weak-reference side effects --------------------------------------------------------

def test_the_entry_cache_never_answers_for_another_code_at_a_reused_address():
    """r17 #2 replaced the weak cache with a digest of the code's own bytes, which owns no application object. Two
    code shapes of equal size alternate at a reused address (measured: every iteration); an id-keyed cache would
    answer for the freed one (O107)."""
    import opcode

    def f(x):
        return x
    template = f.__code__
    first = bytes([opcode.opmap["RESUME"], 0, opcode.opmap["LOAD_FAST"], 0, opcode.opmap["RETURN_VALUE"], 0])
    second = bytes([opcode.opmap["NOP"], 0, opcode.opmap["RESUME"], 0, opcode.opmap["RETURN_VALUE"], 0])
    reused, last = 0, None
    for i in range(200):
        code = template.replace(co_code=first if i % 2 == 0 else second)
        reused += last is not None and id(code) == last
        assert observer._first_resume(code) == (0 if i % 2 == 0 else 2), i
        last = id(code)
        del code
    assert reused, "no address was reused: the test would not distinguish an id-keyed cache"


WEAKCOUNT_HELPER = ("import types,weakref\ndef invoke(transport,rebind):\n    if not rebind:\n"
                    "        transport.send('eric','pick: Turner')\n        return\n"
                    "    template=compile('lambda: None','<qa-weakcount>','eval').co_consts[0]\n"
                    "    code=template.replace(co_filename='<qa-weakcount-live>')\n    fn=types.FunctionType(code,{})\n"
                    "    before=weakref.getweakrefcount(code)\n    fn()\n    after=weakref.getweakrefcount(code)\n"
                    "    print('weakref_counts',before,after)\n    transport.send('eric',ALERT if after!=before else None)\n")


def test_observation_never_changes_an_application_code_weak_reference_count(tmp_path):
    """Codex r17 #2: the weak entry cache attached a weak reference to the application's own code, so
    weakref.getweakrefcount changed 0 -> 1 only when observed, and the alert the application then sent was certified.
    The digest cache attaches nothing: both twins see 0 -> 0, and nothing is witnessed."""
    res = _defend(tmp_path, WEAKCOUNT_HELPER)
    observed = (tmp_path / "out/mutant.stdout.txt").read_text()
    plain = (tmp_path / "out/mutant_unobserved.stdout.txt").read_text()
    assert "weakref_counts 0 0" in observed and "weakref_counts 0 0" in plain, (observed, plain)
    assert res["verdict"] == "rejected" and not res["certificates"][NODE]["ok"], res


def _screen_labels(tmp_path, source):
    """The closure screen's categories for one helper file holding ``source``."""
    import importlib.util
    from pathlib import Path
    tool = Path(observer.__file__).parents[3] / "docs/audit/2026-09-29-incident-register-evidence/tooling/closure_screen.py"
    spec = importlib.util.spec_from_file_location("closure_screen_under_test", tool)
    screen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(screen)
    (tmp_path / "src/bts").mkdir(parents=True)
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/helper.py").write_text(source)
    return {hit[0] for hit in screen.screen(tmp_path)}


def test_the_closure_screen_lists_lifetime_dependent_application_code(tmp_path):
    """Proposed ruling 13 puts behaviour that depends on object lifetimes outside the model (Codex r17 #1: the
    observer keeps a retired boundary's code, and a finalizer on its constant ran unobserved only). The closure
    screen lists such code in a closure, so a prepared spec's reliance on it is visible."""
    labels = _screen_labels(tmp_path, "import weakref\nclass Payload:\n    def __del__(self):\n        pass\n"
                                      "def count(code):\n    return weakref.getweakrefcount(code)\n")
    assert {"finalizer or exit hook", "lifetime introspection"} <= labels, labels


@pytest.mark.parametrize("source,label", [
    ("import weakref\nvalue = weakref.proxy(object())\n", "lifetime introspection"),
    ("import weakref as wr\nvalue = wr.getweakrefcount(object())\n", "lifetime introspection"),
    ("from weakref import getweakrefcount as count\nvalue = count(object())\n", "lifetime introspection"),
    ("import weakref\nvalue = getattr(weakref, 'getweak' + 'refcount')(object())\n", "lifetime introspection"),
    ("import gc\nvalue = gc.is_finalized(object())\n", "lifetime introspection"),
    ("import sys\nslots = [sys.monitoring.get_tool(i) for i in range(6)]\n", "instrumentation introspection"),
    ("from sys import monitoring as m\nslots = m.get_events(0)\n", "instrumentation introspection"),
    ("import dis as d\nops = list(d.get_instructions(f.__code__, adaptive=True))\n", "instrumentation introspection"),
    ("raw = f.__code__._co_code_adaptive\n", "instrumentation introspection"),
    ("import sys\nframes = sys._current_frames()\n", "instrumentation introspection"),
    ("import time\nstarted = time.monotonic()\n", "elapsed time or resource use"),
    ("from time import perf_counter as clock\nstarted = clock()\n", "elapsed time or resource use"),
    ("key = id(object())\n", "object address"),
], ids=["proxy", "module-alias", "function-alias", "computed-name", "gc-query", "monitoring-registry",
        "monitoring-alias", "adaptive-code", "adaptive-bytes", "other-threads-frames", "elapsed", "elapsed-alias",
        "address"])
def test_the_closure_screen_flags_code_that_can_see_the_observer(tmp_path, source, label):
    """Codex r18 #3 measured seven misses in nine of its sources; the screen now flags a named module at its import,
    so module and function aliases and a computed attribute name on an imported module are listed, and r18 #1-#2's
    instrumentation reads have their own category. A name generated without importing the module, and a dependency
    outside src/bts and tests, are still not flagged: the screen is a warning list, and each prepared closure has an
    independent source review (proposed ruling 13)."""
    labels = _screen_labels(tmp_path, source)
    assert label in labels, labels


# --- Codex r18: code that inspects the interpreter can see the observer -----------------------------------------

REGISTRY_HELPER = ("import sys\ndef invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n"
                   "        return\n    occupied=any(sys.monitoring.get_tool(i) is not None for i in range(6))\n"
                   "    print('observation_visible',occupied)\n    transport.send('eric',ALERT if occupied else None)\n")
ADAPTIVE_HELPER = ("import dis\ndef invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n"
                   "        return\n    code=transport.send.__code__\n    transport.send('eric','pick: Turner')\n"
                   "    instrumented=any(i.opname.startswith('INSTRUMENTED_') for i in dis.get_instructions(code,adaptive=True))\n"
                   "    print('observation_visible',instrumented)\n    transport.send('eric',ALERT if instrumented else None)\n")
RETIRED_HELPER = ("import types\nreleased=[]\nclass Payload:\n    def __del__(self):\n        released.append(True)\n"
                  "def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n        return\n"
                  "    original=transport.send\n    payload=Payload()\n"
                  "    code=original.__code__.replace(co_consts=original.__code__.co_consts+(payload,))\n"
                  "    candidate=types.FunctionType(code,original.__globals__)\n    transport.send=candidate\n    del payload\n"
                  "    candidate('eric','pick: Turner')\n"
                  "    transport.send=types.FunctionType(original.__code__.replace(),original.__globals__)\n"
                  "    del candidate,code\n    print('observation_visible',not released)\n"
                  "    transport.send('eric',ALERT if not released else None)\n")


@pytest.mark.parametrize("helper", [REGISTRY_HELPER, ADAPTIVE_HELPER, RETIRED_HELPER],
                         ids=["monitoring-registry", "adaptive-code", "retired-boundary-constant"])
def test_an_alert_sent_only_when_observed_is_refused(tmp_path, helper):
    """Codex r18 #1 (the monitoring-tool registry), r18 #2 (adaptive disassembly of the boundary's own code) and r17 #1
    (the observer keeps a retired boundary's code, so its constant's finalizer runs only unobserved): the application
    sees the observer and sends the alert only when observed. Both twins failed the frozen assertion at the same line,
    so the certificate was accepted. The twins' failure messages must now agree too (D18), and here they differ, so
    the interval is refused. These shapes stay outside the model (proposed ruling 13): the comparison is a backstop on
    normalized message text, not semantic equality and not a proof that observation is invisible."""
    res = _defend(tmp_path, helper)
    observed = (tmp_path / "out/mutant.stdout.txt").read_text()
    plain = (tmp_path / "out/mutant_unobserved.stdout.txt").read_text()
    assert "observation_visible True" in observed and "observation_visible False" in plain, (observed, plain)
    assert res["verdict"] == "rejected", res
    assert any("failed with a different message observed and unobserved" in r for r in res["reasons"]), res["reasons"]


RUN_VARIANT_TEST = ("from unittest.mock import MagicMock\nfrom bts import mod,transport\ndef test_probe(tmp_path):\n"
                    "    shown=(MagicMock(name='shown'),object(),str(tmp_path))\n    result=mod.deliver()\n"
                    "    assert transport.sent==[('eric','pick: Turner')],shown\n    assert result=='done'  # ASSERT-RESULT\n")


def test_a_real_alert_whose_message_shows_run_variant_values_is_still_certified(tmp_path):
    """The cost side of D18. A genuine alert's failing assertion shows a mock's id, an object's address and the
    tmp_path, whose directory pytest numbers per session: all three differ between ANY two runs, observed or not.
    Their repr-shaped forms are blanked before the twins' messages are compared (O111-O113), so the certificate is accepted. Measured on
    Codex r18's prepared runs: without the blanking, I-0830-a (a tmp_path) and I-0830-c (a mock's id) are refused."""
    res = _defend(tmp_path, "def invoke(transport,rebind):\n    transport.send('eric',ALERT if rebind else 'pick: Turner')\n",
                  test=RUN_VARIANT_TEST)
    assert res["verdict"] == "accepted", res


@pytest.mark.parametrize("shown", [("'status=0x111'", "'status=0x222'"), ("\"business id='111'\"", "\"business id='222'\""),
                                   ("'/business/pytest-111/item'", "'/business/pytest-222/item'")],
                         ids=["hex-text", "id-text", "pytest-text"])
def test_application_text_shaped_like_run_metadata_still_counts(tmp_path, shown):
    """Codex r19 #1: the blanking matched ANY hex token, quoted decimal id or /pytest-N/ path component, so an alert sent
    only when observed (the monitoring registry) was accepted when the frozen assertion's message showed application
    text of those shapes that differed between the twins. Only repr-shaped run metadata is blanked now (O111-O113), so
    the messages differ and the interval is refused."""
    observed, plain = shown
    test = ('"""PYTEST_DONT_REWRITE"""\nfrom bts import mod,transport\ndef test_probe():\n    result=mod.deliver()\n'
            "    assert transport.sent==[('eric','pick: Turner')], (" + observed + " if transport.sent[-1][1] is not None else "
            + plain + ")\n    assert result=='done'\n")
    res = _defend(tmp_path, REGISTRY_HELPER, test=test)
    assert "observation_visible True" in (tmp_path / "out/mutant.stdout.txt").read_text()
    assert res["verdict"] == "rejected", res
    assert any("failed with a different message observed and unobserved" in r for r in res["reasons"]), res["reasons"]


def test_the_message_digest_blanks_run_metadata_shapes_not_application_text():
    """Blanked: a default repr's address, the id in a mock's repr, pytest's /pytest-of-<user>/pytest-N/ number. Kept:
    the same characters anywhere else (Codex r19 #1's three shapes), and any other difference."""
    digest = observer._message_digest
    assert digest("x <object object at 0x10a2f> <code object f at 0x1, file 'a', line 1> <MagicMock name='m' id='4415867120'>"
                  " <NonCallableMagicMock spec='S' id='1'> /T/pytest-of-u/pytest-2033/t0") == \
        digest("x <object object at 0x2ffe0> <code object f at 0x2, file 'a', line 1> <MagicMock name='m' id='4477438192'>"
               " <NonCallableMagicMock spec='S' id='2'> /T/pytest-of-u/pytest-2034/t0")
    for a, b in [("status=0x111", "status=0x222"), ("business id='111'", "business id='222'"),
                 ("/business/pytest-111/item", "/business/pytest-222/item"), ("<thing at 0x1 here>", "<thing at 0x2 here>")]:
        assert digest(a) != digest(b), (a, b)
    assert digest("send('eric', 'BTS health CRITICAL: x')") != digest("send('eric', None)")
    assert digest(type("S", (str,), {})("x")) is None
    assert digest(None) is None


def _twin(digest):
    base = {"kind": "report", "nodeid": "tests/t.py::n", "wasxfail": None}
    return [dict(base, when="setup", outcome="passed"),
            dict(base, when="call", outcome="failed", exc_module="builtins", exc_qualname="AssertionError",
                 frames=[], message="AssertionError: x", message_sha256=digest),
            dict(base, when="teardown", outcome="passed")]


@pytest.mark.parametrize("observed,plain,refused", [("a", "a", False), ("a", "b", True), (None, None, True)],
                         ids=["same", "different", "unrecorded"])
def test_twins_must_fail_with_the_same_recorded_message(monkeypatch, tmp_path, observed, plain, refused):
    """A failed node whose observed and unobserved messages differ is refused (D18); so is one whose message was not
    recorded, which is never read as agreement (D19)."""
    monkeypatch.setattr(runner, "run", lambda *a, **k: runner.Run("mutant_unobserved", 1, _twin(plain), {}, tmp_path / "e"))
    monkeypatch.setattr(runner, "gate", lambda *a, **k: [])
    seen = runner.Run("mutant", 1, _twin(observed), {}, tmp_path / "e")
    _, why = defence._conformance(str(tmp_path), seen, tmp_path / "out", [], {}, "mutant_unobserved", "mutant",
                                  ["tests/t.py::n"], [])
    assert bool(why) is refused, why
