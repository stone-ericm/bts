"""Codex phase-1 r6 counterexamples, each pinned in the direction the design requires.

Adapted from the reviewer's retained probes (``test_review_r6.py``). Each of them measured a FALSE
certificate or a broken observational guarantee at cd9456b.
"""
import importlib.util
import marshal
import struct
import subprocess
import sys
import types
from pathlib import Path

import pytest

from scripts.audit.incident_register import defence, deploy_runs, observer, owned
from tests.scripts.incident_register.synth import DM, defended_project
from tests.scripts.incident_register.test_r4_counterexamples import one_spec

NODE = "tests/test_probe.py::test_probe"
PROD = ("from bts import transport\nfrom tests.helper import invoke\ndef deliver():\n"
        "    invoke(transport,False)  # BRANCH\n    return 'done'\n")
TRANSPORT = ("sent=[]\ndef send(recipient,text):\n    sent.append((recipient,text))\n"
             "def replacement(recipient,text):\n    sent.append((recipient,text))\n")
TEST = ("from bts import mod,transport\ndef test_probe():\n    result=mod.deliver()\n"
        "    assert transport.sent==[('eric','pick: Turner')]\n    assert result=='done'  # ASSERT-RESULT\n")
MUTANT = ("invoke(transport,False)  # BRANCH\n    return 'done'", "invoke(transport,True)  # BRANCH\n    return 'defer'")


def _defend(tmp_path, helper, *, transport=TRANSPORT, test=TEST, boundaries=None):
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": PROD, "src/bts/transport.py": transport,
                                        "tests/helper.py": helper, "tests/test_probe.py": test})
    spec = one_spec(*MUTANT)
    if boundaries:
        spec["boundaries"] = boundaries
    return defence.current_defence(wt, spec, tmp_path / "out")


def _refused(res, reason):
    cert = res["certificates"][NODE]
    assert res["verdict"] == "rejected" and not cert["ok"], res
    assert any(reason in r for r in cert["reasons"]), cert["reasons"]
    return cert


# --- finding 1: the namespace root, attribute dispatch and type changes ---------------------------

SYS_MODULES_SWAP = """import sys,types,importlib
def invoke(transport,rebind):
    if not rebind:
        transport.send('eric','pick: Turner')
        return
    old=sys.modules
    alternate=types.ModuleType('bts.transport')
    alternate.send=transport.replacement
    sys.modules=dict(old)
    try:
        sys.modules['bts.transport']=alternate
        imported=importlib.import_module('bts.transport')
        assert imported is alternate
        list(map(imported.send,['eric'],['pick: Turner']))
    finally:
        sys.modules=old
"""


def test_a_temporary_sys_modules_replacement_is_seen(tmp_path):
    """r6 #1 measured: a helper swapped sys.modules for a copy, imported an alternate module from it and
    called its sender through map(); the watchers stayed on the old root and absence was accepted."""
    _refused(_defend(tmp_path, SYS_MODULES_SWAP), "1 qualifying 'dm' call(s) occurred")


def test_a_persistent_sys_modules_replacement_is_seen(tmp_path):
    """r6 #1 (the end-guard probe): the copied root stays in place until fixture teardown."""
    helper = SYS_MODULES_SWAP.replace("    finally:\n        sys.modules=old\n", "    finally:\n        pass\n")
    test = ("import sys,pytest\nfrom bts import mod,transport\nimport tests.helper\n"
            "_saved=sys.modules\n@pytest.fixture(autouse=True)\ndef cleanup():\n    yield\n    sys.modules=_saved\n" + TEST[TEST.index("def test_probe"):])
    _refused(_defend(tmp_path, helper, test=test), "1 qualifying 'dm' call(s) occurred")


def test_a_module_with_overridden_attribute_dispatch_is_a_coverage_gap(tmp_path):
    """r6 #1 measured: a ModuleType subclass returned the replacement from __getattribute__ while its raw
    'send' slot never changed. Only exact ModuleType (and exact type for classes) have the attribute
    dispatch that reading the raw namespace describes."""
    transport = TRANSPORT + ("import sys,types\nclass DynamicModule(types.ModuleType):\n"
                             "    def __getattribute__(self,name):\n"
                             "        if name=='send' and types.ModuleType.__getattribute__(self,'alternate'):\n"
                             "            return types.ModuleType.__getattribute__(self,'replacement')\n"
                             "        return types.ModuleType.__getattribute__(self,name)\n"
                             "module=sys.modules[__name__]\nmodule.__class__=DynamicModule\nmodule.alternate=False\n")
    helper = ("def invoke(transport,rebind):\n    transport.alternate=rebind\n    try:\n"
              "        list(map(transport.send,['eric'],['pick: Turner']))\n    finally:\n        transport.alternate=False\n")
    _refused(_defend(tmp_path, helper, transport=transport), "coverage gap (unsupported namespace")


def test_a_transient_module_type_change_is_a_coverage_gap(tmp_path):
    """r6 #1 measured: the module is exact ModuleType at both ends, but its __class__ is swapped to an
    overriding subclass for the call. The path objects' exact types are re-checked at every function
    start while boundaries are observed, so the swap is seen when the override runs."""
    transport = TRANSPORT + ("import types\nclass DynamicModule(types.ModuleType):\n"
                             "    def __getattribute__(self,name):\n        if name=='send':\n"
                             "            return types.ModuleType.__getattribute__(self,'replacement')\n"
                             "        return types.ModuleType.__getattribute__(self,name)\n")
    helper = ("def invoke(transport,rebind):\n    original=type(transport)\n    try:\n        if rebind:\n"
              "            transport.__class__=transport.DynamicModule\n"
              "        list(map(transport.send,['eric'],['pick: Turner']))\n    finally:\n        transport.__class__=original\n")
    _refused(_defend(tmp_path, helper, transport=transport), "coverage gap (attribute dispatch changed")


def test_a_class_with_a_custom_metaclass_is_a_coverage_gap(tmp_path):
    """r6 #1 measured: the same override through a metaclass's __getattribute__."""
    transport = TRANSPORT + ("class Meta(type):\n    def __getattribute__(cls,name):\n"
                             "        if name=='send' and type.__getattribute__(cls,'alternate'):\n"
                             "            return replacement\n        return type.__getattribute__(cls,name)\n"
                             "class API(metaclass=Meta):\n    send=send\n    alternate=False\n")
    helper = ("def invoke(transport,rebind):\n    transport.API.alternate=rebind\n    try:\n"
              "        list(map(transport.API.send,['eric'],['pick: Turner']))\n    finally:\n        transport.API.alternate=False\n")
    _refused(_defend(tmp_path, helper, transport=transport, boundaries=[dict(DM, binding="bts.transport:API.send")]),
             "coverage gap (unsupported namespace")


def test_every_function_start_is_kept_while_boundaries_are_observed():
    """The dispatch re-check runs at function starts, so while boundaries are observed no code may be
    disabled after its first start: here f's SECOND start is the first one after the swap."""
    mod = types.ModuleType("w15_r7_dispatch")
    mod.send = lambda recipient, text: None

    class Loud(types.ModuleType):
        pass

    def f():
        return 1

    sys.modules[mod.__name__] = mod
    mon = observer._Monitor("n", {"boundaries": [dict(DM, binding="w15_r7_dispatch:send")]}, "")
    try:
        mon.start()
        f()
        mod.__class__ = Loud
        f()
        mod.__class__ = types.ModuleType
    finally:
        mon.stop()
        del sys.modules["w15_r7_dispatch"]
    assert any(e["kind"] == "boundary_gap" and e["reason"].startswith("attribute dispatch changed")
               for e in mon.events), mon.events


# --- finding 2: a code object does not identify its function --------------------------------------

def test_a_function_sharing_the_senders_code_is_not_the_sender(tmp_path):
    """r6 #2 measured: a FunctionType clone of send (same code, other globals) ran; the declared sender
    did not, yet the event was attributed to it and an event certificate accepted."""
    transport = ("import types\nsent=[]\nunrelated=[]\ndef send(recipient,text):\n    sent.append((recipient,text))\n"
                 "clone=types.FunctionType(send.__code__,{'sent':unrelated})\n")
    prod = "from bts import transport\ndef deliver():\n    return 'done'  # BRANCH\n"
    test = ("from bts import mod,transport\ndef test_probe():\n    result=mod.deliver()\n    assert transport.sent==[]\n"
            "    assert len(transport.unrelated)==(0 if result=='done' else 1)\n    assert result=='done'  # ASSERT-RESULT\n")
    _, wt = defended_project(tmp_path, {"src/bts/mod.py": prod, "src/bts/transport.py": transport, "tests/test_probe.py": test})
    spec = one_spec("    return 'done'  # BRANCH", "    transport.clone('eric','pick: Turner')  # BRANCH\n    return 'defer'",
                    kind="event")
    _refused(defence.current_defence(wt, spec, tmp_path / "out"), "no 'pick' event after the branch")


# --- finding 4: callables are recognised by exact type, never by name ------------------------------

def _monitor_of(binding):
    return observer._Monitor("n", {"boundaries": [dict(DM, binding=binding)]}, "")


def test_a_spoofed_function_type_name_runs_no_application_code():
    """r6 #4 measured: a class labelled 'builtins.function' had its __code__ property run inside the real
    dict-watcher callback. FunctionType/MethodType are now recognised by identity."""
    calls = []

    class Fake:
        @property
        def __code__(self):
            calls.append("application __code__")
            return (lambda: None).__code__

    Fake.__module__, Fake.__qualname__ = "builtins", "function"
    mod = types.ModuleType("w15_r7_spoof")
    mod.send = lambda recipient, text: None
    sys.modules[mod.__name__] = mod
    mon = _monitor_of("w15_r7_spoof:send")
    try:
        mon.start()
        mod.send = Fake()
    finally:
        mon.stop()
        del sys.modules["w15_r7_spoof"]
    assert calls == []
    assert any(e["kind"] == "boundary_gap" and e["reason"] == "not a Python-observable callable" for e in mon.events)


# --- finding 5: every live thread, not only threading's registered ones ---------------------------

def test_a_low_level_worker_started_in_the_interval_makes_absence_unavailable(tmp_path):
    """r6 #5 measured: a _thread worker (never registered with threading) was live at the interval's end
    and sent afterwards; the census counted zero threads and absence was accepted."""
    helper = ("import _thread,threading\nrelease=threading.Event()\nfinished=threading.Event()\nstarted=threading.Event()\n"
              "def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n        return\n"
              "    def work():\n        started.set()\n        release.wait()\n        transport.send('eric','pick: Turner')\n"
              "        finished.set()\n    _thread.start_new_thread(work,())\n    assert started.wait(5)\n")
    test = ("import pytest\nfrom bts import mod,transport\nfrom tests import helper\n@pytest.fixture(autouse=True)\n"
            "def cleanup():\n    yield\n    helper.release.set()\n    if helper.started.is_set():\n        assert helper.finished.wait(5)\n"
            "    assert transport.sent==[('eric','pick: Turner')]\ndef test_probe():\n    result=mod.deliver()\n"
            "    assert result=='done'  # ASSERT-RESULT\n")
    _refused(_defend(tmp_path, helper, test=test), "thread(s) started in the interval were still alive")


def test_a_pre_existing_low_level_worker_is_counted():
    import _thread
    import threading
    started, release, done = threading.Event(), threading.Event(), threading.Event()

    def work():
        started.set()
        release.wait()
        done.set()

    _thread.start_new_thread(work, ())
    assert started.wait(5)
    mon = observer._Monitor("n", {}, "")
    try:
        mon.start()
    finally:
        mon.stop()
        release.set()
        assert done.wait(5)
    end = [e for e in mon.events if e["kind"] == "obs_end"][0]
    assert end["preexisting_threads_alive"] >= 1, end


# --- finding 6: a failed start releases everything it acquired ------------------------------------

def test_a_failed_start_releases_its_watchers_and_monitoring_id(monkeypatch):
    """r6 #6 measured: start() raised after taking monitoring id 4 and both CPython watchers, and stop()
    returned at once because the monitor was not yet active."""
    def fail(_binding):
        raise RuntimeError("start failed")

    monkeypatch.setattr(observer, "_resolve_chain", fail)
    mon = _monitor_of("os:getcwd")
    try:
        with pytest.raises(RuntimeError, match="start failed"):
            mon.start()
        try:
            mon.stop()
        except BaseException as e:  # noqa: BLE001
            raise AssertionError(f"stop() must never raise: {e!r}")
        assert mon.dict_watcher is None and mon.func_watcher is None
        assert sys.monitoring.get_tool(observer.TOOL_ID) is None
    finally:
        # whatever a broken stop() leaked is released here, AFTER the assertions: a CPython watcher left
        # installed outlives this monitor's ctypes callback and crashes the whole session later
        mon._remove_watchers()
        if sys.monitoring.get_tool(observer.TOOL_ID) is not None:
            sys.monitoring.free_tool_id(observer.TOOL_ID)


# --- finding 3: the reviewed hook's code is what runs ---------------------------------------------

def _venv_with_reviewed_hook(tmp_path, wt, extra_code):
    """A synthetic venv with uv's reviewed _virtualenv source and a VALID cache of other code."""
    site = next((wt / ".venv/lib").glob("python*/site-packages"))
    real = next(Path(sys.prefix).glob("lib/python*/site-packages/_virtualenv.py"), None)
    if real is None or owned._sha(real.read_bytes()) not in owned.REVIEWED_PTH_IMPORTS["_virtualenv"]:
        pytest.skip("the reviewed _virtualenv hook is not in this interpreter's venv")
    hook = site / "_virtualenv.py"
    hook.write_bytes(real.read_bytes())
    (site / "_virtualenv.pth").write_text("import _virtualenv\n")
    cache = Path(importlib.util.cache_from_source(str(hook)))
    cache.parent.mkdir(exist_ok=True)
    st = hook.stat()
    cache.write_bytes(importlib.util.MAGIC_NUMBER + struct.pack("<III", 0, int(st.st_mtime), st.st_size)
                      + marshal.dumps(compile(extra_code, str(hook), "exec")))
    return site, cache


def test_a_cached_form_of_a_reviewed_hook_is_refused_until_purged(tmp_path):
    """r6 #3 measured: the reviewed source was hashed, but Python ran a valid cache of OTHER code that put
    an external root on sys.path; the fingerprint never moved. A cache of a reviewed hook is refused;
    owned.reset purges it (evidence runs never write bytecode), and then the reviewed source runs."""
    _, wt = defended_project(tmp_path)
    marker = tmp_path / "cache-executed.txt"
    site, cache = _venv_with_reviewed_hook(tmp_path, wt, f"open({str(marker)!r},'w').write('unreviewed cache ran')\n")
    cmd = [str(wt / ".venv/bin/python"), "-B", "-c", "pass"]
    subprocess.check_call(cmd, cwd=wt)
    assert marker.exists()                                           # the cache really is what runs
    with pytest.raises(owned.ClosureRefused, match="cached"):
        owned.venv_fingerprint(wt)
    marker.unlink()
    owned.reset(wt, "HEAD")                                          # purges the hook's cache
    assert not cache.exists()
    owned.venv_fingerprint(wt)
    subprocess.check_call(cmd, cwd=wt)
    assert not marker.exists()                                       # the reviewed source ran instead


def test_a_pair_over_a_cached_hook_is_refused(tmp_path):
    """r6 #3 measured: a complete pair was accepted while the cached hook's external dependency changed."""
    from scripts.audit.incident_register import acceptance
    from tests.scripts.incident_register.test_expected_failure import GOOD, REGISTRY, setup_project
    _, wt = setup_project(tmp_path, GOOD)
    external = tmp_path / "external"
    external.mkdir()
    (external / "local_dep.py").write_text("VALUE=1\n")
    _venv_with_reviewed_hook(tmp_path, wt, f"import sys\nsys.path.insert(0,{str(external)!r})\n")
    with pytest.raises(owned.ClosureRefused, match="cached"):
        acceptance.run_pair(wt, REGISTRY, ["tests/test_incident.py", "-q"], tmp_path / "out")


def test_another_importable_form_of_a_reviewed_hook_is_refused(tmp_path):
    """An extension module or package of the hook's name would be imported instead of the reviewed .py."""
    _, wt = defended_project(tmp_path)
    site, cache = _venv_with_reviewed_hook(tmp_path, wt, "pass\n")
    cache.unlink()
    (site / "_virtualenv.cpython-312-darwin.so").write_bytes(b"\x00")
    with pytest.raises(owned.ClosureRefused, match="another importable form"):
        owned.venv_fingerprint(wt)


# --- finding 7: precision through every consumer ---------------------------------------------------

def _obs_run(i, ts, sha):
    return {"run_id": i, "created_at": ts, "log": "retained", "pre_sha": None, "deployed_sha": sha, "deployed_at": ts,
            "rollback": "none"}


def test_a_non_adjacent_overlapping_disagreement_is_refused():
    """r6 #7.1 measured: A at 00Z (the whole second) and B at .500Z overlap, but an A at .100Z between
    them shielded the pair from an adjacent-only check."""
    with pytest.raises(ValueError, match="cannot be ordered"):
        deploy_runs.observations([_obs_run(1, "2026-07-01T00:00:00Z", "aaaaaaa"),
                                  _obs_run(2, "2026-07-01T00:00:00.100Z", "aaaaaaa"),
                                  _obs_run(3, "2026-07-01T00:00:00.500Z", "bbbbbbb")])


def test_live_at_inside_an_observations_precision_is_not_observed():
    """r6 #7.2 measured: a deployed line stamped 00:00:01Z happened somewhere in [01, 02); live_at at its
    floor claimed 'observed'."""
    run = {"run_id": 1, "created_at": "2026-07-01T00:00:00Z", "log": "retained", "pre_sha": "aaaaaaa",
           "pre_at": "2026-07-01T00:00:00Z", "deployed_sha": "bbbbbbb", "deployed_at": "2026-07-01T00:00:01Z",
           "rollback": "none", "canary": "passed"}
    timeline = deploy_runs.installed_timeline([run])
    got = deploy_runs.live_at(timeline, "2026-07-01T00:00:01Z")
    assert got["basis"] != "observed" and got["sha"] is None, got
    assert deploy_runs.live_at(timeline, "2026-07-01T00:00:02Z")["sha"] == "bbbbbbb"      # after the unit ends
