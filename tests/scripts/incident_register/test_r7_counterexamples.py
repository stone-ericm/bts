"""Codex phase-1 r7 counterexamples, and the narrowing that answers them (plan ruling 10).

Adapted from the reviewer's retained probes (``test_review_r7.py``, ``audit_cost.py``, ``stop_audit.py``).
At 1c6c183 probes #1-#4 each obtained a FALSE absence certificate and #5 made ``stop()`` raise. Since plan
ruling 10 no absence is certified: each probe is refused as an absence and also runs as an event, where
its mechanism must not create a witness. A call the recorder cannot see (fallback or inherited lookup,
dispatch changed between two Python starts) is missed, never false; a mock whose ``__call__`` is not the
standard one is unattributed; and an application audit hook makes the whole observation impure.
"""
import json
import signal
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

from scripts.audit.incident_register import certify, observer, runner
from tests.scripts.incident_register.synth import DM, defended_project
from tests.scripts.incident_register.test_r4_counterexamples import ALERT
from tests.scripts.incident_register.test_r6_counterexamples import TRANSPORT, _defend, _refused

BUILD = Path(__file__).resolve().parents[3]


def _mutant_events(tmp_path):
    return runner.load(tmp_path / "out" / "mutant.events.jsonl")


# --- findings 1 and 2: lookup the raw namespace does not describe ---------------------------------

def test_an_exact_modules_getattr_fallback_is_missed_not_false(tmp_path):
    """r7 #1 measured: deleting the watched entry exposed a module ``__getattr__`` fallback; the send
    through it was unrecorded and absence was accepted."""
    transport = TRANSPORT + "def __getattr__(name):\n    if name=='send':\n        return replacement\n    raise AttributeError(name)\n"
    helper = ("def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n        return\n"
              "    saved=transport.send\n    del transport.send\n    try:\n        list(map(transport.send,['eric'],[ALERT]))\n"
              "    finally:\n        transport.send=saved\n")
    _refused(_defend(tmp_path, helper, transport=transport), "no 'alert' event")


def test_an_exact_classs_inherited_attribute_is_missed_not_false(tmp_path):
    """r7 #1 measured: deleting a class's own attribute exposed an inherited sender."""
    transport = TRANSPORT + "class Base:\n    send=replacement\nclass API(Base):\n    send=send\n"
    helper = ("def invoke(transport,rebind):\n    if not rebind:\n        transport.API.send('eric','pick: Turner')\n        return\n"
              "    saved=transport.API.send\n    del transport.API.send\n    try:\n        list(map(transport.API.send,['eric'],[ALERT]))\n"
              "    finally:\n        transport.API.send=saved\n")
    _refused(_defend(tmp_path, helper, transport=transport, boundaries=[dict(DM, binding="bts.transport:API.send")]),
             "no 'alert' event")


def test_dispatch_changed_between_two_python_starts_is_missed_not_false(tmp_path):
    """r7 #2 measured: a C-implemented ``__getattribute__`` swapped in and out with no Python start in
    between; the captured replacement's send was unrecorded and absence was accepted."""
    transport = TRANSPORT + ("import types,functools\nclass DynamicModule(types.ModuleType):\n"
                             "    __getattribute__=functools.partial(dict.__getitem__,{'send':replacement})\n")
    helper = ("def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n        return\n"
              "    original=type(transport)\n    transport.__class__=transport.DynamicModule\n    callback=transport.send\n"
              "    transport.__class__=original\n    list(map(callback,['eric'],[ALERT]))\n")
    _refused(_defend(tmp_path, helper, transport=transport), "no 'alert' event")


# --- finding 3: a mock's effective __call__ ---------------------------------------------------------

CUSTOM_MOCKS = {
    # r7 #3 as measured: the override calls a real sender, so the mock records nothing
    "sends-elsewhere": "    def __call__(self,recipient,text):\n        replacement(recipient,text)\n",
    # the event direction: the override hands the standard code other arguments than the call's
    "rewrites-arguments": "    def __call__(self,recipient,text):\n        return super().__call__(recipient,'BTS health CRITICAL: x')\n",
}


@pytest.mark.parametrize("body", list(CUSTOM_MOCKS.values()), ids=list(CUSTOM_MOCKS))
def test_a_mock_whose_call_was_overridden_before_observation_witnesses_nothing(tmp_path, body):
    transport = TRANSPORT + "from unittest.mock import Mock\nclass Custom(Mock):\n" + body + "custom=Custom()\n"
    helper = ("def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n        return\n"
              "    transport.send=transport.custom\n    transport.send('eric','pick: Turner')\n")
    _refused(_defend(tmp_path, helper, transport=transport), "no 'alert' event")
    events = _mutant_events(tmp_path)
    assert any(e["kind"] == "boundary_gap" and "effective __call__ is not the standard one" in e["reason"] for e in events)
    for e in events:                                   # a recorded call of it is never attributed
        if e["kind"] == "boundary":
            assert e["identity"]["category"] == "unavailable" or e["identity"]["value"] != ALERT, e


def _monitor(binding):
    return observer._Monitor("n", {"boundaries": [dict(DM, binding=binding)]}, "")


def _calls(mon):
    return [e["identity"] for e in mon.events if e["kind"] == "boundary"]


@pytest.fixture
def module():
    from unittest.mock import MagicMock
    mod = types.ModuleType("w15_r8_mock")
    mod.send = MagicMock()
    sys.modules[mod.__name__] = mod
    yield mod
    del sys.modules[mod.__name__]


def test_a_standard_mock_call_is_attributed(module):
    """Control: patch()'s MagicMock (the shape the real fixtures use) keeps the standard ``__call__``."""
    mon = _monitor("w15_r8_mock:send")
    try:
        mon.start()
        module.send("eric", ALERT)
    finally:
        mon.stop()
    assert [c["category"] for c in _calls(mon)] == ["alert"]


def test_a_held_mocks_class_changed_after_it_was_held_is_unattributed(module):
    mon = _monitor("w15_r8_mock:send")
    other = type("Other", (type(module.send),), {})
    try:
        mon.start()
        object.__dict__["__class__"].__set__(module.send, other)   # a mock's own __class__ is a property
        module.send("eric", ALERT)
    finally:
        mon.stop()
    calls = _calls(mon)
    assert len(calls) == 1 and calls[0]["category"] == "unavailable", calls
    assert "effective __call__" in calls[0]["reason"]


def test_a_call_override_stored_after_the_mock_was_held_is_unattributed(module):
    """The later-store case (O21's gap) now also withholds attribution at the call itself."""
    mon = _monitor("w15_r8_mock:send")
    cls = type(module.send)
    standard = cls.__mro__[1].__call__
    try:
        mon.start()
        cls.__call__ = lambda self, recipient, text: standard(self, recipient, "BTS health CRITICAL: x")
        module.send("eric", "pick: Turner")
    finally:
        mon.stop()
        del cls.__call__
    calls = _calls(mon)
    assert [c["category"] for c in calls] == ["unavailable"], calls


# --- finding 4: observer purity ---------------------------------------------------------------------

def test_an_application_audit_hook_makes_the_observation_impure(tmp_path):
    """r7 #4 measured: the observer's own audited primitives ran an application audit hook, which sent a
    pick inside the monitoring callback, unrecorded. Now any audit hook added after the trusted
    bootstrap makes the interval unavailable: here even the genuinely recorded alert is not certified."""
    transport = TRANSPORT + ("import sys\narmed=False\ndef audit(event,args):\n"
                             "    if event=='gc.get_referrers' and armed:\n        send('eric','pick: Turner')\n"
                             "sys.addaudithook(audit)\n")
    helper = ("def invoke(transport,rebind):\n    if not rebind:\n        transport.send('eric','pick: Turner')\n        return\n"
              "    transport.armed=True\n    try:\n        transport.send('eric',ALERT)\n    finally:\n        transport.armed=False\n")
    res = _defend(tmp_path, helper, transport=transport)
    cert = _refused(res, "audit hook(s) added after the trusted bootstrap")
    assert cert["linked"] is not None                     # the alert WAS linked: only purity refuses it


def test_the_observers_own_primitives_run_application_audit_hooks():
    """r7 #4's direct control (``audit_cost.py``), in its own interpreter: why the census exists."""
    script = textwrap.dedent(f"""
        import sys, types, json
        sys.path.insert(0, {str(BUILD)!r})
        from scripts.audit.incident_register import observer
        from tests.scripts.incident_register.synth import DM
        mod = types.ModuleType('r8_audit')
        def send(recipient, text): pass
        mod.send = send
        sys.modules[mod.__name__] = mod
        seen = set()
        armed = [False]
        sys.addaudithook(lambda event, args: seen.add(event) if armed[0] else None)
        mon = observer._Monitor('n', {{'boundaries': [dict(DM, binding='r8_audit:send')]}}, '')
        mon.start()
        armed[0] = True
        mod.send('eric', 'pick: Turner')
        armed[0] = False
        mon.stop()
        print(json.dumps(sorted(seen)))
    """)
    seen = json.loads(subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True).stdout)
    assert "gc.get_referrers" in seen and "builtins.id" in seen, seen


def test_stop_never_raises_under_a_raising_audit_hook():
    """r7 #5 measured: an audit hook raising on the final census made stop() raise. The census is gone;
    every audited step left in stop() is guarded, and the interval ends impure (no census here)."""
    script = textwrap.dedent(f"""
        import sys, types, json
        sys.path.insert(0, {str(BUILD)!r})
        from scripts.audit.incident_register import observer, certify
        from tests.scripts.incident_register.synth import DM
        class API:
            def send(recipient, text): pass
        mod = types.ModuleType('r8_stop')
        mod.API = API
        sys.modules[mod.__name__] = mod
        armed = [False]
        def audit(event, args):
            if armed[0]:
                raise RuntimeError('application audit hook refuses ' + event)
        sys.addaudithook(audit)
        mon = observer._Monitor('n', {{'boundaries': [dict(DM, binding='r8_stop:API.send')]}}, '')
        mon.start()
        armed[0] = True
        try:
            mon.stop()
            out = 'returned'
        except BaseException as e:
            out = 'raised ' + type(e).__name__
        armed[0] = False
        for i in range(2000):                      # the watchers' callbacks must be gone, not dangling
            API.__dict__.get('x'); mod.__dict__[f'k{{i}}'] = i
        kinds = [e['kind'] for e in mon.events]
        print(json.dumps({{'stop': out, 'tool': sys.monitoring.get_tool(observer.TOOL_ID),
                           'watchers': [mon.dict_watcher, mon.func_watcher], 'kinds': kinds,
                           'why': certify.interval(mon.events, 'n')[1]}}))
    """)
    got = json.loads(subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True).stdout)
    assert got["stop"] == "returned" and got["tool"] is None and got["watchers"] == [None, None], got
    assert got["kinds"][0] == "obs_start" and got["kinds"][-1] == "obs_end", got
    assert "observer_error" in got["kinds"], got            # the refused end check is recorded, not raised
    assert got["why"], got


def test_automatic_collection_is_off_for_an_observed_or_quiesced_node_and_back_on_after(project_gc, tmp_path):
    """Automatic collection is off during the call phase of an observed node and of its observer-off twin
    (``quiesce``), and back on for the next node, so the twins differ only by observation."""
    wt = project_gc
    nodes = ["tests/test_gc.py::test_inside"]
    observe = {"nodes": nodes, "entries": [], "boundaries": [], "returns": []}
    for kw in ({"observe": observe}, {"quiesce": nodes}):
        r = runner.run(wt, ["tests/test_gc.py", "-q"], tmp_path / "o", "g", **kw)
        assert {runner.node_state(r.events, n) for n in runner.collected(r.events)} == {"passed"}, r.events
    plain = runner.run(wt, ["tests/test_gc.py", "-q"], tmp_path / "o", "plain")
    assert runner.node_state(plain.events, nodes[0]) == "failed"


@pytest.fixture
def project_gc(tmp_path):
    tests = ("import gc\n\ndef test_inside():\n    assert not gc.isenabled()\n\n"
             "def test_zafter():\n    assert gc.isenabled()\n")
    return defended_project(tmp_path, {"tests/test_gc.py": tests})[1]


def test_collection_switched_back_on_or_a_signal_handler_is_recorded_as_impure():
    mon = observer._Monitor("n", {}, "")
    gc_was = observer.gc.isenabled()
    previous = signal.getsignal(signal.SIGUSR1)
    observer.gc.disable()
    try:
        mon.start()
        observer.gc.enable()
        signal.signal(signal.SIGUSR1, lambda *a: None)
    finally:
        mon.stop()
        signal.signal(signal.SIGUSR1, previous)
        if not gc_was:
            observer.gc.disable()
    start, end = (e["purity"] for e in mon.events if e["kind"] in ("obs_start", "obs_end"))
    assert start["gc_enabled"] is False and start["signal_handlers"] == []
    assert end["gc_enabled"] is True and end["signal_handlers"] == [int(signal.SIGUSR1)]
    why = certify.interval(mon.events, "n")[1]
    assert any("garbage collection on at obs_end" in w for w in why) and any("signal handler" in w for w in why), why


def test_the_observer_off_twin_runs_with_collection_off_too(tmp_path):
    """The conformance twin quiesces the same killing nodes, so a node whose behaviour depends on the
    collector's state behaves alike observed and unobserved (else the twins would disagree for a reason
    that is not observation)."""
    from tests.scripts.incident_register.test_defence import run as defence_run
    extra = {"tests/test_gcstate.py": (
        "import gc\nfrom unittest.mock import patch\nfrom bts import mod\n\n\ndef test_gc_sensitive():\n"
        "    with patch('bts.transport.send') as send:\n        mod.run(True)\n    assert not gc.isenabled()\n"
        "    assert [c.args[1] for c in send.call_args_list] == ['pick: Turner']  # ASSERT-GC\n")}
    project = defended_project(tmp_path, extra=extra)
    res = defence_run(project, tmp_path, tests=["tests/test_mod.py", "tests/test_gcstate.py", "-q"],
                      killing=[{"node": "tests/test_gcstate.py::test_gc_sensitive",
                                "assertion": {"path": "tests/test_gcstate.py", "text": "# ASSERT-GC"}}])
    assert res["verdict"] == "accepted", res["reasons"]
