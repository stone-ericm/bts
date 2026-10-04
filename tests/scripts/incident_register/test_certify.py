"""certify() on hand-built observer events: linking, ordering, file identity (Codex phase-1 r2 #5);
purity (r7 #4); absence refused (plan ruling 10)."""
import pytest

from scripts.audit.incident_register.certify import ABSENCE_REFUSAL, AbsenceRefused, certify

N = "tests/test_x.py::test_y"
F = "/wt/src/bts/mod.py"
OTHER = "/wt/src/bts/impostor.py"
ENTRY = {"file": F, "qualname": "deliver"}
BAD = {"boundary": "dm", "category": "alert"}


def ev(kind, seq, **kw):
    return {"kind": kind, "seq": seq, "node": N, "thread": 1, "pid": 1, "async": False, **kw}


def entry(seq, frame, file=F):
    return ev("entry", seq, file=file, qualname="deliver", frame=frame,
              stack=[["deliver", file, 4, frame], ["test_y", "/wt/tests/test_x.py", 3, 900]])


def stack(*frames):
    return [[q, f, 1, fid] for q, f, fid in frames] + [["test_y", "/wt/tests/test_x.py", 3, 900]]


def branch(seq, *frames):
    return ev("branch", seq, file=F, line=6, stack=stack(*frames))


def call(seq, cat, *frames):
    return ev("boundary", seq, name="dm", caller=["deliver", F, 7], stack=stack(*frames),
              identity={"category": cat, "value": "'x'", "sha256": "0"})


def exit_(seq, frame, how="return"):
    return ev("entry_exit", seq, file=F, qualname="deliver", frame=frame, how=how)


PURE = {"audit_hooks_added": 0, "gc_enabled": False, "signal_handlers": [], "tracing": False, "tracers_installed": 0}


def window(*events, start=PURE, end=PURE):
    return [ev("obs_start", 0, purity=start), *events, ev("obs_end", 1000, purity=end)]


IN = ("deliver", F, 10)


def test_linked_event_is_certified():
    got = certify(window(entry(1, 10), branch(2, IN), call(3, "alert", IN)), node=N, kind="event", entry=ENTRY, bad=BAD)
    assert got["ok"], got


def test_same_name_entry_in_another_file_does_not_count():
    imp = ("deliver", OTHER, 10)
    got = certify(window(entry(1, 10, file=OTHER), branch(2, imp), call(3, "alert", imp)),
                  node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"] and any("never invoked" in r for r in got["reasons"])


def test_branch_outside_the_invocation_is_rejected():
    got = certify(window(entry(1, 10), branch(2, ("helper", F, 55)), call(3, "alert", IN)),
                  node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"] and any("never executed inside" in r for r in got["reasons"])


def test_event_before_the_branch_is_rejected():
    got = certify(window(entry(1, 10), call(2, "alert", IN), branch(3, IN)), node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"]


def test_reused_frame_id_splits_invocations():
    """Frame ids of dead frames are reused: the branch belongs to invocation 1, the event to 2."""
    events = window(entry(1, 10), branch(2, IN), entry(5, 10), call(6, "alert", IN))
    got = certify(events, node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"], got


def test_wrong_category_is_not_the_event():
    got = certify(window(entry(1, 10), branch(2, IN), call(3, "pick", IN)), node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"]


def test_incomplete_window_is_rejected():
    got = certify([ev("obs_start", 0, purity=PURE), entry(1, 10), branch(2, IN), call(3, "alert", IN)],
                  node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"] and any("observation" in r for r in got["reasons"])


def test_unidentified_boundary_call_is_never_the_event():
    unidentified = call(3, "alert", IN)
    unidentified["identity"] = {"value": None, "category": "unavailable"}
    got = certify(window(entry(1, 10), branch(2, IN), unidentified), node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"]


def test_return_kind_links_the_returning_invocation():
    ret = ev("return", 3, file=F, qualname="deliver", frame=10, value="'miss'", type="str", stack=stack(IN))
    bad = {"file": F, "qualname": "deliver", "value": "'miss'"}
    assert certify(window(entry(1, 10), branch(2, IN), ret), node=N, kind="return", entry=ENTRY, bad=bad)["ok"]
    wrong = dict(ret, value="'void'")
    assert not certify(window(entry(1, 10), branch(2, IN), wrong), node=N, kind="return", entry=ENTRY, bad=bad)["ok"]


def test_return_kind_by_category():
    ret = ev("return", 3, file=F, qualname="deliver", frame=10, value="PickLockState(locked=True, reason='x')",
             category="locked", type="PickLockState", stack=stack(IN))
    bad = {"file": F, "qualname": "deliver", "category": "locked"}
    assert certify(window(entry(1, 10), branch(2, IN), ret), node=N, kind="return", entry=ENTRY, bad=bad)["ok"]
    assert not certify(window(entry(1, 10), branch(2, IN), dict(ret, category="other")), node=N, kind="return",
                       entry=ENTRY, bad=bad)["ok"]


Q = {"boundary": "dm", "category": "pick"}
LINKED = window(entry(1, 10), branch(2, IN), call(3, "alert", IN), exit_(4, 10))


@pytest.mark.parametrize("events", [
    LINKED,                                                              # the old accepted absence shape
    window(entry(1, 10), branch(2, IN)),                                 # the invocation never completed
    window(entry(1, 10), branch(2, IN), ev("boundary_gap", 3, name="dm", reason="x"), exit_(4, 10)),
    window(entry(1, 10), branch(2, IN), exit_(3, 10), call(4, "pick", ("test_y", "/wt/tests/test_x.py", 3))),
], ids=["zero-calls", "incomplete", "gap", "qualifying-call"])
def test_absence_is_refused_whatever_its_events(events):
    """Plan ruling 10: Phase 1 certifies no missing event, however complete the recorded interval looks.
    The refusal is exactly ``AbsenceRefused`` with the report's reason: any other exception (the unknown-kind
    ``ValueError`` below it, sweep C14 at ``c468430``) or a result fails the assertion, not the call."""
    try:
        got = certify(events, node=N, kind="absence", entry=ENTRY, bad=Q)
    except Exception as e:  # noqa: BLE001 - the exact refusal is the contract
        got = e
    assert type(got) is AbsenceRefused and str(got) == ABSENCE_REFUSAL


def test_an_unknown_kind_is_an_error():
    with pytest.raises(ValueError):
        certify(LINKED, node=N, kind="missing", entry=ENTRY, bad=BAD)


LINKED_EVENT = (entry(1, 10), branch(2, IN), call(3, "alert", IN))


@pytest.mark.parametrize("where", ["start", "end"])
@pytest.mark.parametrize("purity,needle", [
    (None, "purity not recorded"),
    (dict(PURE, audit_hooks_added=None), "no audit-hook census"),
    (dict(PURE, audit_hooks_added=1), "1 audit hook(s) added after the trusted bootstrap"),
    (dict(PURE, gc_enabled=True), "automatic garbage collection on"),
    (dict(PURE, gc_enabled=None), "automatic garbage collection on"),
    (dict(PURE, signal_handlers=[15]), "application signal handler(s)"),
    (dict(PURE, tracing=True), "an application trace, profile or monitoring function"),
    (dict(PURE, tracing=None), "an application trace, profile or monitoring function"),
    (dict(PURE, tracers_installed=None), "no trace/profile/monitoring install census"),
], ids=["missing", "no-census", "hook", "gc-on", "gc-unknown", "signal", "tracing", "tracing-unknown", "no-install-census"])
def test_an_impure_observation_certifies_nothing(purity, needle, where):
    """Codex phase-1 r7 #4: application code the observation itself could run (an audit hook, a finalizer
    in a collection, a signal handler) makes the interval unavailable, at either end."""
    events = window(*LINKED_EVENT, **{where: purity})
    got = certify(events, node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"] and any(needle in r for r in got["reasons"]), got["reasons"]


def test_a_tracer_installed_inside_the_call_phase_certifies_nothing():
    """Codex phase-1 r16: a trace, profile or monitoring function installed and removed again inside the call phase
    is gone at both ends; the bootstrap's install count differs between them, so the interval is unavailable."""
    events = window(*LINKED_EVENT, start=dict(PURE, tracers_installed=0), end=dict(PURE, tracers_installed=2))
    got = certify(events, node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"] and any("2 trace, profile or monitoring install(s) during the call phase" in r
                                 for r in got["reasons"]), got["reasons"]


def test_a_tracer_installed_before_the_call_phase_certifies_nothing():
    """Codex phase-1 r16: a hook installed earlier, on a thread whose trace state the endpoint checks cannot read,
    may still run inside observer callbacks; any install since the trusted bootstrap refuses the interval."""
    events = window(*LINKED_EVENT, start=dict(PURE, tracers_installed=1), end=dict(PURE, tracers_installed=1))
    got = certify(events, node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"] and any("1 trace, profile or monitoring install(s) before the call phase" in r
                                 for r in got["reasons"]), got["reasons"]


def test_a_pure_observation_is_certified():
    assert certify(window(*LINKED_EVENT), node=N, kind="event", entry=ENTRY, bad=BAD)["ok"]


def test_recursion_links_through_the_common_outer_invocation():
    """r3 #7: the outer invocation executes the branch; a nested invocation sends."""
    INNER = ("deliver", F, 11)
    events = window(entry(1, 10), branch(2, IN), entry(3, 11), call(4, "alert", INNER, IN))
    assert certify(events, node=N, kind="event", entry=ENTRY, bad=BAD)["ok"]


def test_exit_then_frame_reuse_splits_invocations():
    events = window(entry(1, 10), branch(2, IN), exit_(3, 10), entry(4, 10), call(5, "alert", IN))
    assert not certify(events, node=N, kind="event", entry=ENTRY, bad=BAD)["ok"]
