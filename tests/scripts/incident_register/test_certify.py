"""certify() on hand-built observer events: linking, ordering, file identity (Codex phase-1 r2 #5)."""
from scripts.audit.incident_register.certify import certify

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


def window(*events):
    return [ev("obs_start", 0), *events, ev("obs_end", 1000, pending_identity=[])]


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
    got = certify([ev("obs_start", 0), entry(1, 10), branch(2, IN), call(3, "alert", IN)],
                  node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"] and any("observation" in r for r in got["reasons"])


def test_unresolved_identity_is_rejected():
    events = [ev("obs_start", 0), entry(1, 10), branch(2, IN), call(3, "alert", IN), ev("obs_end", 9, pending_identity=[3])]
    got = certify(events, node=N, kind="event", entry=ENTRY, bad=BAD)
    assert not got["ok"] and any("unresolved" in r for r in got["reasons"])


def test_return_kind_links_the_returning_invocation():
    ret = ev("return", 3, file=F, qualname="deliver", frame=10, value="'miss'", type="str", stack=stack(IN))
    bad = {"file": F, "qualname": "deliver", "value": "'miss'"}
    assert certify(window(entry(1, 10), branch(2, IN), ret), node=N, kind="return", entry=ENTRY, bad=bad)["ok"]
    wrong = dict(ret, value="'void'")
    assert not certify(window(entry(1, 10), branch(2, IN), wrong), node=N, kind="return", entry=ENTRY, bad=bad)["ok"]


def test_absence_needs_zero_qualifying_calls_and_a_positive_control():
    q = {"boundary": "dm", "category": "pick"}
    base = window(entry(1, 10), call(2, "pick", IN))
    ok = certify(window(entry(1, 10), branch(2, IN), call(3, "alert", IN)), node=N, kind="absence", entry=ENTRY,
                 bad=q, positive_events=base)
    assert ok["ok"], ok
    unidentified = call(3, "pick", IN)
    unidentified["identity"] = None
    got = certify(window(entry(1, 10), branch(2, IN), unidentified), node=N, kind="absence", entry=ENTRY,
                  bad=q, positive_events=base)
    assert not got["ok"] and any("without identity" in r for r in got["reasons"])
