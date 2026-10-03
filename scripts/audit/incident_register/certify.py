"""Causal-path certificates over observer events (design §9.3 as amended; Codex phase-1 r2 #5, r3 #1, #7).

**Phase 1 certifies POSITIVE witnesses only** (plan ruling 10: Eric, 2026-09-30, after the Codex
consult on the absence loop). The r5–r7 reviews found absence-coverage failures and a false positive event
attribution. Phase 1 defers absence certification; attribution, observational purity and execution closure
remain independent obligations for retained positive witnesses (Codex phase-1 r8 #5, verbatim). An absence
request is refused (``AbsenceRefused``): its link reads unavailable with ``ABSENCE_REFUSAL``. A call the
recorder misses can only fail to witness a positive event; it never creates one.

Every certificate needs:
* a complete observed CALL phase of the killing node: exactly one ``obs_start`` and ``obs_end``, no
  observer error, no coroutine frame;
* clean endpoint purity fields (``purity``, recorded at both ends; Codex phase-1 r7 #4), which do not by
  themselves prove that no application callback ran: the trusted bootstrap's audit-hook census reports no
  hook added after it (the observer's own primitives are audited, so any such hook would run inside the
  observer); (Codex phase-1 r14, verbatim:) Automatic garbage collection stays off; reference-count
  finalizers remain possible, and endpoint fields do not prove callback-free observation. No application
  signal handler is installed; no application trace, profile or monitoring function is active at either end
  or installed in any thread since the trusted bootstrap (its census counts installs; Codex phase-1 r16);
* an execution of the mutated line inside a LIVE invocation of the declared entry (matched by file
  realpath and qualname). An invocation E is live at event ev when E started before ev on the same
  thread, E's frame is on ev's stack, and no exit of E and no newer entry record for the same frame id
  lies between them (a reused frame id belongs to a different invocation).

Then:
* ``event`` (wrong or extra event): a boundary call with the declared bad category after the branch,
  with some invocation live at both (recursion: the common live invocation counts);
* ``return`` (wrong value): a return from the declared function's source (file and qualname) with the
  declared category (or safe value), after the branch, with a common live invocation.

A witness is never an unattributed call (its identity carries a reason), an incomplete value, or the
reserved category ``unavailable``, which no request may name (Codex phase-1 r8 #2).

Each certificate states its coverage and model. A certificate is necessary, not sufficient: the
defence runner also requires the killing failure at the declared assertion and the observer-on/off
conformance runs (the same node states and, for a node failing both ways, the same exception, innermost
frame and message once what differs between any two runs is blanked), and the reviewer reads the patch,
frames and linked events.
"""
from __future__ import annotations

ABSENCE_REFUSAL = ("absence (a missing event) is not certifiable by the Phase 1 recorder (plan ruling 10): "
                   "report this link unavailable with this reason and keep its regression evidence")


class AbsenceRefused(ValueError):
    pass


# The bounded model (Codex's consult, 2026-09-30, adapted): what a certificate claims, what makes it
# unavailable, and what lies outside the model and so outside any claim.
COVERAGE = {
    "claims": "a positive witness in this pinned synthetic execution: a recorded call of a declared boundary with the "
              "declared bad category, or a recorded return from the declared function's source with the declared "
              "value or category, linked to the mutated line through a common live invocation of the declared entry",
    "return_identity": "Entry and return identity are source-file and qualname identities, not Python function-object, "
                       "globals or receiver identities. The fixture review must establish that the observed code "
                       "invocation belongs to the declared contract and selection. A claim requiring exact callable or "
                       "receiver identity needs an additional witness or reads unavailable.",
    "boundary_call": "a call that started the code of a value the declared binding held in the interval (the binding "
                     "is tracked at every store on its path from sys's own namespace, by CPython dict watchers; a "
                     "held function's __code__ replaced in place is registered at the new code's first start "
                     "while the thread is alone, never by a function watcher, whose Python callback replaces an "
                     "exception pending in C (own review during r14)); a call's start is a PY_START at the code's first RESUME "
                     "carrying the entry argument, and a start reported anywhere else is not read (Codex phase-1 r15 "
                     "#1); for a mock, the arguments its standard __call__ received, which is what the mock "
                     "itself records",
    "unattributed_when": ["a callable bound to several boundaries", "a callable the binding held earlier in the interval",
                          "a boundary code object shared by several live functions", "a call on another receiver",
                          "a mock whose effective __call__ is not the standard one, or whose class changed",
                          "a value incomplete at any depth",
                          "a binding whose path passes through a namespace holding a key that is not an exact str, "
                          "or was changed by a store through such a key (until a per-key store re-resolves it)",
                          "a call observed while another thread was alive (only the reads `observation` admits then "
                          "happen; plan ruling 12, Codex phase-1 r13 #3)",
                          "an argument the observer could not read: never read as a null or as an empty call (Codex "
                          "phase-1 r13 #1)",
                          "a call the binding does not resolve to AT THE CALL, through watched namespaces that are all "
                          "supported, read under the lock every watched store takes before its change (Codex phase-1 "
                          "r10 part 2 #1)",
                          "an identity read from a keyword dict holding a key that is not an exact str"],
    "unavailable_when": ["an absence claim (ABSENCE_REFUSAL)", "an audit hook added after the trusted bootstrap",
                         "no audit-hook census", "automatic garbage collection on at either end of the call phase",
                         "an application signal handler at either end of the call phase",
                         "an application trace, profile or monitoring function active at either end of the call phase, "
                         "or installed in any thread at any time since the trusted bootstrap (it runs inside the "
                         "observer's callbacks and can set f_lineno to re-run a frame's entry; Codex phase-1 r16)", "observer errors",
                         "an incomplete observation window",
                         "a coroutine or async-generator frame, the boundary callee's own included (Codex phase-1 r13 #2)",
                         "a return observed while another thread was alive (its value is not read; plan ruling 12)",
                         "a run whose bts import provenance is unreadable without application dispatch (refused at "
                         "every stage's gate)"],
    "observation": "inside a certifiable call phase the observer hashes, compares or dispatches only on values it owns or has "
                   "type-checked as exact builtins: a dict it reads is walked item by item and is unsupported when it "
                   "holds a key that is not an exact str; heap class module metadata is read by raw namespace "
                   "iteration, and instance dicts through the standard C getset found by raw MRO namespace "
                   "iteration; a code object's name that is not an exact str reads as unnamed, "
                   "and code objects themselves are never hashed or compared (the observer's maps key them by id: "
                   "a code object's hash and equality reach its co_name and co_consts) (plan ruling 11: Codex "
                   "phase-1 r9; code objects: self-review during r10). Arguments are read from the frame's own slots, "
                   "never through frame.f_locals, whose refresh can release an application reference (Codex phase-1 "
                   "r12 #1). Other application reads happen only after a sample shows the observed thread is the "
                   "interpreter's only registered thread, since another thread could make a temporary reference it "
                   "holds the last one and run a finalizer (plan ruling 12, Codex phase-1 r11 #1). (Codex phase-1 "
                   "r13 #3, verbatim:) While another thread is alive, application arguments, cell contents and "
                   "binding namespaces are not read. A mock boundary may retain only `self` from its verified "
                   "non-cell frame slot and compare its identity with observer-held targets to select an "
                   "unavailable record. The paused frame owns that reference. Code/frame identity and a live "
                   "replacement-code operand are also admitted metadata. Calls and returns requiring other "
                   "application reads are unavailable; stores cut the binding before key/value conversion. (Since "
                   "the r14 answer there is no function watcher, so no replacement-code operand is read; a held "
                   "function's swapped __code__ is compared by identity at its first start, only while alone.) The "
                   "sample is a point predicate: a Python-accessing native thread of the reviewed venv that "
                   "attaches after it is not excluded (open: Codex phase-1 r13 #4)",
    "missed_not_false": "a call through anything the recorder does not observe (an unsupported namespace, lookup that "
                        "bypasses the raw namespace, a C-implemented callable, another process) is not recorded: it "
                        "can only fail to witness an event",
    "outside_the_model": ["deliberate tampering with the observer, its evidence or the interpreter",
                          "native code other than the reviewed venv", "audit hooks installed before the trusted "
                          "bootstrap (only site initialisation and the reviewed .pth hooks run before it)",
                          "garbage collection or signal handlers switched on and off again inside the call phase",
                          "a test or helper that replaces a standard-library function the observer itself calls",
                          "a held mock's class changed and restored between the moments the recorder reads it",
                          "application behaviour that can see the observer: a boundary event or returned-value decision "
                          "that changes with observable finalization, weak-reference liveness, callbacks or counts, "
                          "reference counts, or garbage-collector introspection (Codex phase-1 r18's wording); with the "
                          "sys.monitoring tool registry or event sets, instrumented or adaptive bytecode, or another "
                          "thread's frames; with elapsed time or resource use; or with object addresses. An in-process "
                          "observer holds every value a binding held, and its code, until the observation ends; it "
                          "occupies a monitoring tool and instruments the code it watches; and it takes time and memory "
                          "(proposed ruling 13; Codex phase-1 r17 #1, r18 #1-#3). The defence refuses such a divergence "
                          "when it reaches the failing assertion's message (observer-on/off conformance). The closure "
                          "screen flags selected literal spellings in src/bts and tests; it does not resolve aliases, "
                          "imports or generated code, so each prepared closure requires independent source review for "
                          "this exclusion (Codex phase-1 r18 #3, verbatim)"],
}


def _purity_errs(rec: dict, where: str) -> list[str]:
    p = rec.get("purity")
    if type(p) is not dict:
        return [f"observer purity not recorded at {where}"]
    why = []
    hooks = p.get("audit_hooks_added")
    if hooks is None:
        why.append(f"no audit-hook census at {where} (the trusted bootstrap did not load the observer): purity unavailable")
    elif hooks:
        why.append(f"{hooks} audit hook(s) added after the trusted bootstrap: the observer's audited primitives would run "
                   "them inside observation, purity unavailable")
    if p.get("gc_enabled") is not False:
        why.append(f"automatic garbage collection on at {where}: a collection could run application finalizers inside "
                   "the observer, purity unavailable")
    if p.get("signal_handlers"):
        why.append(f"application signal handler(s) at {where} ({p['signal_handlers'][:3]}): purity unavailable")
    if p.get("tracing") is not False:
        why.append(f"an application trace, profile or monitoring function at {where}: it runs inside the observer's "
                   "callbacks and can re-run a frame's entry, purity unavailable (Codex phase-1 r16)")
    if p.get("tracers_installed") is None:
        why.append(f"no trace/profile/monitoring install census at {where}: purity unavailable")
    return why


def interval(events: list[dict], node: str) -> tuple[list[dict], list[str]]:
    mine = sorted((e for e in events if e.get("node") == node and "seq" in e), key=lambda e: e["seq"])
    starts = [e for e in mine if e["kind"] == "obs_start"]
    ends = [e for e in mine if e["kind"] == "obs_end"]
    if len(starts) != 1 or len(ends) != 1:
        return [], [f"{len(starts)} observation starts / {len(ends)} ends for {node} (need exactly one each)"]
    why = _purity_errs(starts[0], "obs_start") + _purity_errs(ends[0], "obs_end")
    installed = [((e.get("purity") or {}) if type(e.get("purity")) is dict else {}).get("tracers_installed")
                 for e in (starts[0], ends[0])]
    if installed[0] is not None and installed[0] > 0:
        why.append(f"{installed[0]} trace, profile or monitoring install(s) before the call phase, in any thread: "
                   "one may still be active on a thread the endpoint checks cannot see, purity unavailable "
                   "(Codex phase-1 r16)")
    if installed[0] is not None and installed[1] is not None and installed[0] != installed[1]:
        why.append(f"{installed[1] - installed[0]} trace, profile or monitoring install(s) during the call phase: "
                   "purity unavailable (Codex phase-1 r16)")
    inside = [e for e in mine if starts[0]["seq"] < e["seq"] < ends[0]["seq"]]
    if any(e["kind"] == "observer_error" for e in inside):
        why.append("observer error during the observed interval")
    if any(e.get("async") for e in inside):
        why.append("asynchronous path: unavailable (coroutine frame observed)")
    return inside + [ends[0]], why


def _live(en: dict, ev: dict, records: list[dict]) -> bool:
    if not (en["seq"] < ev["seq"] and en["thread"] == ev["thread"]):
        return False
    if (en["frame"], en["qualname"], en["file"]) not in {(f, q, p) for q, p, _l, f in ev.get("stack", [])}:
        return False
    for r in records:
        if en["seq"] < r["seq"] < ev["seq"] and r.get("frame") == en["frame"] and \
                r["kind"] in ("entry_exit", "entry") and r["thread"] == en["thread"]:
            return False
    return True


RESERVED_CATEGORY = "unavailable"   # an unattributed or incomplete witness's category: never requestable


def _category(ev: dict) -> str:
    return (ev.get("identity") or {}).get("category", RESERVED_CATEGORY)


def _complete(safe) -> bool:
    """True when no part of an observer ``_safe`` form was omitted at any depth (as ``observer._complete``)."""
    if type(safe) is not dict:
        return True
    if safe.get("incomplete") or "unavailable" in safe:
        return False
    if "seq" in safe:
        return all(_complete(v) for v in safe["seq"])
    for key in ("map", "fields"):
        if key in safe:
            return all(_complete(v) for v in safe[key].values())
    return True


def _witness(x: dict) -> bool:
    """An attributed, complete record (Codex phase-1 r8 #2): no attribution-failure reason, no omitted part.
    (The reserved category itself is refused as a request, below.)"""
    if x["kind"] == "boundary":
        ident = x.get("identity") or {}
        return "reason" not in ident and _complete(ident.get("value"))
    return _complete(x.get("value"))


def certify(events: list[dict], *, node: str, kind: str, entry: dict, bad: dict) -> dict:
    """``entry`` = {"file": realpath, "qualname"}; ``bad`` = {"boundary", "category"} for event,
    {"file", "qualname", "category" or "value"} for return. Returns {"ok", "reasons", "linked",
    "coverage"}. An absence request raises ``AbsenceRefused``."""
    if kind == "absence":
        raise AbsenceRefused(ABSENCE_REFUSAL)
    if kind not in ("event", "return"):
        raise ValueError(kind)
    inside, why = interval(events, node)
    if not inside:
        return {"ok": False, "reasons": why, "linked": None, "coverage": COVERAGE}
    body = inside[:-1]
    entries = [e for e in body if e["kind"] == "entry" and e["file"] == entry["file"]
               and e["qualname"] == entry["qualname"]]
    if not entries:
        why.append("the declared production entry was never invoked in the observed interval")
    branches = [b for b in body if b["kind"] == "branch"]
    live_at_branch = [(b, {en["seq"] for en in entries if _live(en, b, body)}) for b in branches]
    live_at_branch = [(b, s) for b, s in live_at_branch if s]
    if not live_at_branch:
        why.append("the mutated line never executed inside a live declared entry invocation")
    if bad.get("category") == RESERVED_CATEGORY:
        why.append(f"the category {RESERVED_CATEGORY!r} is reserved for unattributed or incomplete records and "
                   "cannot be requested")
    if kind == "event":
        candidates = [x for x in body if x["kind"] == "boundary" and x["name"] == bad["boundary"]
                      and _witness(x) and _category(x) == bad["category"]]
    else:
        candidates = [x for x in body if x["kind"] == "return" and x["file"] == bad["file"]
                      and x["qualname"] == bad["qualname"] and _witness(x)
                      and (("category" in bad and x.get("category") == bad["category"])
                           or ("value" in bad and x.get("value") == bad["value"]))]
    linked = None
    for x in candidates:
        live_x = {en["seq"] for en in entries if _live(en, x, body)}
        common = [(b, s & live_x) for b, s in live_at_branch if b["seq"] < x["seq"] and s & live_x]
        if common:
            b, s = common[0]
            linked = {"entry": min(s), "branch": b["seq"], kind: x["seq"]}
            break
    if linked is None:
        what = bad.get("category") or bad.get("value")
        why.append(f"no {what!r} {kind} after the branch within a common live entry invocation")
    return {"ok": not why, "reasons": why, "linked": linked, "coverage": COVERAGE}
