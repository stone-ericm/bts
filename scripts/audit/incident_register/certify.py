"""Causal-path certificates over observer events (design §9.3 as amended; Codex phase-1 r3 #1, #7).

Every kind needs a complete observed CALL phase of the killing node (exactly one ``obs_start`` and
``obs_end``, no observer error, no coroutine frame) and an execution of the mutated line inside a
LIVE invocation of the declared entry (matched by file realpath and qualname). An invocation E is
live at event ev when E started before ev on the same thread, E's frame is on ev's stack, and no exit
of E and no newer entry record for the same frame id lies between them (a reused frame id belongs to
a different invocation).

* ``event`` (wrong or extra event): a boundary call with the declared bad category after the branch,
  with some invocation live at both (recursion: the common live invocation counts);
* ``return`` (wrong value): a return of the declared function with the declared category (or safe
  value), after the branch, with a common live invocation;
* ``absence`` (missing event): an invocation live at the branch EXITED inside the interval
  (completion), no other thread — started in the interval or already running before it — is still alive
  at its end (Codex phase-1 r4 #1.3), the boundary had no
  coverage gap (not Python-observable, or rebound to an unseen callable), there is NO boundary call of
  that name with the qualifying category from ANY caller in the interval (callee-side recording sees
  C-invoked and threaded calls), none with an unavailable identity, and the same node's baseline
  interval DOES contain a qualifying call (positive control).

Each certificate states its coverage. A certificate is necessary, not sufficient: the defence
runner also requires the killing failure at the declared assertion and the observer-on/off
conformance runs, and the reviewer reads the patch, frames and linked events.
"""
from __future__ import annotations

COVERAGE = {"recorder": "callee-side, Python-observable boundaries (mocks, Python functions, bound methods of the declared receiver)",
            "unavailable_when": ["a C-implemented or unresolvable boundary", "a call on another receiver of the boundary's code",
                                 "a value too large or too deep to serialize completely", "any other thread alive at the end",
                                 "subprocesses", "async tasks"]}


def interval(events: list[dict], node: str) -> tuple[list[dict], list[str]]:
    mine = sorted((e for e in events if e.get("node") == node and "seq" in e), key=lambda e: e["seq"])
    starts = [e for e in mine if e["kind"] == "obs_start"]
    ends = [e for e in mine if e["kind"] == "obs_end"]
    if len(starts) != 1 or len(ends) != 1:
        return [], [f"{len(starts)} observation starts / {len(ends)} ends for {node} (need exactly one each)"]
    why = []
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


def _category(ev: dict) -> str:
    return (ev.get("identity") or {}).get("category", "unavailable")


def certify(events: list[dict], *, node: str, kind: str, entry: dict, bad: dict | None = None,
            positive_events: list[dict] | None = None) -> dict:
    """``entry`` = {"file": realpath, "qualname"}; ``bad`` = {"boundary", "category"} for event and
    absence (for absence: the QUALIFYING category that must not occur), {"file", "qualname",
    "category" or "value"} for return. Returns {"ok", "reasons", "linked", "coverage"}."""
    if kind not in ("event", "absence", "return"):
        raise ValueError(kind)
    inside, why = interval(events, node)
    if not inside:
        return {"ok": False, "reasons": why, "linked": None, "coverage": COVERAGE}
    end = inside[-1]
    body = inside[:-1]
    entries = [e for e in body if e["kind"] == "entry" and e["file"] == entry["file"]
               and e["qualname"] == entry["qualname"]]
    exits = [e for e in body if e["kind"] == "entry_exit"]
    if not entries:
        why.append("the declared production entry was never invoked in the observed interval")
    branches = [b for b in body if b["kind"] == "branch"]
    live_at_branch = [(b, {en["seq"] for en in entries if _live(en, b, body)}) for b in branches]
    live_at_branch = [(b, s) for b, s in live_at_branch if s]
    if not live_at_branch:
        why.append("the mutated line never executed inside a live declared entry invocation")
    linked = None
    if kind in ("event", "return"):
        if kind == "event":
            candidates = [x for x in body if x["kind"] == "boundary" and x["name"] == bad["boundary"]
                          and _category(x) == bad["category"]]
        else:
            candidates = [x for x in body if x["kind"] == "return" and x["file"] == bad["file"]
                          and x["qualname"] == bad["qualname"]
                          and (("category" in bad and x.get("category") == bad["category"])
                               or ("value" in bad and x.get("value") == bad["value"]))]
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
    else:
        completed = [(b, s) for b, s in live_at_branch
                     if any(ex["seq"] > b["seq"] and any(en["seq"] in s and en["frame"] == ex["frame"]
                                                        and en["thread"] == ex["thread"] for en in entries)
                            for ex in exits)]
        if live_at_branch and not completed:
            why.append("the invocation that executed the mutated line did not complete inside the interval")
        if end.get("outstanding_threads"):
            why.append(f"{end['outstanding_threads']} thread(s) started in the interval were still alive at its end: absence unavailable")
        if end.get("preexisting_threads_alive"):
            why.append(f"{end['preexisting_threads_alive']} thread(s) already running when the observed interval began "
                       "were still alive at its end and could do the declared work afterwards: absence unavailable")
        gaps = [g for g in body if g["kind"] == "boundary_gap" and g["name"] == bad["boundary"]]
        if gaps:
            why.append(f"boundary {bad['boundary']!r} coverage gap ({gaps[0]['reason']}): absence unavailable")
        same = [x for x in body if x["kind"] == "boundary" and x["name"] == bad["boundary"]]
        qualifying = [x for x in same if _category(x) == bad["category"]]
        unidentified = [x for x in same if _category(x) == "unavailable"]
        if qualifying:
            why.append(f"{len(qualifying)} qualifying {bad['boundary']!r} call(s) occurred: not an absence")
        if unidentified:
            why.append(f"{len(unidentified)} {bad['boundary']!r} call(s) without identity: absence not established")
        base, base_why = interval(positive_events or [], node)
        if base_why or not any(x["kind"] == "boundary" and x["name"] == bad["boundary"]
                               and _category(x) == bad["category"] for x in base):
            why.append("positive control: the baseline did not produce the qualifying event")
        linked = {"calls_observed": len(same), "completed": bool(completed)}
    return {"ok": not why, "reasons": why, "linked": linked, "coverage": COVERAGE}
