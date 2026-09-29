"""Causal-path certificates over observer events (design §9.3; Codex phase-1 r2 #5).

All three kinds need, inside the killing node's observed CALL phase (``obs_start``…``obs_end``,
complete, no pending identities, no coroutine frames), an invocation of the declared production
entry (matched by file realpath AND qualname) and an execution of the mutated line LINKED to that
invocation: the entry's frame is on the branch event's stack, and it is the latest entry record for
that frame id (a dead frame's id can be reused; a live one cannot). Then:

* ``event`` (wrong or extra event): a boundary call whose identity category is the declared bad
  one, linked to the SAME invocation, after the branch;
* ``absence`` (missing event): NO boundary call of the declared name with the qualifying category
  anywhere in the observed interval — the observer records every call made from production code to
  the boundary, not only calls under the entry — and no boundary call of that name whose identity
  is unavailable; plus a positive control: the same node's baseline interval DOES contain a
  qualifying call;
* ``return`` (wrong value): a return of the declared function with the declared bad value, linked
  to the same invocation, after the branch.

A certificate is necessary, not sufficient: the defence runner also requires the killing failure at
the declared assertion, and the reviewer reads the patch and frames.
"""
from __future__ import annotations


def interval(events: list[dict], node: str) -> tuple[list[dict], list[str]]:
    mine = sorted((e for e in events if e.get("node") == node and "seq" in e), key=lambda e: e["seq"])
    starts = [e for e in mine if e["kind"] == "obs_start"]
    ends = [e for e in mine if e["kind"] == "obs_end"]
    if len(starts) != 1 or len(ends) != 1:
        return [], [f"{len(starts)} observation starts / {len(ends)} ends for {node} (need exactly one each)"]
    why = []
    if ends[0].get("pending_identity"):
        why.append("boundary identity unresolved for some calls")
    inside = [e for e in mine if starts[0]["seq"] < e["seq"] < ends[0]["seq"]]
    if any(e["kind"] == "observer_error" for e in inside):
        why.append("observer error during the observed interval")
    if any(e.get("async") for e in inside):
        why.append("asynchronous path: unavailable (coroutine frame observed)")
    return inside, why


def _linked(ev: dict, entries: list[dict]) -> dict | None:
    """The live entry invocation whose frame is on ``ev``'s stack, if any."""
    on_stack = {(fid, q, f) for q, f, _line, fid in ev.get("stack", [])}
    best = None
    for en in entries:
        if en["seq"] < ev["seq"] and en["thread"] == ev["thread"] and \
                (en["frame"], en["qualname"], en["file"]) in on_stack:
            if best is None or en["seq"] > best["seq"]:
                best = en
    return best


def _category(ev: dict) -> str:
    return (ev.get("identity") or {}).get("category", "unavailable")


def certify(events: list[dict], *, node: str, kind: str, entry: dict, bad: dict | None = None,
            positive_events: list[dict] | None = None) -> dict:
    """``entry`` = {"file": realpath, "qualname"}; ``bad`` = {"boundary", "category"} for event and
    absence (for absence: the QUALIFYING category that must not occur), {"file", "qualname", "value"}
    for return. Returns {"ok", "reasons", "linked": {...}}."""
    if kind not in ("event", "absence", "return"):
        raise ValueError(kind)
    inside, why = interval(events, node)
    if not inside and why:
        return {"ok": False, "reasons": why, "linked": None}
    entries = [e for e in inside if e["kind"] == "entry" and e["file"] == entry["file"]
               and e["qualname"] == entry["qualname"]]
    if not entries:
        why.append("the declared production entry was never invoked in the observed interval")
    branches = [(b, _linked(b, entries)) for b in inside if b["kind"] == "branch"]
    linked_branches = [(b, en) for b, en in branches if en is not None]
    if not linked_branches:
        why.append("the mutated line never executed inside a declared entry invocation")
    linked = None
    if kind == "event":
        for x in inside:
            if x["kind"] != "boundary" or x["name"] != bad["boundary"] or _category(x) != bad["category"]:
                continue
            en = _linked(x, entries)
            if en is not None and any(e is en and b["seq"] < x["seq"] for b, e in linked_branches):
                linked = {"entry": en["seq"], "boundary": x["seq"], "identity": x.get("identity")}
                break
        if linked is None:
            why.append(f"no {bad['category']!r} {bad['boundary']!r} call after the branch in the same invocation")
    elif kind == "absence":
        same = [x for x in inside if x["kind"] == "boundary" and x["name"] == bad["boundary"]]
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
        linked = {"calls_observed": len(same)}
    else:
        for x in inside:
            if x["kind"] != "return" or x["file"] != bad["file"] or x["qualname"] != bad["qualname"] \
                    or x["value"] != bad["value"]:
                continue
            en = _linked(x, entries)
            if en is not None and any(e is en and b["seq"] < x["seq"] for b, e in linked_branches):
                linked = {"entry": en["seq"], "return": x["seq"], "value": x["value"]}
                break
        if linked is None:
            why.append(f"no return of {bad['value']!r} from {bad['qualname']} after the branch in the same invocation")
    return {"ok": not why, "reasons": why, "linked": linked}
