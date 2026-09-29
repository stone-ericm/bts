"""Acceptance rules over the evidence plugin's structured reports (Codex phase-1 r1 #4, #5).

These are the only functions that may turn a run into a certificate input. Every rule looks at
ALL phases of a node (setup, call, teardown), requires the node to be collected exactly once,
and names the reason when it refuses.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def load(path) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                events.append(json.loads(line))
    return events


def collected(events: list[dict]) -> list[str]:
    ids: list[str] = []
    for e in events:
        if e["kind"] == "collected":
            ids.extend(e["nodeids"])
    return ids


def reports(events: list[dict], nodeid: str) -> dict[str, list[dict]]:
    by_when: dict[str, list[dict]] = {}
    for e in events:
        if e["kind"] == "report" and e["nodeid"] == nodeid:
            by_when.setdefault(e["when"], []).append(e)
    return by_when


def _base(events: list[dict], nodeid: str) -> tuple[dict | None, list[str]]:
    """The single setup/call/teardown triple of a node, or the reasons it is not usable."""
    why = []
    n = collected(events).count(nodeid)
    if n != 1:
        why.append(f"collected {n} times")
    by_when = reports(events, nodeid)
    for when in ("setup", "call", "teardown"):
        got = by_when.get(when, [])
        if when == "call" and not got and by_when.get("setup") and by_when["setup"][0]["outcome"] != "passed":
            continue
        if len(got) != 1:
            why.append(f"{len(got)} {when} reports")
    if why:
        return None, why
    return {w: by_when[w][0] for w in by_when}, why


def node_state(events: list[dict], nodeid: str) -> str:
    triple, why = _base(events, nodeid)
    if triple is None:
        return "missing" if "collected 0 times" in why else "malformed"
    setup, call, teardown = triple.get("setup"), triple.get("call"), triple.get("teardown")
    if setup["outcome"] != "passed":
        return "setup_xfail" if setup.get("wasxfail") else "setup_error"
    if teardown["outcome"] != "passed":
        return "teardown_error"
    if call["outcome"] == "passed":
        return "passed"
    if call["outcome"] == "skipped":
        return "xfail" if call.get("wasxfail") else "skipped"
    return "failed"


def accept_green(events: list[dict], nodeids: list[str], expected_xfail: set[str] = frozenset()) -> tuple[bool, list[str]]:
    """Every listed node collected once and passed (or xfailed where that is its baseline state)."""
    why = []
    for nid in nodeids:
        state = node_state(events, nid)
        want = "xfail" if nid in expected_xfail else "passed"
        if state != want:
            why.append(f"{nid}: {state} (want {want})")
    return (not why), why


def accept_killed(events: list[dict], nodeid: str) -> tuple[bool, list[str]]:
    """The killing node failed in its CALL phase, with clean setup and teardown."""
    triple, why = _base(events, nodeid)
    if triple is None:
        return False, why
    state = node_state(events, nodeid)
    if state != "failed":
        return False, [f"state {state}, not a call-phase failure"]
    return True, []


def accept_expected_failure(events: list[dict], nodeid: str, *, exc_class: str, oracle_func: str,
                            test_file: str) -> tuple[bool, list[str]]:
    """A strict xfail counts only as: collected once; setup and teardown passed; the CALL phase
    raised exactly ``exc_class`` (not an imperative ``pytest.xfail``), raised inside
    ``oracle_func`` in ``test_file``; and the node's marker is strict with raises=exc_class."""
    triple, why = _base(events, nodeid)
    if triple is None:
        return False, why
    setup, call, teardown = triple.get("setup"), triple.get("call"), triple.get("teardown")
    if setup["outcome"] != "passed":
        why.append(f"setup {setup['outcome']}")
    if teardown["outcome"] != "passed":
        why.append(f"teardown {teardown['outcome']}")
    if call is None:
        return False, why + ["no call report"]
    if call["outcome"] != "skipped" or not call.get("wasxfail"):
        why.append(f"call outcome {call['outcome']} (not an xfail)")
    if call.get("imperative_xfail"):
        why.append("imperative pytest.xfail()")
    if call.get("exc_type") != exc_class:
        why.append(f"exception {call.get('exc_type')!r} != {exc_class!r}")
    frames = call.get("frames") or []
    if not frames or frames[-1]["func"] != oracle_func or not frames[-1]["path"].endswith(test_file):
        last = frames[-1] if frames else None
        why.append(f"raised at {last and (last['path'], last['func'])!r}, not in {oracle_func} of {test_file}")
    marker = call.get("marker") or {}
    if marker.get("strict") is not True or marker.get("raises") != [exc_class]:
        why.append(f"marker {marker!r} is not strict with raises=[{exc_class}]")
    return (not why), why


def imports_under(events: list[dict], root: Path) -> tuple[bool, list[str]]:
    """Every imported ``bts`` module resolved inside ``root`` (the certified worktree's src)."""
    root = os.path.realpath(root)
    offenders = []
    seen = False
    for e in events:
        if e["kind"] != "imports":
            continue
        seen = True
        for name, info in e["modules"].items():
            if not info["file"].startswith(root + os.sep):
                offenders.append(f"{name} -> {info['file']}")
    if not seen:
        return False, ["no imports record (session did not finish?)"]
    return (not offenders), offenders
