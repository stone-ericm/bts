"""Observational call-stack witnesses for current-defence mutants (design §9.3; Codex phase-1 r1 #3).

The harness copies this file into the mutant worktree as ``src/bts/_w15_witness.py``. A patch
imports ``hook`` and calls ``hook("<tag>")`` at:

* ``entry``    — the FIRST statement of the declared production entry function;
* ``branch``   — the mutated decision (also placed in the witness-only patch, same spot);
* ``boundary`` — immediately before the production call/write that produces the checked event
  (for a missing-event symptom it marks where the event WOULD be produced);
* ``done``     — where the entry invocation completes (missing-event symptoms only).

Each call appends one JSON line — tag, a per-process sequence number, pid, thread id, run id
(``$W15_RUN_ID``), pytest node id + phase, and the caller-outward stack as
``[qualname, filename, lineno, frame_id]`` — to ``$W15_WITNESS_PATH``; with it unset the hook
does nothing. It never changes control flow.

``certify`` is the acceptance rule. It is a synchronous, single-thread certificate: asynchronous
or multi-process paths are ``unavailable`` rather than guessed.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import threading

ENV = "W15_WITNESS_PATH"
RUN_ENV = "W15_RUN_ID"
_SEQ = itertools.count(1)


def hook(tag: str) -> None:
    path = os.environ.get(ENV)
    if not path:
        return
    frame = sys._getframe(1)
    stack = []
    while frame is not None:
        code = frame.f_code
        stack.append([getattr(code, "co_qualname", code.co_name), os.path.realpath(code.co_filename),
                      frame.f_lineno, id(frame)])
        frame = frame.f_back
    record = {"tag": tag, "seq": next(_SEQ), "pid": os.getpid(), "thread": threading.get_ident(),
              "run": os.environ.get(RUN_ENV, ""), "node": os.environ.get("PYTEST_CURRENT_TEST", ""),
              "stack": stack}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load(path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def _node_phase(record: dict) -> tuple[str, str]:
    node = record.get("node", "")
    base, _, phase = node.rpartition(" (")
    return base, phase.rstrip(")")


def _under(record: dict, entry: str, entry_frame: int) -> bool:
    return any(q == entry and fid == entry_frame for q, _f, _l, fid in record["stack"])


def certify(records: list[dict], *, node: str, run: str, entry: str, entry_file: str,
            branch_file: str, kind: str) -> dict:
    """Certificate for one killing node.

    ``kind="event"``   (wrong/extra event): one ``entry`` → ``branch`` → ``boundary``, in that
                       sequence order, all under the same entry invocation;
    ``kind="absence"`` (missing event): one ``entry`` → ``branch`` → ``done`` under the same
                       invocation AND no ``boundary`` record under it (the event set is empty).
    ``entry_file`` / ``branch_file`` are the realpaths of the production files that must own the
    entry function and the mutated decision. Returns {"ok", "reasons", "entry_frame"}.
    """
    if kind not in ("event", "absence"):
        raise ValueError("kind must be 'event' or 'absence'")
    why: list[str] = []
    mine = [r for r in records if r.get("run") == run and _node_phase(r) == (node, "call")]
    if len({(r["pid"], r["thread"]) for r in mine}) > 1:
        return {"ok": False, "reasons": ["records from more than one process/thread"], "entry_frame": None}
    entries = [r for r in mine if r["tag"] == "entry"]
    if len(entries) != 1:
        return {"ok": False, "reasons": [f"{len(entries)} entry records (need exactly 1)"], "entry_frame": None}
    entry_rec = entries[0]
    q, f, _l, entry_frame = entry_rec["stack"][0]
    if q != entry or os.path.realpath(f) != os.path.realpath(entry_file):
        return {"ok": False, "reasons": [f"entry hook in {q!r} of {f!r}, not {entry!r} of {entry_file!r}"],
                "entry_frame": None}
    inside = [r for r in mine if _under(r, entry, entry_frame) and r["seq"] > entry_rec["seq"]]
    branches = [r for r in inside if r["tag"] == "branch"
                and os.path.realpath(r["stack"][0][1]) == os.path.realpath(branch_file)]
    if not branches:
        why.append(f"no branch record from {branch_file!r} under the entry invocation")
        return {"ok": False, "reasons": why, "entry_frame": None}
    first_branch = min(r["seq"] for r in branches)
    boundaries = [r for r in inside if r["tag"] == "boundary"]
    if kind == "event":
        if not any(r["seq"] > first_branch for r in boundaries):
            why.append("no boundary record after the branch under the entry invocation")
    else:
        dones = [r for r in inside if r["tag"] == "done" and r["seq"] > first_branch]
        if not dones:
            why.append("no completion record after the branch under the entry invocation")
        if boundaries:
            why.append(f"{len(boundaries)} boundary record(s): the event occurred")
    return {"ok": not why, "reasons": why, "entry_frame": entry_frame if not why else None}


def positive_control(records: list[dict], *, node: str, run: str, entry: str, entry_file: str) -> dict:
    """Witness-only (no mutation) run of the same node: the event DOES occur under one entry
    invocation — the baseline half of a missing-event certificate."""
    mine = [r for r in records if r.get("run") == run and _node_phase(r) == (node, "call")]
    entries = [r for r in mine if r["tag"] == "entry"]
    if len(entries) != 1:
        return {"ok": False, "reasons": [f"{len(entries)} entry records (need exactly 1)"]}
    q, f, _l, entry_frame = entries[0]["stack"][0]
    if q != entry or os.path.realpath(f) != os.path.realpath(entry_file):
        return {"ok": False, "reasons": [f"entry hook in {q!r} of {f!r}"]}
    if not any(r["tag"] == "boundary" and _under(r, entry, entry_frame) for r in mine):
        return {"ok": False, "reasons": ["the baseline event did not occur"]}
    return {"ok": True, "reasons": []}


def connected(records: list[dict], *, node: str, entry: str, branch_tag: str,
              boundary_tag: str | None = None, completion_tag: str | None = None,
              entry_tag: str = "entry") -> dict:
    """LEGACY (phase-1 r0) reachability check kept only so preliminary batch runs keep working.
    It is NOT an acceptance rule: no ordering, no file identity, no absence proof (Codex phase-1
    r1 #3). Certificates use ``certify`` / ``positive_control``."""
    if (boundary_tag is None) == (completion_tag is None):
        raise ValueError("pass exactly one of boundary_tag / completion_tag")
    in_node = [r for r in records if _node_phase(r) == (node, "call")]
    entries = [r for r in in_node if r["tag"] == entry_tag]
    if len(entries) != 1:
        return {"ok": False, "reason": f"{len(entries)} {entry_tag!r} records (need exactly 1)",
                "entry_frame": None}
    top_qualname, _f, _l, entry_frame = entries[0]["stack"][0]
    if top_qualname != entry:
        return {"ok": False, "reason": f"{entry_tag!r} recorded in {top_qualname!r}", "entry_frame": None}
    second = boundary_tag or completion_tag
    needed = {branch_tag: False, second: False}
    for r in in_node:
        if r["tag"] in needed and _under(r, entry, entry_frame):
            needed[r["tag"]] = True
    missing = [t for t, seen in needed.items() if not seen]
    if missing:
        return {"ok": False, "reason": f"not under the one {entry!r} invocation: {missing}", "entry_frame": None}
    return {"ok": True, "reason": "same entry invocation (legacy reachability only)", "entry_frame": entry_frame}
