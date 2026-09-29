"""Observational call-stack witnesses for current-defence mutants (design §9.3).

A mutant patch imports ``hook`` (the harness copies this file into the mutant worktree as
``src/bts/_w15_witness.py``) and calls ``hook("<tag>")`` at the mutated decision and, for a
wrong/extra-event symptom, at the production call site of the checked boundary. Each call
appends one JSON line — tag, pytest node id, and the full stack as
``[qualname, filename, lineno, frame_id]`` from the hook's caller outward — to the file named
by ``$W15_WITNESS_PATH``; with the variable unset the hook does nothing. It never changes
control flow.

Frame ids (``id(frame)``) identify one live invocation: two records whose stacks contain the
same entry-function frame id were made inside the same call of that entry function.
"""
from __future__ import annotations

import json
import os
import sys

ENV = "W15_WITNESS_PATH"


def hook(tag: str) -> None:
    path = os.environ.get(ENV)
    if not path:
        return
    frame = sys._getframe(1)
    stack = []
    while frame is not None:
        code = frame.f_code
        stack.append([getattr(code, "co_qualname", code.co_name), code.co_filename,
                      frame.f_lineno, id(frame)])
        frame = frame.f_back
    record = {"tag": tag, "node": os.environ.get("PYTEST_CURRENT_TEST", ""), "stack": stack}
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def load(path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))
    return records


def _node_id(record: dict) -> str:
    # PYTEST_CURRENT_TEST is "<nodeid> (setup|call|teardown)"
    node = record.get("node", "")
    return node.rsplit(" (", 1)[0]


def _entry_frames(record: dict, entry_qualname: str) -> set[int]:
    return {frame_id for qualname, _file, _line, frame_id in record["stack"]
            if qualname == entry_qualname}


def connected(records: list[dict], *, node: str, entry: str, branch_tag: str,
              boundary_tag: str | None = None, completion_tag: str | None = None,
              entry_tag: str = "entry") -> dict:
    """Certify, within one killing node, that ONE invocation of the declared entry function
    reached the mutated decision and then either the checked boundary (wrong/extra event) or
    its own completion (missing event; ``completion_tag`` is recorded at the entry's return).

    The mutant records ``entry_tag`` at the top of the entry function; exactly one such record
    must exist in the node's call phase, so a reused frame id from an earlier, finished
    invocation cannot fake a connection. Returns {"ok", "reason", "entry_frame"}.
    """
    if (boundary_tag is None) == (completion_tag is None):
        raise ValueError("pass exactly one of boundary_tag / completion_tag")
    in_node = [r for r in records if _node_id(r) == node and r.get("node", "").endswith("(call)")]
    entries = [r for r in in_node if r["tag"] == entry_tag]
    if len(entries) != 1:
        return {"ok": False, "reason": f"{len(entries)} {entry_tag!r} records (need exactly 1)",
                "entry_frame": None}
    # The entry hook's own caller frame is the entry function itself.
    top_qualname, _f, _l, entry_frame = entries[0]["stack"][0]
    if top_qualname != entry:
        return {"ok": False, "reason": f"{entry_tag!r} recorded in {top_qualname!r}, not {entry!r}",
                "entry_frame": None}
    second_tag = boundary_tag or completion_tag
    needed = {branch_tag: False, second_tag: False}
    for r in in_node:
        if r["tag"] in needed and entry_frame in _entry_frames(r, entry):
            needed[r["tag"]] = True
    missing = [tag for tag, seen in needed.items() if not seen]
    if missing:
        return {"ok": False, "reason": f"not under the one {entry!r} invocation: {missing}",
                "entry_frame": None}
    return {"ok": True, "reason": "same entry invocation", "entry_frame": entry_frame}
