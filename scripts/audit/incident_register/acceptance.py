"""Strict expected-failure acceptance (design §9.7; Codex phase-1 r2 #3, #4).

A registered node is accepted as a reproduction only from a PAIR of observed runs of the same
worktree state — the marked run and a ``--runxfail`` run — each passing the session gate:

* marked: the node is XFAIL in its CALL phase (setup and teardown passed), not imperative, under a
  strict marker whose ``raises`` is exactly the registered exception (``module.qualname``); every
  other selected node passes;
* ``--runxfail``: the node fails in its CALL phase with exactly the registered exception class
  (module AND qualname), raised in the registered oracle function of the registered file (exact
  realpath, not a suffix), and the message carries the declared bad value and the required value;
  the declared production entry was invoked inside the node's observed call phase;
* both runs collected the same inventory, the same test-file bytes and the same ``bts`` modules.

The registry (``expected_failures.json``) binds each node to its exception, oracle, entry, bad and
required values; it is reviewed data, hashed into the acceptance output.
"""
from __future__ import annotations

import os

from scripts.audit.incident_register import certify, runner


def observe_config(worktree, registry: list[dict]) -> dict:
    wt = os.path.realpath(worktree)
    return {"nodes": [r["node"] for r in registry],
            "entries": [{"file": os.path.join(wt, r["entry"]["path"]), "qualname": r["entry"]["qualname"]}
                        for r in registry], "boundaries": [], "returns": []}


def accept(marked: runner.Run, unmarked: runner.Run, *, worktree, registry: list[dict]) -> dict:
    """{node: [reasons]} (empty list = accepted) plus a "_session" key for run-level reasons."""
    wt = os.path.realpath(worktree)
    nodes = {r["node"] for r in registry}
    out: dict[str, list[str]] = {"_session": []}
    inventory = runner.collected(marked.events)
    out["_session"] += [f"marked: {r}" for r in runner.gate(marked, worktree=worktree, mode="expected_failure",
                                                            expected_failures=frozenset(nodes))]
    un_gate = runner.gate(unmarked, worktree=worktree, mode="mutant", expected=inventory)
    out["_session"] += [f"--runxfail: {r}" for r in un_gate]
    killed = {n for n in inventory if runner.node_state(unmarked.events, n) == "failed"}
    if killed != nodes:
        out["_session"].append(f"--runxfail failures {sorted(killed ^ nodes)[:5]} differ from the registry")
    files = [e for e in marked.events if e["kind"] == "collected"]
    files_u = [e for e in unmarked.events if e["kind"] == "collected"]
    if not files or not files_u or files[0].get("files") != files_u[0].get("files"):
        out["_session"].append("the two runs collected different test-file bytes")
    imp = [e["modules"] for e in marked.events if e["kind"] == "imports"]
    imp_u = [e["modules"] for e in unmarked.events if e["kind"] == "imports"]
    if not imp or imp != imp_u:
        out["_session"].append("the two runs imported different bts modules")
    for r in registry:
        n, why = r["node"], []
        call = runner.phases(marked.events, n).get("call", [])
        if len(call) != 1 or not call[0].get("wasxfail") or call[0]["outcome"] != "skipped":
            why.append("marked run: not an XFAIL in the call phase")
        else:
            c = call[0]
            marker = c.get("marker") or {}
            if c.get("imperative_xfail"):
                why.append("marked run: imperative pytest.xfail()")
            if marker.get("raises") != [r["exception"]]:
                why.append(f"marked run: marker raises {marker.get('raises')} != [{r['exception']}]")
            strict = marker.get("strict")
            if not (strict is True or (strict is None and _ini_strict(marked))):
                why.append("marked run: marker is not strict")
        ucall = runner.phases(unmarked.events, n).get("call", [])
        if len(ucall) != 1 or ucall[0]["outcome"] != "failed":
            why.append("--runxfail: the node did not fail in its call phase")
        else:
            u = ucall[0]
            if f"{u.get('exc_module')}.{u.get('exc_qualname')}" != r["exception"]:
                why.append(f"--runxfail: raised {u.get('exc_module')}.{u.get('exc_qualname')}")
            last = u["frames"][-1] if u.get("frames") else None
            oracle_file = os.path.join(wt, r["oracle"]["path"])
            if not last or last[0] != oracle_file or last[2] != r["oracle"]["qualname"]:
                why.append(f"--runxfail: raised in {last[0] if last else None}::{last[2] if last else None}, "
                           f"not {oracle_file}::{r['oracle']['qualname']}")
            msg = u.get("message") or ""
            if f"got the declared bad value {r['bad']}" not in msg or f"required {r['required']}" not in msg:
                why.append("--runxfail: the message lacks the declared bad/required values")
        inside, iv_why = certify.interval(unmarked.events, n)
        why += [f"--runxfail observation: {x}" for x in iv_why]
        entry_file = os.path.join(wt, r["entry"]["path"])
        if not any(e["kind"] == "entry" and e["file"] == entry_file and e["qualname"] == r["entry"]["qualname"]
                   for e in inside):
            why.append(f"--runxfail: production entry {r['entry']['qualname']} was never invoked")
        out[n] = why
    return out


def _ini_strict(run: runner.Run) -> bool:
    starts = [e for e in run.events if e["kind"] == "session_start"]
    return bool(starts and starts[0].get("xfail_strict_ini"))
