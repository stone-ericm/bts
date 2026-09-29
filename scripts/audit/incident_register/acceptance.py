"""Strict expected-failure acceptance (design §9.7; Codex phase-1 r2 #3, #4; r3 #4, #8).

A registered node is accepted as a reproduction only from a PAIR of observed runs of the same
worktree state — the marked run and a ``--runxfail`` run — each passing the session gate (the pair's
execution closure is frozen and re-checked by the driver around both runs):

* marked: the node is XFAIL in its CALL phase (setup and teardown passed), not imperative, under a
  strict marker whose ``raises`` is exactly the registered exception (``module.qualname``); every
  other selected node passes;
* ``--runxfail``: the node fails in its CALL phase with exactly the registered exception class
  (module AND qualname), raised in the registered oracle function of the registered file (exact
  realpath), the message carries the declared bad and required values, and the declared production
  entry was invoked inside the node's observed call phase;
* the oracle's structured record (``$W15_ORACLE_OUT``) shows actual == declared bad == the registry's
  ``bad_json`` and required == ``required_json``;
* CONNECTION (plan ruling 7 as amended after Codex phase-1 r4 #3): a VALUE MATCH, never a dataflow
  proof — ``return``: the last complete return of the declared production function in the call phase
  equals the oracle's actual; ``reads``: the complete values the fixture read back through production
  loaders, outside any live entry invocation and after the entry ran, equal it component by component
  (zipped components with equal cardinality). The tool cannot see which invocation or selection the
  oracle consumed, so a match counts (``value_match``) only with the fixture review recorded in the
  registry, which states that dataflow. ``derived``: no value is matched (the outcome is a derived
  label) — ``exception_shape``, again only with a recorded review. Anything else is ``unmatched``;
* both runs collected the same inventory, test-file bytes and ``bts`` modules.

The registry (``expected_failures.json``) is reviewed data; the driver hashes it into its output.
"""
from __future__ import annotations

import json
import os

from scripts.audit.incident_register import certify, runner

UNAVAILABLE = object()


def observe_config(worktree, registry: list[dict]) -> dict:
    wt = os.path.realpath(worktree)
    returns, seen = [], set()
    for r in registry:
        conn = r.get("connection", {})
        fns = [conn] if conn.get("kind") == "return" else conn.get("components", [])
        for c in fns:
            key = (c["path"], c["qualname"])
            if key not in seen:
                seen.add(key)
                returns.append({"file": os.path.join(wt, c["path"]), "qualname": c["qualname"]})
    return {"nodes": [r["node"] for r in registry],
            "entries": [{"file": os.path.join(wt, r["entry"]["path"]), "qualname": r["entry"]["qualname"]}
                        for r in registry], "boundaries": [], "returns": returns}


def unwrap(safe):
    """The plain JSON value behind an observer ``_safe`` form, or UNAVAILABLE when any part of it is
    incomplete or unavailable (Codex phase-1 r4 #2: a truncated 50-of-51 list matched a wrong oracle)."""
    t = type(safe)
    if safe is None or t in (bool, int, float, str):
        return safe
    if t is not dict or safe.get("incomplete") or "unavailable" in safe:
        return UNAVAILABLE
    if "seq" in safe:
        parts = [unwrap(v) for v in safe["seq"]]
        return UNAVAILABLE if any(v is UNAVAILABLE for v in parts) else parts
    for key in ("map", "fields"):
        if key in safe:
            parts = {k: unwrap(v) for k, v in safe[key].items()}
            return UNAVAILABLE if any(v is UNAVAILABLE for v in parts.values()) else parts
    return UNAVAILABLE


def _field(safe, name: str):
    """One named field of an observed object, complete on its own (the object may omit private fields)."""
    fields = safe.get("fields") if type(safe) is dict else None
    if type(fields) is not dict or name not in fields:
        return UNAVAILABLE
    return unwrap(fields[name])


def _returns(inside: list[dict], wt: str, comp: dict) -> list[dict]:
    path = os.path.join(wt, comp["path"])
    return [e for e in inside if e["kind"] == "return" and e["file"] == path and e["qualname"] == comp["qualname"]]


def connection(unmarked: runner.Run, node: str, reg: dict, actual, wt: str) -> tuple[str, list[str]]:
    """(status, reasons): ``value_match`` | ``exception_shape`` (derived) | ``unmatched``."""
    conn = reg.get("connection") or {}
    kind = conn.get("kind")
    if not str(conn.get("review", "")).strip():
        return "unmatched", [f"{kind} connection without a recorded fixture review"]
    if kind == "derived":
        return "exception_shape", []
    inside, why = certify.interval(unmarked.events, node)
    if why:
        return "unmatched", [f"connection: {w}" for w in why]
    body = inside[:-1]
    entry_file = os.path.join(wt, reg["entry"]["path"])
    entries = [e for e in body if e["kind"] == "entry" and e["file"] == entry_file
               and e["qualname"] == reg["entry"]["qualname"]]
    if kind == "return":
        rets = _returns(body, wt, conn)
        if not rets:
            return "unmatched", [f"connection: no return of {conn['qualname']} observed"]
        got = unwrap(rets[-1]["value"])
        if got is UNAVAILABLE:
            return "unmatched", [f"connection: {conn['qualname']}'s return was not observed completely"]
        return ("value_match", []) if got == actual else \
            ("unmatched", [f"connection: {conn['qualname']} returned {got!r}, the oracle saw {actual!r}"])
    if kind == "reads":
        if not entries:
            return "unmatched", ["connection: the production entry never ran"]
        first = entries[0]["seq"]
        values = []
        for comp in conn["components"]:
            outside = [r for r in _returns(body, wt, comp) if r["seq"] > first
                       and not any(certify._live(en, r, body) for en in entries)]
            if not outside:
                return "unmatched", [f"connection: the fixture never read {comp['qualname']} after the entry ran"]
            vals = [_field(r["value"], comp["field"]) if comp.get("field") else unwrap(r["value"]) for r in outside]
            if any(v is UNAVAILABLE for v in (vals if comp.get("all") else vals[-1:])):
                return "unmatched", [f"connection: a read of {comp['qualname']} was not observed completely"]
            values.append(vals if comp.get("all") else vals[-1])
        if conn.get("shape") == "zip":
            if len({len(v) for v in values}) != 1:
                return "unmatched", ["connection: zipped read-backs have different cardinalities"]
            got = [list(t) for t in zip(*values)]
        else:
            got = values
        return ("value_match", []) if got == actual else \
            ("unmatched", [f"connection: production read-back {got!r} != the oracle's actual {actual!r}"])
    return "unmatched", [f"unknown connection kind {kind!r}"]


def load_oracle_records(path) -> dict[str, dict]:
    """Last oracle record per node from a ``$W15_ORACLE_OUT`` file."""
    out: dict[str, dict] = {}
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    out[rec["node"]] = rec
    return out


def accept(marked: runner.Run, unmarked: runner.Run, *, worktree, registry: list[dict],
           oracle_records: dict | None = None) -> dict:
    """{node: [reasons]} (empty = accepted), plus "_session" (run-level reasons) and
    "_connections" ({node: value_match | exception_shape | unmatched})."""
    wt = os.path.realpath(worktree)
    nodes = {r["node"] for r in registry}
    out: dict = {"_session": [], "_connections": {}}
    inventory = runner.collected(marked.events)
    out["_session"] += [f"marked: {r}" for r in runner.gate(marked, worktree=worktree, mode="expected_failure",
                                                            expected_failures=frozenset(nodes))]
    out["_session"] += [f"--runxfail: {r}" for r in runner.gate(unmarked, worktree=worktree, mode="mutant",
                                                                expected=inventory)]
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
    records = oracle_records or {}
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
        rec = records.get(n)
        if rec is None:
            why.append("--runxfail: no structured oracle record")
            out["_connections"][n] = "unmatched"
        else:
            if "bad_json" in r and rec["bad"] != r["bad_json"]:
                why.append("oracle record: bad value differs from the registry")
            if "required_json" in r and rec["required"] != r["required_json"]:
                why.append("oracle record: required value differs from the registry")
            if rec["actual"] != rec["bad"]:
                why.append("oracle record: actual is not the declared bad value")
            status, cwhy = connection(unmarked, n, r, rec["actual"], wt)
            out["_connections"][n] = status
            why += cwhy
        out[n] = why
    return out


def _ini_strict(run: runner.Run) -> bool:
    starts = [e for e in run.events if e["kind"] == "session_start"]
    return bool(starts and starts[0].get("xfail_strict_ini"))


def run_pair(worktree, registry: list[dict], tests: list[str], out_dir, *, env: dict | None = None) -> dict:
    """Marked + ``--runxfail`` runs of ONE frozen execution closure, then ``accept``.

    Every tracked file's working bytes, the untracked list and the environment's content fingerprint
    are taken before the pair and re-checked after EACH run (Codex phase-1 r3 #4: a helper changed
    between the runs was accepted). A node counts as a reproduction only when the pair verdict is
    ``accepted``; ``connections`` says, per node, ``value_match`` / ``exception_shape`` (each only with
    its recorded fixture review) or ``unmatched``."""
    import hashlib
    from pathlib import Path

    from scripts.audit.incident_register import owned
    from scripts.audit.incident_register.defence import _drift

    wt, out_dir = Path(worktree), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(env or {})
    reg_text = json.dumps(registry, sort_keys=True)
    res = {"verdict": "rejected", "reasons": [], "registry_sha256": hashlib.sha256(reg_text.encode()).hexdigest()}
    m0 = owned.manifest(wt)
    v0 = owned.venv_fingerprint(wt)
    cfg = observe_config(wt, registry)
    res["observe_sha256"] = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    oracle_path = out_dir / "oracle.jsonl"
    if oracle_path.exists():
        oracle_path.unlink()
    marked = runner.run(wt, tests, out_dir, "marked", observe=cfg, env_extra=env)
    res["reasons"] += _drift(wt, m0["files"], m0["untracked"], v0, (), "after marked")
    unmarked = runner.run(wt, tests + ["--runxfail"], out_dir, "unmarked", observe=cfg,
                          env_extra={**env, "W15_ORACLE_OUT": str(oracle_path)})
    res["reasons"] += _drift(wt, m0["files"], m0["untracked"], v0, (), "after unmarked")
    got = accept(marked, unmarked, worktree=wt, registry=registry, oracle_records=load_oracle_records(oracle_path))
    res["session"] = got["_session"]
    res["connections"] = got["_connections"]
    res["per_node"] = {n: w for n, w in got.items() if not n.startswith("_")}
    res["reasons"] += got["_session"]
    rejected = sorted(n for n, w in res["per_node"].items() if w)
    if rejected:
        res["reasons"].append(f"{len(rejected)} registered node(s) rejected: {rejected[:3]}")
    res["returncodes"] = {"marked": marked.returncode, "unmarked": unmarked.returncode}
    res["raw_events_sha256"] = {r.stage: hashlib.sha256(r.events_path.read_bytes()).hexdigest()
                                for r in (marked, unmarked) if r.events_path.exists()}
    res["verdict"] = "accepted" if not res["reasons"] else "rejected"
    return res
