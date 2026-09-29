"""Historical replay (design §9.2; Codex phase-1 r2 #8).

A *semantic regression replay*: the complete fix set's tests (at the last fix commit F) run on the
``src/`` of the parent P of the FIRST fix commit, inside an owned worktree under the trusted
observer. It is never a deployed-closure replay: cron, script and unit defects are ``unavailable``
here, and the record keeps the deployed ref and its basis as given.

``acceptance.json`` is written on every path. ``accepted`` requires:

* GREEN (F's ``src``): the session gate accepts it — every selected node passes;
* RED (P's ``src``, F's tests): a sound session (no collection or observer errors, same node
  inventory); every declared symptom node fails in its CALL phase with a builtin
  ``AssertionError`` whose last frame is its declared assertion line (an observable-contract
  assertion named by the reviewer); a ``TypeError``/``ImportError`` etc. never counts;
* a harness-compatibility audit covering EVERY P→F change outside ``src/`` (renames keep both
  paths) with a decision in {neutral, irrelevant, adapter} and a non-empty reason (an adapter also
  names its evidence); the entries are retained verbatim in the output;
* the restored worktree equals the pre-run manifest.

Other nodes are classified (``symptom_candidate`` needs the reviewer; it is not a verdict).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from scripts.audit.incident_register import owned, runner
from scripts.audit.incident_register.defence import SpecError, anchor_line, innermost_repo_frame

NEW_API = {"ImportError", "ModuleNotFoundError", "AttributeError", "TypeError", "NameError"}
DECISIONS = {"neutral", "irrelevant", "adapter"}


def _git(cwd, *args) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def harness_changes(repo, parent: str, fix: str) -> list[dict]:
    """Every P→F change outside ``src/`` (``-z``, renames/copies keep both paths)."""
    raw = subprocess.run(["git", "diff", "-z", "--name-status", "-M", parent, fix, "--", ".", ":(exclude)src"],
                         cwd=repo, check=True, capture_output=True, text=True).stdout
    parts, changes, i = raw.split("\0"), [], 0
    while i < len(parts) and parts[i]:
        status = parts[i]
        if status.startswith(("R", "C")):
            changes.append({"status": status, "from": parts[i + 1], "path": parts[i + 2]})
            i += 3
        else:
            changes.append({"status": status, "path": parts[i + 1]})
            i += 2
    return changes


def audit_problems(changes: list[dict], audit: dict) -> list[str]:
    why = []
    for c in changes:
        for path in {c["path"], c.get("from") or c["path"]}:
            entry = (audit or {}).get(path)
            if not entry:
                why.append(f"unaudited harness change: {path}")
            elif entry.get("decision") not in DECISIONS:
                why.append(f"{path}: decision {entry.get('decision')!r} is not one of {sorted(DECISIONS)}")
            elif not str(entry.get("reason", "")).strip():
                why.append(f"{path}: audit decision without a reason")
            elif entry["decision"] == "adapter" and not str(entry.get("adapter_evidence", "")).strip():
                why.append(f"{path}: adapter decision without adapter evidence")
    return sorted(set(why))


def _drift(worktree, m0: dict, v0: str, where: str, *, src_too: bool) -> list[str]:
    """Frozen working bytes, untracked files and the environment, compared right after a run."""
    m = owned.manifest(worktree)
    files = m["files"] if src_too else {k: v for k, v in m["files"].items() if not k.startswith("src/")}
    why = []
    changed = sorted(k for k in set(files) | set(m0["files"]) if files.get(k) != m0["files"].get(k))
    if changed:
        why.append(f"{where}: frozen files changed: {changed[:5]}")
    moved = sorted(k for k in set(m["untracked"]) | set(m0["untracked"])
                   if m["untracked"].get(k) != m0["untracked"].get(k) and (src_too or not k.startswith("src/")))
    if moved:                                            # added, removed or changed in content
        why.append(f"{where}: untracked files changed: {moved[:5]}")
    if owned.venv_fingerprint(worktree) != v0:
        why.append(f"{where}: the environment changed")
    return why


def classify(red: runner.Run, green: runner.Run, node: str) -> str:
    g = runner.node_state(green.events, node)
    if g != "passed":
        return f"green_{g}"
    r = runner.node_state(red.events, node)
    if r == "missing":
        return "absent_at_parent"
    if r == "passed":
        return "passes_at_parent"
    if r != "failed":
        return r
    call = runner.phases(red.events, node)["call"][0]
    return "new_api" if call.get("exc_qualname") in NEW_API else "symptom_candidate"


def historical_replay(repo, worktree, spec: dict, out_dir) -> dict:
    """``spec``: label, fix_set (oldest→newest), tests, symptom_nodes [{node, assertion {path, text}}],
    audit {path: {decision, reason, adapter_evidence?}}, deployed_ref {sha, basis}, env."""
    worktree, out_dir = Path(worktree), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(json.dumps(spec))
    wt = os.path.realpath(worktree)
    res = {"label": spec.get("label"), "verdict": "rejected", "reasons": [], "closure": "src_swap",
           "label_kind": "semantic_regression_replay", "spec": spec,
           "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()}
    why = res["reasons"]
    owned_ok, fix, completed = False, None, False
    try:
        for key in ("label", "fix_set", "tests", "symptom_nodes", "audit", "deployed_ref"):
            if key not in spec:
                raise SpecError(f"spec lacks {key!r}")
        if not spec["fix_set"] or not spec["symptom_nodes"]:
            raise SpecError("empty fix_set or symptom_nodes")
        owned.assert_owned(worktree)
        owned_ok = True
        fix = _git(repo, "rev-parse", spec["fix_set"][-1])
        first = _git(repo, "rev-parse", spec["fix_set"][0])
        parent = _git(repo, "rev-parse", f"{first}^")
        for c in spec["fix_set"]:
            if subprocess.run(["git", "merge-base", "--is-ancestor", c, fix], cwd=repo).returncode != 0:
                raise SpecError(f"fix-set commit {c} is not an ancestor of {fix}")
        res.update(fix=fix, parent=parent)
        owned.reset(worktree, fix)
        for s in spec["symptom_nodes"]:
            rel = s["assertion"]["path"]
            if rel.startswith("src/"):
                raise SpecError(f"{rel}: symptom assertions live in frozen (non-src) files")
            s["_line"] = anchor_line(worktree / rel, s["assertion"]["text"], s["assertion"].get("line"))
            s["_path"] = os.path.join(wt, rel)
        m0 = owned.manifest(worktree)
        v0 = owned.venv_fingerprint(worktree)
        frozen0 = {k: v for k, v in m0["files"].items() if not k.startswith("src/")}
        changes = harness_changes(repo, parent, fix)
        res["harness_changes"] = changes
        problems = audit_problems(changes, spec["audit"])
        res["audit_problems"] = problems
        if problems:
            res["label_kind"] = "semantic_regression_replay_unaudited"
            why += problems
        env = spec.get("env", {})
        res["src_tree"] = {"fix": _git(repo, "rev-parse", f"{fix}:src"), "parent": _git(repo, "rev-parse", f"{parent}:src")}

        green = runner.run(worktree, spec["tests"], out_dir, "green", env_extra=env)
        inventory = runner.collected(green.events)
        why += [f"green: {r}" for r in runner.gate(green, worktree=worktree, mode="green")]
        why += _drift(worktree, m0, v0, "after green", src_too=True)       # before any swap or reset
        owned.swap_src(worktree, parent)
        red = runner.run(worktree, spec["tests"], out_dir, "red", env_extra=env)
        # the red run may fail CALLS (that is the replay); setup/teardown/collection errors, missing
        # or duplicate phases and an unexplained return code still reject (Codex phase-1 r3 #3)
        why += [f"red: {r}" for r in runner.gate(red, worktree=worktree, mode="mutant", expected=inventory)]
        why += _drift(worktree, {"files": frozen0, "untracked": m0["untracked"]}, v0, "after red", src_too=False)
        if subprocess.run(["git", "diff", "--quiet", parent, "--", "src"], cwd=worktree).returncode != 0:
            why.append("after red: src is no longer exactly the parent's src")
        res["classes"] = {n: classify(red, green, n) for n in inventory}
        res["symptoms"] = {}
        for s in spec["symptom_nodes"]:
            n, bad = s["node"], []
            if runner.node_state(red.events, n) != "failed":
                bad.append(f"state {runner.node_state(red.events, n)} at the parent")
            else:
                call = runner.phases(red.events, n)["call"][0]
                if (call.get("exc_module"), call.get("exc_qualname")) != ("builtins", "AssertionError"):
                    bad.append(f"raised {call.get('exc_module')}.{call.get('exc_qualname')}")
                last = innermost_repo_frame(call.get("frames") or [], wt)
                if not last or last[0] != s["_path"] or last[1] != s["_line"]:
                    bad.append(f"raised at {last[:2] if last else None}, not {s['_path']}:{s['_line']}")
            res["symptoms"][n] = bad
            why += [f"symptom {n}: {b}" for b in bad]
        owned.reset(worktree, fix)
        m1 = owned.manifest(worktree)
        if m1 != m0:
            why.append("the restored worktree differs from the pre-run manifest")
        res["events_sha256"] = {r.stage: hashlib.sha256(r.events_path.read_bytes()).hexdigest()
                                for r in (green, red) if r.events_path.exists()}
        completed = True
    except (SpecError, owned.OwnershipError, FileNotFoundError, ValueError, KeyError,
            subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        why.append(f"refused: {type(e).__name__}: {e}")
    except BaseException as e:
        why.append(f"aborted: {type(e).__name__}: {e}")     # recorded, then surfaced to the caller
        raise
    finally:
        if owned_ok and fix:
            try:
                owned.reset(worktree, fix)
            except Exception as e:  # noqa: BLE001
                why.append(f"final reset failed: {type(e).__name__}: {e}")
        res["verdict"] = "accepted" if completed and not why else "rejected"     # see current_defence
        (out_dir / "acceptance.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=str) + "\n")
    return res
