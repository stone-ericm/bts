"""Current-defence certificates (design §9.3; Codex phase-1 r2 #1–#3, #5).

One run certifies one causal link of one incident at the pinned baseline, inside an OWNED worktree
(``owned``), with the trusted observer (``runner``) and no source instrumentation:

1. spec checks — mutation targets obey the hard rule, assertion anchors are in frozen files, the
   baseline is reset and every tracked file's working bytes and the venv are fingerprinted;
2. GREEN — the selected tests under the observer (entry + boundary recorder on the killing
   nodes); the session gate accepts it; this run is also the positive control of an absence claim;
3. MUTANT — the mutation edits (the smallest semantic mutant restoring the pre-fix decision) touch
   only allowed ``src/bts`` files, every other file is byte-identical; the observer also watches the
   mutated line (a unique anchor in the mutated file); the gate accepts the session with the SAME
   inventory; each declared killing node fails in its CALL phase with a builtin ``AssertionError``
   whose last frame is the declared assertion line of its frozen test file; each has a certificate;
4. RESTORE — reset, identical manifest, GREEN again with the same inventory.

The manifest and venv fingerprint are compared after every subprocess. ``acceptance.json`` is
written on every path; the verdict is ``accepted`` only when no reason was recorded. The runner's
verdict is necessary, not sufficient: a certificate also needs the reviewer's decision.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from scripts.audit.incident_register import certify, owned, runner

KINDS = ("event", "absence", "return")


class SpecError(ValueError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def anchor_line(path: Path, text: str) -> int:
    hits = [i for i, line in enumerate(Path(path).read_text().splitlines(), 1) if text in line]
    if len(hits) != 1:
        raise SpecError(f"{path}: anchor {text!r} matches {len(hits)} lines (need exactly 1)")
    return hits[0]


def _check_spec(spec: dict) -> None:
    for key in ("label", "baseline", "tests", "allowed_paths", "mutation_edits", "branch", "entry", "symptom", "killing"):
        if key not in spec:
            raise SpecError(f"spec lacks {key!r}")
    if spec["symptom"].get("kind") not in KINDS:
        raise SpecError(f"symptom kind must be one of {KINDS}")
    if not spec["killing"]:
        raise SpecError("no killing node declared")
    if spec["branch"]["path"] not in {e[0] for e in spec["mutation_edits"]}:
        raise SpecError("the branch anchor must be in a mutated file")


def _drift(worktree, frozen: dict, untracked: list, venv: str, exclude, where: str) -> list[str]:
    m = owned.manifest(worktree, exclude=exclude)
    why = []
    changed = sorted(k for k in set(frozen) | set(m["files"]) if frozen.get(k) != m["files"].get(k))
    if changed:
        why.append(f"{where}: frozen files changed: {changed[:5]}")
    if m["untracked"] != untracked:
        why.append(f"{where}: untracked files changed: {sorted(set(m['untracked']) ^ set(untracked))[:5]}")
    if owned.venv_fingerprint(worktree) != venv:
        why.append(f"{where}: the venv changed")
    return why


def _assertion_ok(run: runner.Run, node: str, path: str, line: int) -> list[str]:
    calls = runner.phases(run.events, node).get("call", [])
    if len(calls) != 1:
        return [f"{node}: {len(calls)} call reports"]
    call = calls[0]
    why = []
    if (call.get("exc_module"), call.get("exc_qualname")) != ("builtins", "AssertionError"):
        why.append(f"{node}: killed by {call.get('exc_module')}.{call.get('exc_qualname')}, not AssertionError")
    last = call["frames"][-1] if call.get("frames") else None
    if not last or last[0] != path or last[1] != line:
        why.append(f"{node}: failure raised at {last[:2] if last else None}, not the declared assertion {path}:{line}")
    return why


def current_defence(worktree, spec: dict, out_dir) -> dict:
    worktree, out_dir = Path(worktree), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wt = os.path.realpath(worktree)
    spec = json.loads(json.dumps(spec))          # private copy: derived fields never leak back
    res = {"label": spec.get("label"), "verdict": "rejected", "reasons": [],
           "spec_sha256": _sha(json.dumps(spec, sort_keys=True).encode()), "spec": spec, "stages": {}}
    why = res["reasons"]
    owned_ok = False
    try:
        _check_spec(spec)
        owned.assert_owned(worktree)
        owned_ok = True
        owned.reset(worktree, spec["baseline"])
        res["baseline"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=worktree, check=True,
                                         capture_output=True, text=True).stdout.strip()
        allowed = set(spec["allowed_paths"])
        for rel in sorted(allowed):
            owned.check_mutation_path(worktree, rel)
        for k in spec["killing"]:
            rel = k["assertion"]["path"]
            if rel.startswith("src/"):
                raise SpecError(f"{rel}: the killing assertion must be in a frozen (non-src) file")
            k["_line"] = anchor_line(worktree / rel, k["assertion"]["text"])
            k["_path"] = os.path.join(wt, rel)
        knodes = [k["node"] for k in spec["killing"]]
        entry = {"file": os.path.join(wt, spec["entry"]["path"]), "qualname": spec["entry"]["qualname"]}
        returns = [{"file": os.path.join(wt, r["path"]), "qualname": r["qualname"]} for r in spec.get("returns", [])]
        observe = {"nodes": knodes, "entries": [entry], "boundaries": spec.get("boundaries", []), "returns": returns}
        env = spec.get("env", {})
        m0 = owned.manifest(worktree)
        v0 = owned.venv_fingerprint(worktree)
        res["manifest_sha256"] = owned.manifest_digest(m0)

        green = runner.run(worktree, spec["tests"], out_dir, "green", observe=observe, env_extra=env)
        inventory = runner.collected(green.events)
        g = runner.gate(green, worktree=worktree, mode="green")
        res["stages"]["green"] = {"returncode": green.returncode, "nodes": len(inventory), "gate": g}
        why += [f"green: {r}" for r in g]
        why += _drift(worktree, m0["files"], m0["untracked"], v0, (), "after green")
        why += [f"killing node {n} not in the green inventory" for n in knodes if n not in inventory]
        if why:
            return res

        touched = owned.apply_edits(worktree, spec["mutation_edits"], allowed)
        frozen = {k: v for k, v in m0["files"].items() if k not in touched}
        why += _drift(worktree, frozen, m0["untracked"], v0, touched, "after mutation")
        patch = subprocess.run(["git", "diff", "--", *touched], cwd=worktree, check=True,
                               capture_output=True).stdout
        (out_dir / "mutant.patch").write_bytes(patch)
        res["patch_sha256"] = _sha(patch)
        res["touched"] = touched
        line = anchor_line(worktree / spec["branch"]["path"], spec["branch"]["text"])
        res["branch_line"] = line
        observe_m = {**observe, "branch": {"file": os.path.join(wt, spec["branch"]["path"]), "line": line}}
        mutant = runner.run(worktree, spec["tests"], out_dir, "mutant", observe=observe_m, env_extra=env)
        mg = runner.gate(mutant, worktree=worktree, mode="mutant", expected=inventory)
        kills = [n for n in inventory if runner.node_state(mutant.events, n) == "failed"]
        res["stages"]["mutant"] = {"returncode": mutant.returncode, "kills": kills, "gate": mg}
        why += [f"mutant: {r}" for r in mg]
        why += _drift(worktree, frozen, m0["untracked"], v0, touched, "after mutant")
        sym = spec["symptom"]
        if sym["kind"] == "return":
            bad = {"file": os.path.join(wt, sym["path"]), "qualname": sym["qualname"], "value": sym["value"]}
        else:
            bad = {"boundary": sym["boundary"], "category": sym["category"]}
        res["certificates"] = {}
        for k in spec["killing"]:
            n = k["node"]
            if n not in kills:
                why.append(f"declared killing node {n} was not killed")
                continue
            why += _assertion_ok(mutant, n, k["_path"], k["_line"])
            cert = certify.certify(mutant.events, node=n, kind=sym["kind"], entry=entry, bad=bad,
                                   positive_events=green.events)
            res["certificates"][n] = cert
            why += [f"certificate {n}: {r}" for r in cert["reasons"]]

        owned.reset(worktree, spec["baseline"])
        why += _drift(worktree, m0["files"], m0["untracked"], v0, (), "after restore")
        restored = runner.run(worktree, spec["tests"], out_dir, "restored", observe=observe, env_extra=env)
        rg = runner.gate(restored, worktree=worktree, mode="green", expected=inventory)
        res["stages"]["restored"] = {"returncode": restored.returncode, "gate": rg}
        why += [f"restored: {r}" for r in rg]
        why += _drift(worktree, m0["files"], m0["untracked"], v0, (), "after restored run")
        res["events_sha256"] = {r.stage: _sha(r.events_path.read_bytes()) for r in (green, mutant, restored)
                                if r.events_path.exists()}
    except (SpecError, owned.OwnershipError, owned.PathRefused, FileNotFoundError, ValueError, KeyError,
            subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        why.append(f"refused: {type(e).__name__}: {e}")
    finally:
        if owned_ok:
            try:
                owned.reset(worktree, spec["baseline"])
            except Exception as e:  # noqa: BLE001 - recorded; never widens what was reset
                why.append(f"final reset failed: {type(e).__name__}: {e}")
        res["verdict"] = "accepted" if not why else "rejected"
        (out_dir / "acceptance.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=str) + "\n")
    return res
