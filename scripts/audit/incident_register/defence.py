"""Current-defence certificates (design §9.3; Codex phase-1 r2 #1–#3, #5).

One run certifies one causal link of one incident at the pinned baseline, inside an OWNED worktree
(``owned``), with the trusted observer (``runner``) and no source instrumentation:

1. spec checks — mutation targets obey the hard rule, assertion anchors are in frozen files, the
   baseline is reset and every tracked file's working bytes and the venv are fingerprinted;
2. GREEN — the selected tests under the observer (entry + boundary recorder on the killing
   nodes); the session gate accepts it;
3. MUTANT — the mutation edits (the smallest semantic mutant restoring the pre-fix decision) touch
   only allowed ``src/bts`` files, every other file is byte-identical; the observer also watches the
   mutated line (a unique anchor in the mutated file); the gate accepts the session with the SAME
   inventory; each declared killing node fails in its CALL phase with a builtin ``AssertionError``
   whose last frame is the declared assertion line of its frozen test file; each has a certificate;
4. RESTORE — reset, identical manifest, GREEN again with the same inventory.

Each observed run has an observer-off twin (same per-node states, same failure); automatic garbage
collection is off during the killing nodes' call phase in both, so the twins differ only by observation.
An absence spec is refused before anything runs (plan ruling 10, ``certify.ABSENCE_REFUSAL``).

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

KINDS = ("event", "return")


class SpecError(ValueError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def anchor_line(path: Path, text: str, line: int | None = None) -> int:
    """The anchor's 1-based line: ``text`` must be unique in the file, or — for repeated assertion
    text — ``line`` names the line (in a frozen file) and must contain ``text``."""
    lines = Path(path).read_text().splitlines()
    if line is not None:
        if not (1 <= line <= len(lines)) or text not in lines[line - 1]:
            raise SpecError(f"{path}: line {line} does not contain the anchor {text!r}")
        return line
    hits = [i for i, ln in enumerate(lines, 1) if text in ln]
    if len(hits) != 1:
        raise SpecError(f"{path}: anchor {text!r} matches {len(hits)} lines (need exactly 1)")
    return hits[0]


def _check_spec(spec: dict) -> None:
    for key in ("label", "baseline", "tests", "allowed_paths", "mutation_edits", "branch", "entry", "symptom", "killing"):
        if key not in spec:
            raise SpecError(f"spec lacks {key!r}")
    if spec["symptom"].get("kind") == "absence":
        raise SpecError(certify.ABSENCE_REFUSAL)
    if spec["symptom"].get("kind") not in KINDS:
        raise SpecError(f"symptom kind must be one of {KINDS}")
    reserved = certify.RESERVED_CATEGORY
    if spec["symptom"].get("category") == reserved or any(
            rule[0] == reserved for item in [*spec.get("boundaries", []), *spec.get("returns", [])]
            for rule in item.get("classify", [])):
        raise SpecError(f"the category {reserved!r} is reserved for unattributed or incomplete records: no "
                        "symptom or classify rule may name it (Codex phase-1 r8 #2)")
    if not spec["killing"]:
        raise SpecError("no killing node declared")
    if spec["branch"]["path"] not in {e[0] for e in spec["mutation_edits"]}:
        raise SpecError("the branch anchor must be in a mutated file")


def _drift(worktree, frozen: dict, untracked: dict, venv: str, exclude, where: str) -> list[str]:
    m = owned.manifest(worktree, exclude=exclude)
    why = []
    changed = sorted(k for k in set(frozen) | set(m["files"]) if frozen.get(k) != m["files"].get(k))
    if changed:
        why.append(f"{where}: frozen files changed: {changed[:5]}")
    moved = sorted(k for k in set(untracked) | set(m["untracked"]) if untracked.get(k) != m["untracked"].get(k))
    if moved:                                            # added, removed or changed in content
        why.append(f"{where}: untracked files changed: {moved[:5]}")
    if owned.venv_fingerprint(worktree) != venv:
        why.append(f"{where}: the venv changed")
    return why


def innermost_repo_frame(frames: list, worktree: str) -> list | None:
    """The innermost traceback frame inside the worktree (its own venv excluded): the test line that
    raised, even when the AssertionError itself comes from a stdlib helper such as unittest.mock."""
    root, venv = worktree.rstrip(os.sep) + os.sep, os.path.join(worktree, ".venv") + os.sep
    inside = [f for f in frames if f[0].startswith(root) and not f[0].startswith(venv)]
    return inside[-1] if inside else None


def _failure(run: runner.Run, node: str, worktree: str):
    """(exception module.qualname, innermost worktree frame [path, line]) of a failed call phase."""
    calls = runner.phases(run.events, node).get("call", [])
    if len(calls) != 1:
        return None
    c = calls[0]
    last = innermost_repo_frame(c.get("frames") or [], worktree)
    return (f"{c.get('exc_module')}.{c.get('exc_qualname')}", last[:2] if last else None)


def _message(run: runner.Run, node: str) -> str | None:
    """The digest of a failed call phase's message, run-variant parts blanked (``observer._message_digest``)."""
    calls = runner.phases(run.events, node).get("call", [])
    return calls[0].get("message_sha256") if len(calls) == 1 else None


def _conformance(worktree, observed: runner.Run, out_dir, tests, env, stage: str, mode: str,
                 inventory: list[str], quiet: list[str]) -> tuple[runner.Run, list[str]]:
    """Re-run ``tests`` with NO observation and require identical per-node states — and, for a node that
    fails both ways, the same failure: exception type and innermost worktree frame (design §9.3 as
    amended: observer-on behaviour is validated against an observer-off control; Codex phase-1 r4 #2) and the
    same message, once what differs between any two runs is blanked. Code that inspects the interpreter can see
    the observer (Codex phase-1 r18 #1, #2; r17 #1) and is outside the model (proposed ruling 13); this refuses
    such a divergence when it reaches the failing assertion's message, and a missing message is never agreement."""
    plain = runner.run(worktree, tests, out_dir, stage, observe=None, quiesce=quiet, env_extra=env)
    why = [f"{stage}: {r}" for r in runner.gate(plain, worktree=worktree, mode=mode, expected=inventory)]
    wt = os.path.realpath(worktree)
    for n in inventory:
        a, b = runner.node_state(observed.events, n), runner.node_state(plain.events, n)
        if a != b:
            why.append(f"{stage}: {n} is {a} observed but {b} unobserved")
        elif a == "failed":
            fa, fb = _failure(observed, n, wt), _failure(plain, n, wt)
            if fa != fb:
                why.append(f"{stage}: {n} failed differently observed ({fa}) and unobserved ({fb})")
                continue
            ma, mb = _message(observed, n), _message(plain, n)
            if ma is None or ma != mb:
                why.append(f"{stage}: {n} failed with a different message observed and unobserved, or one not "
                           "recorded: observation may have changed what the failing assertion saw (proposed ruling 13)")
    return plain, why


def _assertion_ok(run: runner.Run, node: str, path: str, line: int, worktree: str) -> list[str]:
    calls = runner.phases(run.events, node).get("call", [])
    if len(calls) != 1:
        return [f"{node}: {len(calls)} call reports"]
    call = calls[0]
    why = []
    if (call.get("exc_module"), call.get("exc_qualname")) != ("builtins", "AssertionError"):
        why.append(f"{node}: killed by {call.get('exc_module')}.{call.get('exc_qualname')}, not AssertionError")
    last = innermost_repo_frame(call.get("frames") or [], worktree)
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
    owned_ok = completed = False
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
            k["_line"] = anchor_line(worktree / rel, k["assertion"]["text"], k["assertion"].get("line"))
            k["_path"] = os.path.join(wt, rel)
        knodes = [k["node"] for k in spec["killing"]]
        entry = {"file": os.path.join(wt, spec["entry"]["path"]), "qualname": spec["entry"]["qualname"]}
        returns = [{"file": os.path.join(wt, r["path"]), "qualname": r["qualname"], "classify": r.get("classify", [])}
                   for r in spec.get("returns", [])]
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
        green_plain, cw = _conformance(worktree, green, out_dir, spec["tests"], env, "green_unobserved", "green",
                                       inventory, knodes)
        why += cw
        why += _drift(worktree, m0["files"], m0["untracked"], v0, (), "after green_unobserved")
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
        mutant_plain, cw = _conformance(worktree, mutant, out_dir, spec["tests"], env, "mutant_unobserved", "mutant",
                                        inventory, knodes)
        why += cw
        why += [f"mutant: {r}" for r in mg]
        why += _drift(worktree, frozen, m0["untracked"], v0, touched, "after mutant")
        sym = spec["symptom"]
        if sym["kind"] == "return":
            bad = {"file": os.path.join(wt, sym["path"]), "qualname": sym["qualname"]}
            bad.update({k: sym[k] for k in ("category", "value") if k in sym})
        else:
            bad = {"boundary": sym["boundary"], "category": sym["category"]}
        res["certificates"] = {}
        for k in spec["killing"]:
            n = k["node"]
            if n not in kills:
                why.append(f"declared killing node {n} was not killed")
                continue
            why += _assertion_ok(mutant, n, k["_path"], k["_line"], wt)
            cert = certify.certify(mutant.events, node=n, kind=sym["kind"], entry=entry, bad=bad)
            res["certificates"][n] = cert
            why += [f"certificate {n}: {r}" for r in cert["reasons"]]

        owned.reset(worktree, spec["baseline"])
        why += _drift(worktree, m0["files"], m0["untracked"], v0, (), "after restore")
        restored = runner.run(worktree, spec["tests"], out_dir, "restored", observe=observe, env_extra=env)
        rg = runner.gate(restored, worktree=worktree, mode="green", expected=inventory)
        res["stages"]["restored"] = {"returncode": restored.returncode, "gate": rg}
        why += [f"restored: {r}" for r in rg]
        why += _drift(worktree, m0["files"], m0["untracked"], v0, (), "after restored run")
        res["events_sha256"] = {r.stage: _sha(r.events_path.read_bytes())
                                for r in (green, green_plain, mutant, mutant_plain, restored) if r.events_path.exists()}
        completed = True
    except (SpecError, owned.OwnershipError, owned.PathRefused, FileNotFoundError, ValueError, KeyError,
            subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        why.append(f"refused: {type(e).__name__}: {e}")
    except BaseException as e:
        why.append(f"aborted: {type(e).__name__}: {e}")     # recorded, then surfaced to the caller
        raise
    finally:
        if owned_ok:
            try:
                owned.reset(worktree, spec["baseline"])
            except Exception as e:  # noqa: BLE001 - recorded; never widens what was reset
                why.append(f"final reset failed: {type(e).__name__}: {e}")
        # accepted only when every stage ran to the end with no reason recorded (self-review 2026-09-29:
        # an unhandled exception before the first reason used to leave an 'accepted' artifact)
        res["verdict"] = "accepted" if completed and not why else "rejected"
        (out_dir / "acceptance.json").write_text(json.dumps(res, indent=2, sort_keys=True, default=str) + "\n")
    return res
