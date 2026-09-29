"""Fixture evidence runs for the W1.5 incident register (design §9.1–§9.5).

Two kinds of run, each in an isolated git worktree with its own environment:

* ``historical_replay`` — the fix commit's tests run on the parent's ``src/`` (red) and on the
  fix's ``src/`` (green). ``src/`` is replaced wholesale and verified equal to the ref: a plain
  ``git checkout <ref> -- src/`` keeps files the ref lacks, which made a pre-fix run pass on the
  fix's new module during the pilot. Results are per node and per phase, so a collection error or
  a new-API ``TypeError`` is never mistaken for the incident's symptom. The F^→F change set outside
  ``src/`` is listed (harness-compatibility audit); any change there keeps the run labelled a
  *semantic regression replay* until it is reviewed.
* ``current_defence`` — at the pinned baseline: green run, apply one reviewed mutant patch (plus
  the witness module), red run with ``$W15_WITNESS_PATH`` set, restore, green run again. Test files
  are hashed before and after so a patch can never touch them.

Nothing here reads data outside the repository.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

NEW_API = ("ImportError", "ModuleNotFoundError", "AttributeError", "TypeError", "NameError")
WITNESS_SRC = Path(__file__).with_name("witness.py")


def sh(args: list[str], cwd: Path, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=cwd, env=env, check=check, capture_output=True, text=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def use_src(worktree: Path, ref: str) -> str:
    """Make ``worktree/src`` exactly ``ref:src`` (no extra files). Returns the ref's src tree id."""
    shutil.rmtree(worktree / "src", ignore_errors=True)
    sh(["git", "checkout", ref, "--", "src/"], worktree)
    diff = sh(["git", "diff", "--quiet", ref, "--", "src"], worktree, check=False)
    extra = sh(["git", "ls-files", "--others", "--exclude-standard", "src"], worktree).stdout.strip()
    if diff.returncode != 0 or extra:
        raise RuntimeError(f"src/ is not exactly {ref}: diff={diff.returncode} extra={extra!r}")
    return sh(["git", "rev-parse", f"{ref}:src"], worktree).stdout.strip()


@dataclass
class NodeResult:
    outcome: str          # passed | failed | error | skipped
    phase: str            # call | setup | teardown | collection | ""
    exc_type: str = ""
    message: str = ""


def parse_junit(path: Path) -> dict[str, NodeResult]:
    """Per-node results from pytest's junit XML (a local file the run just wrote)."""
    results: dict[str, NodeResult] = {}
    root = ET.parse(path).getroot()
    for tc in root.iter("testcase"):
        node = f"{tc.get('classname')}::{tc.get('name')}"
        res = NodeResult("passed", "call")
        for child in tc:
            msg = child.get("message") or ""
            if child.tag == "failure":
                res = NodeResult("failed", "call", _exc_type(child), msg)
            elif child.tag == "error":
                phase = ("setup" if "failed on setup" in msg else
                         "teardown" if "failed on teardown" in msg else "collection")
                res = NodeResult("error", phase, _exc_type(child), msg)
            elif child.tag == "skipped":
                res = NodeResult("skipped", "", child.get("type") or "", msg)
            else:
                continue
            break
        results[node] = res
    return results


def _exc_type(el) -> str:
    """The exception class name: the head of junit's ``message`` ("Type: text"), else the first
    ``E   Type:`` traceback line. Continuation lines of a multi-line message (e.g. mock's
    ``Calls: [...]``) are never read as the type."""
    import re
    head = re.match(r"\s*([A-Za-z_][\w.]*)\s*:", el.get("message") or "")
    if head:
        return head.group(1).rsplit(".", 1)[-1]
    for line in (el.text or "").splitlines():
        m = re.match(r"\s*E\s+([A-Za-z_][\w.]*)\s*:", line)
        if m:
            return m.group(1).rsplit(".", 1)[-1]
    return ""


def classify(red: NodeResult | None, green: NodeResult | None) -> str:
    """Mechanical pre-classification; `symptom_candidate` still needs a human symptom review."""
    if green is None or green.outcome != "passed":
        return "green_not_passing"
    if red is None:
        return "absent_at_parent"
    if red.outcome == "passed":
        return "passes_at_parent"
    if red.outcome == "skipped":
        return "skipped"
    if red.phase != "call":
        return f"{red.phase}_error"
    if red.exc_type in NEW_API or any(t in red.message for t in NEW_API):
        return "new_api"
    return "symptom_candidate"


def run_pytest(worktree: Path, args: list[str], junit: Path, python: list[str],
               env: dict | None = None) -> dict:
    """Run pytest via ``python`` (e.g. ["uv", "run", "python"]); record which bts was imported."""
    full_env = dict(os.environ, **(env or {}))
    junit.unlink(missing_ok=True)
    proc = sh([*python, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider", f"--junitxml={junit}"],
              worktree, env=full_env, check=False)
    imported = sh([*python, "-c", "import bts; print(bts.__file__)"], worktree, env=full_env,
                  check=False).stdout.strip()
    return {"exit": proc.returncode, "imported_bts": imported,
            "nodes": {k: vars(v) for k, v in parse_junit(junit).items()} if junit.exists() else {}}


def harness_changes(repo: Path, parent: str, fix: str) -> list[dict]:
    """Every F^→F change outside src/ (tests, conftest, fixtures, config, lock, scripts)."""
    out = sh(["git", "diff", "--name-status", parent, fix, "--", ".", ":(exclude)src"], repo).stdout
    changes = []
    for line in out.splitlines():
        status, _, path = line.partition("\t")
        kind = ("conftest" if path.endswith("conftest.py") else
                "test_data" if path.startswith("tests/") and not path.endswith(".py") else
                "test" if path.startswith("tests/") else
                "lock_or_config" if path in ("uv.lock", "pyproject.toml") else
                "script" if path.startswith("scripts/") else "other")
        changes.append({"status": status, "path": path, "kind": kind})
    return changes


@dataclass
class Replay:
    label: str
    fix: str
    parent: str
    tests: list[str]
    pins: dict = field(default_factory=dict)
    red: dict = field(default_factory=dict)
    green: dict = field(default_factory=dict)
    harness: list = field(default_factory=list)
    classes: dict = field(default_factory=dict)

    @property
    def label_kind(self) -> str:
        risky = [c for c in self.harness if c["kind"] in ("conftest", "test_data", "lock_or_config")
                 or (c["kind"] == "test" and c["path"] not in self.tests)]
        return "semantic_regression_replay_unaudited" if risky else "semantic_regression_replay"


def historical_replay(repo: Path, worktree: Path, label: str, fix: str, tests: list[str],
                      out_dir: Path, python: list[str], sync: list[str] | None = None) -> Replay:
    out_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "checkout", "-q", "--detach", "-f", fix], worktree)
    sh(["git", "clean", "-qfdx", "src", "tests"], worktree)
    if sync:
        sh(sync, worktree)
    parent = sh(["git", "rev-parse", f"{fix}^"], worktree).stdout.strip()
    fix_full = sh(["git", "rev-parse", fix], worktree).stdout.strip()
    rep = Replay(label, fix_full, parent, tests)
    rep.pins = {"tests": {t.split("::")[0]: sha256_file(worktree / t.split("::")[0]) for t in tests},
                "lock": sha256_file(worktree / "uv.lock") if (worktree / "uv.lock").exists() else None}
    rep.harness = harness_changes(repo, parent, fix_full)
    rep.pins["src_parent"] = use_src(worktree, parent)
    rep.red = run_pytest(worktree, tests, out_dir / "red.xml", python)
    rep.pins["src_fix"] = use_src(worktree, fix_full)
    rep.green = run_pytest(worktree, tests, out_dir / "green.xml", python)
    for node in sorted(set(rep.red["nodes"]) | set(rep.green["nodes"])):
        r = rep.red["nodes"].get(node)
        g = rep.green["nodes"].get(node)
        rep.classes[node] = classify(NodeResult(**r) if r else None, NodeResult(**g) if g else None)
    (out_dir / "replay.json").write_text(json.dumps(
        {**vars(rep), "label_kind": rep.label_kind}, indent=2, sort_keys=True) + "\n")
    return rep


def current_defence(worktree: Path, baseline: str, label: str, patch: Path, tests: list[str],
                    out_dir: Path, python: list[str]) -> dict:
    """Green baseline → mutant (with witness) → restored baseline, same tests, hashed tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sh(["git", "checkout", "-q", "--detach", "-f", baseline], worktree)
    sh(["git", "clean", "-qfdx", "src", "tests"], worktree)
    test_files = sorted({t.split("::")[0] for t in tests})
    hashes = {f: sha256_file(worktree / f) for f in test_files}
    result = {"label": label, "baseline": sh(["git", "rev-parse", "HEAD"], worktree).stdout.strip(),
              "patch": str(patch), "patch_sha256": sha256_file(patch), "test_sha256": hashes}
    result["green_before"] = run_pytest(worktree, tests, out_dir / "green_before.xml", python)
    touched = sh(["git", "apply", "--numstat", str(patch)], worktree).stdout.split()
    if any(p.startswith("tests/") for p in touched[2::3]):
        raise RuntimeError("a mutant patch may not touch tests/")
    sh(["git", "apply", str(patch)], worktree)
    shutil.copy(WITNESS_SRC, worktree / "src" / "bts" / "_w15_witness.py")
    witness_path = out_dir / "witness.jsonl"
    witness_path.unlink(missing_ok=True)
    result["mutant"] = run_pytest(worktree, tests, out_dir / "mutant.xml", python,
                                  env={"W15_WITNESS_PATH": str(witness_path)})
    result["tests_unchanged_after_mutant"] = all(
        sha256_file(worktree / f) == h for f, h in hashes.items())
    (worktree / "src" / "bts" / "_w15_witness.py").unlink()
    use_src(worktree, baseline)
    result["green_after"] = run_pytest(worktree, tests, out_dir / "green_after.xml", python)
    result["witness_records"] = sum(1 for _ in open(witness_path)) if witness_path.exists() else 0
    (out_dir / "defence.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


# ---------------------------------------------------------------------------
# v2 (Codex phase-1 r1 #2, #4, #6): structured plugin runs, allowlisted mutations, full restore,
# witness-only positive control, explicit acceptance object. v1 functions above are legacy.
# ---------------------------------------------------------------------------
import tempfile
import uuid

PLUGIN_SRC = Path(__file__).with_name("evidence_plugin.py")
WITNESS_DEST = "src/bts/_w15_witness.py"


def _plugin_dir() -> Path:
    """The plugin as a standalone module (never via the worktree's own ``scripts`` package)."""
    d = Path(tempfile.mkdtemp(prefix="w15plugin-"))
    shutil.copy(PLUGIN_SRC, d / "w15_evidence_plugin.py")
    return d


def run_structured(worktree: Path, args: list[str], out: Path, python: list[str],
                   env: dict | None = None) -> list[dict]:
    """pytest with the evidence plugin; returns its structured events (session, collected,
    per-phase reports, imports) recorded inside the pytest process."""
    from scripts.audit.incident_register.acceptance import load
    out.unlink(missing_ok=True)
    plugin = _plugin_dir()
    full_env = dict(os.environ, **(env or {}))
    full_env["W15_EVIDENCE_OUT"] = str(out)
    full_env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(plugin), full_env.get("PYTHONPATH", "")]))
    proc = sh([*python, "-m", "pytest", *args, "-q", "-p", "no:cacheprovider", "-p", "w15_evidence_plugin"],
              worktree, env=full_env, check=False)
    (out.parent / (out.stem + ".stdout.txt")).write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr)
    shutil.rmtree(plugin, ignore_errors=True)
    return load(out) if out.exists() else []


def reset_worktree(worktree: Path, ref: str) -> None:
    """Exactly ``ref``: every tracked path restored, every untracked/ignored file removed except the venv."""
    sh(["git", "reset", "-q", "--hard", ref], worktree)
    sh(["git", "clean", "-qfdx", "-e", ".venv"], worktree)
    dirty = sh(["git", "status", "--porcelain=v1", "--untracked-files=all", "--ignored"], worktree).stdout
    leftovers = [ln for ln in dirty.splitlines() if not ln[3:].startswith(".venv")]
    if leftovers:
        raise RuntimeError(f"worktree not clean after reset to {ref}: {leftovers[:5]}")


def changed_paths(worktree: Path) -> list[str]:
    raw = sh(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"], worktree).stdout
    paths, parts, i = [], raw.split("\0"), 0
    while i < len(parts):
        entry = parts[i]
        if not entry:
            i += 1
            continue
        status, path = entry[:2], entry[3:]
        paths.append(path)
        if "R" in status or "C" in status:       # rename/copy: the next field is the source path
            paths.append(parts[i + 1])
            i += 1
        i += 1
    return sorted(set(paths))


def apply_edits(worktree: Path, edits: list[list[str]]) -> None:
    for rel, old, new in edits:
        p = worktree / rel
        s = p.read_text()
        n = s.count(old)
        if n != 1:
            raise RuntimeError(f"edit anchor in {rel} matched {n} times: {old[:60]!r}")
        p.write_text(s.replace(old, new))


def _stage(worktree: Path, baseline: str, edits: list[list[str]], allowed: set[str],
           with_witness: bool) -> list[str]:
    reset_worktree(worktree, baseline)
    apply_edits(worktree, edits)
    touched = changed_paths(worktree)
    bad = [p for p in touched if p not in allowed]
    if bad:
        reset_worktree(worktree, baseline)
        raise RuntimeError(f"mutation touches paths outside its allowlist: {bad}")
    if with_witness:
        shutil.copy(Path(__file__).with_name("witness.py"), worktree / WITNESS_DEST)
    return touched


def current_defence_v2(worktree: Path, spec: dict, out_dir: Path, python: list[str]) -> dict:
    """spec: label, baseline, allowed_paths, witness_edits, mutation_edits, tests, killing_node,
    entry, entry_file, branch_file, kind ('event'|'absence'), expected_xfail (optional list).
    Returns an acceptance object whose ``verdict`` is 'accepted' only when every check passes."""
    from scripts.audit.incident_register.acceptance import (accept_green, accept_killed,
                                                            imports_under, node_state)
    from scripts.audit.incident_register.witness import certify, load as load_w, positive_control

    out_dir.mkdir(parents=True, exist_ok=True)
    baseline = sh(["git", "rev-parse", spec["baseline"]], worktree).stdout.strip()
    allowed = set(spec["allowed_paths"])
    run_id = f"{spec['label']}-{uuid.uuid4().hex[:8]}"
    src_root = (worktree / "src").resolve()
    reasons: list[str] = []
    result = {"label": spec["label"], "baseline": baseline, "run_id": run_id, "spec": spec,
              "witness_sha256": sha256_file(Path(__file__).with_name("witness.py")),
              "plugin_sha256": sha256_file(PLUGIN_SRC)}
    try:
        reset_worktree(worktree, baseline)
        result["lock_sha256"] = sha256_file(worktree / "uv.lock")
        tracked_before = sh(["git", "ls-files", "-s"], worktree).stdout
        result["tracked_manifest_sha256"] = hashlib.sha256(tracked_before.encode()).hexdigest()
        expected_xfail = set(spec.get("expected_xfail", []))

        green = run_structured(worktree, spec["tests"], out_dir / "green_before.jsonl", python)
        nodes = [n for n in collected_ids(green)]
        result["nodes"] = len(nodes)
        baseline_states = {n: node_state(green, n) for n in nodes}
        result["baseline_states"] = baseline_states
        bad = {n: st for n, st in baseline_states.items()
               if st not in ("passed", "xfail") or (st == "xfail" and expected_xfail and n not in expected_xfail)}
        if not nodes or bad:
            reasons.append(f"baseline not green ({len(nodes)} nodes): {dict(list(bad.items())[:3])}")
        ok, why = imports_under(green, src_root)
        if not ok:
            reasons.append(f"baseline imports outside the worktree: {why[:3]}")

        wit_path = out_dir / "witness.jsonl"
        wit_path.unlink(missing_ok=True)
        env = {"W15_WITNESS_PATH": str(wit_path), "W15_RUN_ID": run_id + "-witness-only"}
        result["witness_only_touched"] = _stage(worktree, baseline, spec["witness_edits"], allowed, True)
        wonly = run_structured(worktree, spec["tests"], out_dir / "witness_only.jsonl", python, env)
        same = {n: node_state(wonly, n) for n in nodes} == {n: node_state(green, n) for n in nodes}
        if not same:
            reasons.append("witness-only run changed test outcomes (hooks are not observational)")

        env = {"W15_WITNESS_PATH": str(wit_path), "W15_RUN_ID": run_id + "-mutant"}
        result["mutant_touched"] = _stage(worktree, baseline,
                                          spec["witness_edits"] + spec["mutation_edits"], allowed, True)
        mutant = run_structured(worktree, spec["tests"], out_dir / "mutant.jsonl", python, env)
        ok, why = imports_under(mutant, src_root)
        if not ok:
            reasons.append(f"mutant imports outside the worktree: {why[:3]}")
        result["kills"] = sorted(n for n in nodes
                                 if node_state(green, n) == "passed" and node_state(mutant, n) != "passed")
        ok, why = accept_killed(mutant, spec["killing_node"])
        if not ok:
            reasons.append(f"killing node not killed in call: {why}")
        call = next((e for e in mutant if e["kind"] == "report" and e["nodeid"] == spec["killing_node"]
                     and e["when"] == "call"), None)
        result["killing_call"] = call
        records = load_w(wit_path) if wit_path.exists() else []
        entry_file = str((worktree / spec["entry_file"]).resolve())
        branch_file = str((worktree / spec["branch_file"]).resolve())
        cert = certify(records, node=spec["killing_node"], run=run_id + "-mutant", entry=spec["entry"],
                       entry_file=entry_file, branch_file=branch_file, kind=spec["kind"])
        result["witness"] = cert
        if not cert["ok"]:
            reasons.append(f"witness: {cert['reasons']}")
        if spec["kind"] == "absence":
            ctrl = positive_control(records, node=spec["killing_node"], run=run_id + "-witness-only",
                                    entry=spec["entry"], entry_file=entry_file)
            result["positive_control"] = ctrl
            if not ctrl["ok"]:
                reasons.append(f"positive control: {ctrl['reasons']}")
    finally:
        reset_worktree(worktree, baseline)
    tracked_after = sh(["git", "ls-files", "-s"], worktree).stdout
    if hashlib.sha256(tracked_after.encode()).hexdigest() != result.get("tracked_manifest_sha256"):
        reasons.append("tracked manifest differs after restore")
    green2 = run_structured(worktree, spec["tests"], out_dir / "green_after.jsonl", python)
    after = {n: node_state(green2, n) for n in collected_ids(green2)}
    if after != result.get("baseline_states"):
        reasons.append("restored baseline outcomes differ from the first baseline run")
    reset_worktree(worktree, baseline)          # leave the worktree pristine (no run artefacts)
    result["verdict"] = "accepted" if not reasons else "rejected"
    result["reasons"] = reasons
    (out_dir / "acceptance.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    return result


def collected_ids(events: list[dict]) -> list[str]:
    from scripts.audit.incident_register.acceptance import collected
    return collected(events)


def harness_changes_v2(repo: Path, parent: str, fix: str) -> list[dict]:
    """Every F^→F change outside src/ from ``git diff -z --name-status -M`` (renames keep both paths)."""
    raw = sh(["git", "diff", "-z", "--name-status", "-M", parent, fix, "--", ".", ":(exclude)src"], repo).stdout
    parts = [p for p in raw.split("\0")]
    changes, i = [], 0
    while i < len(parts) and parts[i]:
        status = parts[i]
        if status.startswith(("R", "C")):
            changes.append({"status": status, "path": parts[i + 2], "from": parts[i + 1]})
            i += 3
        else:
            changes.append({"status": status, "path": parts[i + 1]})
            i += 2
    return changes


AUDIT_DECISIONS = {"neutral", "irrelevant", "adapter"}


def replay_label(changes: list[dict], audit: dict[str, dict] | None) -> tuple[str, list[str]]:
    """Clean label only when EVERY outside-src change carries a recorded audit decision —
    including changes inside the selected test files (Codex phase-1 r1 #6)."""
    audit = audit or {}
    missing = [c["path"] for c in changes
               if audit.get(c["path"], {}).get("decision") not in AUDIT_DECISIONS]
    return ("semantic_regression_replay" if not missing else "semantic_regression_replay_unaudited"), missing


def classify_structured(red: list[dict], green: list[dict], nodeid: str) -> str:
    from scripts.audit.incident_register.acceptance import node_state
    g = node_state(green, nodeid)
    if g != "passed":
        return f"green_{g}"
    r = node_state(red, nodeid)
    if r == "missing":
        return "absent_at_parent"
    if r == "passed":
        return "passes_at_parent"
    if r != "failed":
        return r
    call = next(e for e in red if e["kind"] == "report" and e["nodeid"] == nodeid and e["when"] == "call")
    return "new_api" if call.get("exc_type") in NEW_API else "symptom_candidate"


def historical_replay_v2(repo: Path, worktree: Path, label: str, fix: str, tests: list[str],
                         out_dir: Path, python: list[str], audit: dict[str, dict] | None = None,
                         sync: list[str] | None = None) -> dict:
    """Fix F's tests on F^'s src (red) and F's src (green), structured and import-checked."""
    from scripts.audit.incident_register.acceptance import collected, imports_under
    out_dir.mkdir(parents=True, exist_ok=True)
    fix_full = sh(["git", "rev-parse", fix], repo).stdout.strip()
    parent = sh(["git", "rev-parse", f"{fix_full}^"], repo).stdout.strip()
    reset_worktree(worktree, fix_full)
    if sync:
        sh(sync, worktree)
    files = sorted({t.split("::")[0] for t in tests})
    changes = harness_changes_v2(repo, parent, fix_full)
    label_kind, unaudited = replay_label(changes, audit)
    res = {"label": label, "fix": fix_full, "parent": parent, "tests": tests,
           "test_sha256": {f: sha256_file(worktree / f) for f in files if (worktree / f).exists()},
           "lock_sha256": sha256_file(worktree / "uv.lock") if (worktree / "uv.lock").exists() else None,
           "harness_changes": changes, "unaudited": unaudited, "label_kind": label_kind}
    try:
        res["src_parent"] = use_src(worktree, parent)
        red = run_structured(worktree, tests, out_dir / "red.jsonl", python)
        res["src_fix"] = use_src(worktree, fix_full)
        green = run_structured(worktree, tests, out_dir / "green.jsonl", python)
    finally:
        reset_worktree(worktree, fix_full)
    src_root = (worktree / "src").resolve()
    res["imports_ok"] = {"red": imports_under(red, src_root)[0], "green": imports_under(green, src_root)[0]}
    nodes = sorted(set(collected(red)) | set(collected(green)))
    res["classes"] = {n: classify_structured(red, green, n) for n in nodes}
    (out_dir / "replay_v2.json").write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    return res
