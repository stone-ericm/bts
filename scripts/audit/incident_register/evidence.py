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
    """The exception's class name from the traceback text's last 'E   Name:' line or the message."""
    text = (el.text or "") + "\n" + (el.get("message") or "")
    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if stripped.startswith("E ") and ":" in stripped:
            return stripped[1:].strip().split(":", 1)[0].rsplit(".", 1)[-1].strip()
    head = (el.get("message") or "").split(":", 1)[0]
    return head.rsplit(".", 1)[-1].strip()


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
