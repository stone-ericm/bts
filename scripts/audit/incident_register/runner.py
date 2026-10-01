"""Evidence runs: pytest under the trusted observer, in an owned worktree's own venv (Codex phase-1 r2 #3, #6).

``run`` copies ``observer.py`` to a fresh directory OUTSIDE the worktree under a random module name,
then starts ``<worktree>/.venv/bin/python bootstrap.py`` which loads that exact file with
``spec_from_file_location`` and hands the module object to ``pytest.main(plugins=[...])``. Nothing
is imported by name, so no module in the worktree can stand in for the observer; the observer
reports its own loaded file and sha256, and ``gate`` compares them with the trusted copy.

The bootstrap's FIRST statement installs an audit-hook census (Codex phase-1 r7 #4): the observer's own
primitives (``id``, ``sys._getframe``, ``gc.get_referrers`` …) raise audit events, so any other audit
hook would run inside observation. Only interpreter start-up — site initialisation and the reviewed
``.pth`` hooks (``owned.REVIEWED_PTH_IMPORTS``) — runs before the census; every later
``sys.addaudithook`` raises the ``sys.addaudithook`` event through it and is counted. The observer
records the count, and a certificate needs it to be zero (``certify``).

``gate`` is the one session-level acceptance check every stage shares: the observer identity, the
worktree's own venv (prefix and executable), rootdir, pytest not shadowed from the worktree, a
finished session whose exit status equals the process return code, no collection or observer
errors, the exact node inventory, one setup/call/teardown report per node with the per-mode rules,
and every imported ``bts`` module under the worktree's ``src`` with unchanged bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from scripts.audit.incident_register import owned
from scripts.audit.incident_register.observer import src_digest

OBSERVER_SRC = Path(__file__).with_name("observer.py")
BOOTSTRAP = """import sys
_census = {"hooks_added": 0}
def _w15_audit_census(event, args, _c=_census):
    if event == "sys.addaudithook":
        _c["hooks_added"] += 1
sys.addaudithook(_w15_audit_census)
import importlib.util
path, name = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location(name, path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
module._AUDIT_CENSUS = _census
import pytest
sys.exit(pytest.main(sys.argv[3:], plugins=[module]))
"""
_SCRUB = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
          "W15_OBS_CONFIG")
MODES = ("green", "mutant", "expected_failure")


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class Run:
    stage: str
    returncode: int
    events: list
    trusted: dict
    events_path: Path
    src_digest: str | None = None      # production src tree digest taken just before the run


def load(path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _inifile(worktree: Path) -> Path:
    for name in ("pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg"):
        p = worktree / name
        if p.is_file() and (name == "pytest.ini" or "pytest" in p.read_text(errors="replace")):
            return p
    raise FileNotFoundError(f"{worktree}: no pytest configuration file")


def run(worktree, args: list[str], out_dir, stage: str, *, observe: dict | None = None,
        quiesce: list[str] | None = None, env_extra: dict | None = None, timeout: int = 3600) -> Run:
    """``quiesce``: nodes whose call phase runs with automatic garbage collection off although nothing
    observes them (an observed run's twin; observed nodes are always quiesced)."""
    worktree = Path(worktree)
    owned.assert_owned(worktree)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wt = os.path.realpath(worktree)
    if os.path.commonpath([os.path.realpath(out_dir), wt]) == wt:
        raise ValueError("evidence output must live outside the worktree")
    python = worktree / ".venv" / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(f"{python}: the worktree has no own venv")
    tmp = Path(tempfile.mkdtemp(prefix="w15obs-"))
    try:
        name = f"w15obs_{secrets.token_hex(6)}"
        obs = tmp / f"{name}.py"
        shutil.copyfile(OBSERVER_SRC, obs)
        trusted = {"file": os.path.realpath(obs), "sha256": _sha(obs)}
        boot = tmp / "bootstrap.py"
        boot.write_text(BOOTSTRAP)
        events_path = out_dir / f"{stage}.events.jsonl"
        if events_path.exists():
            events_path.unlink()
        cfg = tmp / "config.json"
        cfg.write_text(json.dumps({"out": str(events_path), "run_id": f"{stage}-{name}",
                                   "prod_root": os.path.join(wt, "src"), "observe": observe or {},
                                   "quiesce": list(quiesce or [])}))
        env = {k: v for k, v in os.environ.items() if k not in _SCRUB}
        env.update({"W15_OBS_CONFIG": str(cfg), "PYTHONDONTWRITEBYTECODE": "1"})
        env.update(env_extra or {})
        cmd = [str(python), str(boot), str(obs), name, "-p", "no:cacheprovider",
               "--rootdir", wt, "-c", str(_inifile(worktree)), *args]
        before = src_digest(os.path.join(wt, "src"))
        proc = subprocess.run(cmd, cwd=worktree, env=env, capture_output=True, text=True, timeout=timeout)
        (out_dir / f"{stage}.stdout.txt").write_text(proc.stdout)
        (out_dir / f"{stage}.stderr.txt").write_text(proc.stderr)
        events = load(events_path) if events_path.exists() else []
        return Run(stage, proc.returncode, events, trusted, events_path, before)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def collected(events: list[dict]) -> list[str]:
    recs = [e for e in events if e["kind"] == "collected"]
    return recs[0]["nodeids"] if len(recs) == 1 else []


def phases(events: list[dict], nodeid: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        if e["kind"] == "report" and e["nodeid"] == nodeid:
            out[e["when"]].append(e)
    return out


def node_state(events: list[dict], nodeid: str) -> str:
    """passed | failed | xfail | skipped | setup_error | teardown_error | missing | malformed."""
    ph = phases(events, nodeid)
    if not ph:
        return "missing"
    if any(len(ph.get(w, [])) > 1 for w in ("setup", "call", "teardown")):
        return "malformed"
    setup, call, teardown = (ph.get(w, [None])[0] for w in ("setup", "call", "teardown"))
    if setup is None:
        return "malformed"
    if setup["outcome"] != "passed":
        return "xfail" if setup.get("wasxfail") else ("skipped" if setup["outcome"] == "skipped" else "setup_error")
    if call is None:
        return "malformed"
    if teardown is None or teardown["outcome"] != "passed":
        return "teardown_error"
    if call.get("wasxfail"):
        return "xfail" if call["outcome"] == "skipped" else "xpass"
    return call["outcome"]


def gate(run: Run, *, worktree, mode: str, expected: list[str] | None = None,
         expected_failures: frozenset = frozenset()) -> list[str]:
    """Reasons to reject the run (empty = acceptable). ``mode``: ``green`` (every node passes),
    ``mutant`` (clean setup/teardown, calls pass or fail — also the historical red stage),
    ``expected_failure`` (``green`` except the registered nodes, which must be XFAIL)."""
    if mode not in MODES:
        raise ValueError(mode)
    ev = run.events
    why: list[str] = []
    starts = [e for e in ev if e["kind"] == "session_start"]
    if len(starts) != 1:
        return [f"{len(starts)} session_start records (process did not start pytest under the observer)"]
    s = starts[0]
    wt = os.path.realpath(worktree)
    if s["observer_file"] != run.trusted["file"] or s["observer_sha256"] != run.trusted["sha256"]:
        why.append("the loaded observer is not the trusted copy")
    if s["prefix"] != os.path.join(wt, ".venv"):
        why.append(f"interpreter prefix {s['prefix']} is not the worktree's own venv")
    if os.path.abspath(s["executable"]) != os.path.join(wt, ".venv", "bin", "python"):
        why.append("interpreter executable is not the worktree's own venv python")
    if s["rootdir"] != wt:
        why.append(f"rootdir {s['rootdir']} is not the worktree")
    pf = s["pytest_file"]
    if pf.startswith(wt + os.sep) and not pf.startswith(os.path.join(wt, ".venv") + os.sep):
        why.append("pytest was imported from the worktree (shadowed)")
    finished = [e for e in ev if e["kind"] == "session_finish"]
    if len(finished) != 1:
        why.append("the session did not finish")
    elif finished[0]["exitstatus"] != run.returncode:
        why.append(f"exit status {finished[0]['exitstatus']} != process return code {run.returncode}")
    if run.src_digest is None or s.get("src_digest") != run.src_digest:
        why.append("the production src tree at session start differs from the tree the runner prepared")
    if finished and finished[0].get("src_digest") != run.src_digest:
        why.append("the production src tree changed during the session")
    if any(e["kind"] == "collect_error" for e in ev):
        why.append("collection errors")
    if any(e["kind"] == "observer_error" for e in ev):
        why.append("observer errors")
    coll = [e for e in ev if e["kind"] == "collected"]
    nodes = coll[0]["nodeids"] if len(coll) == 1 else []
    if len(coll) != 1 or not nodes:
        why.append("no collection record / no nodes")
    if len(nodes) != len(set(nodes)):
        why.append("duplicate node ids")
    if expected is not None and sorted(nodes) != sorted(expected):
        missing = sorted(set(expected) - set(nodes))[:3]
        extra = sorted(set(nodes) - set(expected))[:3]
        why.append(f"node inventory differs (missing {missing}, extra {extra})")
    imports = [e for e in ev if e["kind"] == "imports"]
    mods = imports[0]["modules"] if len(imports) == 1 else {}
    if len(imports) == 1 and "unavailable" in imports[0]:
        why.append(f"import provenance unavailable: {imports[0]['unavailable']}")
    if not mods:
        why.append("no bts modules imported")
    src = os.path.join(wt, "src") + os.sep
    for name, m in sorted(mods.items()):
        if "file" not in m:
            why.append(f"module {name}: import provenance unavailable ({m.get('unavailable')})")
        elif not m["file"].startswith(src):
            why.append(f"module {name} imported from outside the worktree src: {m['file']}")
        elif not os.path.exists(m["file"]) or m["sha256"] != _sha(m["file"]):
            why.append(f"module {name} changed after import")
    any_failed = False
    for n in nodes:
        state = node_state(ev, n)
        if mode == "expected_failure" and n in expected_failures:
            if state != "xfail":
                why.append(f"{n}: registered expected failure is {state}, not xfail")
            continue
        if state == "failed" and mode == "mutant":
            any_failed = True
            continue
        if state != "passed":
            why.append(f"{n}: {state}")
    want = 1 if any_failed else 0
    if run.returncode != want:
        why.append(f"return code {run.returncode}, expected {want}")
    return why
