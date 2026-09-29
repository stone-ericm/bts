"""Expected-failure pair acceptance for the registered W1.5 nodes (plan rev 3 Task 1).

    python -m scripts.audit.incident_register.run_expected_failures <owned-worktree> <ref> <out-dir>

Resets the OWNED evidence worktree to ``ref``, runs ``tests/test_incident_register_2026.py`` marked
and with ``--runxfail`` under the trusted observer, applies ``acceptance.accept`` with the registry
in the evidence directory, restores the worktree and writes ``expected_failures_acceptance.json``
(summary, per-node reasons, raw event sha256s) to ``out-dir``.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from scripts.audit.incident_register import acceptance, owned, runner

REGISTRY = "docs/audit/2026-09-29-incident-register-evidence/expected_failures.json"
FIXTURES = "tests/test_incident_register_2026.py"


def main(worktree: str, ref: str, out: str) -> dict:
    wt, out_dir = Path(worktree), Path(out)
    owned.reset(wt, ref)
    reg = json.loads((wt / REGISTRY).read_text())["entries"]
    cfg = acceptance.observe_config(wt, reg)
    env = {"TZ": "America/New_York"}
    t0 = time.time()
    marked = runner.run(wt, [FIXTURES, "-q"], out_dir, "marked", observe=cfg, env_extra=env)
    unmarked = runner.run(wt, [FIXTURES, "-q", "--runxfail"], out_dir, "unmarked", observe=cfg, env_extra=env)
    got = acceptance.accept(marked, unmarked, worktree=wt, registry=reg)
    owned.reset(wt, ref)
    summary = {"ref": ref, "seconds": round(time.time() - t0, 1), "session_reasons": got["_session"],
               "accepted": sum(1 for n, why in got.items() if n != "_session" and not why),
               "registered": len(reg),
               "rejected": {n: why for n, why in got.items() if n != "_session" and why},
               "returncodes": {"marked": marked.returncode, "unmarked": unmarked.returncode},
               "raw_events_sha256": {r.stage: hashlib.sha256(r.events_path.read_bytes()).hexdigest()
                                     for r in (marked, unmarked)}}
    (out_dir / "expected_failures_acceptance.json").write_text(
        json.dumps({"summary": summary, "per_node": got}, indent=1) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(main(*sys.argv[1:4]), indent=1))
