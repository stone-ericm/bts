"""Expected-failure pair acceptance for the registered W1.5 nodes (plan Task 1; Codex phase-1 r3 #8).

    python -m scripts.audit.incident_register.run_expected_failures <owned-worktree> <ref> <out-dir>

Builds a REJECTED result first; after proving ownership it resets the evidence worktree to ``ref``,
runs ``acceptance.run_pair`` (marked + ``--runxfail`` over one frozen closure) with the registry in the
evidence directory, restores the worktree in ``finally``, and ALWAYS writes
``expected_failures_acceptance.json``: resolved ref, registry sha256, overall verdict, per-node reasons
and connection status. A session-level reason rejects the whole pair; no node is then reported as an
accepted reproduction.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.audit.incident_register import acceptance, owned

REGISTRY = "docs/audit/2026-09-29-incident-register-evidence/expected_failures.json"
FIXTURES = "tests/test_incident_register_2026.py"


def main(worktree: str, ref: str, out: str) -> dict:
    wt, out_dir = Path(worktree), Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {"ref": ref, "verdict": "rejected", "reasons": [], "accepted_nodes": []}
    owned_ok = False
    t0 = time.time()
    try:
        owned.assert_owned(wt)
        owned_ok = True
        owned.reset(wt, ref)
        result["resolved_ref"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, check=True,
                                                capture_output=True, text=True).stdout.strip()
        registry = json.loads((wt / REGISTRY).read_text())["entries"]
        pair = acceptance.run_pair(wt, registry, [FIXTURES, "-q"], out_dir, env={"TZ": "America/New_York"})
        result.update(pair)
        result["registered"] = len(registry)
        result["accepted_nodes"] = (sorted(n for n, w in pair["per_node"].items() if not w)
                                    if pair["verdict"] == "accepted" else [])
    except Exception as e:  # noqa: BLE001 - every handled failure still produces a rejected artifact
        result["reasons"] = result.get("reasons", []) + [f"refused: {type(e).__name__}: {e}"]
        result["verdict"] = "rejected"
    finally:
        if owned_ok:
            try:
                owned.reset(wt, ref)
            except Exception as e:  # noqa: BLE001
                result["reasons"].append(f"final reset failed: {type(e).__name__}: {e}")
                result["verdict"] = "rejected"
        result["seconds"] = round(time.time() - t0, 1)
        (out_dir / "expected_failures_acceptance.json").write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    summary = main(*sys.argv[1:4])
    print(json.dumps({k: summary.get(k) for k in ("ref", "resolved_ref", "verdict", "registered", "connections")},
                     indent=1))
    print("accepted:", len(summary.get("accepted_nodes", [])), "| reasons:", summary.get("reasons", [])[:5])
