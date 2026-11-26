"""Re-run certified current-defence specs at a deploy candidate with the FROZEN W1.5 tooling (D7 re-certification).

    python run_at_candidate.py <tool-checkout> <candidate> <specs-dir> <out-dir> LABEL[,LABEL...]

The copy of `task56/run_specs.py`'s defence branch (2026-10-03), differing only in the commit the specs run at. The tooling is
imported from <tool-checkout>, a clean checkout of the frozen pin f453283; the owned worktree is created at <candidate>
(synced once with the candidate's own lock), and every spec runs with `baseline` set to <candidate>. The spec files are
used unchanged (`docs/ops/reconcile-receipt-v1.md` § Re-certification: each function body is unchanged and each mutation
still applies at an offset). summary.json binds the tool pin, the candidate, the observer source sha256 and each spec's
sha256 and verdict. A certificate still needs the runner's `accepted` AND an `accept` decision in REVIEW-RUNS.md.
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

tool, candidate, specs_dir, out = Path(sys.argv[1]).resolve(), sys.argv[2], *map(lambda p: Path(p).resolve(), sys.argv[3:5])
labels = sys.argv[5].split(",")
pin = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tool, check=True, capture_output=True, text=True).stdout.strip()
assert pin == "f453283d0a0ec39a939b22c6500b5fb36f5452c0", f"the tooling checkout is not at the frozen pin: {pin}"
if subprocess.run(["git", "status", "--porcelain"], cwd=tool, check=True, capture_output=True, text=True).stdout:
    raise SystemExit(f"{tool}: not a clean checkout")
candidate = subprocess.run(["git", "rev-parse", candidate], cwd=tool, check=True, capture_output=True,
                           text=True).stdout.strip()
sys.path.insert(0, str(tool))
from scripts.audit.incident_register import defence, owned, runner  # noqa: E402
assert Path(runner.__file__).resolve().parents[3] == tool, runner.__file__
ENV = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


out.mkdir(parents=True, exist_ok=True)
summary_path = out / "summary.json"
summary = {"kind": "defence", "tool_pin": pin, "candidate": candidate,
           "observer_sha256": sha(Path(runner.OBSERVER_SRC).read_bytes()), "results": []}
shared = out / "wt-defence"
try:
    owned.create(tool, candidate, shared)
    p = subprocess.run(["uv", "sync", "--extra", "model"], cwd=shared, env=ENV, capture_output=True, text=True)
    (out / "sync-defence.log").write_text(p.stdout + p.stderr)
    if p.returncode:
        raise SystemExit("uv sync failed for the defence worktree")
    for label in labels:
        raw = (specs_dir / f"{label}.json").read_bytes()
        spec = json.loads(raw)
        folder = out / label
        t0 = time.time()
        spec["baseline"] = candidate
        res = defence.current_defence(shared, spec, folder)
        acc = folder / "acceptance.json"
        item = {"label": label, "spec_sha256": sha(raw), "verdict": res.get("verdict"), "reasons": res.get("reasons", []),
                "acceptance_sha256": sha(acc.read_bytes()) if acc.exists() else None, "seconds": round(time.time() - t0)}
        summary["results"].append(item)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: item[k] for k in ("label", "verdict", "seconds")}), (item["reasons"] or [""])[0][:160], flush=True)
finally:
    if shared.exists():
        owned.destroy(shared)
