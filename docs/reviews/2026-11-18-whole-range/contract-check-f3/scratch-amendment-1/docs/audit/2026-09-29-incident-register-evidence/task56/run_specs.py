"""Run the Task 5 replay specs or the Task 6 defence specs sequentially with the FROZEN tooling (W1.5 Phase 1).

    python run_specs.py replay  <tool-checkout> <specs-dir> <manifest> <out-dir> [LABEL,...]
    python run_specs.py defence <tool-checkout> <specs-dir> <manifest> <out-dir> [LABEL,...]

The tooling is imported from <tool-checkout>, a clean checkout of the frozen pin. Replay: one owned worktree per spec,
created at the spec's last fix commit and synced with that commit's own lock (`uv sync --extra model`), then
`replay.historical_replay`. Defence: one owned worktree at the pin, synced once; every spec runs there with
`baseline` set to the pin (`defence.current_defence`). Each spec's output directory keeps the runner's acceptance.json and
raw records; summary.json binds the pin, the observer source hash, each spec's hash and its verdict. Specs run one at a
time (no CPU contention).
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

kind, tool, specs_dir, manifest, out = sys.argv[1], *map(lambda p: Path(p).resolve(), sys.argv[2:6])
only = set(sys.argv[6].split(",")) if len(sys.argv) > 6 else None
assert kind in ("replay", "defence"), kind
pin = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tool, check=True, capture_output=True, text=True).stdout.strip()
if subprocess.run(["git", "status", "--porcelain"], cwd=tool, check=True, capture_output=True, text=True).stdout:
    raise SystemExit(f"{tool}: not a clean checkout")
sys.path.insert(0, str(tool))
from scripts.audit.incident_register import defence, owned, replay, runner  # noqa: E402
assert Path(runner.__file__).resolve().parents[3] == tool, runner.__file__
ENV = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "TZ": "America/New_York", "PYTHONDONTWRITEBYTECODE": "1"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sync(wt: Path, log: Path) -> int:
    p = subprocess.run(["uv", "sync", "--extra", "model"], cwd=wt, env=ENV, capture_output=True, text=True)
    log.write_text(p.stdout + p.stderr)
    return p.returncode


labels = [e["spec"] for e in json.loads(manifest.read_text()) if e.get("spec")]
labels = [l for l in dict.fromkeys(labels) if not only or l in only]
out.mkdir(parents=True, exist_ok=True)
summary_path = out / "summary.json"
summary = json.loads(summary_path.read_text()) if summary_path.exists() else {
    "kind": kind, "pin": pin, "observer_sha256": sha(Path(runner.OBSERVER_SRC).read_bytes()), "results": []}
assert summary["pin"] == pin, (summary["pin"], pin)
done = {r["label"] for r in summary["results"]}
repo = tool
shared = None
try:
    if kind == "defence":
        shared = out / "wt-defence"
        owned.create(repo, pin, shared)
        if sync(shared, out / "sync-defence.log"):
            raise SystemExit("uv sync failed for the defence worktree")
    for label in labels:
        if label in done:
            continue
        raw = (specs_dir / f"{label}.json").read_bytes()
        spec = json.loads(raw)
        folder = out / label
        t0 = time.time()
        if kind == "replay":
            wt = out / f"wt-{label}"
            fix = subprocess.run(["git", "rev-parse", spec["fix_set"][-1]], cwd=repo, check=True, capture_output=True,
                                 text=True).stdout.strip()
            owned.create(repo, fix, wt)
            try:
                rc = sync(wt, out / f"sync-{label}.log")
                res = replay.historical_replay(repo, wt, spec, folder) if rc == 0 else {
                    "verdict": "rejected", "reasons": [f"uv sync at {fix[:7]} failed ({rc})"]}
            finally:
                owned.destroy(wt)
        else:
            spec["baseline"] = pin
            res = defence.current_defence(shared, spec, folder)
        acc = folder / "acceptance.json"
        item = {"label": label, "spec_sha256": sha(raw), "verdict": res.get("verdict"), "reasons": res.get("reasons", []),
                "acceptance_sha256": sha(acc.read_bytes()) if acc.exists() else None, "seconds": round(time.time() - t0)}
        summary["results"].append(item)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps({k: item[k] for k in ("label", "verdict", "seconds")}), (item["reasons"] or [""])[0][:160], flush=True)
finally:
    if shared is not None and shared.exists():
        owned.destroy(shared)
