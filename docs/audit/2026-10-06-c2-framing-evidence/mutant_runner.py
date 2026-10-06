"""Mutant ledger runner (C2 framing screen; revised after reviews r1 and r2). Each mutant must make its named tests FAIL,
cleanly: RED only when pytest exits 1 with at least one `FAILED` line and no `ERROR` line, and its summary reports no
error, interruption or skip (r2 R2-4: a failure elsewhere must not hide a test that never ran). A pass is SURVIVED;
anything else is INCONCLUSIVE. Every named test runs; each failure and error is printed; the file is restored by hash
after each mutant; the runner exits 1 if any mutant is not RED or has an invalid anchor. Spec JSON: [{"id", "rule",
"file", "old", "new", "tests": [...]}]; "old" must occur exactly once."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
UNCLEAN = (" error", "interrupted", " skipped", " xfailed", " xpassed", "no tests ran")


def classify(returncode: int, stdout: str) -> tuple[str, list, list]:
    lines = stdout.splitlines()
    failed = [l for l in lines if l.startswith("FAILED ")]
    errored = [l for l in lines if l.startswith("ERROR ")]
    summary = (stdout.strip().splitlines() or [""])[-1]
    clean = not errored and not any(w in summary for w in UNCLEAN)
    if returncode == 1 and failed and clean:
        return "RED", failed, errored
    if returncode == 0 and clean:
        return "SURVIVED", failed, errored
    return f"INCONCLUSIVE(exit {returncode}{', errors' if errored else ''})", failed, errored


def main(spec: str, only: str | None = None) -> int:
    wanted = set(only.split(",")) if only else None
    bad = []
    for m in json.loads(Path(spec).read_text()):
        if wanted and m["id"] not in wanted:
            continue
        f = REPO / m["file"]; orig = f.read_bytes(); h = hashlib.sha256(orig).hexdigest(); s = orig.decode()
        if s.count(m["old"]) != 1:
            print(m["id"], "ANCHOR x", s.count(m["old"]), flush=True); bad.append(m["id"]); continue
        f.write_text(s.replace(m["old"], m["new"]))
        try:
            env = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "TZ": "America/New_York", "OMP_NUM_THREADS": "1",
                   "PYTHONPYCACHEPREFIX": tempfile.mkdtemp(), "PYTHONDONTWRITEBYTECODE": "1"}
            r = subprocess.run(["uv", "run", "python", "-B", "-m", "pytest", "-q", "-rfE", "-p", "no:cacheprovider",
                                *m["tests"]], cwd=REPO, env=env, capture_output=True, text=True)
            verdict, failed, errored = classify(r.returncode, r.stdout)
            print(m["id"], verdict, (r.stdout.strip().splitlines() or ["?"])[-1], flush=True)
            for line in failed + errored:
                print("    " + line[:300], flush=True)
            if verdict != "RED":
                bad.append(m["id"])
        finally:
            f.write_bytes(orig); assert hashlib.sha256(f.read_bytes()).hexdigest() == h
    print("NOT RED:", ",".join(bad) if bad else "none", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
