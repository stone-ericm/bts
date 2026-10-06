"""C2 side item (e) framing-screen mutant ledger runner (revised after review r1). Each mutant must make its named tests
FAIL: a mutant is RED only when pytest exits 1 (tests failed) with at least one `FAILED` line; any other exit
(collection or usage error, interruption, no tests) is INCONCLUSIVE, and a pass is SURVIVED. Every named test runs and
each failure is printed with its reason; the file is restored by hash after each mutant. The runner exits 1 if any
mutant is not RED or has an invalid anchor. Spec JSON: [{"id", "rule", "file", "old", "new", "tests": [...]}]; "old"
must occur exactly once."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
bad = []
for m in json.loads(Path(sys.argv[1]).read_text()):
    if only and m["id"] not in only:
        continue
    f = REPO / m["file"]; orig = f.read_bytes(); h = hashlib.sha256(orig).hexdigest(); s = orig.decode()
    if s.count(m["old"]) != 1:
        print(m["id"], "ANCHOR x", s.count(m["old"]), flush=True); bad.append(m["id"]); continue
    f.write_text(s.replace(m["old"], m["new"]))
    try:
        env = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "TZ": "America/New_York", "OMP_NUM_THREADS": "1",
               "PYTHONPYCACHEPREFIX": tempfile.mkdtemp(), "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run(["uv", "run", "python", "-B", "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider",
                            *m["tests"]], cwd=REPO, env=env, capture_output=True, text=True)
        failed = [l for l in r.stdout.splitlines() if l.startswith("FAILED ")]
        verdict = "RED" if r.returncode == 1 and failed else ("SURVIVED" if r.returncode == 0 else
                                                            f"INCONCLUSIVE(exit {r.returncode})")
        print(m["id"], verdict, (r.stdout.strip().splitlines() or ["?"])[-1], flush=True)
        for line in failed:
            print("    " + line[:300], flush=True)
        if verdict != "RED":
            bad.append(m["id"])
    finally:
        f.write_bytes(orig); assert hashlib.sha256(f.read_bytes()).hexdigest() == h
print("NOT RED:", ",".join(bad) if bad else "none", flush=True)
sys.exit(1 if bad else 0)
