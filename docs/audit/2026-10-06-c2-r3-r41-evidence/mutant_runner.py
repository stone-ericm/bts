"""Each mutant must make its named tests fail (every named test runs; each failure is printed with its reason); the file is restored by hash after each (kickoff brief §6 runner,
pointed at this worktree). spec JSON: [{"id", "file", "old", "new", "tests": [...]}]; "old" must occur exactly once."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[3]
for m in json.loads(Path(sys.argv[1]).read_text()):
    f = REPO / m["file"]; orig = f.read_bytes(); h = hashlib.sha256(orig).hexdigest(); s = orig.decode()
    if s.count(m["old"]) != 1:
        print(m["id"], "ANCHOR x", s.count(m["old"])); continue
    f.write_text(s.replace(m["old"], m["new"]))
    try:
        env = {**os.environ, "UV_CACHE_DIR": "/tmp/uv-cache", "TZ": "America/New_York",
               "PYTHONPYCACHEPREFIX": tempfile.mkdtemp(), "PYTHONDONTWRITEBYTECODE": "1"}
        r = subprocess.run(["uv", "run", "python", "-B", "-m", "pytest", "-q", "-rf", "-p", "no:cacheprovider", *m["tests"]],
                           cwd=REPO, env=env, capture_output=True, text=True)
        print(m["id"], "RED" if r.returncode else "SURVIVED", (r.stdout.strip().splitlines() or ["?"])[-1], flush=True)
        for line in r.stdout.splitlines():           # attribution: each failing test and its reason (C2 r3 review r1)
            if line.startswith("FAILED "):
                print("    " + line[:300], flush=True)
    finally:
        f.write_bytes(orig); assert hashlib.sha256(f.read_bytes()).hexdigest() == h
