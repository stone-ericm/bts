"""Mutant ledger runner (C2 framing screen; revised after reviews r1, r2 and r3).

Each mutant's named tests are its intended set: before mutating, `pytest --collect-only` on the unmutated source lists
exactly the nodes they select. The mutant is RED only when pytest exits 1, every intended node executed and reported
PASSED or FAILED (`-rA`), at least one FAILED, there is no `ERROR` line, and the summary reports no error,
interruption, skip, xfail or xpass (r2 R2-4). A run where an intended node never executed (fail-fast, a crash, a
changed selection) is INCONCLUSIVE, whatever its summary says (r3 R3-4). A complete clean pass is SURVIVED; anything
else is INCONCLUSIVE.

Inherited selection and early-stop options cannot apply: `PYTEST_ADDOPTS` is removed from the subprocess environment,
the ini `addopts` are cleared (`-o addopts=`) for both the collection and the run, and a named test list that contains
any option is INVALID before anything is mutated. Every failure and error is printed; the file is restored by hash after
each mutant; the runner exits 1 if any mutant is not RED. Spec JSON: [{"id", "rule", "file", "old", "new",
"tests": [...]}]; "old" must occur exactly once; "tests" are paths or node ids only."""
import hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
UNCLEAN = (" error", "interrupted", " skipped", " xfailed", " xpassed", "no tests ran")
# One rootdir for both passes, so the collected node ids and the -rA summary's node ids are the same strings.
PYTEST = ["uv", "run", "python", "-B", "-m", "pytest", f"--rootdir={REPO}", "-o", "addopts=", "-p", "no:cacheprovider"]


def _env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_ADDOPTS"}
    env.update(UV_CACHE_DIR="/tmp/uv-cache", TZ="America/New_York", OMP_NUM_THREADS="1",
               PYTHONPYCACHEPREFIX=tempfile.mkdtemp(), PYTHONDONTWRITEBYTECODE="1")
    return env


def _node(rest: str, intended) -> str | None:
    """The intended node a summary line names (node ids may themselves contain ' - ')."""
    for n in sorted(intended, key=len, reverse=True):
        if rest == n or rest.startswith(n + " - "):
            return n
    return None


def classify(returncode: int, stdout: str, intended) -> tuple[str, list, list]:
    lines = stdout.splitlines()
    failed = [l for l in lines if l.startswith("FAILED ")]
    errored = [l for l in lines if l.startswith("ERROR ")]
    ran = {_node(l[len("PASSED "):], intended) for l in lines if l.startswith("PASSED ")}
    ran |= {_node(l[len("FAILED "):], intended) for l in failed}
    missing = set(intended) - ran
    summary = (stdout.strip().splitlines() or [""])[-1]
    if errored:
        return f"INCONCLUSIVE(exit {returncode}, errors)", failed, errored
    if any(w in summary for w in UNCLEAN):
        return f"INCONCLUSIVE(exit {returncode})", failed, errored
    if missing:
        return f"INCONCLUSIVE(exit {returncode}, {len(missing)} not run)", failed, errored
    if returncode == 1 and failed:
        return "RED", failed, errored
    if returncode == 0 and not failed:
        return "SURVIVED", failed, errored
    return f"INCONCLUSIVE(exit {returncode})", failed, errored


def collect(tests: list[str]) -> list[str] | None:
    """The nodes the named tests select, on the unmutated source; None when collection fails or selects nothing."""
    r = subprocess.run([*PYTEST, "--collect-only", "-q", *tests], cwd=REPO, env=_env(), capture_output=True, text=True)
    nodes = [l.strip() for l in r.stdout.splitlines() if "::" in l and not l.startswith(" ")]
    return nodes if r.returncode == 0 and nodes else None


def main(spec: str, only: str | None = None) -> int:
    wanted = set(only.split(",")) if only else None
    bad = []
    for m in json.loads(Path(spec).read_text()):
        if wanted and m["id"] not in wanted:
            continue
        options = [t for t in m["tests"] if t.startswith("-")]
        if options:
            print(m["id"], "INVALID: options in the named test list", options, flush=True); bad.append(m["id"]); continue
        f = REPO / m["file"]; orig = f.read_bytes(); h = hashlib.sha256(orig).hexdigest(); s = orig.decode()
        if s.count(m["old"]) != 1:
            print(m["id"], "ANCHOR x", s.count(m["old"]), flush=True); bad.append(m["id"]); continue
        intended = collect(m["tests"])
        if intended is None:
            print(m["id"], "INVALID: the named tests do not collect", flush=True); bad.append(m["id"]); continue
        f.write_text(s.replace(m["old"], m["new"]))
        try:
            r = subprocess.run([*PYTEST, "-q", "-rA", *m["tests"]], cwd=REPO, env=_env(), capture_output=True, text=True)
            verdict, failed, errored = classify(r.returncode, r.stdout, intended)
            print(m["id"], verdict, (r.stdout.strip().splitlines() or ["?"])[-1], f"[{len(intended)} intended]",
                  flush=True)
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
