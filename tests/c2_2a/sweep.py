"""C2 step 2a, code review round 4 (Eric's row C2-2a-review-r4): the per-line fault sweep.

The signed design (§1, §3.0) promises that a witness failure never costs the forecast, never changes the cache, the
picks or what is delivered, and never makes the witness claim what did not happen. Rounds 1 to 3 each found one more
statement where an injected fault broke that promise. This sweep looks for every such statement mechanically.

The fault model. A fault is an exception (MemoryError) raised by a sys.settrace hook:
- at the `line` event of every executed line that the candidate adds or changes relative to the deployed baseline
  (`git diff -U0 f882411` over the pick path's five files), and at the `call` event of every function whose definition
  is new. Each point is a (line, call site) pair, so a shared helper is faulted once per place that calls it;
- one at a time, at the first time that point executes in a scenario;
- in the plain scenarios and in the gate's designed-fault scenarios. The latter already carry one fault, so the points
  only they reach (fallbacks, handlers) are faulted on top of it.
A line whose bytecode is only NOP (`try:`, `pass`) performs no operation, so nothing there can fail. Since Python 3.11,
entering a `try` costs nothing, so such lines are not fault points (`nop_lines`).

Three statements are the computation itself, changed by the design, not witness code (`COMPUTATION`): the held cache
loader's unpickle (one read: it unpickles the hashed bytes), the hashing writer's forwarded write (the cache's bytes
go through it), and R10's explicit projected flag. A fault there is a genuine failure of that operation; the gate's
genuine-failure scenarios (named beside each) compare it with the baseline's. They are listed in the results, never
silently skipped.

The verdict for each point. The scenario's whole compared surface must equal the deployed baseline's golden
(`test_golden._compare`: predictions, selection, locks, transports, files, cache and the slate's rows). The slate's
witness may only lose information: every part of the faulted run's witness (recursively, other than its error lists
and timestamps) is either null or equal to the plain candidate run's (`witness_only_loses`).

Run from the repository root:
    TZ=America/New_York OMP_NUM_THREADS=1 UV_CACHE_DIR=/tmp/uv-cache uv run python -B -m tests.c2_2a.sweep \\
        --workers 8 --out <results.jsonl>
"""
from __future__ import annotations

import argparse
import concurrent.futures
import dis
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BASELINE = "f882411"
FILES = ("src/bts/model/predict.py", "src/bts/model/calibrate.py", "src/bts/orchestrator.py", "src/bts/slate.py",
         "src/bts/serving_witness.py")
# Plain scenarios first (cheapest first), then the designed-fault scenarios. A point is faulted in the first scenario
# that reaches it. Scenarios that install their own sys.settrace hook are left out (they would replace this one).
SCENARIOS = (
    "model_cached", "model_cold", "calibration_on", "calibration_off_explicit", "calibration_no_pa_file",
    "calibration_insufficient_support", "calibration_no_sklearn", "calibration_two_thresholds", "day_dm",
    "calibration_empty_pa", "genuine_pick_unreadable",
    "fault_parquet_buffer", "fault_parquet_buffer_alloc", "fault_parquet_hash", "fault_cache_buffer",
    "fault_cache_hash", "fault_hashing_writer_construction", "fault_hashing_writer_hash",
    "fault_hashing_writer_finalisation", "fault_short_write", "fault_calibration_pa_buffer", "fault_pick_buffer",
    "fault_pick_decoder", "fault_pick_decoder_oserror", "fault_collector_appends", "fault_pa_append_landed",
    "fault_sample_canonicalisation", "fault_map_extraction", "fault_map_hash", "fault_error_recording",
    "fault_build", "fault_attrs_assignment", "fault_attrs_copy_calibration", "fault_omitted_input_lost_error",
    "fault_undescribable_parquet", "fault_undescribable_pick", "fault_undescribable_cache", "fault_package_query",
    "fault_calibration_record", "fault_provenance_take", "calibration_apply_failure",
    "calibration_error_after_assignment", "calibration_fit_failure",
)
UNSTABLE_KEYS = {"errors", "built_at"}
# (file, stripped source line, or "def <name>" for its call event) -> the genuine-failure scenario covering it
COMPUTATION = {
    ("serving_witness.py", "def load_blend"): "genuine_cache_unpickle",
    ("serving_witness.py", "return pickle.loads(raw)  # noqa: S301 — the deployed load_blend's operation, on the held "
                           "bytes"): "genuine_cache_unpickle",
    ("serving_witness.py", "def write"): "genuine_partial_write",
    ("serving_witness.py", "return self._f.write(b)"): "genuine_partial_write",
    ("predict.py", "slot[\"projected\"] = is_projected    # explicit for every slot (R10): false = posted lineup"):
        "the deployed `if is_projected:` at the same position",
}


def computation(event: str, file: str, line: int):
    """The genuine-failure scenario covering a computation statement, or None for a witness statement."""
    text = Path(file).read_text().splitlines()[line - 1].strip()
    if event == "call":
        text = "def " + text.split("def ", 1)[1].split("(")[0] if "def " in text else text
    return COMPUTATION.get((Path(file).name, text))


class Injected(MemoryError):
    """The sweep's fault."""


def new_lines(repo: Path = REPO) -> dict[str, set[int]]:
    """Every candidate line added or changed relative to the baseline, per resolved file path."""
    out = {}
    for rel in FILES:
        diff = subprocess.run(["git", "-C", str(repo), "diff", "-U0", BASELINE, "--", rel], capture_output=True,
                              text=True, check=True).stdout
        lines = set()
        for m in re.finditer(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", diff, re.M):
            start, count = int(m[1]), int(m[2]) if m[2] is not None else 1
            lines.update(range(start, start + count))
        out[str((repo / rel).resolve())] = lines
    return out


def _code_objects(code):
    yield code
    for c in code.co_consts:
        if isinstance(c, types.CodeType):
            yield from _code_objects(c)


def executable_lines(repo: Path = REPO) -> dict[str, set[int]]:
    """Lines with at least one non-NOP instruction inside a function (not module or class bodies), per file."""
    out = {}
    for rel in FILES:
        path = (repo / rel).resolve()
        lines = set()
        for code in _code_objects(compile(path.read_bytes(), str(path), "exec")):
            if not code.co_flags & inspect.CO_OPTIMIZED:
                continue
            for ins in dis.get_instructions(code):
                if ins.positions and ins.positions.lineno is not None and ins.opname not in ("NOP", "RESUME"):
                    lines.add(ins.positions.lineno)
        out[str(path)] = lines
    return out


def nop_lines(repo: Path = REPO) -> dict[str, set[int]]:
    """Lines whose every instruction is NOP (nothing executes there that could fail), per resolved file path."""
    out = {}
    for rel in FILES:
        path = (repo / rel).resolve()
        ops: dict[int, set[str]] = {}
        for code in _code_objects(compile(path.read_bytes(), str(path), "exec")):
            for ins in dis.get_instructions(code):
                if ins.positions and ins.positions.lineno is not None:
                    ops.setdefault(ins.positions.lineno, set()).add(ins.opname)
        out[str(path)] = {line for line, names in ops.items() if names <= {"NOP"}}
    return out


def _point(event, frame) -> tuple:
    back = frame.f_back
    return (event, frame.f_code.co_filename, frame.f_lineno,
            back.f_code.co_filename if back else None, back.f_lineno if back else None)


def record(name: str, new: dict, nops: dict) -> tuple[list[tuple], dict]:
    """The fault points one plain run of `name` executes, in first-execution order, and that run's observation."""
    seen, order = set(), []

    def keep(event, frame):
        f = frame.f_code.co_filename
        if not frame.f_code.co_flags & inspect.CO_OPTIMIZED:    # module and class bodies run at import, not on
            return                                             # the pick path
        if frame.f_lineno in new.get(f, ()) and frame.f_lineno not in nops.get(f, ()):
            p = _point(event, frame)
            if p not in seen:
                seen.add(p)
                order.append(p)

    def local(frame, event, arg):
        if event == "line":
            keep(event, frame)
        return local

    def glob(frame, event, arg):
        if frame.f_code.co_filename not in new:
            return None
        if event == "call" and frame.f_code.co_firstlineno in new[frame.f_code.co_filename]:
            keep("call", frame)
        return local
    obs = _run(name, glob)
    return order, obs


def _run(name: str, tracer) -> dict:
    from tests.c2_2a.golden import scenarios as S
    with tempfile.TemporaryDirectory(prefix="c2-2a-sweep-") as tmp:
        sys.settrace(tracer)
        try:
            return S.run(name, REPO, Path(tmp) / name)
        finally:
            sys.settrace(None)


def inject(name: str, point: tuple) -> dict:
    """Run `name` once with the fault at `point`'s first execution."""
    event, file, line, cfile, cline = point
    fired = []

    def match(frame):
        back = frame.f_back
        return (not fired and frame.f_lineno == line and frame.f_code.co_filename == file
                and (back.f_code.co_filename if back else None) == cfile and (back.f_lineno if back else None) == cline)

    def local(frame, ev, arg):
        if ev == "line" and event == "line" and match(frame):
            fired.append(1)
            raise Injected(f"sweep: {Path(file).name}:{line}")      # CPython then unsets the hook: one fault
        return local

    def glob(frame, ev, arg):
        if ev == "call" and event == "call" and match(frame):
            fired.append(1)
            raise Injected(f"sweep: call {Path(file).name}:{line}")
        return local if frame.f_code.co_filename == file else None
    obs = _run(name, glob)
    return {"obs": obs, "fired": bool(fired)}


def witness_only_loses(plain, faulted, path="serving") -> list[str]:
    """Where the faulted witness says something other than the plain run's (null, or a missing envelope key, is
    always allowed: it withholds)."""
    if faulted is None or faulted == "<absent>":
        return []
    if isinstance(faulted, dict) and isinstance(plain, dict):
        bad = []
        for k, v in faulted.items():
            if k in UNSTABLE_KEYS:
                continue
            if k not in plain:
                bad.append(f"{path}.{k}: not in the plain witness")
            else:
                bad += witness_only_loses(plain[k], v, f"{path}.{k}")
        return bad
    if isinstance(faulted, list) and isinstance(plain, list):
        if len(faulted) != len(plain):
            return [f"{path}: {len(faulted)} entries != plain {len(plain)}"]
        return [p for i, (a, b) in enumerate(zip(plain, faulted)) for p in witness_only_loses(a, b, f"{path}[{i}]")]
    return [] if faulted == plain else [f"{path}: {faulted!r} != plain {plain!r}"[:300]]


def _reread_once(golden: dict, obs: dict) -> dict:
    """The observation with the reviewed fallback normalised: a pick file whose held read was faulted is read once
    more from its path (design r1 F2; the gate's FALLBACK_REREAD). Any other read count stays a difference."""
    g, c = golden.get("pick_reads"), obs.get("pick_reads")
    if not isinstance(g, dict) or not isinstance(c, dict) or set(g) != set(c):
        return obs
    if all(c[f] in (g[f], g[f] + 1) for f in g) and sum(c[f] - g[f] for f in g) <= 1:
        return {**obs, "pick_reads": dict(g)}
    return obs


def verdict(name: str, point: tuple, result: dict, golden: dict, plain: dict) -> dict:
    from tests.c2_2a.test_golden import _compare
    problems = []
    if not result["fired"]:
        problems.append("the fault did not fire")
    try:
        _compare(golden, _reread_once(golden, result["obs"]), name)
    except AssertionError as e:
        problems.append("differs from the baseline: " + (str(e) or "surface")[:300])
    fs, ps = (result["obs"].get("slate") or {}), (plain.get("slate") or {})
    problems += witness_only_loses(ps.get("serving"), fs.get("serving"))
    return {"scenario": name, "point": [point[0], Path(point[1]).name, point[2],
                                        Path(point[3]).name if point[3] else None, point[4]],
            "ok": not problems, "problems": problems}


def _task(args):
    name, point = args
    data = REPO / "tests/c2_2a/golden/data"
    golden = json.loads((data / f"{name}.json").read_text())
    plain = json.loads(Path(os.environ["C2_2A_SWEEP_PLAIN"]).read_text())[name]
    try:
        return verdict(name, point, inject(name, point), golden, plain)
    except BaseException as e:                      # the scenario itself crashed: a failure of the point
        return {"scenario": name, "point": [point[0], Path(point[1]).name, point[2],
                                            Path(point[3]).name if point[3] else None, point[4]],
                "ok": False, "problems": [f"harness crash: {type(e).__name__}: {e}"[:300],
                                          traceback.format_exc()[-1200:]]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenarios", default=None, help="comma-separated subset (diagnostic only)")
    args = ap.parse_args(argv)
    if os.environ.get("TZ") != "America/New_York" or os.environ.get("OMP_NUM_THREADS") != "1":
        print("refusing: set TZ=America/New_York and OMP_NUM_THREADS=1", file=sys.stderr)
        return 2
    names = args.scenarios.split(",") if args.scenarios else list(SCENARIOS)
    new, nops = new_lines(), nop_lines()
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    plain, assigned, reached, excluded = {}, {}, set(), {}
    for name in names:
        t = time.time()
        points, plain[name] = record(name, new, nops)
        fresh = [p for p in points if p not in assigned and p not in excluded]
        for p in points:
            cover = computation(p[0], p[1], p[2])
            if cover is not None and p not in excluded:
                excluded[p] = cover
        fresh = [p for p in fresh if p not in excluded]
        for p in fresh:
            assigned[p] = name
        reached |= {(p[1], p[2]) for p in points}
        print(f"[sweep] {name}: {len(points)} points, {len(fresh)} new, {time.time() - t:.1f}s", file=sys.stderr,
              flush=True)
    plain_path = Path(tempfile.mkstemp(prefix="c2-2a-sweep-plain-", suffix=".json")[1])
    plain_path.write_text(json.dumps(plain))
    os.environ["C2_2A_SWEEP_PLAIN"] = str(plain_path)
    code_lines = executable_lines()
    executable = {(f, l) for f, ls in new.items() for l in ls if l in code_lines.get(f, set())}
    out = Path(args.out)
    results = []
    with out.open("w") as fh, concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        fh.write(json.dumps({"head": head, "baseline": BASELINE, "scenarios": names, "points": len(assigned),
                             "computation": [[p[0], Path(p[1]).name, p[2], Path(p[3]).name if p[3] else None, p[4],
                                              cover] for p, cover in excluded.items()]}) + "\n")
        for r in pool.map(_task, [(n, p) for p, n in assigned.items()]):
            results.append(r)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            if not r["ok"]:
                print("[sweep] FAIL", r["scenario"], r["point"], r["problems"][:2], file=sys.stderr, flush=True)
        never = sorted(f"{Path(f).name}:{l}" for f, l in executable - reached)
        failed = [r for r in results if not r["ok"]]
        summary = {"summary": True, "points": len(results), "failed": len(failed),
                   "lines_never_reached": len(never), "never_reached": never}
        fh.write(json.dumps(summary) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "never_reached"}), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
