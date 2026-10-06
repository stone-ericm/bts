"""Drive the phase-level memory method (README § "Phase-level memory method"): per phase, 5 baseline/candidate pairs
(alternating order) and 5 baseline/baseline control pairs, each run a fresh process; then the declared rules."""
import json, os, statistics, subprocess, sys, time
from pathlib import Path

PHASES = ("load", "cache", "save", "tail_off", "tail_on", "slate")
REPEATS, LIMIT, AIM = 5, 250_000_000, 125_000_000
ENV = {**os.environ, "TZ": "America/New_York", "OMP_NUM_THREADS": "1", "UV_CACHE_DIR": "/tmp/uv-cache"}


def one(side, phase, inputs):
    r = subprocess.run(["uv", "run", "python", "-m", "tests.c2_2a.bench.phases", "measure", "--phase", phase,
                        "--inputs", str(inputs)], cwd=side, env=ENV, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{side} {phase}:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def run(base, cand, inputs, out):
    with Path(out).open("a") as fh:
        for phase in PHASES:
            for i in range(REPEATS):
                order = [("baseline", base), ("candidate", cand)] if i % 2 == 0 else [("candidate", cand), ("baseline", base)]
                for label, side in order + [("control_a", base), ("control_b", base)]:
                    m = one(side, phase, inputs)
                    m.update(side=label, repeat=i, at=time.strftime("%H:%M:%S"))
                    fh.write(json.dumps(m) + "\n"); fh.flush()
                    print(phase, i, label, m["rss_peak"] // 10**6, flush=True)


def summarise(rows):
    out = {}
    for phase in PHASES:
        def pairs(a, b):
            res = []
            for i in range(REPEATS):
                x = [r for r in rows if r["phase"] == phase and r["repeat"] == i and r["side"] == a]
                y = [r for r in rows if r["phase"] == phase and r["repeat"] == i and r["side"] == b]
                if x and y:
                    res.append(y[0]["rss_peak"] - x[0]["rss_peak"])
            return res
        d, c = pairs("baseline", "candidate"), pairs("control_a", "control_b")
        if not d:
            continue
        cmax = max(abs(v) for v in c) if c else None
        resolved = cmax is not None and cmax < LIMIT
        out[phase] = {"pairs": d, "control_pairs": c, "max": max(d), "median": statistics.median(d),
                      "control_max_abs": cmax, "control_within_aim": cmax is not None and cmax <= AIM,
                      "verdict": ("UNRESOLVED" if not resolved else ("PASS" if max(d) <= LIMIT else "EXCEEDED")),
                      "elapsed_s": {s: statistics.median([r["elapsed_s"] for r in rows if r["phase"] == phase
                                                          and r["side"] == s]) for s in ("baseline", "candidate")}}
    return out


if __name__ == "__main__":
    if sys.argv[1] == "summarise":
        print(json.dumps(summarise([json.loads(l) for l in Path(sys.argv[2]).read_text().splitlines()]), indent=1))
    else:
        run(*sys.argv[1:5])
