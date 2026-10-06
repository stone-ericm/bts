"""C2 step 2a §6.1: drive the paired cost benchmark (tests/c2_2a/bench/bench.py) and apply the predeclared acceptance.

For each case, five repeats; each repeat runs the baseline (f882411 worktree) and the candidate in fresh processes,
alternating which goes first. A sixth series runs baseline against baseline for warm_off (the A/A noise control).
Acceptance (fixed in the design before measurement): for every case, paired incremental wall time <= 2.0 s median and
<= 5.0 s maximum, and paired incremental peak RSS <= 250,000,000 bytes at its maximum."""
import json, os, statistics, subprocess, sys, time
from pathlib import Path

BASE, CAND = Path(sys.argv[1]), Path(sys.argv[2])
INPUTS, OUT = Path(sys.argv[3]), Path(sys.argv[4])
CASES = ("cold_small", "warm_off", "warm_on", "warm_unavailable")
REPEATS = 5
ENV = {**os.environ, "TZ": "America/New_York", "OMP_NUM_THREADS": "1", "UV_CACHE_DIR": "/tmp/uv-cache"}


def one(side: Path, case: str) -> dict:
    r = subprocess.run(["uv", "run", "python", "-m", "tests.c2_2a.bench.bench", "measure", "--case", case,
                        "--inputs", str(INPUTS)], cwd=side, env=ENV, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"{side} {case} failed:\n{r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    rows = []
    with OUT.open("a") as fh:
        for case in CASES:
            for i in range(REPEATS):
                order = [("baseline", BASE), ("candidate", CAND)] if i % 2 == 0 else [("candidate", CAND), ("baseline", BASE)]
                for label, side in order:
                    m = one(side, case)
                    m.update(side=label, repeat=i, at=time.strftime("%H:%M:%S"))
                    fh.write(json.dumps(m) + "\n"); fh.flush(); rows.append(m)
                    print(case, i, label, round(m["total_s"], 2), m["rss_peak"], flush=True)
        for i in range(REPEATS):
            for label in ("baseline", "baseline_aa"):
                m = one(BASE, "warm_off")
                m.update(side=label, repeat=i, case="warm_off_aa", at=time.strftime("%H:%M:%S"))
                fh.write(json.dumps(m) + "\n"); fh.flush(); rows.append(m)
                print("warm_off_aa", i, label, round(m["total_s"], 2), m["rss_peak"], flush=True)
    print(json.dumps(summarise(rows), indent=1))


def summarise(rows):
    out = {}
    for case in CASES + ("warm_off_aa",):
        a, b = ("baseline", "candidate") if case != "warm_off_aa" else ("baseline", "baseline_aa")
        pairs = []
        for i in range(REPEATS):
            x = [r for r in rows if r["case"] == case and r["repeat"] == i and r["side"] == a]
            y = [r for r in rows if r["case"] == case and r["repeat"] == i and r["side"] == b]
            if x and y:
                pairs.append((y[0]["total_s"] - x[0]["total_s"], y[0]["rss_peak"] - x[0]["rss_peak"]))
        if not pairs:
            continue
        dt, dr = [p[0] for p in pairs], [p[1] for p in pairs]
        res = {"n": len(pairs), "time_median_s": statistics.median(dt), "time_max_s": max(dt),
               "rss_median": statistics.median(dr), "rss_max": max(dr), "time_pairs": dt, "rss_pairs": dr}
        if case != "warm_off_aa":
            res["accept"] = res["time_median_s"] <= 2.0 and res["time_max_s"] <= 5.0 and res["rss_max"] <= 250_000_000
        out[case] = res
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "summarise":
        print(json.dumps(summarise([json.loads(l) for l in Path(sys.argv[2]).read_text().splitlines()]), indent=1))
    else:
        main()
