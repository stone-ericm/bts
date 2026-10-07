"""The declared phase driver (`phases_drive.py`), imported unchanged, restricted to the load phase for its re-run
attempts (manager's row C2-2a-cost-r4-load: at most 3 attempts, spaced apart; the gate is on the control only, within
the 125 MB aim; every attempt reported, kept or discarded, with its conditions snapshot).

Eric's row C2-2a-cost-r4-warmup (after attempt 1, run with the plain mode, counted EXCEEDED on one low reading of the
first process of its series): attempts 2 and 3 use `warmup-run`, which first runs ONE discarded process of the
f882411 code (the load phase, baseline side), records its peak in `<OUT stem>.warmup.json`, and then runs the declared
driver unchanged. The rule is unchanged (max pair <= 250 MB; control <= 125 MB). The cause is a HYPOTHESIS, not
established: that the first large process of a series under-reads under memory pressure. The 16:41 series had low
readings at many positions, not only the first.

    python phases_r4_load.py BASE CAND INPUTS OUT              # one attempt, plain (attempt 1)
    python phases_r4_load.py warmup-run BASE CAND INPUTS OUT   # one attempt with the discarded warm-up (attempts 2, 3)
    python phases_r4_load.py summarise OUT                     # the declared rules, on the load phase
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phases_drive as D  # noqa: E402

D.PHASES = ("load",)


def warmup(base, inputs, out) -> dict:
    """One discarded f882411 process; its peak is recorded, never paired."""
    m = D.one(base, "load", inputs)
    m.update(side="warmup_discarded", at=time.strftime("%H:%M:%S"))
    Path(out).with_suffix(".warmup.json").write_text(json.dumps(m) + "\n")
    print("load warmup (discarded)", m["rss_peak"] // 10**6, flush=True)
    return m


if __name__ == "__main__":
    if sys.argv[1] == "summarise":
        print(json.dumps(D.summarise([json.loads(l) for l in Path(sys.argv[2]).read_text().splitlines()]), indent=1))
    elif sys.argv[1] == "warmup-run":
        base, cand, inputs, out = sys.argv[2:6]
        warmup(base, inputs, out)
        D.run(base, cand, inputs, out)
    else:
        D.run(*sys.argv[1:5])
