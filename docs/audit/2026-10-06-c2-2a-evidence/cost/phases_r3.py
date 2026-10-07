"""The declared phase driver (`phases_drive.py`), imported unchanged, restricted to the phases the code review r2 fixes
reach (row C2-2a-review-r3: "a cost re-run of the touched phases"):
- load: run_pipeline's PA reads, now carrying the collection flag (R2-2);
- tail_off / tail_on: predict_local after run_pipeline, now with the contained provenance take (R2-1), and on tail_on
  calibration's PA read and its record's completeness check (R2-2).
cache stops at run_pipeline's entry, save is save_blend, and slate is save_slate: none of them reaches the changed code.

    python phases_r3.py BASE CAND INPUTS OUT        # run
    python phases_r3.py summarise OUT               # the declared rules, on these three phases
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phases_drive as D  # noqa: E402

D.PHASES = ("load", "tail_off", "tail_on")

if __name__ == "__main__":
    if sys.argv[1] == "summarise":
        print(json.dumps(D.summarise([json.loads(l) for l in Path(sys.argv[2]).read_text().splitlines()]), indent=1))
    else:
        D.run(*sys.argv[1:5])
