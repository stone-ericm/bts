"""The declared phase driver (`phases_drive.py`), imported unchanged, restricted to the load phase for its re-run
attempts (manager's row C2-2a-cost-r4-load: at most 3 attempts, spaced apart; the gate is on the control only, within
the 125 MB aim; every attempt reported, kept or discarded, with its conditions snapshot).

    python phases_r4_load.py BASE CAND INPUTS OUT        # one attempt
    python phases_r4_load.py summarise OUT               # the declared rules, on the load phase
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phases_drive as D  # noqa: E402

D.PHASES = ("load",)

if __name__ == "__main__":
    if sys.argv[1] == "summarise":
        print(json.dumps(D.summarise([json.loads(l) for l in Path(sys.argv[2]).read_text().splitlines()]), indent=1))
    else:
        D.run(*sys.argv[1:5])
