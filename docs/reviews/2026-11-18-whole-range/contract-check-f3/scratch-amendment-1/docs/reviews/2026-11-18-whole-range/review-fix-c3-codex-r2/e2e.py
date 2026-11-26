"""Run bounded CLI cutoff, recovery, concurrency, and special-file probes."""

import json
from pathlib import Path
import subprocess
import sys

from compare_probes import offline, scratch
import probe_support
import reconcile_probe as original

original.REPO = probe_support.REPO


def special(root):
    for part in ("old_pick", "recent_pick", "old_decision", "recent_decision", "decision_parent", "streak",
                 "picks_root", "scoring_lock", "lineup_evolution", "receipt_directory"):
        for kind in ("fifo", "directory"):
            work = root / f"{part}-{kind}"
            work.mkdir()
            try:
                result = subprocess.run([sys.executable, __file__, "special-worker", str(work / "picks"), part, kind],
                                        text=True, capture_output=True, timeout=6, check=True)
                out = json.loads(result.stdout) if result.stdout else {"fixture": "expected directory"}
            except subprocess.TimeoutExpired:
                out = {"timeout": 6}
            print(json.dumps(dict(part=part, kind=kind, **out)), flush=True)


if __name__ == "__main__":
    with offline():
        mode = sys.argv[1]
        if mode == "special-worker":
            root = Path(sys.argv[2])
            if sys.argv[3] == "receipt_directory":
                from bts.reconcile_receipt import receipts_root

                receipts_root(root).parent.mkdir(parents=True)
            original.special_worker(root, sys.argv[3], sys.argv[4])
        else:
            with scratch(mode) as root:
                (special if mode == "special" else getattr(original, mode))(Path(root))
