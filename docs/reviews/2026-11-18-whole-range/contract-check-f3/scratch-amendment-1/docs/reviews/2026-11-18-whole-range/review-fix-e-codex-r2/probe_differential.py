"""Compare full normalized behavior snapshots from two isolated source trees."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from fixtures import HERE, ROOT, fixture

probe = fixture("probe_differential")


if __name__ == "__main__":
    family = sys.argv[1]
    if len(sys.argv) > 2:
        print(json.dumps(probe.worker(sys.argv[2] == "base", family, Path(sys.argv[3])),
                         default=probe.serialization, sort_keys=True))
        sys.exit(0)
    snapshots = []
    with tempfile.TemporaryDirectory(dir=HERE) as name:
        for revision, tree in (("base", HERE / "base/src"), ("head", ROOT / "src")):
            directory = Path(name) / revision
            directory.mkdir()
            run = subprocess.run(
                [sys.executable, __file__, family, revision, str(directory)],
                cwd=ROOT, env=os.environ | {"PYTHONPATH": str(tree)},
                stdout=subprocess.PIPE, text=True, timeout=12, check=True,
            )
            snapshots.append(json.loads(run.stdout))
            (HERE / f"{family}-{revision}.json").write_text(json.dumps(snapshots[-1], indent=2) + "\n")
    assert snapshots[0] == snapshots[1], family
    print(f"{family}: identical normalized results ({len(snapshots[0])} entries)")
