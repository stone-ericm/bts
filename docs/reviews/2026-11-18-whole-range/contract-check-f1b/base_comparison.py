"""Run the exact base entry modules against the same file-backed cases."""
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = "1e69d0f0"
sys.path.insert(0, str(REPO))


def run(root):
    import bts

    overlay = root / "base" / "bts"
    overlay.mkdir(parents=True)
    for file in ("cli.py", "entry_receipt.py"):
        source = subprocess.run(["git", "show", f"{BASE}:src/bts/{file}"], cwd=REPO, stdout=subprocess.PIPE,
                                check=True, timeout=2).stdout
        (overlay / file).write_bytes(source)
    bts.__path__.insert(0, str(overlay))
    from bts import cli, entry_receipt
    from tests.test_wr_f1_wrong_game import DAY, MATCH, PITCH, RIGHT, WRONG, World
    assert Path(cli.__file__).parent == Path(entry_receipt.__file__).parent == overlay

    late = World(root / "last-slot")
    late.commit(pitch=f"{DAY}T19:05:00-04:00")
    assert late.check("18:30", fetch_failure="pending")["marker"] is None
    completed = late.check("18:45", elapsed=850)
    assert "NOT entered" in completed["dm"][0]
    print("base 850-second observation", completed["marker"], completed["dm"])

    wrong = World(root / "morning")
    wrong.commit(pitch=PITCH)
    wrong.capture([WRONG])
    first = wrong.check("18:30", MATCH)
    assert first["marker"]["status"] == "confirmed"
    wrong.capture([RIGHT], f"{DAY}T230000Z")
    assert wrong.check("18:45", MATCH)["fetches"] == []
    outcome = wrong.morning()
    assert outcome["outcome"]["state"] == "entry_unverifiable" and outcome["dm"]
    print("base morning after terminal batter-only confirmation", outcome)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(dir=HERE) as tmp:
        run(Path(tmp))
