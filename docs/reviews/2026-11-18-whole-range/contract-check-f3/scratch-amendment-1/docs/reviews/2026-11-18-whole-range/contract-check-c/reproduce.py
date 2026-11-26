"""Run the cited slice probes without writing under docs or using pytest."""

from contextlib import redirect_stderr
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SLICES = ROOT / "docs/reviews/2026-11-18-whole-range"


def carbon(scratch):
    import test_slice1_probes_v3 as probes

    probes._H.REPO_ROOT = ROOT
    names = [name for name in vars(probes) if name.startswith("test_probe_")]
    for number in (1, 2, 3, 6):
        name, = (name for name in names if name.startswith(f"test_probe_{number}_"))
        try:
            getattr(probes, name)(scratch)
        except Exception:
            traceback.print_exc()
            print(f"{name}: original assertion did not reproduce", flush=True)
        else:
            print(f"{name}: reproduced", flush=True)
    print(json.dumps(probes.EVIDENCE, indent=2))


def codex_replay(scratch):
    import clocks_and_scoring
    import offline

    offline.HERE = scratch
    offline.ROOT = ROOT
    clocks_and_scoring.refused_preview_blocks_replay()
    print("Codex refused-preview replay: reproduced", flush=True)


def codex_restart(scratch):
    from bts.dm import send_dm
    import opening_edges as edges
    import opening_rehearsal as opening

    opening.REPO = ROOT
    opening.HERE = scratch
    with subprocess.Popen(
        ["git", "archive", "619abda7", "src"], cwd=ROOT, stdout=subprocess.PIPE,
    ) as archive:
        with tarfile.open(fileobj=archive.stdout, mode="r|") as snapshot:
            snapshot.extractall(scratch / "base", filter="data")
        assert archive.wait() == 0
    for phase in ("before_remote_accept", "after_remote_accept", "corrected_after_marker"):
        with TemporaryDirectory(dir=scratch) as case:
            season = opening.Season(Path(case))
            season.real_transport = send_dm
            season.old_seed()
            with season.run_offline(), redirect_stderr(season.journal):
                season.restore()
                result = edges.restart_at_send(season, phase)
            messages = [m["body"] for m in result["messages"]]
            assert not any("DELIVERY OUTCOME UNKNOWN" in body for body in messages)
            assert any("Fix it now!" in body for body in messages)
            print(phase, "unknown-delivery alerts=0", result["next_morning"]["state"], flush=True)


if __name__ == "__main__":
    mode = sys.argv[1]
    folder, run = {
        "carbon": ("slice-1-carbon", carbon),
        "replay": ("slice-3-codex", codex_replay),
        "restart": ("slice-1-codex", codex_restart),
    }[mode]
    sys.path.insert(0, str(SLICES / folder))
    sys.path.insert(0, str(ROOT / "src"))
    with TemporaryDirectory(dir=HERE) as directory:
        run(Path(directory))
