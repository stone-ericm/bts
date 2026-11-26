from pathlib import Path
import shutil
import subprocess


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
ARCHIVED = REVIEW / "round2"
ROUND1_CARBON = (
    ROOT.parent / "wr-c2-r2-codex" / ".codex-review" / "wr-c2-r2" / "round1" / "carbon"
)


def stage_probes(source, destination, pattern):
    destination.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob(pattern))
    assert files, source
    for path in files:
        shutil.copyfile(path, destination / path.name)


stage_probes(ARCHIVED / "codex", REVIEW, "probe_receipt_*.py")
stage_probes(ARCHIVED / "codex", REVIEW, "replay_round1.py")
stage_probes(ARCHIVED / "round1-codex", REVIEW / "round1" / "codex", "probe_*.py")
stage_probes(ROUND1_CARBON, REVIEW / "round1" / "carbon", "probe_*.py")
subprocess.run([str(ROOT / ".venv/bin/python"), str(REVIEW / "replay_round1.py"), "prepare"],
               cwd=ROOT, check=True, timeout=15)

previous = REVIEW / "previous"
previous.mkdir(exist_ok=True)
archive = subprocess.run(["git", "archive", "e897c766", "src"], cwd=ROOT,
                         capture_output=True, check=True, timeout=10)
subprocess.run(["tar", "-x", "-C", str(previous)], input=archive.stdout,
               check=True, timeout=10)
