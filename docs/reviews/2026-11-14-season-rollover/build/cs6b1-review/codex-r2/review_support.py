"""Scratch archives and environments shared by the review runners."""

from io import BytesIO
import os
from pathlib import Path
import subprocess
import tarfile

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]


def archive(commit, directory):
    directory.mkdir()
    raw = subprocess.check_output(["git", "archive", commit, "src"], cwd=ROOT)
    with tarfile.open(fileobj=BytesIO(raw)) as bundle:
        bundle.extractall(directory, filter="data")
    return directory / "src"


def environment(source, name):
    return {
        **os.environ,
        "TZ": "America/New_York",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": os.pathsep.join(map(str, [REVIEW / "guard", source, ROOT])),
        "CS6B1R2_SOURCE": str(source),
        "CS6B1R2_DENIED": str(REVIEW / f"{name}-denied.log"),
    }
