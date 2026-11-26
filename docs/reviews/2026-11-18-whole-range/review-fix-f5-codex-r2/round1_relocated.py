"""Find the original fixtures from the review record's checked-in location."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from round1 import alerts, run
from support import ROOT, legacy, source


if __name__ == "__main__":
    version = sys.argv[1]
    source(version)
    alerts.ROOT = ROOT
    with TemporaryDirectory() as directory:
        run(Path(directory), version)
    if version == "f5":
        legacy("mutation_witness").replay("candidate")
