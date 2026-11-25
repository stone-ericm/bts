"""Isolate the cutoff control from the confirmed pick in the retained probe."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import probe_contest as probe
from scripts.audit.season_walk import walk


def isolated_delivery(directory, before, after=None, at_cutoff=False):
    if at_cutoff:
        directory = directory.parent / "cutoff-control" / "picks"
        probe.stale(directory, before)
    return real_delivery(directory, before, after, at_cutoff)


real_delivery = probe.send_stale
with TemporaryDirectory(dir=probe.WORK) as temp, walk.offline(probe.at("2027-07-15T11:40")), \
        patch.object(probe, "send_stale", isolated_delivery):
    probe.settlements(Path(temp))
    probe.later_check(Path(temp))
