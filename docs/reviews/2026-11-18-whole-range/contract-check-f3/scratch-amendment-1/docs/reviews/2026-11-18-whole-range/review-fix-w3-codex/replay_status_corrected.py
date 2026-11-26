"""Run the archived status-delay example while counting only pick DMs."""

from pathlib import Path
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "base/.codex-review/contract-check-w3/r2"))

import probe
import remaining


original_sends = probe.offline.Offline.sends


def pick_sends(world, kind):
    sends = original_sends(world, kind)
    if kind == "dm":
        return [body for body in sends if body.startswith("Today's pick")]
    return sends


with patch.object(probe.offline.Offline, "sends", pick_sends):
    remaining.status_delay(False)
    remaining.status_delay(True)
