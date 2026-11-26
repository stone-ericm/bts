"""Replay the archived pre-implementation examples in the round-two base archive."""

from pathlib import Path
import sys
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import replay_contract as archived

archived.BASE = HERE / "base"
archived.CHECK = archived.BASE / ".codex-review" / "contract-check-w3"

group = sys.argv[1]
if group != "corrected":
    raise SystemExit(archived.main(group))

sys.path.insert(0, str(archived.CHECK / "r2"))
import probe
import remaining

original = probe.offline.Offline.sends


def pick_only(world, kind):
    captured = original(world, kind)
    if kind == "dm":
        return [body for body in captured if body.startswith("Today's pick")]
    return captured


with patch.object(probe.offline.Offline, "sends", pick_only):
    remaining.status_delay(False)
    remaining.status_delay(True)
