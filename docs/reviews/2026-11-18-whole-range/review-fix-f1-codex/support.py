"""Use the repository's captured transports while making independent assertions."""
from itertools import count
import json
from pathlib import Path
import runpy
import shutil

import bts

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
fixture = runpy.run_path(str(REPO / "tests/test_wr_f1_wrong_game.py"))
DAY = fixture["DAY"]
PITCH = fixture["PITCH"]
World = fixture["World"]
row = fixture["row"]
MATCH = [row()]
DOUBLE = [row(), row(78, 2, 2)]
RIGHT = {"id": 77, "roundId": 1, "feedId": 101}
WRONG = {**RIGHT, "feedId": 102}
RIGHT_DD = {"id": 78, "roundId": 1, "feedId": 201}
WRONG_DD = {**RIGHT_DD, "feedId": 202}
ALL_CLEAR = f"✅ BTS pick entry confirmed for {DAY} (match)."
FIRST = "⚠️ BTS entry game needs verification."
REMINDER = "⏰ BTS entry game verification is still needed."


def print_source():
    print(f"bts.__file__ = {bts.__file__}", flush=True)


class Arena:
    """Reuse the same fixture path across cells and when comparing revisions."""
    def __init__(self, tag):
        self.root = HERE / f"state-{tag}"
        self.ids = count()

    def world(self, day=DAY):
        root = self.root / str(next(self.ids))
        return World(root, day)

    def clear(self, world):
        shutil.rmtree(world.root)


def read_marker_bytes(world):
    if not world.marker.is_file():
        return None
    raw = world.marker.read_bytes()
    # Normalize only the receipt ID and the marker clock, preserving raw layout.
    import re
    return re.sub(rb'("(?:receipt|checked_at)"\s*:\s*)"[^"]*"', rb'\1"<normalized>"', raw).hex()


def normalized(value, root):
    if isinstance(value, dict):
        return {key: normalized(item, root) for key, item in value.items()
                if key not in {"receipt", "attempt_id", "confirmed_by", "checked_at", "started_at",
                               "published_at", "response_completed_at", "failed_at"}}
    if isinstance(value, list):
        return [normalized(item, root) for item in value]
    if isinstance(value, str):
        return value.replace(str(root), "<fixture>")
    return value


def dump(label, value):
    print(label, json.dumps(value, ensure_ascii=False, sort_keys=True))
