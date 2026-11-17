"""Check which ET date a dry-run uses when the real TOML load spans midnight."""

import local_guard
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo
import json

from startup_probe import GAME, HERE, REV, captured, cli, fixture_config, invoke, orchestrator, scheduler

ET = ZoneInfo("America/New_York")
BEFORE = datetime(2027, 7, 15, 23, 59, 59, 900_000, tzinfo=ET)
AFTER = datetime(2027, 7, 16, 0, 0, 0, 100_000, tzinfo=ET)


def probe(root):
    config = fixture_config(root, "[scheduler]\ncontest_state_required=true\n")
    real_load = orchestrator.load_config
    current = BEFORE
    dates = []

    class CrossingClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return current.replace(tzinfo=None) if tz is None else current.astimezone(tz)

    def load(path):
        nonlocal current
        result = real_load(path)
        current = AFTER
        return result

    def fetch(day):
        dates.append(day)
        return [GAME | {"gameDate": "2027-07-16T03:55:00Z"}] if day == "2027-07-15" else []

    with ExitStack() as stack:
        events = captured(stack)
        stack.enter_context(patch.object(orchestrator, "load_config", load))
        stack.enter_context(patch.object(cli, "datetime", CrossingClock))
        stack.enter_context(patch.object(scheduler, "datetime", CrossingClock))
        stack.enter_context(patch.object(scheduler, "_now_et", lambda: current))
        stack.enter_context(patch.object(scheduler, "fetch_schedule", fetch))
        outcome = invoke("schedule", config, "--dry-run", date=False)
    assert outcome["exit"] == 0, outcome
    assert not local_guard.network
    return dict(dates=dates, events=events, outcome=outcome,
                heartbeat=json.loads((root / "heartbeat").read_text())["state"])


if __name__ == "__main__":
    with TemporaryDirectory(dir=HERE) as folder:
        data = probe(Path(folder))
    (HERE / f"midnight-{REV}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(REV, data)
