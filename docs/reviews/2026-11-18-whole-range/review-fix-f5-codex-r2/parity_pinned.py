"""Compare byte parity with the fixture writer's lineup-capture clock pinned as well."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from parity import HERE, matrix, source


class FixtureDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2027, 3, 24, 8, tzinfo=timezone.utc).astimezone(tz)


if __name__ == "__main__":
    version = sys.argv[1]
    source(version)
    with TemporaryDirectory() as directory, patch("bts.picks.datetime", FixtureDateTime):
        results = matrix(Path(directory))
    (HERE / f"parity-pinned-{version}.json").write_text(json.dumps(results, indent=2) + "\n")
    if version != "base":
        base = json.loads((HERE / "parity-pinned-base.json").read_text())
        assert base.keys() == results.keys()
        mismatch = [key for key in base if base[key] != results[key]]
        assert not mismatch, mismatch
    print(len(results), "sequences", sum(len(case["calls"]) for case in results.values()), "invocations")
