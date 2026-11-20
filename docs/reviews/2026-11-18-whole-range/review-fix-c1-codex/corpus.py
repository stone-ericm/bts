"""Compare base end-of-slate and HEAD after-midnight classifications."""

from datetime import date, datetime, timedelta
from itertools import product
import json
import logging
from pathlib import Path
import random
import shutil
import signal
import socket
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
DAYS = (date(2027, 7, 12), date(2027, 2, 10), date(2027, 12, 31))
STATE_NAMES = (
    "contest_streak.manual.json", "contest_streak.json", "season_contract.json", "saver_state.json",
)


def fixture(probe, picks, day, spec):
    manual, automatic, contract, saver, coverage = spec
    manual_path = picks / "account_state" / STATE_NAMES[0]
    if manual == "malformed":
        manual_path.write_text("{")
    elif manual == "non-UTF8":
        manual_path.write_bytes(b"\xff")
    elif manual in ("dated", "dated saver", "dated next"):
        probe.dated(picks, day + timedelta(days=manual == "dated next"),
                    streak=12 if manual == "dated saver" else 0)
    elif manual != "absent":
        offset = {"before": -1, "at": 0, "after": 1, "undated": 0, "legacy": -1}[manual]
        body = {"streak": 12, "source_date": str(day)}
        if manual != "legacy":
            body["override_expires_at"] = (probe.end(day) + timedelta(microseconds=offset)).isoformat()
        if manual == "undated":
            del body["source_date"]
        probe.put(picks, STATE_NAMES[0], body)

    if automatic != "absent":
        fields = {
            "valid": {}, "low": {"active_streak": 3}, "high": {"active_streak": 25, "best_streak": 26},
            "malformed": {}, "non-UTF8": {}, "bad streak": {"active_streak": True},
            "bad best": {"best_streak": 58}, "bad saver": {"saver_available": "used"},
            "unsupported": {"schema_version": "bts season-contract\n--date 2041-07-01"},
            "missing schema": {"schema_version": None}, "bad season": {"season": True},
            "wrong season": {"season": day.year + 1}, "mismatch": {"activation_id": "b" * 32},
            "bad activation": {"activation_id": "bad"}, "timestamp only": {"source_date": None},
            "no date": {"source_date": None, "recorded_at": None},
            "grace": {"source_date": str(day + timedelta(days=2))},
            "future": {"source_date": str(day + timedelta(days=3))},
        }[automatic]
        target = probe.auto(picks, day, **fields)
        if automatic == "malformed":
            target.write_text("{")
        if automatic == "non-UTF8":
            target.write_bytes(b"\xff")

    if contract != "absent":
        fields = {"invalid switch": {"season": False}, "other season": {"season": day.year + 1}}
        target = probe.switch(picks, day, status="revoked" if contract == "revoked" else "automatic",
                              **fields.get(contract, {}))
        if contract == "malformed":
            target.write_text("{")
    if saver != "absent":
        target = probe.saver_flag(picks, day.replace(year=day.year + (saver == "other season")),
                                  state=saver if saver in ("active", "used", "not_earned") else "active")
        if saver == "malformed":
            target.write_text("{")
        elif saver == "invalid flag":
            probe.saver_flag(picks, day, state="garbage")
    for offset in range(coverage):
        probe.settled(picks, day - timedelta(days=offset))


def spec_cases():
    axes = (
        ("absent", "dated", "dated saver", "dated next", "before", "at", "after", "legacy", "undated",
         "malformed", "non-UTF8"),
        ("valid", "absent", "low", "high", "malformed", "non-UTF8", "bad streak", "bad best", "bad saver",
         "unsupported", "missing schema", "bad season", "wrong season", "mismatch", "bad activation",
         "timestamp only", "no date", "grace", "future"),
        ("valid", "absent", "revoked", "other season", "malformed", "invalid switch"),
        ("active", "used", "not_earned", "absent", "other season", "malformed", "invalid flag"),
        (0, 1, 2),
    )
    baseline = tuple(axis[0] for axis in axes)
    cases = {baseline}
    for first in range(len(axes)):
        for second in range(first + 1, len(axes)):
            for left, right in product(axes[first], axes[second]):
                fields = list(baseline)
                fields[first], fields[second] = left, right
                cases.add(tuple(fields))
    choices = random.Random(20)
    for _ in range(200):
        cases.add(tuple(choices.choice(axis) for axis in axes))
    return sorted(cases)


def evaluate(revision):
    source = HERE / "base/src" if revision == "base" else HERE.parents[1] / "src"
    sys.path.insert(0, str(source))
    import bts
    from bts.health import contest_state

    assert Path(bts.__file__).resolve().is_relative_to(source)
    print(bts.__file__, file=sys.stderr)
    sys.path.insert(0, str(HERE / "contract-check-c1"))
    import probe

    rows = {}
    with TemporaryDirectory(dir=HERE) as scratch:
        picks = Path(scratch) / "picks"
        for day, spec in product(DAYS, spec_cases()):
            shutil.rmtree(picks, ignore_errors=True)
            (picks / "account_state").mkdir(parents=True)
            fixture(probe, picks, day, spec)
            for expected in (False, True):
                key = repr((day.isoformat(), expected, spec))
                kwargs = {"historical_day": day} if revision == "head" else {}
                alerts = contest_state.check(picks, expected=expected, now=probe.end(day), **kwargs)
                rows[key] = [(item.level, item.source, item.incident_key) for item in alerts]
    return rows


if __name__ == "__main__":
    signal.alarm(55)
    logging.disable(logging.CRITICAL)
    if sys.argv[1] == "compare":
        base = json.loads((HERE / "corpus-base.json").read_text())
        head = json.loads((HERE / "corpus-head.json").read_text())
        assert base.keys() == head.keys()
        differences = [(key, base[key], head[key]) for key in base if base[key] != head[key]]
        print(f"{len(base)} matched inputs, {sum(map(len, base.values()))} base alerts, "
              f"{len(differences)} classification differences")
        for row in differences:
            print(row)
        assert not differences
    else:
        with patch.object(socket, "socket", side_effect=AssertionError("offline")), \
             patch.object(socket, "create_connection", side_effect=AssertionError("offline")):
            rows = evaluate(sys.argv[1])
        print(json.dumps(rows, sort_keys=True))
