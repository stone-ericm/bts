"""Compare the manual CLI and unchanged-reader inputs on archived base and candidate."""
from datetime import date, datetime, timezone
from itertools import product
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from support import HERE, DAY, Replay, source
from bts.entry_receipt import discover
from bts.health import pick_entry
from tests.test_wr_f1_wrong_game import RIGHT, WRONG, row


def normalize(value, world):
    body = json.dumps(value, sort_keys=True).replace(str(world.root), "<data>")
    receipts = sorted(discover(world.picks, DAY), key=lambda rec: rec["started_at"])
    for index, receipt in enumerate(receipts):
        body = body.replace(receipt["attempt_id"], f"attempt-{index}")
    return json.loads(body)


def observe(world, events):
    for event in events:
        assert "early_cron_run" not in (event["marker"] or {})
        assert "early_cron_run" not in event["receipt"]
    files = {}
    for path in world.root.rglob("*"):
        if path.is_file():
            raw = path.read_bytes()
            try:
                raw = raw.decode()
            except UnicodeDecodeError:
                raw = raw.hex()
            files[str(path.relative_to(world.root))] = raw
    alerts = pick_entry.check(world.picks, date.fromisoformat(DAY), datetime(2030, 1, 1, tzinfo=timezone.utc))
    return normalize({"calls": events, "files": files, "morning": world.morning(),
                      "health": [(alert.level, alert.message) for alert in alerts]}, world)


def matrix(root):
    traces = {}
    for pick_state, prior, account in product(
        ("no_pick", "preview", "early", "original", "double", "cutoff"),
        ("absent", "same_alerted", "same_confirmed", "previous_dm_failed"),
        ("match", "wrong_game", "missing", "crosswalk", "login_fails"),
    ):
        name = "/".join((pick_state, prior, account))
        world = Replay(root / name)
        if pick_state != "no_pick":
            pitch = {"preview": "10:50", "early": "10:05", "original": "10:50", "double": "21:05",
                     "cutoff": "10:05"}[pick_state]
            world.commit_at(pitch, delivery=None if pick_state == "preview" else "delivered",
                            dd=(1020, 202, "10:50") if pick_state == "double" else None)
        if prior != "absent":
            world.set_marker({"date": "2027-03-23" if prior == "previous_dm_failed" else DAY,
                              "status": prior.removeprefix("same_").removeprefix("previous_"), "reason": "no_pick"})
        if account in ("match", "wrong_game", "crosswalk"):
            world.capture([WRONG if account == "wrong_game" else RIGHT, {"id": 78, "roundId": 1, "feedId": 202}])
            pending = [row(), row(78, 2, 2)] if pick_state == "double" else [row()]
        else:
            pending = []
        at = "09:45" if pick_state == "early" else "10:00"
        event = world.invoke(at, manual=True, pending=pending, crosswalk={} if account == "crosswalk" else None,
                             auth="login" if account == "login_fails" else "ok")
        traces[name] = observe(world, [event])
    world = Replay(root / "retry")
    world.commit_at("10:50")
    traces["retry"] = observe(world, [world.invoke("10:00", manual=True, dm="failed"),
                                      world.invoke("10:15", manual=True),
                                      world.invoke("10:30", manual=True, pending=[row()])])
    world = Replay(root / "elapsed")
    world.commit_at("10:20")
    traces["elapsed"] = observe(world, [world.invoke("10:00", manual=True, elapsed=901)])
    return traces


if __name__ == "__main__":
    version = sys.argv[1]
    source(version)
    with TemporaryDirectory() as directory:
        results = matrix(Path(directory))
    (HERE / f"parity-{version}.json").write_text(json.dumps(results, indent=2) + "\n")
    if version != "base":
        base = json.loads((HERE / "parity-base.json").read_text())
        assert base.keys() == results.keys()
        mismatch = [key for key in base if base[key] != results[key]]
        assert not mismatch, mismatch
    print(len(results), "sequences", sum(len(case["calls"]) for case in results.values()), "invocations")
