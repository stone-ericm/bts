"""Compare byte-for-byte artifacts, alert keys and routes on two real revisions."""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
from bts.features import compute


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))
import probe_health as fixture


DATA = Path("custom") / ("room-" + chr(96) + "-source")
PICKS = DATA / "selections"
CACHE = DATA / "models/league-cache.json"
STATUS = DATA / "health_state" / fixture.SOURCE
RAW = Path("test-raw")
FEED = RAW / "2026/100.json"
MALFORMED = f"{fixture.DAY}.copy-{chr(96)}.json"
SECOND_MALFORMED = f"{fixture.DAY}.copy-room.json"
OLD = b'{"100":null,"800":{"away":1,"home":null,"away_tid":51,"home_tid":52}}'


def snapshot():
    evidence = CACHE.parent / "rejected"
    return {
        "active": CACHE.read_bytes().hex(),
        "status": [(path.name, path.read_bytes().hex()) for path in sorted(STATUS.glob("*.json"))],
        "evidence": [(path.name, path.read_bytes().hex()) for path in sorted(evidence.glob("*.json"))],
    }


def build(at):
    out = io.StringIO()
    fixture.Clock.at = at
    with (
        contextlib.redirect_stderr(out),
        patch.object(fixture.producer, "datetime", fixture.Clock),
        patch.object(compute, "_probable_pitcher_cache_path", CACHE),
    ):
        lookup = compute._build_probable_pitcher_lookup(RAW)
    print("Producer stderr:", repr(out.getvalue()), file=sys.stderr)
    return sorted(lookup)


def production_sequence():
    CACHE.parent.mkdir(parents=True)
    CACHE.write_bytes(OLD)
    at = datetime.fromisoformat(fixture.AT)
    assert build(at) == [800]
    first, = STATUS.glob("*.json")
    FEED.parent.mkdir(parents=True)
    FEED.write_text("{}")
    assert build(at) == [100, 800]
    # A new incident D+1, after the D rewrite. Same record/evidence semantics on both revisions.
    CACHE.write_bytes(b"\xff")
    assert build(at + timedelta(minutes=2)) == [100]
    (STATUS / MALFORMED).write_bytes(b"{")
    (STATUS / SECOND_MALFORMED).write_bytes(b"}")
    before = snapshot()
    by_day = {day: [json.loads(bytes.fromhex(payload)) for name, payload in before["status"]
                    if name.startswith(str(day)) and chr(96) not in name and "copy-room" not in name]
              for day in (fixture.DAY, fixture.NEXT)}
    assert len(by_day[fixture.DAY]) == 2 and len(by_day[fixture.NEXT]) == 1
    recovered, = [row for row in by_day[fixture.DAY] if row["preserved"]]
    assert recovered["cache_path"] == str(CACHE)
    evidence = Path(recovered["cache_path"]).parent / "rejected" / recovered["preserved"]
    assert evidence.read_bytes() == OLD
    assert any(name == first.name for name, _ in before["status"])
    expected_keys = [
        fixture.summary_key(day, MALFORMED, SECOND_MALFORMED,
                            *[row["identity"] for row in by_day[fixture.DAY]],
                            *([by_day[fixture.NEXT][0]["identity"]] if day == fixture.NEXT else []))
        for day in (fixture.DAY, fixture.NEXT)
    ]
    customer_picks = PICKS.absolute()
    elsewhere = Path("other-health-cwd")
    elsewhere.mkdir()
    cwd = Path.cwd()
    os.chdir(elsewhere)
    try:
        with fixture.captured() as (packets, rejected, logs):
            keys = [fixture.run(day, picks=customer_picks)[0].incident_key for day in (fixture.DAY, fixture.NEXT)]
            log = logs.getvalue()
            assert keys == expected_keys and len(packets) == 2 and not rejected
            assert recovered["preserved"] in log and not evidence.exists()
            assert (customer_picks.parent / "health_state" / fixture.SOURCE / first.name).is_file()
            print("Cross-cwd health: relative status pointer from configured picks parent; in producer cwd",
                  "record supplies", recovered["cache_path"], "and evidence", recovered["preserved"], file=sys.stderr)
            malformed = [line for line in log.splitlines() if "status unreadable" in line
                         and line.startswith("Probable-pitcher cache status unreadable")]
            print("D and D+1 malformed details:", repr(malformed), file=sys.stderr)
    finally:
        os.chdir(cwd)
    assert snapshot() == before
    return dict(artifacts=before, keys=keys, packets=packets)


def obstructed_sequence():
    CACHE.parent.mkdir(parents=True)
    CACHE.write_bytes(OLD)
    (CACHE.parent / "rejected").write_bytes(b"not a directory")
    FEED.parent.mkdir(parents=True)
    FEED.write_text("{}")
    assert build(datetime.fromisoformat(fixture.AT)) == [100, 800]
    bad, = STATUS.glob("*.json")
    row = json.loads(bad.read_bytes())
    assert row["preserved"] is None and CACHE.read_bytes() == OLD
    before = snapshot()
    # Preserve again successfully but block status publishing at the status directory.
    bad.rename(bad.with_name("held.tmp"))
    (STATUS / "held.tmp").rename(CACHE.parent / "held.tmp")
    STATUS.rmdir()
    os.mkfifo(STATUS)
    (CACHE.parent / "rejected").unlink()
    assert build(datetime.fromisoformat(fixture.AT)) == [100, 800]
    assert CACHE.read_bytes() == OLD and STATUS.is_fifo()
    copy, = (CACHE.parent / "rejected").iterdir()
    assert copy.read_bytes() == OLD
    with fixture.captured() as (packets, rejected, logs):
        (alert,) = fixture.run(picks=PICKS)
        assert alert.incident_key == fixture.summary_key(fixture.DAY, STATUS.name)
        assert len(packets) == 1 and not rejected
        print("Unreadable custom root:", repr(logs.getvalue()), file=sys.stderr)
    direct = Path("separate-bad-" + chr(96))
    os.mkfifo(direct)
    handler = io.StringIO()
    with contextlib.redirect_stderr(handler):
        (direct_alert,) = fixture.health.check(direct, fixture.DAY)
    assert direct_alert.incident_key == fixture.summary_key(fixture.DAY, direct.name)
    return dict(before=before, blocked_recovery=copy.read_bytes().hex(), blocked_evidence_name=copy.name,
                key=alert.incident_key, direct_key=direct_alert.incident_key, packets=packets)


def child(source):
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py"
    outcome = {}
    for name, run in (("sequence", production_sequence), ("obstructed", obstructed_sequence)):
        case = Path(name)
        case.mkdir()
        os.chdir(case)
        try:
            outcome[name] = run()
        finally:
            os.chdir("..")
    print(json.dumps(outcome, sort_keys=True))


def main():
    sources = [HERE / "previous/src", ROOT / "src"]
    with TemporaryDirectory(dir=HERE, prefix="identity-") as work:
        outputs = []
        for label, source in zip(("284dc97a", "0ce9e5ee"), sources, strict=True):
            cwd = Path(work) / label
            cwd.mkdir()
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), str(source)], cwd=cwd,
                env=dict(os.environ, PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1"),
                capture_output=True, timeout=15,
            )
            print(label, str(source), result.stderr.decode())
            assert result.returncode == 0, result.stdout
            outputs.append(result.stdout)
        assert outputs[0] == outputs[1]
        print("Byte-identical comparison digest", hashlib.sha256(outputs[0]).hexdigest())
        states = json.loads(outputs[0])
        for name, state in states.items():
            print(name, "keys", repr(state.get("keys") or [state["key"], state["direct_key"]]),
                  "packets", [len(packet) for packet in state["packets"]])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        child(Path(sys.argv[1]).resolve())
