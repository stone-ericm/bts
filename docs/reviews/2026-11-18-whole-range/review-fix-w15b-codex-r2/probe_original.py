"""Run the archived round-one restart replay and its timestamp/path fixture."""
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import bts


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import probe_health as archived_health
import probe_journal as archived_journal


def journal(source, previous):
    if previous:
        archived_journal.main(source)
        return

    # Same fixture, journal transcript, dispatch state, receipt and clock as the archived probe.
    a = archived_journal
    a.STATUS.parent.mkdir(parents=True)
    a.STATUS.write_bytes(b"{")
    a.STATE_DIR.mkdir(parents=True)
    (a.STATE_DIR / "decision.json").write_text(json.dumps(dict(
        schema_version=archived_health.DECISION_SCHEMA, date=str(a.DAY), action="single",
        source="mdp", delivery_status="delivered", scoreable=True,
    )))
    (a.STATE_DIR / "scheduler_state.json").write_text(json.dumps(dict(
        date=str(a.DAY), pick_locked=True, analytics_jobs={
            "shadow": {"status": "dispatched", "dispatched_at": a.DISPATCHED_AT},
        },
    )))
    receipt = Path("health-receipt.json")
    first, sent, rejected, log = a.run("", "2026-07-22T23:40:00-04:00", "first-eod", receipt)
    assert not first and [len(p) for p in sent] == [741] and not rejected
    (detail,) = (line for line in log.splitlines() if "status unreadable" in line
                 and line.startswith("Probable-pitcher cache status unreadable"))
    assert "an entry for 2026-07-22 whose name is not a status record name" in detail
    journal = "Jul 22 23:40:00 synthetic bts[1]: " + detail + "\n"
    second, sent, rejected, _ = a.run(journal, "2026-07-22T23:50:00-04:00", "eod-restart", receipt)
    assert not second and not sent and not rejected
    saved = json.loads(receipt.read_bytes())
    assert saved["status"] == "sent" and saved["critical_count"] == 0
    genuine, sent, rejected, _ = a.run(
        journal + a.GENUINE + "\n", "2026-07-22T23:50:00-04:00", "genuine-oom", Path("genuine.json"))
    assert len(genuine) == 1 and a.GENUINE in genuine[0] and sent and not rejected


def timestamp(previous):
    fixture = archived_health
    if previous:
        fixture.timestamp_and_log_controls(False)
        return
    path = fixture.write_record(
        fixture.STATUS,
        timestamp=fixture.AT.replace("T", fixture.TICK),
        cache_path="vendor\nnext/models/cache.json",
        preserved="probable_pitcher_lookup.20261231T140000Z.bbbbbbbbbbbb.json",
    )
    before = path.read_bytes()
    with fixture.captured() as (packets, rejected, logs):
        result, = fixture.run()
        text = logs.getvalue()
    assert fixture.AT in text and fixture.TICK not in text and "vendor" not in text
    assert "rejected folder beside the cache" in text and len(text.splitlines()) == 2
    assert len(packets) == 1 and not rejected and path.read_bytes() == before
    assert result.incident_key == fixture.summary_key(fixture.DAY, "a" * 64)
    print("Original timestamp/path fixture:", repr(text), "wire graphemes", len(packets[0]))


def main(source):
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py", bts.__file__
    previous = source == HERE / "previous/src"
    print("Imported", bts.__file__)
    start = Path.cwd()
    with TemporaryDirectory(dir=HERE, prefix="original-") as work:
        for name, run in (("journal", lambda: journal(source, previous)),
                          ("timestamp", lambda: timestamp(previous))):
            folder = Path(work) / name
            folder.mkdir()
            os.chdir(folder)
            try:
                run()
            finally:
                os.chdir(start)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
