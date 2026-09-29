import pytest

from scripts.audit.season_ledger.acquire import acquire, fetch_schedule
from scripts.audit.season_ledger.bundle import open_bundle
from tests.scripts.season_ledger.builders import gz


def test_acquire_copies_sources_declares_missing_inputs_and_seals(tmp_path):
    snap = tmp_path / "final-20260928"
    picks = snap / "data" / "picks"
    (picks / "2026-05-01").mkdir(parents=True)
    (picks / "2026-05-01.json").write_bytes(b'{"pick": {}}')
    (picks / "2026-05-01" / "decision.json").write_bytes(b"{}")
    static = snap / "data" / "leaderboard" / "static_snapshots"
    for feed in ("rounds", "units", "players"):
        (static / feed).mkdir(parents=True)
        (static / feed / ".last_sha256").write_text("x\n")
    (static / "rounds" / "20260704T030011Z.json.gz").write_bytes(gz(b'{"rounds": []}'))
    for stamp in ("20260704T030011Z", "20260801T030011Z", "20260928T123001Z"):
        (static / "players" / f"{stamp}.json.gz").write_bytes(gz(b'{"players": []}'))
    grab = snap / "data" / "leaderboard" / "final_grab_20260927" / "raw" / "static"
    grab.mkdir(parents=True)
    (grab / "002_players.json.gz").write_bytes(gz(b'{"players": []}'))
    (snap / "cron.log").write_text("log\n")

    def fetch(day):
        if day == "2026-05-02":
            raise OSError("timeout")
        return b'{"dates": []}'

    out = tmp_path / "bundle"
    acquire(snapshot_root=snap, out_root=out, dates=["2026-05-01", "2026-05-02"], fetch=fetch,
            now_utc=lambda: "2026-09-28T16:00:00.000000Z")
    manifest, files = open_bundle(out)
    assert files["picks/2026-05-01.json"] == b'{"pick": {}}'
    assert sorted(k for k in files if k.startswith("static/players/")) == [
        "static/players/20260704T030011Z.json.gz", "static/players/20260928T123001Z.json.gz"]
    assert files["static/units/NO_CAPTURES"] is None and "static/rounds/.last_sha256" not in files
    assert files["static/grab_20260927/002_players.json.gz"].startswith(b"\x1f\x8b")
    assert files["logs/cron.log"] == b"log\n" and files["logs/journal_bts-scheduler_retained.txt"] is None
    assert files["schedules/2026-05-01.json"] == b'{"dates": []}' and files["schedules/2026-05-02.json"] is None
    entries = {e["rel_path"]: e for e in manifest["entries"]}
    assert entries["schedules/2026-05-02.json"]["note"] == "fetch_failed:OSError"
    assert entries["picks/2026-05-01.json"]["source_path"] == "data/picks/2026-05-01.json"
    assert entries["picks/2026-05-01.json"]["source_mtime_utc"].endswith("Z")
    with pytest.raises(FileExistsError):
        acquire(snapshot_root=snap, out_root=out, dates=[], fetch=fetch, now_utc=lambda: "x")


def test_schedule_fetch_retries_with_backoff_and_seals_only_a_schedule():
    # Final review #3: a transient error or a non-schedule body is retried; only the last failure becomes a declared
    # `missing` entry, and nothing but a JSON object with a `dates` list is ever returned for sealing.
    sleeps, answers = [], [OSError("timeout"), b"<html>challenge</html>", b'{"dates": []}']

    def get(day):
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    assert fetch_schedule("2026-05-01", get, sleep=sleeps.append) == b'{"dates": []}'
    assert sleeps == [2, 4]
    with pytest.raises(ValueError):
        fetch_schedule("2026-05-01", lambda day: b"<html>", sleep=lambda seconds: None)
    with pytest.raises(ValueError, match="dates"):
        fetch_schedule("2026-05-01", lambda day: b'{"copyright": "x"}', sleep=lambda seconds: None)
    # Codex code r1 #5: a response for another date is no schedule for this one.
    with pytest.raises(ValueError, match="requested date"):
        fetch_schedule("2026-05-01", lambda day: b'{"dates": [{"date": "2026-04-30", "games": []}]}',
                       sleep=lambda seconds: None)
