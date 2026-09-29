import pytest

from scripts.audit.season_ledger.acquire import acquire
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
