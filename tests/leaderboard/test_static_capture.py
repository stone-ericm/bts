"""Tests for the BTS static-JSON capture (pregame consensus archival)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bts.leaderboard.static_capture import (
    FEEDS,
    capture_all,
    snapshot_filename,
    validate_payload,
)

NOW = datetime(2026, 7, 3, 23, 30, 0, tzinfo=timezone.utc)
NOW2 = datetime(2026, 7, 4, 0, 0, 0, tzinfo=timezone.utc)


def _payloads(**overrides) -> dict[str, bytes]:
    base = {
        "most_selected_players": {"mostSelectedPlayers": [
            {"roundId": 923, "playerId": 94, "probabilityStarter": 0.65, "numberSelections": 396}]},
        "suggested_players": {"suggestedPlayers": [
            {"roundId": 924, "playerId": 377, "probabilityStarter": 0.72}]},
        "rounds": {"rounds": [{"id": 923, "date": "2026-07-03T08:00:00-04:00"}]},
        "units": {"units": [{"id": 1323, "homeSquadId": 1, "awaySquadId": 2}]},
        "players": {"players": [{"id": 377, "feedId": 650333, "name": "Luis Arraez"}]},
        "checksums": {"rounds": "abc123", "units": "def456"},
    }
    base.update(overrides)
    return {k: json.dumps(v).encode() for k, v in base.items()}


def _fetcher(payloads: dict[str, bytes], fail: set[str] = frozenset()):
    def fetch(name: str, url: str) -> bytes:
        if name in fail:
            raise OSError(f"boom fetching {name}")
        return payloads[name]
    return fetch


class TestSnapshotFilename:
    def test_utc_sortable_no_colons(self):
        fn = snapshot_filename(NOW)
        assert fn == "20260703T233000Z.json"
        assert ":" not in fn
        assert snapshot_filename(NOW2) > fn  # lexicographic == chronological


class TestMaybeGunzip:
    def test_gzip_bytes_decompressed(self):
        import gzip
        from bts.leaderboard.static_capture import _maybe_gunzip
        payload = json.dumps({"rounds": []}).encode()
        assert _maybe_gunzip(gzip.compress(payload)) == payload

    def test_plain_bytes_untouched(self):
        from bts.leaderboard.static_capture import _maybe_gunzip
        assert _maybe_gunzip(b'{"rounds": []}') == b'{"rounds": []}'


class TestValidatePayload:
    def test_accepts_expected_key(self):
        assert validate_payload("mostSelectedPlayers", {"mostSelectedPlayers": []})

    def test_rejects_missing_key(self):
        assert not validate_payload("mostSelectedPlayers", {"other": []})

    def test_rejects_non_dict(self):
        assert not validate_payload("mostSelectedPlayers", [1, 2])

    def test_no_key_requires_nonempty_dict(self):
        assert validate_payload(None, {"anything": 1})
        assert not validate_payload(None, {})
        assert not validate_payload(None, "<html>error</html>")


class TestCaptureAll:
    def test_first_run_stores_every_feed(self, tmp_path):
        results = capture_all(tmp_path, fetch=_fetcher(_payloads()), now=NOW)
        assert {r["feed"] for r in results} == set(FEEDS)
        assert all(r["status"] == "stored" for r in results)
        for name in FEEDS:
            files = list((tmp_path / name).glob("*.json"))
            assert len(files) == 1, name
            assert files[0].name == "20260703T233000Z.json"

    def test_unchanged_content_not_restored(self, tmp_path):
        pay = _payloads()
        capture_all(tmp_path, fetch=_fetcher(pay), now=NOW)
        results = capture_all(tmp_path, fetch=_fetcher(pay), now=NOW2)
        assert all(r["status"] == "unchanged" for r in results)
        for name in FEEDS:
            assert len(list((tmp_path / name).glob("*.json"))) == 1, name

    def test_changed_feed_stores_second_snapshot(self, tmp_path):
        capture_all(tmp_path, fetch=_fetcher(_payloads()), now=NOW)
        changed = _payloads(most_selected_players={"mostSelectedPlayers": [
            {"roundId": 924, "playerId": 94, "numberSelections": 402}]})
        results = capture_all(tmp_path, fetch=_fetcher(changed), now=NOW2)
        by_feed = {r["feed"]: r for r in results}
        assert by_feed["most_selected_players"]["status"] == "stored"
        assert len(list((tmp_path / "most_selected_players").glob("*.json"))) == 2
        assert by_feed["rounds"]["status"] == "unchanged"
        assert len(list((tmp_path / "rounds").glob("*.json"))) == 1

    def test_invalid_payload_not_stored(self, tmp_path):
        pay = _payloads()
        pay["most_selected_players"] = b"<html>maintenance</html>"
        pay["units"] = json.dumps({"wrong_key": []}).encode()
        results = capture_all(tmp_path, fetch=_fetcher(pay), now=NOW)
        by_feed = {r["feed"]: r for r in results}
        assert by_feed["most_selected_players"]["status"] == "invalid"
        assert by_feed["units"]["status"] == "invalid"
        assert not (tmp_path / "most_selected_players").exists() or \
            not list((tmp_path / "most_selected_players").glob("*.json"))
        assert by_feed["rounds"]["status"] == "stored"

    def test_fetch_error_isolated_to_that_feed(self, tmp_path):
        results = capture_all(
            tmp_path, fetch=_fetcher(_payloads(), fail={"players"}), now=NOW)
        by_feed = {r["feed"]: r for r in results}
        assert by_feed["players"]["status"] == "fetch_error"
        assert by_feed["most_selected_players"]["status"] == "stored"

    def test_status_file_written_and_last_stored_survives_unchanged_run(self, tmp_path):
        pay = _payloads()
        capture_all(tmp_path, fetch=_fetcher(pay), now=NOW)
        capture_all(tmp_path, fetch=_fetcher(pay), now=NOW2)
        status = json.loads((tmp_path / "capture_status.json").read_text())
        assert status["last_run_utc"].startswith("2026-07-04T00:00:00")
        feed = status["feeds"]["most_selected_players"]
        assert feed["status"] == "unchanged"
        assert feed["last_stored_utc"].startswith("2026-07-03T23:30:00")

    def test_stored_bytes_roundtrip(self, tmp_path):
        pay = _payloads()
        capture_all(tmp_path, fetch=_fetcher(pay), now=NOW)
        stored = (tmp_path / "rounds" / "20260703T233000Z.json").read_bytes()
        assert stored == pay["rounds"]
