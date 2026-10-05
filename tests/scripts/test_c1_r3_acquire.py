"""C1 rank 3 feed re-acquisition (register row C1-r3-acquire): per-attempt receipts, receipt-bound resume, and the
fail-closed 403/429 stop that pauses C1 (code review r1 F2, F3, F8, F9)."""
import gzip
import io
import json
import urllib.error
from datetime import datetime, timezone

import pytest

from scripts.audit.c1 import launch
from scripts.audit.c1_r3 import acquire as aq


def sched(*rows):
    """rows = (listing date, gamePk, detailedState, officialDate or None)."""
    dates = {}
    for d, pk, st, off in rows:
        dates.setdefault(d, []).append({"gamePk": pk, "officialDate": off or d, "status": {"detailedState": st}})
    return {"dates": [{"date": d, "games": g} for d, g in sorted(dates.items())]}


def feed(pk):
    return json.dumps({"gamePk": pk, "gameData": {"game": {"pk": pk}}, "liveData": {}}).encode()


def http_error(code):
    return urllib.error.HTTPError("https://statsapi.mlb.com/x", code, "err", {}, io.BytesIO(b""))


class Clock:
    def __call__(self):
        return datetime(2026, 10, 5, 1, 0, tzinfo=timezone.utc)


# ---------- the feed inventory ----------
def test_inventory_dates_a_suspended_game_by_its_completion_listing_not_official_date():
    s = sched(("2023-06-01", 3, "Suspended: Rain", "2023-06-01"), ("2023-06-02", 3, "Final", "2023-06-01"),
              ("2023-09-28", 716404, "Postponed", None), ("2023-10-02", 716404, "Completed Early: Rain", None),
              ("2023-10-01", 1, "Final", None), ("2023-10-01", 2, "Cancelled", None))
    games = aq.schedule_games(s)
    assert [(g["gamePk"], g["date"]) for g in games] == [(3, "2023-06-02"), (1, "2023-10-01"), (716404, "2023-10-02")]


@pytest.mark.parametrize("bad", [{"pk": 0}, {"pk": -4}, {"pk": True}, {"pk": 7.0}])
def test_inventory_refuses_invalid_game_ids(bad):
    with pytest.raises(ValueError):
        aq.schedule_games({"dates": [{"date": "2023-06-01", "games": [
            {"gamePk": bad["pk"], "status": {"detailedState": "Final"}}]}]})


def test_inventory_refuses_an_unsupported_status():
    with pytest.raises(aq.UnsupportedStatus):
        aq.schedule_games(sched(("2023-06-01", 5, "In Progress", None)))


# ---------- acquisition ----------
def run(tmp_path, responses, games=(101, 102, 103)):
    calls, sleeps = [], []

    def fetch(url):
        calls.append(url)
        r = responses.get(url.rsplit("/game/", 1)[1].split("/")[0])
        if isinstance(r, list):
            r = r.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    rc = aq.acquire([{"gamePk": pk, "season": 2021, "statuses": ["Final"]} for pk in games],
                    out_dir=tmp_path / "c1" / "r3", feeds_dir=tmp_path / "raw", pause_root=tmp_path / "c1",
                    fetch=fetch, sleep=sleeps.append, jitter=lambda: 1.5, now=Clock())
    return rc, calls, sleeps


def recs(tmp_path):
    return aq.read_receipts(tmp_path / "c1" / "r3")


def test_stores_feeds_with_per_attempt_receipts_and_a_bound_store_record(tmp_path):
    rc, calls, sleeps = run(tmp_path, {"101": feed(101), "102": feed(102), "103": feed(103)})
    assert rc == 0 and len(calls) == 3 and sleeps == [1.5, 1.5]
    assert json.loads(gzip.decompress((tmp_path / "raw" / "2021" / "101.json.gz").read_bytes()))["gamePk"] == 101
    rs = recs(tmp_path)
    assert sum(r["kind"] == "intent" for r in rs) == 3 and aq.unresolved_intents(rs) == []
    stored = [r for r in rs if r.get("outcome") == "stored"]
    assert [r["gamePk"] for r in stored] == [101, 102, 103]
    assert stored[0]["decoded_sha256"] == aq.sha256(feed(101)) and stored[0]["stored_path"] == "2021/101.json.gz"


def test_every_retry_has_its_own_receipt(tmp_path):
    rc, calls, _ = run(tmp_path, {"101": [http_error(503), http_error(503), feed(101)], "102": feed(102)},
                       games=(101, 102))
    assert rc == 0 and len(calls) == 4
    rs = recs(tmp_path)
    intents = [r for r in rs if r["kind"] == "intent"]
    assert len(intents) == 4 and aq.unresolved_intents(rs) == []
    assert [r["outcome"] for r in rs if r["kind"] == "completion" and r.get("gamePk") == 101][:3] == \
        ["http_error", "http_error", "response"]


def test_resume_skips_only_receipt_bound_files_and_quarantines_orphans(tmp_path):
    run(tmp_path, {"101": feed(101), "102": feed(102), "103": feed(103)})
    rc, calls, _ = run(tmp_path, {})
    assert rc == 0 and calls == []
    (tmp_path / "raw" / "2021" / "102.json.gz").write_bytes(gzip.compress(feed(102) + b" "))   # changed bytes
    rc, calls, _ = run(tmp_path, {"102": feed(102)})
    assert rc == 0 and len(calls) == 1
    assert [r for r in recs(tmp_path) if r["kind"] == "reconciled_orphan"][0]["gamePk"] == 102
    assert list((tmp_path / "c1" / "r3" / "quarantine").iterdir())


@pytest.mark.parametrize("code", [403, 429])
def test_rate_limit_stops_everything_and_writes_the_shared_stop(tmp_path, code):
    rc, calls, _ = run(tmp_path, {"101": feed(101), "102": http_error(code), "103": feed(103)})
    assert rc == 3 and len(calls) == 2
    for p in (tmp_path / "c1" / aq.STOP_NAME, tmp_path / "c1" / "r3" / aq.STOP_NAME):
        assert json.loads(p.read_text())["http_status"] == code
    rc2, calls2, _ = run(tmp_path, {"103": feed(103)})
    assert rc2 == 3 and calls2 == []


def test_a_rate_limit_whose_stop_cannot_be_written_leaves_an_unresolved_intent(tmp_path, monkeypatch):
    real = aq._write_durable

    def broken(path, data):
        if path.name == aq.STOP_NAME:
            raise OSError("disk full")
        return real(path, data)
    monkeypatch.setattr(aq, "_write_durable", broken)
    with pytest.raises(OSError):
        run(tmp_path, {"101": http_error(429)}, games=(101, 102))
    assert aq.unresolved_intents(recs(tmp_path))                  # fail-closed witness for the launcher
    monkeypatch.setattr(aq, "_write_durable", real)
    rc, calls, _ = run(tmp_path, {"102": feed(102)})
    assert rc == 3 and calls == []


def test_a_rate_limit_whose_completion_cannot_be_written_still_leaves_the_stop(tmp_path, monkeypatch):
    real = aq._append

    def flaky(path, rec):
        if rec.get("outcome") == "rate_limited":
            raise OSError("disk full")
        return real(path, rec)
    monkeypatch.setattr(aq, "_append", flaky)
    with pytest.raises(OSError):
        run(tmp_path, {"101": http_error(429)}, games=(101, 102))
    assert (tmp_path / "c1" / aq.STOP_NAME).exists()


@pytest.mark.parametrize("body", [json.dumps({"gamePk": 101.0}).encode(), json.dumps({"gamePk": True}).encode(),
                                  json.dumps({"gamePk": 999}).encode(), b"not json"])
def test_a_feed_without_the_exact_requested_game_id_is_refused(tmp_path, body):
    rc, _, _ = run(tmp_path, {"101": body}, games=(101,))
    assert rc == 1 and not (tmp_path / "raw" / "2021" / "101.json.gz").exists()


def test_a_second_concurrent_run_is_refused(tmp_path):
    (tmp_path / "r3").mkdir()
    with aq.writer_lock(tmp_path / "r3"):
        with pytest.raises(aq.Busy):
            with aq.writer_lock(tmp_path / "r3"):
                pass


def test_main_refuses_an_output_outside_the_c1_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(aq, "C1_ROOT", tmp_path / "c1")
    assert aq.main(["--seasons", "2021", "--out", str(tmp_path / "elsewhere")]) == 2


# ---------- the cycle-wide pause ----------
def test_launcher_refuses_any_c1_job_while_a_stop_or_an_unresolved_intent_exists(tmp_path):
    base = dict(name="r4b-run", cpu_hours=1.0, max_hours=1.0, rows=[], acked=False, active_units=[], sched_lines=[],
                now=datetime(2026, 10, 5, tzinfo=timezone.utc), cwd="/home/bts/projects/bts-c1", command=["true"])
    assert launch.plan_launch(**base)["ok"] is True
    p = launch.plan_launch(**base, rate_limit_stops=["hetzner_results/c1/STOP_403_429.json"])
    assert p["ok"] is False and any("403/429" in r for r in p["reasons"])
    root = tmp_path / "data"
    r3 = root / "hetzner_results" / "c1" / "r3" / "receipts"
    r3.mkdir(parents=True)
    (r3 / "2026-10-05.jsonl").write_text(json.dumps({"kind": "intent", "attempt_id": "a1"}) + "\n")
    assert launch.rate_limit_stops(root) == ["unresolved request receipts: hetzner_results/c1/r3 (1)"]


def test_a_receipt_write_failure_during_an_ordinary_request_aborts_the_run(tmp_path, monkeypatch):
    """Fail closed: a receipt that cannot be written is never counted as an ordinary request failure."""
    real = aq._append

    def flaky(path, rec):
        if rec.get("kind") == "completion" and rec.get("outcome") == "response":
            raise OSError("disk full")
        return real(path, rec)
    monkeypatch.setattr(aq, "_append", flaky)
    calls = []
    with pytest.raises(OSError):
        aq.acquire([{"gamePk": pk, "season": 2021} for pk in (101, 102)], out_dir=tmp_path / "c1" / "r3",
                   feeds_dir=tmp_path / "raw", pause_root=tmp_path / "c1",
                   fetch=lambda url: (calls.append(url), feed(101))[1], sleep=lambda s: None, jitter=lambda: 0.0,
                   now=Clock())
    assert len(calls) == 1                                      # no second request after the receipt failure
