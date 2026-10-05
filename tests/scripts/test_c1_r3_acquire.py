"""C1 rank 3 feed re-acquisition (register row C1-r3-acquire): polite pacing, receipts, and the 403/429 cycle stop."""
import gzip
import io
import json
import urllib.error
from datetime import datetime, timezone

import pytest

from scripts.audit.c1 import launch
from scripts.audit.c1_r3 import acquire as aq


def sched(*games):
    """A schedule response: games = (date, gamePk, detailedState)."""
    dates = {}
    for d, pk, st in games:
        dates.setdefault(d, []).append({"gamePk": pk, "officialDate": d, "status": {"detailedState": st}})
    return {"dates": [{"date": d, "games": g} for d, g in sorted(dates.items())]}


def feed(pk):
    return json.dumps({"gamePk": pk, "gameData": {"game": {"pk": pk}}, "liveData": {}}).encode()


def http_error(code):
    return urllib.error.HTTPError("https://statsapi.mlb.com/x", code, "err", {}, io.BytesIO(b""))


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 5, 1, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t


# ---------- which games ----------
def test_schedule_games_keeps_every_played_regular_season_game_once():
    s = sched(("2023-09-28", 716404, "Postponed"), ("2023-10-01", 1, "Final"),
              ("2023-10-02", 716404, "Completed Early: Rain"), ("2023-10-01", 2, "Cancelled"),
              ("2023-06-01", 3, "Suspended: Rain"), ("2023-06-02", 3, "Final"))
    games = aq.schedule_games(s)
    assert [g["gamePk"] for g in games] == [3, 1, 716404]          # completed-early kept; postponed/cancelled-only dropped
    assert {g["gamePk"]: g["date"] for g in games}[3] == "2023-06-02"  # the date it was completed
    assert {g["gamePk"]: g["statuses"] for g in games}[716404] == ["Completed Early: Rain", "Postponed"]


# ---------- acquisition ----------
def run(tmp_path, responses, games=(101, 102, 103)):
    calls, sleeps = [], []

    def fetch(url):
        calls.append(url)
        r = responses.get(url.rsplit("/game/", 1)[1].split("/")[0])
        if isinstance(r, Exception):
            raise r
        return r

    rc = aq.acquire([{"gamePk": pk, "date": "2021-04-01", "season": 2021, "statuses": ["Final"]} for pk in games],
                    out_dir=tmp_path / "r3", feeds_dir=tmp_path / "raw", fetch=fetch, sleep=sleeps.append, jitter=lambda: 1.5, now=Clock())
    return rc, calls, sleeps


def receipts(tmp_path):
    return [json.loads(l) for f in sorted((tmp_path / "r3" / "receipts").glob("*.jsonl")) for l in f.read_text().splitlines()]


def test_stores_gzipped_feeds_with_intent_and_completion_receipts(tmp_path):
    rc, calls, sleeps = run(tmp_path, {"101": feed(101), "102": feed(102), "103": feed(103)})
    assert rc == 0 and len(calls) == 3 and sleeps == [1.5, 1.5]     # paced between real requests only
    stored = tmp_path / "raw" / "2021" / "101.json.gz"
    assert json.loads(gzip.decompress(stored.read_bytes()))["gamePk"] == 101
    rs = receipts(tmp_path)
    assert [r["kind"] for r in rs] == ["intent", "completion"] * 3
    done = [r for r in rs if r["kind"] == "completion"]
    assert all(r["outcome"] == "stored" and r["attempt_id"] for r in done)
    assert done[0]["decoded_sha256"] == aq.sha256(feed(101)) and done[0]["stored_path"] == "2021/101.json.gz"


def test_resume_skips_verified_files_without_a_request(tmp_path):
    run(tmp_path, {"101": feed(101), "102": feed(102), "103": feed(103)})
    rc, calls, sleeps = run(tmp_path, {})
    assert rc == 0 and calls == [] and sleeps == []


@pytest.mark.parametrize("code", [403, 429])
def test_rate_limit_stops_everything_and_writes_the_stop_marker(tmp_path, code):
    rc, calls, _ = run(tmp_path, {"101": feed(101), "102": http_error(code), "103": feed(103)})
    assert rc == 3 and len(calls) == 2                               # no retry, no further request
    stop = json.loads((tmp_path / "r3" / aq.STOP_NAME).read_text())
    assert stop["http_status"] == code and stop["gamePk"] == 102
    assert receipts(tmp_path)[-1]["outcome"] == "rate_limited"
    rc2, calls2, _ = run(tmp_path, {"103": feed(103)})               # a later run refuses before any request
    assert rc2 == 3 and calls2 == []


def test_server_errors_retry_then_record_failure_and_continue(tmp_path):
    rc, calls, sleeps = run(tmp_path, {"101": http_error(503), "102": feed(102)}, games=(101, 102))
    assert rc == 1                                                    # completed with failures
    assert sum("101" in c for c in calls) == 3 and sum("102" in c for c in calls) == 1
    done = [r for r in receipts(tmp_path) if r["kind"] == "completion"]
    assert [r["outcome"] for r in done] == ["failed", "stored"]


def test_a_feed_for_the_wrong_game_is_refused(tmp_path):
    rc, _, _ = run(tmp_path, {"101": feed(999)}, games=(101,))
    assert rc == 1
    assert not (tmp_path / "raw" / "2021" / "101.json.gz").exists()
    assert receipts(tmp_path)[-1]["outcome"] == "invalid"


def test_a_second_concurrent_run_is_refused(tmp_path):
    (tmp_path / "r3").mkdir()
    with aq.writer_lock(tmp_path / "r3"):
        with pytest.raises(aq.Busy):
            with aq.writer_lock(tmp_path / "r3"):
                pass


# ---------- the cycle-wide pause ----------
def test_launcher_refuses_any_c1_job_while_a_rate_limit_stop_exists():
    base = dict(name="r4b-run", cpu_hours=1.0, max_hours=1.0, rows=[], acked=False, active_units=[], sched_lines=[],
                now=datetime(2026, 10, 5, tzinfo=timezone.utc), cwd="/home/bts/projects/bts-c1", command=["true"])
    assert launch.plan_launch(**base)["ok"] is True
    p = launch.plan_launch(**base, rate_limit_stops=["c1/r3/STOP_403_429.json"])
    assert p["ok"] is False and any("403/429" in r for r in p["reasons"])


def test_a_rate_limit_writes_the_stop_marker_even_if_the_receipt_write_fails(tmp_path, monkeypatch):
    real = aq._append

    def flaky(path, rec):
        if rec.get("outcome") == "rate_limited":
            raise OSError("disk full")
        return real(path, rec)
    monkeypatch.setattr(aq, "_append", flaky)
    rc, calls, _ = run(tmp_path, {"101": http_error(429)}, games=(101, 102))
    assert rc == 3 and len(calls) == 1
    assert json.loads((tmp_path / "r3" / aq.STOP_NAME).read_text())["http_status"] == 429
