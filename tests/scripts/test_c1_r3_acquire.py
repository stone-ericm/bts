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


def test_verify_binds_every_stored_file_to_its_receipt_and_flags_the_rest(tmp_path):
    run(tmp_path, {"101": feed(101), "102": feed(102), "103": feed(103)})
    out, feeds = tmp_path / "c1" / "r3", tmp_path / "raw"
    v = aq.verify(out, feeds)
    assert {k: v[k] for k in ("receipted", "bound", "mismatched", "missing", "unreceipted")} == \
        {"receipted": 3, "bound": 3, "mismatched": [], "missing": [], "unreceipted": []}
    assert v["responses"] == {"promoted": 3, "retained": 0, "missing": [], "legacy_unretained": 0}
    (feeds / "2021" / "102.json.gz").write_bytes(gzip.compress(feed(102) + b" "))
    (feeds / "2021" / "103.json.gz").unlink()
    (feeds / "2021" / "999.json.gz").write_bytes(gzip.compress(feed(999)))
    v = aq.verify(out, feeds)
    assert v["mismatched"] == ["2021/102.json.gz"] and v["missing"] == ["2021/103.json.gz"]
    assert v["unreceipted"] == ["2021/999.json.gz"] and v["bound"] == 1



# ---------- code review r2 N7: response durability and schedule resume ----------
def test_a_response_is_durably_retained_before_its_completion_receipt(tmp_path, monkeypatch):
    real, seen = aq._append, []

    def spy(path, rec):
        if rec.get("outcome") == "response":
            f = tmp_path / "c1" / "r3" / rec["response_path"]
            seen.append(f.exists() and aq.sha256(f.read_bytes()) == rec["response_sha256"]
                        and gzip.decompress(f.read_bytes()) == feed(rec["gamePk"]))
        return real(path, rec)
    monkeypatch.setattr(aq, "_append", spy)
    rc, _, _ = run(tmp_path, {"101": feed(101), "102": feed(102)}, games=(101, 102))
    assert rc == 0 and seen == [True, True]
    stored = [r for r in recs(tmp_path) if r.get("outcome") == "stored"]
    resp = {r["attempt_id"]: r for r in recs(tmp_path) if r.get("outcome") == "response"}
    for s in stored:                                     # the stored artifact is bound to the attempt that fetched it
        assert s["from_attempt_id"] in resp and s["stored_sha256"] == resp[s["from_attempt_id"]]["response_sha256"]
    assert not list((tmp_path / "c1" / "r3" / "responses").glob("*.json.gz"))   # promoted, then released


def test_a_failure_retaining_the_response_leaves_the_attempt_unresolved(tmp_path, monkeypatch):
    real = aq._write_durable

    def broken(path, data):
        if path.parent.name == "responses":
            raise OSError("disk full")
        return real(path, data)
    monkeypatch.setattr(aq, "_write_durable", broken)
    with pytest.raises(OSError):
        run(tmp_path, {"101": feed(101)}, games=(101, 102))
    assert aq.unresolved_intents(recs(tmp_path))                   # no completion without durable bytes
    assert not [r for r in recs(tmp_path) if r.get("outcome") == "response"]


def test_a_crash_between_the_response_receipt_and_storage_keeps_the_bytes_behind_the_receipt(tmp_path, monkeypatch):
    real = aq._write_durable

    def broken(path, data):
        if path.suffixes == [".json", ".gz"] and path.parent.name == "2021":
            raise OSError("crash before the feed is stored")
        return real(path, data)
    monkeypatch.setattr(aq, "_write_durable", broken)
    with pytest.raises(OSError):
        run(tmp_path, {"101": feed(101)}, games=(101,))
    monkeypatch.setattr(aq, "_write_durable", real)
    v = aq.verify(tmp_path / "c1" / "r3", tmp_path / "raw")
    assert v["responses"]["retained"] == 1 and v["responses"]["missing"] == [] and v["receipted"] == 0
    rc, calls, _ = run(tmp_path, {"101": feed(101)}, games=(101,))   # resume fetches again; the old bytes stay
    assert rc == 0 and len(calls) == 1
    assert aq.verify(tmp_path / "c1" / "r3", tmp_path / "raw")["responses"]["retained"] == 1


def test_verify_flags_a_response_receipt_whose_bytes_are_gone(tmp_path, monkeypatch):
    real = aq._write_durable
    monkeypatch.setattr(aq, "_write_durable", lambda p, d: (_ for _ in ()).throw(OSError("x"))
                        if p.parent.name == "2021" else real(p, d))
    with pytest.raises(OSError):
        run(tmp_path, {"101": feed(101)}, games=(101,))
    (f,) = (tmp_path / "c1" / "r3" / "responses").glob("*.json.gz")
    f.unlink()
    assert len(aq.verify(tmp_path / "c1" / "r3", tmp_path / "raw")["responses"]["missing"]) == 1


def main_env(tmp_path, monkeypatch, schedules: dict, feeds: dict):
    """aq.main with the network replaced: schedule and feed bodies keyed by season / gamePk."""
    calls = []

    def fetch(url):
        calls.append(url)
        if "/schedule?" in url:
            return json.dumps(schedules[int(url.split("season=")[1].split("&")[0])]).encode()
        return feeds[url.rsplit("/game/", 1)[1].split("/")[0]]
    monkeypatch.setattr(aq, "C1_ROOT", tmp_path / "c1")
    monkeypatch.setattr(aq, "_fetch", fetch)
    monkeypatch.setattr(aq.time, "sleep", lambda s: None)
    argv = ["--seasons", "2021", "--out", str(tmp_path / "c1" / "r3"), "--feeds", str(tmp_path / "raw"),
            "--min-gap", "0", "--max-gap", "0"]
    return calls, argv


def test_a_fetched_schedule_gets_a_stored_receipt_and_is_reused_only_while_bound(tmp_path, monkeypatch):
    s = sched(("2021-04-01", 101, "Final", None))
    calls, argv = main_env(tmp_path, monkeypatch, {2021: s}, {"101": feed(101)})
    assert aq.main(argv) == 0 and sum("/schedule?" in c for c in calls) == 1
    st = [r for r in recs(tmp_path) if r.get("outcome") == "stored" and r.get("kind_of") == "schedule"]
    sp = tmp_path / "c1" / "r3" / "schedules" / "sched_2021.json"
    assert len(st) == 1 and st[0]["season"] == 2021 and st[0]["stored_sha256"] == aq.sha256(sp.read_bytes())
    calls.clear()
    assert aq.main(argv) == 0 and calls == []                      # bound schedule reused, feeds already stored
    sp.write_bytes(sp.read_bytes() + b" ")                          # changed bytes are never trusted
    assert aq.main(argv) == 2 and calls == []


def test_an_orphan_schedule_is_refused(tmp_path, monkeypatch):
    calls, argv = main_env(tmp_path, monkeypatch, {}, {})
    sp = tmp_path / "c1" / "r3" / "schedules" / "sched_2021.json"
    sp.parent.mkdir(parents=True)
    sp.write_text(json.dumps(sched(("2021-04-01", 101, "Final", None))))
    assert aq.main(argv) == 2 and calls == []


def test_a_response_receipt_without_retained_bytes_does_not_bind_a_schedule(tmp_path, monkeypatch):
    """A pre-N7 response receipt (no response_path) or a receipt for another season never binds existing bytes."""
    body = json.dumps(sched(("2021-04-01", 101, "Final", None))).encode()
    sp = tmp_path / "c1" / "r3" / "schedules" / "sched_2021.json"
    sp.parent.mkdir(parents=True)
    sp.write_bytes(body)
    for kind, extra in (("intent", {}), ("completion", {"outcome": "response", "decoded_sha256": aq.sha256(body)})):
        aq._append(tmp_path / "c1" / "r3" / "receipts" / "2026-10-04.jsonl",
                   {"kind": kind, "attempt_id": "legacy1", "kind_of": "schedule", "season": 2021, **extra})
    assert aq.schedule_binding(aq.read_receipts(tmp_path / "c1" / "r3"), 2021, aq.sha256(body)) is None
    calls, argv = main_env(tmp_path, monkeypatch, {}, {"101": feed(101)})
    assert aq.main(argv) == 2 and calls == []
    v = aq.verify(tmp_path / "c1" / "r3", tmp_path / "raw")
    assert v["schedules"] == {"sched_2021.json": None} and v["responses"]["legacy_unretained"] == 1


def test_a_crash_after_the_schedule_response_leaves_it_bound_by_the_retained_response(tmp_path, monkeypatch):
    s = sched(("2021-04-01", 101, "Final", None))
    calls, argv = main_env(tmp_path, monkeypatch, {2021: s}, {"101": feed(101)})
    real = aq._append

    def crash(path, rec):
        if rec.get("outcome") == "stored" and rec.get("kind_of") == "schedule":
            raise OSError("crash before the stored receipt")
        return real(path, rec)
    monkeypatch.setattr(aq, "_append", crash)
    with pytest.raises(OSError):
        aq.main(argv)
    monkeypatch.setattr(aq, "_append", real)
    sp = tmp_path / "c1" / "r3" / "schedules" / "sched_2021.json"
    assert aq.schedule_binding(aq.read_receipts(tmp_path / "c1" / "r3"), 2021, aq.sha256(sp.read_bytes())) == "response"


def test_verify_lists_an_unbound_schedule(tmp_path):
    sp = tmp_path / "c1" / "r3" / "schedules" / "sched_2022.json"
    sp.parent.mkdir(parents=True)
    sp.write_text("{}")
    (tmp_path / "raw").mkdir()
    assert aq.verify(tmp_path / "c1" / "r3", tmp_path / "raw")["schedules"] == {"sched_2022.json": None}


# ---------- Eric 2026-10-05 (row C1-4b-deferral; code review r3 B6): two downloader problems ----------
def test_verify_does_not_report_success_while_a_request_is_unfinished(tmp_path, monkeypatch):
    out = tmp_path / "c1" / "r3"
    aq._append(out / "receipts" / "2026-10-05.jsonl", {"kind": "intent", "attempt_id": "a1", "kind_of": "feed",
                                                       "gamePk": 101})
    (tmp_path / "raw").mkdir()
    v = aq.verify(out, tmp_path / "raw")
    assert v["unresolved"] == ["a1"]
    monkeypatch.setattr(aq, "C1_ROOT", tmp_path / "c1")
    assert aq.main(["--verify", "--out", str(out), "--feeds", str(tmp_path / "raw")]) == 1


def test_verify_fails_when_a_receipted_schedule_is_missing(tmp_path, monkeypatch):
    s = sched(("2021-04-01", 101, "Final", None))
    calls, argv = main_env(tmp_path, monkeypatch, {2021: s}, {"101": feed(101)})
    assert aq.main(argv) == 0
    (tmp_path / "c1" / "r3" / "schedules" / "sched_2021.json").unlink()
    v = aq.verify(tmp_path / "c1" / "r3", tmp_path / "raw")
    assert v["schedules_missing"] == ["schedules/sched_2021.json"]
    assert aq.main(["--verify", "--out", str(tmp_path / "c1" / "r3"), "--feeds", str(tmp_path / "raw")]) == 1


def test_a_cached_schedule_is_parsed_from_the_bytes_whose_binding_was_checked(tmp_path, monkeypatch):
    """r3 B6: the binding check read the file, then parsing and logging re-read it, so bytes swapped in between
    (game 202 instead of the receipt-bound game 101) reached the acquisition."""
    s = sched(("2021-04-01", 101, "Final", None))
    calls, argv = main_env(tmp_path, monkeypatch, {2021: s}, {"101": feed(101)})
    assert aq.main(argv) == 0
    sp = tmp_path / "c1" / "r3" / "schedules" / "sched_2021.json"
    swapped = json.dumps(sched(("2021-04-01", 202, "Final", None))).encode()
    real_read, reads = aq.Path.read_bytes, []

    def read_bytes(self):
        if self == sp:
            reads.append(1)
            return real_read(self) if len(reads) == 1 else swapped      # the file changes after the first read
        return real_read(self)
    monkeypatch.setattr(aq.Path, "read_bytes", read_bytes)
    seen = []
    monkeypatch.setattr(aq, "acquire", lambda games, **kw: (seen.extend(g["gamePk"] for g in games), 0)[1])
    assert aq.main(argv) == 0
    assert seen == [101] and len(reads) == 1


def test_verify_holds_the_writer_lock_and_refuses_while_an_acquisition_owns_it(tmp_path, monkeypatch):
    """C1 infrastructure review r1 B6: an intent appended after verify's receipt snapshot could go unseen."""
    out = tmp_path / "c1" / "r3"
    (tmp_path / "raw").mkdir()
    out.mkdir(parents=True)
    with aq.writer_lock(out):
        with pytest.raises(aq.Busy):
            aq.verify(out, tmp_path / "raw")
        monkeypatch.setattr(aq, "C1_ROOT", tmp_path / "c1")
        assert aq.main(["--verify", "--out", str(out), "--feeds", str(tmp_path / "raw")]) == 3


def test_an_acquisition_cannot_start_while_verify_is_reading(tmp_path, monkeypatch):
    out = tmp_path / "c1" / "r3"
    (tmp_path / "raw").mkdir()
    out.mkdir(parents=True)
    real, attempts = aq.read_receipts, []

    def read_then_try_to_acquire(o):
        recs = real(o)
        try:
            run(tmp_path, {"101": feed(101)}, games=(101,))
            attempts.append("acquired")
        except aq.Busy:
            attempts.append("busy")
        return recs
    monkeypatch.setattr(aq, "read_receipts", read_then_try_to_acquire)
    aq.verify(out, tmp_path / "raw")
    assert attempts == ["busy"]
