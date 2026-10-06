"""T5: the count build end to end on a fully synthetic data tree (no real feed, parquet or receipt is read)."""
import gzip
import hashlib
import json

import pandas as pd
import pytest
from pathlib import Path

from scripts.audit.c1_r3 import acquire as aq
from scripts.audit.c1_r3 import count_build as CB
from scripts.audit.c1_r3 import count_meta as M
from tests.scripts.c1_r3.feeds import feed, lineup


def world(tmp_path, n=120, *, drop_parquet=(), corrupt=(), wrong_counts=(), extra_feed=False, plain=(), feed_kw=None,
          unavailable=()):
    """n certified 2023 games: receipt-bound gz (or plain) feeds plus a PA parquet that agrees with them."""
    data = tmp_path / "data"
    feeds, acq = data / "raw_c1", data / "hetzner_results" / "c1" / "r3"
    rows = []
    pks = list(range(700001, 700001 + n)) + ([799999] if extra_feed else [])
    for pk in pks:
        body = json.dumps(feed(pk=pk, date="2023-06-01", **((feed_kw or {}).get(pk, {})))).encode()
        stored = body if pk in plain else gzip.compress(body, mtime=0)
        rel = f"2023/{pk}.json" if pk in plain else f"2023/{pk}.json.gz"
        (feeds / "2023").mkdir(parents=True, exist_ok=True)
        (feeds / rel).write_bytes(stored + (b"x" if pk in corrupt else b""))
        url = f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
        aq._append(acq / "receipts" / "2026-10-04.jsonl", {"kind": "intent", "attempt_id": f"a-{pk}", "gamePk": pk,
                                                           "url": url, "started_utc": "2026-10-04T23:00:00+00:00"})
        aq._append(acq / "receipts" / "2026-10-04.jsonl",
                   {"kind": "completion", "attempt_id": f"a-{pk}", "kind_of": "feed", "gamePk": pk, "url": url,
                    "outcome": "stored", "decoded_sha256": hashlib.sha256(body).hexdigest(), "stored_path": rel,
                    "stored_sha256": hashlib.sha256(stored).hexdigest(),
                    "ended_utc": None if pk in (unavailable or ()) else "2026-10-04T23:00:01+00:00"})
        if pk in drop_parquet or pk == 799999:
            continue
        for p in M.extract(feed(pk=pk, date="2023-06-01")).pas:
            rows.append({"game_pk": pk, "batter_id": p.batter, "is_home": p.side == "home", "season": 2023,
                         "date": "2023-06-01"})
        if pk in wrong_counts:
            rows.append({"game_pk": pk, "batter_id": 101, "is_home": False, "season": 2023, "date": "2023-06-01"})
    (data / "processed").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(data / "processed" / "pa_2023.parquet", index=False)
    return data


@pytest.fixture
def patched(monkeypatch):
    def apply(data, seasons=(2023,)):
        monkeypatch.setattr(CB, "DATA", data)
        monkeypatch.setattr(CB, "SEASONS", seasons)
        monkeypatch.setattr(CB, "admission_gate", lambda: "f" * 40)
        monkeypatch.setattr(CB, "dirty_tree", lambda: [])
        pins = {f"pa_{s}.parquet": hashlib.sha256((data / "processed" / f"pa_{s}.parquet").read_bytes()).hexdigest()
                for s in seasons}
        monkeypatch.setattr(CB, "load_admission", lambda: {"input_pins": pins, "review_report": "r.md"})
        monkeypatch.setattr(CB, "accepted_identity", lambda adm: {"review_report": "r.md", "review_report_sha256": "0" * 64})
    return apply


def only_run(data):
    (d,) = [p for p in (data / "hetzner_results" / "c1" / "r3" / "count_build").iterdir() if p.is_dir()]
    return d


def test_end_to_end_outputs(tmp_path, patched):
    data = world(tmp_path, extra_feed=True)
    patched(data)
    assert CB.main([]) == 0
    d = only_run(data)
    res = json.loads((d / "results.json").read_text())
    assert res["census"]["eligible"] == 120 and res["census"]["certified"] == 120
    assert res["census"]["ineligible_feeds"] == 1                       # a feed with no parquet rows
    tab = json.loads((d / "count_table.json").read_text())
    assert tab["cells"]["1|away"]["counts"]["1"] == 120                  # every starter batted once in the fixture
    starts = json.loads((d / "bf_starts.json").read_text())
    assert len(starts["starts"]) == 240 and starts["league_median_bf"] == 9.0
    prov = json.loads((d / "provenance.json").read_text())
    assert len(prov) == 120 and prov["700001"]["feed_timestamp"] == "20230602_010203" and prov["700001"]["certified"]
    assert "799999" not in prov                                          # ineligible: hash-checked, not parsed
    inv = json.loads((d / "inventory.json").read_text())
    assert len(inv) == 121 and {e["pk"] for e in inv} >= {799999}
    for name in ("count_table.json", "bf_starts.json", "census.json", "provenance.json"):
        assert res["outputs"][name] == hashlib.sha256((d / name).read_bytes()).hexdigest()
    man = json.loads((d / "manifest.json").read_text())
    assert man["inventory"]["count"] == 121 and "pa_2023.parquet" in man["pa_parquets"]
    assert man["inventory"]["sha256"] == hashlib.sha256((d / "inventory.json").read_bytes()).hexdigest()
    assert man["contract"]["registration_sha256"] and man["contract"]["recipe"].startswith("not applicable")
    assert man["expected_pa_pins"]["pa_2023.parquet"] == hashlib.sha256((data / "processed" / "pa_2023.parquet").read_bytes()).hexdigest()
    assert man["accepted_review"]["review_report_sha256"] == "0" * 64
    assert (d / "CLAIM.json").exists()


def test_more_than_one_percent_quarantined_stops_and_the_claim_blocks_a_rerun(tmp_path, patched):
    data = world(tmp_path, wrong_counts=(700001, 700002))                # 2 of 120 = 1.7%
    patched(data)
    assert CB.main([]) == 2
    d = only_run(data)
    assert json.loads((d / "STOPPED_quarantine.json").read_text())["quarantined"] == 2 and not (d / "results.json").exists()
    census = json.loads((d / "census.json").read_text())
    assert set(census["quarantined"]) == {"700001", "700002"}
    with pytest.raises(SystemExit, match="claimed"):
        CB.main([])


def test_a_feed_that_does_not_match_its_receipt_stops_before_counting(tmp_path, patched):
    data = world(tmp_path, corrupt=(700005,))
    patched(data)
    with pytest.raises(CB.ProvenanceError, match="receipt"):
        CB.main([])


def test_an_eligible_game_without_a_feed_is_quarantined_not_dropped(tmp_path, patched):
    data = world(tmp_path, n=150)
    rows = pd.read_parquet(data / "processed" / "pa_2023.parquet")
    extra = rows[rows["game_pk"] == 700001].assign(game_pk=888888)
    pd.concat([rows, extra]).to_parquet(data / "processed" / "pa_2023.parquet", index=False)
    patched(data)
    assert CB.main([]) == 0
    census = json.loads((only_run(data) / "census.json").read_text())
    assert census["quarantined"] == {"888888": ["no re-acquired feed for an eligible game"]}


def test_the_build_holds_the_acquisition_lock_while_reading_feeds(tmp_path, patched):
    data = world(tmp_path, n=10)
    patched(data)
    with aq.writer_lock(data / "hetzner_results" / "c1" / "r3"):
        with pytest.raises(aq.Busy):
            CB.main([])


def test_admission_is_required(tmp_path, monkeypatch):
    monkeypatch.setattr(CB, "DATA", world(tmp_path, n=5))
    with pytest.raises(SystemExit, match="input_pins|reviewed_commit|review_report|exposure"):
        CB.main([])


def test_there_is_no_root_override():
    with pytest.raises(SystemExit):
        CB.main(["--feeds", "/tmp/x"])



# ---------- code review r1 F1, F6, F7, F8, F9 ----------
def test_the_claim_precedes_every_outcome_bearing_read(tmp_path, patched, monkeypatch):
    data = world(tmp_path, n=10)
    patched(data)
    events, real_read, real_claim = [], Path.read_bytes, CB.A.write_claim

    def read_bytes(self):
        if self.suffix == ".parquet" or self.name.endswith(".json.gz"):
            events.append(("read", self.name))
        return real_read(self)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(CB.A, "write_claim", lambda d, h: (events.append(("claim",)), real_claim(d, h))[1])
    assert CB.main([]) == 0
    assert events[0] == ("claim",) and all(e[0] == "read" for e in events[1:])


def test_a_failure_after_the_claim_still_blocks_a_rerun(tmp_path, patched, monkeypatch):
    data = world(tmp_path, n=10)
    patched(data)
    monkeypatch.setattr(CB, "check_schema", lambda b, name: (_ for _ in ()).throw(RuntimeError("killed")))
    with pytest.raises(RuntimeError):
        CB.main([])
    monkeypatch.undo()
    patched(data)
    with pytest.raises(SystemExit, match="claimed"):
        CB.main([])


@pytest.mark.parametrize("mutate,match", [
    (lambda df: df.assign(batter_id=df["batter_id"].astype("float64")), "batter_id"),
    (lambda df: df.assign(is_home=df["is_home"].astype(str)), "is_home"),
    (lambda df: df.assign(is_resumed_portion=False), "enriched"),
    (lambda df: pd.concat([df, df.iloc[:1].assign(game_pk=None)]), "game_pk|null"),
    (lambda df: df.assign(season=2022), "season"),
])
def test_a_malformed_parquet_refuses_instead_of_coercing(tmp_path, patched, mutate, match):
    data = world(tmp_path, n=5)
    f = data / "processed" / "pa_2023.parquet"
    mutate(pd.read_parquet(f)).to_parquet(f, index=False)
    patched(data)
    with pytest.raises(CB.ProvenanceError, match=match):
        CB.main([])


def test_a_game_in_two_seasonal_parquets_refuses(tmp_path, patched, monkeypatch):
    data = world(tmp_path, n=5)
    df = pd.read_parquet(data / "processed" / "pa_2023.parquet")
    df.assign(season=2022).to_parquet(data / "processed" / "pa_2022.parquet", index=False)
    patched(data, seasons=(2022, 2023))
    with pytest.raises(CB.ProvenanceError, match="two seasonal"):
        CB.main([])


def receipt(acq, rec):
    aq._append(acq / "receipts" / "2026-10-04.jsonl", rec)


@pytest.mark.parametrize("bad", [
    {"stored_path": "2023/8.json.gz"},                      # the path names another game
    {"stored_path": "2023/../../x/700001.json.gz"},         # traversal
    {"stored_path": "2023/700001.txt"},                     # not a feed path
    {"decoded_sha256": "zz"},                               # malformed digest
])
def test_a_non_canonical_receipt_refuses(tmp_path, patched, bad):
    data = world(tmp_path, n=3)
    receipt(data / "hetzner_results" / "c1" / "r3",
            {"kind": "completion", "attempt_id": "x", "kind_of": "feed", "gamePk": 700001, "outcome": "stored",
             "stored_path": "2023/700001.json.gz", "stored_sha256": "a" * 64, "decoded_sha256": "b" * 64, **bad})
    patched(data)
    with pytest.raises(CB.ProvenanceError):
        CB.main([])


def test_conflicting_receipts_or_an_unresolved_intent_refuse(tmp_path, patched):
    data = world(tmp_path, n=3)
    acq = data / "hetzner_results" / "c1" / "r3"
    url = "https://statsapi.mlb.com/api/v1.1/game/700001/feed/live"
    receipt(acq, {"kind": "intent", "attempt_id": "x", "gamePk": 700001, "url": url})
    receipt(acq, {"kind": "completion", "attempt_id": "x", "kind_of": "feed", "gamePk": 700001, "outcome": "stored",
                  "url": url, "stored_path": "2023/700001.json.gz", "stored_sha256": "a" * 64,
                  "decoded_sha256": "b" * 64, "ended_utc": "2026-10-04T23:00:01+00:00"})
    patched(data)
    with pytest.raises(CB.ProvenanceError, match="conflicting"):
        CB.main([])
    data2 = world(tmp_path / "w2", n=3)
    receipt(data2 / "hetzner_results" / "c1" / "r3", {"kind": "intent", "attempt_id": "open", "gamePk": 700002})
    patched(data2)
    with pytest.raises(CB.ProvenanceError, match="unresolved"):
        CB.main([])


def test_plain_json_feeds_are_supported(tmp_path, patched):
    data = world(tmp_path, n=5, plain=(700002,))
    patched(data)
    assert CB.main([]) == 0
    assert json.loads((only_run(data) / "census.json").read_text())["certified"] == 5


def test_a_malformed_eligible_feed_is_quarantined_not_raised(tmp_path, patched):
    data = world(tmp_path, n=150, feed_kw={700003: {"status": "In Progress"}})
    patched(data)
    assert CB.main([]) == 0
    census = json.loads((only_run(data) / "census.json").read_text())
    assert list(census["quarantined"]) == ["700003"]


def test_a_missing_receipt_named_file_stops_durably_and_keeps_the_claim(tmp_path, patched):
    data = world(tmp_path, n=5)
    (data / "raw_c1" / "2023" / "700004.json.gz").unlink()
    patched(data)
    with pytest.raises(CB.ProvenanceError, match="No such file"):
        CB.main([])
    d = only_run(data)
    stop = json.loads((d / "STOPPED_incomplete.json").read_text())
    assert stop["source"] == "2023/700004.json.gz" and 700005 in stop["unprocessed_feeds"] and (d / "CLAIM.json").exists()


def test_an_ineligible_feed_is_hash_checked_but_never_parsed(tmp_path, patched, monkeypatch):
    data = world(tmp_path, n=5, extra_feed=True)
    patched(data)
    parsed = []
    real = CB.M.extract
    monkeypatch.setattr(CB.M, "extract", lambda feed: (parsed.append(feed["gamePk"]), real(feed))[1])
    assert CB.main([]) == 0 and 799999 not in parsed and len(parsed) == 5



# ---------- code review r2 N5, N6, N8 ----------
def test_a_stored_record_must_be_bound_to_its_own_request(tmp_path, patched):
    """r2 N5: a completion sharing an attempt id with another game's intent, or naming another game's URL."""
    for mutate in ({"gamePk": 700002}, {"url": "https://statsapi.mlb.com/api/v1.1/game/8/feed/live"}):
        data = world(tmp_path / str(len(mutate)) / mutate.get("url", "a")[-9:], n=3)
        recs = (data / "hetzner_results" / "c1" / "r3" / "receipts" / "2026-10-04.jsonl")
        lines = [json.loads(x) for x in recs.read_text().splitlines()]
        for r in lines:
            if r["kind"] == "intent" and r["gamePk"] == 700001:
                r.update(mutate)
        recs.write_text("".join(json.dumps(r) + "\n" for r in lines))
        patched(data)
        with pytest.raises(CB.ProvenanceError, match="not bound|unresolved"):
            CB.main([])


def test_a_missing_or_garbled_retrieval_time_quarantines_the_game(tmp_path, patched):
    data = world(tmp_path, n=150, unavailable=(700002,))
    patched(data)
    assert CB.main([]) == 0
    census = json.loads((only_run(data) / "census.json").read_text())
    assert census["quarantined"] == {"700002": ["availability: the receipt has no valid retrieval time"]}
    assert CB._utc("garbled") is None and CB._utc("2026-10-04T23:00:01") is None      # naive time refused


def test_an_input_off_its_expected_pin_stops_before_parsing(tmp_path, patched, monkeypatch):
    """r2 N6: the expected parquet pin is published before the run; the bytes must match it before any parse."""
    data = world(tmp_path, n=5)
    patched(data)
    monkeypatch.setattr(CB, "load_admission", lambda: {"input_pins": {"pa_2023.parquet": "0" * 64}, "review_report": "r"})
    parsed = []
    monkeypatch.setattr(CB, "check_schema", lambda b, name: parsed.append(name))
    with pytest.raises(CB.ProvenanceError, match="pinned"):
        CB.main([])
    d = only_run(data)
    assert parsed == [] and json.loads((d / "STOPPED_incomplete.json").read_text())["source"] == "pa_2023.parquet"
    assert json.loads((d / "manifest.json").read_text())["expected_pa_pins"] == {"pa_2023.parquet": "0" * 64}


def test_admission_pins_must_name_exactly_the_registered_parquets(monkeypatch):
    monkeypatch.setattr(CB, "load_admission", lambda: {"input_pins": {"pa_2023.parquet": "0" * 64}})
    with pytest.raises(SystemExit, match="input_pins"):
        CB.admission_gate()


def test_a_bad_gzip_with_a_matching_stored_digest_stops_durably(tmp_path, patched):
    """r2 N8: a decode failure is a provenance stop with a durable account, not an unaccounted exception."""
    data = world(tmp_path, n=5)
    f = data / "raw_c1" / "2023" / "700003.json.gz"
    f.write_bytes(b"not gzip at all")
    recs = data / "hetzner_results" / "c1" / "r3" / "receipts" / "2026-10-04.jsonl"
    lines = [json.loads(x) for x in recs.read_text().splitlines()]
    for r in lines:
        if r.get("gamePk") == 700003 and r["kind"] == "completion":
            r["stored_sha256"] = hashlib.sha256(b"not gzip at all").hexdigest()
    recs.write_text("".join(json.dumps(r) + "\n" for r in lines))
    patched(data)
    with pytest.raises(CB.ProvenanceError):
        CB.main([])
    stop = json.loads((only_run(data) / "STOPPED_incomplete.json").read_text())
    assert stop["source"] == "2023/700003.json.gz" and "Gzip" in stop["error"]


def test_a_schema_failure_stops_durably(tmp_path, patched):
    data = world(tmp_path, n=5)
    f = data / "processed" / "pa_2023.parquet"
    pd.read_parquet(f).assign(is_resumed_portion=False).to_parquet(f, index=False)
    patched(data)
    with pytest.raises(CB.ProvenanceError, match="enriched"):
        CB.main([])
    assert json.loads((only_run(data) / "STOPPED_incomplete.json").read_text())["source"] == "pa_2023.parquet"
