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
        monkeypatch.setattr(CB, "dirty_tree", lambda: [])
        pins = {f"pa_{s}.parquet": hashlib.sha256((data / "processed" / f"pa_{s}.parquet").read_bytes()).hexdigest()
                for s in seasons} if pins_override is None else pins_override
        monkeypatch.setattr(CB, "admission_gate", lambda: ("f" * 40, {"input_pins": pins, "review_report": "r.md"},
                                                           {"review_report": "r.md", "review_report_sha256": "0" * 64}))
    def apply_with(data, seasons=(2023,), pins=None):
        nonlocal pins_override
        pins_override = pins
        apply(data, seasons)
    pins_override = None
    apply.with_pins = apply_with
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
    """A closed admission record refuses. The record is pinned here: since 2026-10-06 the repository's own
    admission.json is populated (rank 3 was admitted), so the test must not depend on it (C2 acceptance a1, item 6)."""
    monkeypatch.setattr(CB, "DATA", world(tmp_path, n=5))
    closed = {"reviewed_commit": None, "review_report": None, "exposure_commit": None}
    monkeypatch.setattr(CB, "load_admission", lambda: (closed, "0" * 64))
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
        with pytest.raises(CB.ProvenanceError, match="not bound|unresolved|different request"):   # C2R1-1: now at the join
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
    patched.with_pins(data, pins={"pa_2023.parquet": "0" * 64})
    parsed = []
    monkeypatch.setattr(CB, "check_schema", lambda b, name: parsed.append(name))
    with pytest.raises(CB.ProvenanceError, match="pinned"):
        CB.main([])
    d = only_run(data)
    assert parsed == [] and json.loads((d / "STOPPED_incomplete.json").read_text())["source"] == "pa_2023.parquet"
    assert json.loads((d / "manifest.json").read_text())["expected_pa_pins"] == {"pa_2023.parquet": "0" * 64}


def test_admission_pins_must_name_exactly_the_registered_parquets(monkeypatch):
    monkeypatch.setattr(CB, "load_admission", lambda: ({"input_pins": {"pa_2023.parquet": "0" * 64}}, "0" * 64))
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



# ---------- code review r3 R3-2, R3-5 ----------
def test_the_run_consumes_the_admitted_record_not_a_later_one(tmp_path, monkeypatch):
    """r3 R3-2: the real gate admits A (its pin is 64 zeroes; the modelled X-34 binds only A's pins digest). If the
    admission file is then replaced by B (whose pin matches the parquet), the run must still consume A: it stops on
    A's pin, and the manifest records A, never B."""
    data = world(tmp_path, n=5)
    good = hashlib.sha256((data / "processed" / "pa_2023.parquet").read_bytes()).hexdigest()
    A_rec = {"reviewed_commit": "a" * 40, "exposure_commit": "b" * 40, "review_report": "docs/A.md",
             "input_pins": {"pa_2023.parquet": "0" * 64}}
    B_rec = {**A_rec, "review_report": "docs/B-block.md", "input_pins": {"pa_2023.parquet": good}}
    reads = []

    def load():
        reads.append(1)
        return (A_rec if len(reads) == 1 else B_rec), f"{len(reads)}" * 64

    def check(repo, adm, *, inputs_digest, **kw):                      # models X-34 binding only A's pins digest
        return "f" * 40, ([] if inputs_digest == CB.pins_digest(A_rec["input_pins"]) else ["not A"])
    monkeypatch.setattr(CB, "DATA", data)
    monkeypatch.setattr(CB, "SEASONS", (2023,))
    monkeypatch.setattr(CB, "dirty_tree", lambda: [])
    monkeypatch.setattr(CB, "load_admission", load)
    monkeypatch.setattr(CB.A, "admission_check", check)
    monkeypatch.setattr(CB, "accepted_identity", lambda adm: {"review_report": adm["review_report"]})
    with pytest.raises(CB.ProvenanceError, match="pinned"):
        CB.main([])
    man = json.loads((only_run(data) / "manifest.json").read_text())
    assert reads == [1]
    assert man["expected_pa_pins"] == A_rec["input_pins"] and man["admission"] == A_rec
    assert man["accepted_review"] == {"review_report": "docs/A.md", "admission_sha256": "1" * 64}
    assert not (only_run(data) / "results.json").exists()


def modern(acq, pk, body_sha, *, response_sha=None, started="2026-10-04T23:00:00+00:00",
           ended="2026-10-04T23:00:01+00:00", stored_path=None, stored_sha="a" * 64):
    """The newer layout: intent -> response completion -> stored record linked by from_attempt_id."""
    url = f"https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"
    base = {"request_id": "q", "attempt_id": f"m-{pk}", "attempt": 1, "kind_of": "feed", "url": url, "gamePk": pk,
            "season": None}
    receipt(acq, {**base, "kind": "intent", "started_utc": started})
    receipt(acq, {**base, "kind": "completion", "ended_utc": ended, "outcome": "response", "http_status": 200,
                  "decoded_sha256": response_sha or body_sha, "response_path": f"responses/m-{pk}.json.gz"})
    receipt(acq, {"kind": "completion", "attempt_id": f"store-{pk}", "kind_of": "feed", "gamePk": pk,
                  "outcome": "stored", "decoded_sha256": body_sha, "stored_path": stored_path or f"2023/{pk}.json.gz",
                  "stored_sha256": stored_sha, "from_attempt_id": f"m-{pk}", "at_utc": ended})


def modern_world(tmp_path, n=150, **kw):
    """world() with game 700001 re-receipted in the newer layout (its legacy receipts removed)."""
    data = world(tmp_path, n=n)
    acq = data / "hetzner_results" / "c1" / "r3"
    recs = acq / "receipts" / "2026-10-04.jsonl"
    keep = [x for x in recs.read_text().splitlines() if json.loads(x).get("gamePk") != 700001]
    recs.write_text("".join(x + "\n" for x in keep))
    stored = (data / "raw_c1" / "2023" / "700001.json.gz").read_bytes()
    body_sha = hashlib.sha256(gzip.decompress(stored)).hexdigest()
    modern(acq, 700001, body_sha, stored_sha=hashlib.sha256(stored).hexdigest(), **kw)
    return data, body_sha


def test_a_complete_modern_receipt_chain_certifies(tmp_path, patched):
    data, body_sha = modern_world(tmp_path)
    patched(data)
    assert CB.main([]) == 0
    prov = json.loads((only_run(data) / "provenance.json").read_text())
    assert prov["700001"]["certified"] and prov["700001"]["attempt_id"] == "m-700001"


def test_a_linked_response_attesting_other_bytes_refuses(tmp_path, patched):
    data, _ = modern_world(tmp_path, response_sha="0" * 64)
    patched(data)
    with pytest.raises(CB.ProvenanceError, match="different decoded bytes"):
        CB.main([])


def test_a_retrieval_time_before_its_request_quarantines(tmp_path, patched):
    data, _ = modern_world(tmp_path, started="2026-10-04T23:00:05+00:00", ended="2026-10-04T23:00:01+00:00")
    patched(data)
    assert CB.main([]) == 0
    census = json.loads((only_run(data) / "census.json").read_text())
    assert census["quarantined"] == {"700001": ["availability: the retrieval time precedes its request"]}


def test_conflicting_duplicate_intents_or_unknown_receipts_refuse(tmp_path, patched):
    """r3 R3-5: two intents with one attempt id (a wrong URL first, the right one last) used to resolve silently."""
    data = world(tmp_path, n=3)
    recs = data / "hetzner_results" / "c1" / "r3" / "receipts" / "2026-10-04.jsonl"
    lines = recs.read_text().splitlines()
    wrong = {**json.loads(lines[0]), "url": "https://statsapi.mlb.com/api/v1.1/game/8/feed/live"}
    recs.write_text(json.dumps(wrong) + "\n" + "".join(x + "\n" for x in lines))
    patched(data)
    with pytest.raises(CB.ProvenanceError, match="conflicting intent"):
        CB.main([])
    for bad in ({"kind": "mystery", "attempt_id": "z"},
                {"kind": "completion", "attempt_id": "z", "outcome": "teleported", "gamePk": 700002},
                {"kind": "intent", "attempt_id": 7, "gamePk": 700002}):
        d2 = world(tmp_path / bad["kind"] / str(bad.get("outcome")) / str(bad["attempt_id"]), n=3)
        receipt(d2 / "hetzner_results" / "c1" / "r3", bad)
        patched(d2)
        with pytest.raises(CB.ProvenanceError, match="unknown|attempt id"):
            CB.main([])


def test_a_boolean_game_id_cannot_bind_a_request(tmp_path, patched):
    """True == 1 in Python: a request naming game True must not bind game 1's stored record."""
    assert CB.FEED_URL.format(pk=True) != CB.FEED_URL.format(pk=1)       # and the typed check refuses it outright
    data = world(tmp_path, n=3)
    recs = data / "hetzner_results" / "c1" / "r3" / "receipts" / "2026-10-04.jsonl"
    lines = [json.loads(x) for x in recs.read_text().splitlines()]
    for r in lines:
        if r["kind"] == "intent" and r["gamePk"] == 700001:
            r["gamePk"] = float(700001)
    recs.write_text("".join(json.dumps(r) + "\n" for r in lines))
    patched(data)
    with pytest.raises(CB.ProvenanceError, match="not bound|unresolved|game id"):     # r4 R4-1: now typed first
        CB.main([])


# ---- C2 (b) step 1: code review r4 R4-1 (docs/audit/2026-10-05-c1-r3-build-codex-r4.md) --------------------------------

URL1 = "https://statsapi.mlb.com/api/v1.1/game/700001/feed/live"
URL2 = "https://statsapi.mlb.com/api/v1.1/game/700002/feed/live"


def _recs(data):
    return data / "hetzner_results" / "c1" / "r3" / "receipts" / "2026-10-04.jsonl"


def _no_run(data):
    """No run directory and no claim: the admission lock file may exist, since the lock is taken before the
    inventory, which refuses before `make_run_dir` and `write_claim`."""
    builds = data / "hetzner_results" / "c1" / "r3" / "count_build"
    return not builds.exists() or (not [p for p in builds.iterdir() if p.is_dir()]
                                    and not list(builds.rglob("CLAIM.json")))


def _refuses_before_claim(data, patched, monkeypatch, match):
    """Refusal before the claim and before any PA or feed byte is read: the actual Path.read_bytes is instrumented
    for both input roots (C2 review r1: a schema spy alone sits downstream of the parquet read)."""
    patched(data)
    read, real = [], Path.read_bytes
    roots = (str(data / "processed"), str(data / "raw_c1"))

    def spy(self):
        if str(self).startswith(roots):
            read.append(str(self))
        return real(self)
    monkeypatch.setattr(Path, "read_bytes", spy)
    with pytest.raises(CB.ProvenanceError, match=match):
        CB.main([])
    assert _no_run(data) and read == []


def test_r41_a_float_request_then_an_equal_integer_duplicate_refuses(tmp_path, patched, monkeypatch):
    """R4-1 A: dict equality made intent a-700001 with gamePk 7.0 equal to its integer twin, so the malformed record
    was overwritten instead of refused."""
    data = world(tmp_path, n=3)
    lines = _recs(data).read_text().splitlines()
    first = json.loads(lines[0])
    assert first["kind"] == "intent" and first["gamePk"] == 700001
    _recs(data).write_text(json.dumps({**first, "gamePk": float(700001)}) + "\n" + "".join(x + "\n" for x in lines))
    _refuses_before_claim(data, patched, monkeypatch, "game id|gamePk")


def test_r41_a_stored_and_an_http_error_completion_for_one_attempt_refuse(tmp_path, patched, monkeypatch):
    """R4-1 B: completions were checked only within the response and stored classes, so a terminal 404 for the same
    attempt as its stored record was ignored."""
    data = world(tmp_path, n=3)
    receipt(data / "hetzner_results" / "c1" / "r3",
            {"kind": "completion", "attempt_id": "a-700001", "gamePk": 700001, "url": URL1, "outcome": "http_error",
             "http_status": 404, "ended_utc": "2026-10-04T23:00:02+00:00"})
    _refuses_before_claim(data, patched, monkeypatch, "conflicting completion")


def test_r41_a_malformed_error_completion_cannot_resolve_an_intent(tmp_path, patched, monkeypatch):
    """R4-1 B: a network_error completion with gamePk 8.0 resolved an integer-8 intent by tuple equality."""
    data = world(tmp_path, n=3)
    acq = data / "hetzner_results" / "c1" / "r3"
    receipt(acq, {"kind": "intent", "attempt_id": "z", "gamePk": 700002, "url": URL2,
                  "started_utc": "2026-10-04T23:00:00+00:00"})
    receipt(acq, {"kind": "completion", "attempt_id": "z", "gamePk": float(700002), "url": URL2,
                  "outcome": "network_error", "ended_utc": "2026-10-04T23:00:01+00:00"})
    _refuses_before_claim(data, patched, monkeypatch, "game id|gamePk")


@pytest.mark.parametrize("field, value", [("attempt_id", ""), ("attempt_id", 7), ("gamePk", True), ("gamePk", 0),
                                          ("gamePk", "700001"), ("kind_of", "mystery"), ("kind_of", "schedule"),
                                          ("from_attempt_id", 7), ("season", 2023)])
def test_r41_every_intent_and_completion_is_typed_before_any_join(tmp_path, patched, monkeypatch, field, value):
    data = world(tmp_path, n=3)
    receipt(data / "hetzner_results" / "c1" / "r3",
            {"kind": "completion", "attempt_id": "e1", "gamePk": 700003, "url": "u", "outcome": "network_error",
             "ended_utc": "2026-10-04T23:00:01+00:00", field: value})
    _refuses_before_claim(data, patched, monkeypatch, "attempt id|game id|kind_of|season")


def test_r41_genuinely_identical_repeats_and_distinct_retries_still_certify(tmp_path, patched):
    """Positive controls: exact repeated records are tolerated, and a failed attempt followed by a successful one
    (distinct attempt ids) is a normal retry."""
    data = world(tmp_path, n=3)
    lines = _recs(data).read_text().splitlines()
    acq = data / "hetzner_results" / "c1" / "r3"
    _recs(data).write_text("".join(x + "\n" for x in lines + lines[:2]))               # 700001's pair, repeated
    receipt(acq, {"kind": "intent", "attempt_id": "r1", "gamePk": 700002, "url": URL2,
                  "started_utc": "2026-10-04T22:59:00+00:00"})
    receipt(acq, {"kind": "completion", "attempt_id": "r1", "gamePk": 700002, "url": URL2,
                  "outcome": "network_error", "ended_utc": "2026-10-04T22:59:30+00:00"})
    patched(data)
    assert CB.main([]) == 0
    census = json.loads((only_run(data) / "census.json").read_text())
    assert census["certified"] == 3 and census["quarantined"] == {}


def test_r41_a_duplicate_differing_only_in_number_type_is_a_conflict(tmp_path, patched, monkeypatch):
    """Canonical comparison: an intent repeated with attempt 1 then 1.0 is not a genuinely identical duplicate."""
    data = world(tmp_path, n=3)
    lines = _recs(data).read_text().splitlines()
    first = json.loads(lines[0])
    _recs(data).write_text(json.dumps({**first, "attempt": 1}) + "\n" + json.dumps({**first, "attempt": 1.0}) + "\n"
                           + "".join(x + "\n" for x in lines[1:]))
    _refuses_before_claim(data, patched, monkeypatch, "conflicting intent")


# ---- C2 (b) step 1, rank-3 review r1 C2R1-1: request kind and primary identity ------------------------------------

def test_c2r1_a_schedule_completion_cannot_carry_a_feed_game_id(tmp_path, patched, monkeypatch):
    """C2R1-1 composition 1: a schedule-declared error completion with gamePk 8.0 resolved an integer-8 feed intent."""
    data = world(tmp_path, n=3)
    acq = data / "hetzner_results" / "c1" / "r3"
    url8 = "https://statsapi.mlb.com/api/v1.1/game/8/feed/live"
    receipt(acq, {"kind": "intent", "attempt_id": "z", "gamePk": 8, "url": url8, "started_utc": "2026-10-04T23:00:00+00:00"})
    receipt(acq, {"kind": "completion", "attempt_id": "z", "kind_of": "schedule", "season": 2023, "gamePk": 8.0,
                  "url": url8, "outcome": "network_error", "ended_utc": "2026-10-04T23:00:01+00:00"})
    _refuses_before_claim(data, patched, monkeypatch, "schedule receipt")


def test_c2r1_a_schedule_completion_for_another_season_cannot_resolve_its_intent(tmp_path, patched, monkeypatch):
    """C2R1-1 composition 2: both game ids null, so (s, None) matched across seasons."""
    data = world(tmp_path, n=3)
    acq = data / "hetzner_results" / "c1" / "r3"
    base = {"attempt_id": "s", "kind_of": "schedule", "gamePk": None, "url": "https://statsapi.mlb.com/sched"}
    receipt(acq, {**base, "kind": "intent", "season": 2023, "started_utc": "2026-10-04T23:00:00+00:00"})
    receipt(acq, {**base, "kind": "completion", "season": 2024, "outcome": "network_error",
                  "ended_utc": "2026-10-04T23:00:01+00:00"})
    _refuses_before_claim(data, patched, monkeypatch, "different request")


@pytest.mark.parametrize("keep_game", [True, False])
def test_c2r1_a_schedule_declared_intent_cannot_anchor_a_feed(tmp_path, patched, monkeypatch, keep_game):
    """C2R1-1 composition 3: the legacy game-7 intent re-declared as a schedule request still anchored its feed."""
    data = world(tmp_path, n=3)
    lines = [json.loads(x) for x in _recs(data).read_text().splitlines()]
    for r in lines:
        if r["kind"] == "intent" and r["gamePk"] == 700001:
            r.update(kind_of="schedule", season=2023)
            if not keep_game:
                del r["gamePk"]
    _recs(data).write_text("".join(json.dumps(r) + "\n" for r in lines))
    _refuses_before_claim(data, patched, monkeypatch, "schedule receipt" if keep_game else "different request")


@pytest.mark.parametrize("outcome, extra", [("network_error", {"error": "URLError: x"}),
                                            ("rate_limited", {"http_status": 429})])
def test_c2r1_a_stored_attempt_with_any_terminal_error_conflicts(tmp_path, patched, monkeypatch, outcome, extra):
    data = world(tmp_path, n=3)
    receipt(data / "hetzner_results" / "c1" / "r3",
            {"kind": "completion", "attempt_id": "a-700001", "gamePk": 700001, "url": URL1, "outcome": outcome,
             "ended_utc": "2026-10-04T23:00:02+00:00", **extra})
    _refuses_before_claim(data, patched, monkeypatch, "conflicting completion")


def test_c2r1_a_producer_generated_schedule_feed_retry_chain_certifies(tmp_path, patched, monkeypatch):
    """Positive control from the actual current acquirer (schedule request, response and store; a feed network error,
    then a retried response and its separately identified store record), consumed by the actual build."""
    import urllib.error
    data = tmp_path / "data"
    out, feeds = data / "hetzner_results" / "c1" / "r3", data / "raw_c1"
    pk = 700001
    body = json.dumps(feed(pk=pk, date="2023-06-01")).encode()
    sched = json.dumps({"dates": [{"date": "2023-06-01", "games": [
        {"gamePk": pk, "officialDate": "2023-06-01", "status": {"detailedState": "Final"}}]}]}).encode()
    calls = {"feed": 0}

    def fetch(url):
        if "/schedule" in url:
            return sched
        calls["feed"] += 1
        if calls["feed"] == 1:
            raise urllib.error.URLError("synthetic reset")
        return body
    monkeypatch.setattr(aq, "C1_ROOT", data / "hetzner_results" / "c1")
    monkeypatch.setattr(aq, "_fetch", fetch)
    monkeypatch.setattr(aq.time, "sleep", lambda s: None)
    assert aq.main(["--seasons", "2023", "--out", str(out), "--feeds", str(feeds)]) == 0
    kinds = [(r.get("kind"), r.get("kind_of", "feed"), r.get("outcome")) for r in aq.read_receipts(out)]
    assert ("completion", "feed", "network_error") in kinds and ("completion", "schedule", "stored") in kinds
    rows = [{"game_pk": pk, "batter_id": p.batter, "is_home": p.side == "home", "season": 2023, "date": "2023-06-01"}
            for p in M.extract(feed(pk=pk, date="2023-06-01")).pas]
    (data / "processed").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(data / "processed" / "pa_2023.parquet", index=False)
    patched(data)
    assert CB.main([]) == 0
    census = json.loads((only_run(data) / "census.json").read_text())
    assert census["certified"] == 1 and census["quarantined"] == {}


@pytest.mark.parametrize("season", [None, True, 0, -1, "2023", 2023.0])
@pytest.mark.parametrize("role", ["intent", "completion"])
def test_c2r1_a_schedule_receipt_needs_an_exact_positive_int_season(tmp_path, patched, monkeypatch, season, role):
    """The schedule branch's own primary id, isolated: no game id, so only the season rule can refuse."""
    data = world(tmp_path, n=3)
    rec = {"kind": role, "attempt_id": "sch", "kind_of": "schedule", "gamePk": None, "season": season,
           "url": "https://statsapi.mlb.com/sched"}
    rec.update({"started_utc": "2026-10-04T23:00:00+00:00"} if role == "intent"
               else {"outcome": "network_error", "ended_utc": "2026-10-04T23:00:01+00:00"})
    receipt(data / "hetzner_results" / "c1" / "r3", rec)
    _refuses_before_claim(data, patched, monkeypatch, "season")
