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


def world(tmp_path, n=120, *, drop_parquet=(), corrupt=(), wrong_counts=(), extra_feed=False, plain=(), feed_kw=None):
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
        aq._append(acq / "receipts" / "2026-10-04.jsonl",
                   {"kind": "completion", "attempt_id": f"store-{pk}", "kind_of": "feed", "gamePk": pk,
                    "outcome": "stored", "decoded_sha256": hashlib.sha256(body).hexdigest(), "stored_path": rel,
                    "stored_sha256": hashlib.sha256(stored).hexdigest()})
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
    def apply(data):
        monkeypatch.setattr(CB, "DATA", data)
        monkeypatch.setattr(CB, "SEASONS", (2023,))
        monkeypatch.setattr(CB, "admission_gate", lambda: "f" * 40)
        monkeypatch.setattr(CB, "dirty_tree", lambda: [])
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
    pins = json.loads((d / "pins.json").read_text())
    assert pins["pa_2023.parquet"] == hashlib.sha256((data / "processed" / "pa_2023.parquet").read_bytes()).hexdigest()
    assert (d / "CLAIM.json").exists()


def test_more_than_one_percent_quarantined_stops_and_the_claim_blocks_a_rerun(tmp_path, patched):
    data = world(tmp_path, wrong_counts=(700001, 700002))                # 2 of 120 = 1.7%
    patched(data)
    assert CB.main([]) == 2
    d = only_run(data)
    assert (d / "STOPPED_quarantine.txt").exists() and not (d / "results.json").exists()
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
    with pytest.raises(SystemExit, match="reviewed_commit|review_report|exposure"):
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
    patched(data)
    monkeypatch.setattr(CB, "SEASONS", (2022, 2023))
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
    receipt(acq, {"kind": "completion", "attempt_id": "x", "kind_of": "feed", "gamePk": 700001, "outcome": "stored",
                  "stored_path": "2023/700001.json.gz", "stored_sha256": "a" * 64, "decoded_sha256": "b" * 64})
    patched(data)
    with pytest.raises(CB.ProvenanceError, match="conflicting"):
        CB.main([])
    data2 = world(tmp_path / "w2", n=3)
    receipt(data2 / "hetzner_results" / "c1" / "r3", {"kind": "intent", "attempt_id": "open"})
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
    with pytest.raises(CB.ProvenanceError, match="cannot be read"):
        CB.main([])
    d = only_run(data)
    assert (d / "STOPPED_provenance.txt").exists() and (d / "CLAIM.json").exists()


def test_an_ineligible_feed_is_hash_checked_but_never_parsed(tmp_path, patched, monkeypatch):
    data = world(tmp_path, n=5, extra_feed=True)
    patched(data)
    parsed = []
    real = CB.M.extract
    monkeypatch.setattr(CB.M, "extract", lambda feed: (parsed.append(feed["gamePk"]), real(feed))[1])
    assert CB.main([]) == 0 and 799999 not in parsed and len(parsed) == 5
