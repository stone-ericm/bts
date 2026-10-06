"""T5: the count build end to end on a fully synthetic data tree (no real feed, parquet or receipt is read)."""
import gzip
import hashlib
import json

import pandas as pd
import pytest

from scripts.audit.c1_r3 import acquire as aq
from scripts.audit.c1_r3 import count_build as CB
from scripts.audit.c1_r3 import count_meta as M
from tests.scripts.c1_r3.feeds import feed, lineup


def world(tmp_path, n=120, *, drop_parquet=(), corrupt=(), wrong_counts=(), extra_feed=False):
    """n certified 2023 games: receipt-bound gz feeds plus a PA parquet that agrees with them."""
    data = tmp_path / "data"
    feeds, acq = data / "raw_c1", data / "hetzner_results" / "c1" / "r3"
    rows = []
    pks = list(range(700001, 700001 + n)) + ([799999] if extra_feed else [])
    for pk in pks:
        body = json.dumps(feed(pk=pk, date="2023-06-01")).encode()
        stored = gzip.compress(body, mtime=0)
        rel = f"2023/{pk}.json.gz"
        (feeds / "2023").mkdir(parents=True, exist_ok=True)
        (feeds / rel).write_bytes(stored + (b"x" if pk in corrupt else b""))
        aq._append(acq / "receipts" / "2026-10-04.jsonl",
                   {"kind": "completion", "attempt_id": f"store-{pk}", "kind_of": "feed", "gamePk": pk,
                    "outcome": "stored", "decoded_sha256": hashlib.sha256(body).hexdigest(), "stored_path": rel,
                    "stored_sha256": hashlib.sha256(stored).hexdigest()})
        if pk in drop_parquet or pk == 799999:
            continue
        for p in M.extract(feed(pk=pk, date="2023-06-01")).pas:
            rows.append({"game_pk": pk, "batter_id": p.batter, "is_home": p.side == "home", "season": 2023})
        if pk in wrong_counts:
            rows.append({"game_pk": pk, "batter_id": 101, "is_home": False, "season": 2023})
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
    assert len(prov) == 121 and prov["700001"]["feed_timestamp"] == "20230602_010203" and prov["700001"]["certified"]
    assert prov["799999"]["certified"] is False
    for name in ("count_table.json", "bf_starts.json", "census.json", "provenance.json"):
        assert res["outputs"][name] == hashlib.sha256((d / name).read_bytes()).hexdigest()
    man = json.loads((d / "manifest.json").read_text())
    assert man["feeds"]["count"] == 121 and "pa_2023.parquet" in man["pa_parquets"]
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
