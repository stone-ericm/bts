"""W2.1 item 1: the census gate over the retained board receipts and the season-best summary (design B-E5)."""
from __future__ import annotations

import gzip
import hashlib
import json

import pandas as pd

from scripts.audit.field_products import census as C
from tests.scripts import test_field_products_builders as B

OURS = 1000 + 150   # season best 15 in board_rows(420)


def _grab(tmp_path, **kw):
    rows = B.board_rows(420)
    return B.make_grab(tmp_path, board_pages=B.pages(rows, **kw), early=[(1001, "a")], cohort_a=1, cohort_b=0)


def test_complete_board_is_a_census_and_reports_exact_counts_percentile_and_id_bound_rank(tmp_path):
    grab = _grab(tmp_path)
    rec = C.load_board_receipts(grab)
    gate = C.census_gate(rec)
    assert gate["census"] is True and gate["failures"] == [] and gate["reported"] == 420 and gate["listed"] == 420
    s = C.season_best_summary(rec["board"], gate, our_user_id=OURS)
    assert s["basis"] == "census" and s["N"] == 420 and s["missing_season_best"] == 0
    assert s["thresholds"] == {"ge_20": 110, "ge_30": 10, "ge_40": 0}
    assert s["max"] == 30 and s["tie_at_max"] == 10 and s["listing_floor"] == 0
    own = s["own"]
    assert own["matched_by_user_id"] and own["season_best"] == 15 and own["stored_rank"] == 151
    assert (own["below"], own["equal"], own["above"]) == (260, 10, 150)
    assert own["percentile_interval"] == [100 * 260 / 420, 100 * 270 / 420]
    assert own["tied_stored_ranks"] == [151]
    assert sum(r["users"] for r in s["distribution"]) == 420


def test_invalid_updated_at_on_one_page_downgrades_to_observed_lower_bounds(tmp_path):
    grab = _grab(tmp_path, updated_per_page=[B.UPDATED, "not-a-timestamp"])
    rec = C.load_board_receipts(grab)
    gate = C.census_gate(rec)
    assert gate["census"] is False
    assert "recomputed_population:unknown_incomplete_participant_metadata" in gate["failures"]
    s = C.season_best_summary(rec["board"], gate, our_user_id=OURS)
    assert s["basis"] == "observed_lower_bound" and s["N"] is None
    assert s["own"]["percentile_interval"] is None and s["thresholds"]["ge_20"] == 110


def test_retained_census_claim_is_checked_against_the_raw_pages(tmp_path):
    """A raw page whose updatedAt drifts (hash re-forged in status.json) is caught by the independent recompute even
    though the retained status still says census."""
    grab = _grab(tmp_path)
    page2 = grab / "raw" / "board" / "page_002.json.gz"
    body = json.loads(gzip.decompress(page2.read_bytes()))
    body["success"]["updatedAt"] = "2026-09-28T08:11:55-04:00"
    raw = json.dumps(body).encode()
    page2.write_bytes(gzip.compress(raw))
    status = json.loads((grab / "status.json").read_text())
    for e in status["requests"]:
        if e["name"] == "page_002":
            e["archived_sha256"] = hashlib.sha256(raw).hexdigest()
    (grab / "status.json").write_text(json.dumps(status))
    gate = C.census_gate(C.load_board_receipts(grab))
    assert gate["checks"]["retained_status_census"] is True and gate["checks"]["raw_page_hashes"] is True
    assert gate["census"] is False and "recomputed_population:unknown_server_version_drift" in gate["failures"]


def test_tampered_raw_page_fails_the_hash_check(tmp_path):
    grab = _grab(tmp_path)
    page1 = grab / "raw" / "board" / "page_001.json.gz"
    body = json.loads(gzip.decompress(page1.read_bytes()))
    body["success"]["ranks"][0]["streak"] = 99
    page1.write_bytes(gzip.compress(json.dumps(body).encode()))
    gate = C.census_gate(C.load_board_receipts(grab))
    assert gate["census"] is False and "raw_page_hashes" in gate["failures"]


def test_board_output_must_equal_the_raw_pages_and_its_recorded_hash(tmp_path):
    grab = _grab(tmp_path)
    rec = C.load_board_receipts(grab)
    path = grab / rec["board_relpath"]
    df = pd.read_parquet(path)
    df.loc[0, "season_best_streak"] = 31
    df.to_parquet(path)
    gate = C.census_gate(C.load_board_receipts(grab))
    assert gate["census"] is False and {"board_output_hash", "board_rows_equal_raw"} <= set(gate["failures"])


def test_short_walk_with_count_match_is_not_a_census(tmp_path):
    """Equal row and participant counts alone are insufficient: the last page claims more pages exist."""
    rows = B.board_rows(300)
    pages = B.pages(rows)
    pages[0][1]["success"]["nextPage"] = True
    grab = B.make_grab(tmp_path, board_pages=pages + [(500, b"oops")], early=[(1001, "a")], cohort_a=1, cohort_b=0)
    gate = C.census_gate(C.load_board_receipts(grab))
    assert gate["census"] is False and "recomputed_walk_exhausted" in gate["failures"]


def test_missing_season_best_is_counted_never_imputed_zero_and_widens_the_interval():
    board = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "rank": [1, 2, 3, 3, 5], "tab": ["all_season"] * 5,
                          "season_best_streak": pd.array([20, 18, None, 0, 0], dtype="Int64")})
    gate = {"census": True, "reported": 5, "listed": 5}
    s = C.season_best_summary(board, gate, our_user_id=2)
    assert s["missing_season_best"] == 1 and {r["season_best"]: r["users"] for r in s["distribution"]} == {0: 2, 18: 1, 20: 1}
    assert s["listing_floor"] == 0 and s["own"]["below"] == 2 and s["own"]["above"] == 1
    assert s["own"]["percentile_interval"] == [40.0, 80.0]   # [100*2/5, 100*(2+1+1 missing)/5]


def test_own_rank_requires_the_stable_user_id():
    board = pd.DataFrame({"user_id": [1, 2], "rank": [1, 2], "tab": ["all_season"] * 2,
                          "season_best_streak": pd.array([20, 18], dtype="Int64")})
    s = C.season_best_summary(board, {"census": True, "reported": 2, "listed": 2}, our_user_id=99)
    assert s["own"]["matched_by_user_id"] is False and s["own"]["stored_rank"] is None
    assert s["own"]["percentile_interval"] is None
