"""W2.1 item 1: the census gate over the retained board receipts and the season-best summary (design B-E5; code review
r1 F5: integrity failures are not coverage gaps, and failed output rows are never lower-bound evidence)."""
from __future__ import annotations

import gzip
import hashlib
import json

import pandas as pd

from scripts.audit.field_products import census as C
from tests.scripts import test_field_products_builders as B

OURS = 1000 + 150   # season best 15 in board_rows(420)


def _grab(tmp_path, rows=None, **kw):
    rows = rows or B.board_rows(420)
    return B.make_grab(tmp_path, board_pages=B.pages(rows, **kw), early=[(1001, "a")], cohort_a=1, cohort_b=0)


def _summary(grab, our=OURS):
    rec = C.load_board_receipts(grab)
    gate = C.census_gate(rec)
    return gate, C.season_best_summary(gate["qualified"], gate, our_user_id=our)


def test_complete_board_is_a_census_and_reports_exact_counts_percentile_and_id_bound_rank(tmp_path):
    gate, s = _summary(_grab(tmp_path))
    assert gate["census"] is True and gate["failures"] == [] and gate["reported"] == 420 and gate["listed"] == 420
    assert gate["integrity_ok"] and gate["coverage_ok"]
    assert s["basis"] == "census" and s["N"] == 420 and s["missing_season_best"] == 0
    assert s["thresholds"] == {"ge_20": {"known": 110, "upper_bound": 110}, "ge_30": {"known": 10, "upper_bound": 10},
                               "ge_40": {"known": 0, "upper_bound": 0}}
    assert s["max_known"] == 30 and s["field_max"] == 30 and s["tie_at_max_known"] == 10 and s["listing_floor"] == 0
    own = s["own"]
    assert own["matched_by_user_id"] and own["season_best"] == 15 and own["stored_rank"] == 151
    assert (own["below"], own["equal"], own["above"]) == (260, 10, 150)
    assert own["percentile_interval"] == [100 * 260 / 420, 100 * 270 / 420]
    assert own["tied_stored_ranks"] == [151]
    assert sum(r["users"] for r in s["distribution"]) == 420


def test_invalid_updated_at_is_a_coverage_failure_and_downgrades_to_observed_lower_bounds(tmp_path):
    gate, s = _summary(_grab(tmp_path, updated_per_page=[B.UPDATED, "not-a-timestamp"]))
    assert gate["census"] is False and gate["integrity_ok"] and not gate["coverage_ok"]
    assert "recomputed_population:unknown_incomplete_participant_metadata" in gate["failures"]
    assert s["basis"] == "observed_lower_bound" and s["N"] is None
    assert s["own"]["percentile_interval"] is None and s["thresholds"]["ge_20"] == {"known": 110, "upper_bound": None}


def test_retained_census_claim_is_checked_against_the_raw_pages(tmp_path):
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


def test_tampered_raw_page_is_an_integrity_failure_and_its_rows_are_excluded(tmp_path):
    grab = _grab(tmp_path)
    page1 = grab / "raw" / "board" / "page_001.json.gz"
    body = json.loads(gzip.decompress(page1.read_bytes()))
    body["success"]["ranks"][0]["streak"] = 99
    page1.write_bytes(gzip.compress(json.dumps(body).encode()))
    gate, s = _summary(grab)
    assert gate["census"] is False and not gate["integrity_ok"] and "raw_page_hashes" in gate["failures"]
    assert gate["qualification"]["pages_excluded_hash"] == 1 and len(gate["qualified"]) == 120
    assert s["basis"] == "qualified_raw_lower_bound" and s["max_known"] == 0 and s["listed"] == 120


def test_review_probe_altered_output_parquet_is_never_counted(tmp_path):
    """F5 probe: change one output season best to 40 (raw count at 40 is zero): hash and raw/output equality fail;
    the published counts come from the verified raw rows, so ge_40 stays 0."""
    grab = _grab(tmp_path)
    rec = C.load_board_receipts(grab)
    path = grab / rec["board_relpath"]
    df = pd.read_parquet(path)
    df.loc[0, "season_best_streak"] = 40
    df.to_parquet(path)
    gate, s = _summary(grab)
    assert {"board_output_hash", "board_rows_equal_raw"} <= set(gate["failures"]) and not gate["integrity_ok"]
    assert s["basis"] == "qualified_raw_lower_bound" and s["thresholds"]["ge_40"]["known"] == 0
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)          # ...and duplicate the altered row
    df.to_parquet(path)
    gate, s = _summary(grab)
    assert "board_unique_ids" in gate["failures"] and s["thresholds"]["ge_40"]["known"] == 0 and s["listed"] == 420


def test_conflicting_duplicate_raw_rows_are_excluded_and_counted(tmp_path):
    rows = B.board_rows(420)
    dup = dict(rows[5])
    dup.update(rank=400, streak=1)
    rows[399] = dup                                             # user 1005 listed twice with different bests
    grab = B.make_grab(tmp_path, board_pages=B.pages(rows, participants=419), early=[(1001, "a")], cohort_a=1,
                       cohort_b=0)
    gate, s = _summary(grab)
    assert not gate["integrity_ok"] and "recomputed_conflict_free" in gate["failures"]
    assert gate["qualification"]["conflicting_ids_excluded"] == 1 and 1005 not in set(gate["qualified"]["user_id"])
    assert s["basis"] == "qualified_raw_lower_bound"


def test_short_walk_with_count_match_is_not_a_census(tmp_path):
    rows = B.board_rows(300)
    pages = B.pages(rows)
    pages[0][1]["success"]["nextPage"] = True
    grab = B.make_grab(tmp_path, board_pages=pages + [(500, b"oops")], early=[(1001, "a")], cohort_a=1, cohort_b=0)
    gate = C.census_gate(C.load_board_receipts(grab))
    assert gate["census"] is False and "recomputed_walk_exhausted" in gate["failures"] and gate["integrity_ok"]


def test_missing_season_best_is_counted_bounded_and_never_imputed_zero():
    rows = pd.DataFrame({"user_id": [1, 2, 3, 4, 5], "rank": [1, 2, 3, 3, 5],
                         "season_best_streak": pd.array([20, 18, None, 0, 0], dtype="Int64")})
    gate = {"census": True, "reported": 5, "listed": 5, "integrity_ok": True}
    s = C.season_best_summary(rows, gate, our_user_id=2)
    assert s["missing_season_best"] == 1 and {r["season_best"]: r["users"] for r in s["distribution"]} == {0: 2, 18: 1, 20: 1}
    assert s["max_known"] == 20 and s["field_max"] is None and s["tie_at_max_known"] == 1
    assert s["thresholds"]["ge_20"] == {"known": 1, "upper_bound": 2}
    assert s["listing_floor"] == 0 and s["own"]["below"] == 2 and s["own"]["above"] == 1
    assert s["own"]["percentile_interval"] == [40.0, 80.0]   # [100*2/5, 100*(2+1+1 missing)/5]


def test_own_rank_requires_the_stable_user_id():
    rows = pd.DataFrame({"user_id": [1, 2], "rank": [1, 2], "season_best_streak": pd.array([20, 18], dtype="Int64")})
    s = C.season_best_summary(rows, {"census": True, "reported": 2, "listed": 2, "integrity_ok": True}, our_user_id=99)
    assert s["own"]["matched_by_user_id"] is False and s["own"]["stored_rank"] is None
    assert s["own"]["percentile_interval"] is None


def test_review_probe_a_name_only_duplicate_never_certifies_an_excluded_subset(tmp_path):
    """R2-2: the real producer keeps two unique ids (1000 'alice' rank 1/best 30 repeated as 'alice2'; 1001 'bob'
    rank 2/best 18); qualification conservatively excludes 1000. One qualified row is not a census of N=2: no N,
    upper bounds, percentile or field maximum."""
    from tests.scripts.test_final_leaderboard_grab import _rank_row
    rows = [_rank_row(1000, 1, 30, username="alice"), _rank_row(1001, 2, 18, username="bob"),
            _rank_row(1000, 1, 30, username="alice2")]
    grab = B.make_grab(tmp_path, board_pages=B.pages(rows, participants=2), early=[(1001, "a")], cohort_a=1,
                       cohort_b=0)
    gate, s = _summary(grab, our=1001)
    assert gate["census"] is False and gate["listed"] == 1 and gate["reported"] == 2
    assert {"qualified_conflict_free", "qualified_rows_equal_reported"} <= set(gate["failures"])
    assert gate["checks"]["board_rows_equal_raw"] and gate["checks"]["recomputed_population"]   # producer side ok
    assert s["basis"] == "qualified_raw_lower_bound" and s["N"] is None and s["field_max"] is None
    assert s["thresholds"]["ge_30"] == {"known": 0, "upper_bound": None}
    assert s["own"]["percentile_interval"] is None
