"""W2.1 item 2-3: the survivor-selected Cohort A case series from the real final-grab formats (code review r1 F3, F4,
F7): round completeness from the verified raw profile response, run dates unavailable, composition unknown."""
from __future__ import annotations

import gzip
import json

import pandas as pd

from scripts.audit.field_products import leaders as L
from tests.scripts import test_field_products_builders as B

BEST = {0: 4, 1: 6, 2: 3}     # ids 1000, 1001, 1002; everyone else 0 -> A = [1001, 1000, 1002]


def _grab(tmp_path):
    rows = B.board_rows(420, best=lambda i: BEST.get(i, 0))
    p1000 = B.profile([B.pred(1000, 1, [(1, 5, 3, "hit")]), B.pred(1001, 2, [(1, 5, 4, "hit")]),
                       B.pred(1002, 4, [(1, 5, 5, "hit"), (2, 5, 6, "hit")]), B.pred(1003, 0, [(1, 5, 7, "not_hit")]),
                       B.pred(1009, 1, [(1, 5, 3, "hit")])])
    p1001 = B.profile([B.pred(1004, 5, [(1, 5, 3, "hit")]), B.pred(1005, 6, [(1, 5, 3, "hit")])])
    return B.make_grab(tmp_path, board_pages=B.pages(rows), early=[(1001, "x")], cohort_a=3, cohort_b=0,
                       profiles={1000: (200, p1000), 1001: (200, p1001), 1002: (200, B.profile([]))})


def test_case_series_reports_pick_days_dd_runs_and_unknown_composition_with_survivor_labels(tmp_path):
    grab = _grab(tmp_path)
    out = L.case_series(grab)
    users = out["users"].set_index("user_id")
    assert list(out["users"]["user_id"]) == [1001, 1000, 1002]          # A's own (rank, id) order
    u = users.loc[1000]
    assert u["board_rank"] == 2 and u["board_season_best"] == 4 and u["history"] == "usable"
    assert u["raw_witness"] == "verified"
    assert u["pick_days"] == 5 and u["complete_rounds"] == 5 and u["dd_rounds"] == 1 and u["dd_frequency"] == 0.2
    assert u["unobserved_calendar_dates"] == 187 - 5
    assert u["runs_status"] == "dates_unavailable" and u["runs_reported_attainments"] == 1
    assert u["runs_longest_observed_segment"] == 4
    assert u["composition_witnessed"] == 0 and u["composition_unknown"] == 6
    v = users.loc[1001]
    assert v["runs_status"] == "dates_unavailable" and v["runs_longest_observed_segment"] == 2
    w = users.loc[1002]
    assert w["history"] == "no_history" and w["pick_days"] == 0 and pd.isna(w["dd_frequency"])
    assert w["runs_status"] == "history_unavailable"
    r = out["runs"][out["runs"]["user_id"] == 1000].iloc[0]
    assert not r["recoverable"] and r["start_date"] is None and r["reported_attainment_date"] == "2026-09-20"
    assert r["observed_segment_lower_bound"] == 4
    s = out["summary"]
    assert s["survivor_selected"] is True and "not an awarded prize" in s["labels"]["listing"]
    assert s["n_A"] == 3 and s["history"] == {"usable": 2, "no_history": 1}
    assert s["pooled_dd"] == {"complete_rounds": 7, "dd_rounds": 1, "dd_frequency": 1 / 7, "users": 2}
    assert s["composition"]["witnessed"] == 0 and s["composition"]["unknown"] == 8
    assert s["raw_witness"] == {"verified": 2, "not_applicable": 1}


def test_raw_profile_that_fails_its_receipt_hash_gives_no_completeness(tmp_path):
    grab = _grab(tmp_path)
    raw = grab / "raw" / "profiles" / "1000.json.gz"
    body = json.loads(gzip.decompress(raw.read_bytes()))
    body["success"]["predictions"][2]["roundPredictions"].pop()          # the DD leg vanishes from the archive
    raw.write_bytes(gzip.compress(json.dumps(body).encode()))
    u = L.case_series(grab)["users"].set_index("user_id").loc[1000]
    assert u["raw_witness"] == "archive_hash_mismatch" and u["complete_rounds"] == 0 and pd.isna(u["dd_frequency"])
    assert u["pick_days"] == 5


def test_an_a_member_without_a_qualified_board_row_has_unavailable_board_fields(tmp_path):
    from scripts.audit.field_products import census as C
    grab = _grab(tmp_path)
    qualified = C.census_gate(C.load_board_receipts(grab))["qualified"]
    u = L.case_series(grab, board=qualified[qualified["user_id"] != 1000])["users"].set_index("user_id").loc[1000]
    assert pd.isna(u["board_rank"]) and pd.isna(u["board_username"]) and pd.isna(u["board_season_best"])
    assert u["runs_status"] == "best_unavailable" and u["runs_reported_attainments"] == 0
    assert u["history"] == "usable" and u["pick_days"] == 5            # membership/profile coverage preserved
