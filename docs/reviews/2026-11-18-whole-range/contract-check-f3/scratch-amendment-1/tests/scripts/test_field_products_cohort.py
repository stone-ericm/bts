"""W2.2 membership: manifest E, acquisition labels (never an analytical cohort), final-grab history status, and the
capture-time identity binding of daily sanitized-username files (design B-E1/B-E2)."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts.audit.field_products import cohort as K
from scripts.final_leaderboard_grab import allocate_cohort
from tests.scripts import test_field_products_builders as B


def test_manifest_is_loaded_in_recorded_order_with_its_hash(tmp_path):
    p = B.manifest(tmp_path / "e.json", [(5, "x"), (3, "y"), (9, "z")])
    m = K.load_manifest(p)
    assert list(m["members"]["user_id"]) == [5, 3, 9] and list(m["members"]["order"]) == [1, 2, 3]
    assert m["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest() and m["n"] == 3


def test_manifest_with_duplicate_ids_or_broken_order_is_refused(tmp_path):
    p = B.manifest(tmp_path / "e.json", [(5, "x"), (5, "y")])
    with pytest.raises(ValueError, match="distinct"):
        K.load_manifest(p)
    doc = json.loads(B.manifest(tmp_path / "f.json", [(5, "x"), (6, "y")]).read_text())
    doc["users"][1]["order"] = 7
    (tmp_path / "f.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="order"):
        K.load_manifest(tmp_path / "f.json")


def test_allocation_labels_are_recomputed_and_kept_as_labels(tmp_path):
    early = [(i, f"u{i}") for i in range(1, 7)]
    m = K.load_manifest(B.manifest(tmp_path / "e.json", early))
    cj = allocate_cohort([{"user_id": 3, "rank": 1}, {"user_id": 9, "rank": 2}, {"user_id": 1, "rank": 3}],
                         [{"user_id": i} for i, _ in early], cohort_a=3, cohort_b=2)
    cj.update(early_cohort_sha256=m["sha256"], E_n=6)
    labels, checks = K.allocation(m, cj)
    assert dict(zip(labels["user_id"], labels["allocation"])) == {1: "E_in_A", 2: "B", 3: "E_in_A", 4: "B",
                                                                  5: "E_unfetched", 6: "E_unfetched"}
    assert checks == {"manifest_sha_matches_grab": True, "E_n_matches": True, "recomputed_allocation_matches": True,
                      "B_shortfall": 0, "cohort_b": 2}
    cj["B"] = [2, 5]
    labels2, checks2 = K.allocation(m, cj)
    assert checks2["recomputed_allocation_matches"] is False
    assert dict(zip(labels2["user_id"], labels2["allocation"]))[5] == "E_unfetched"   # labels follow the rule


def test_final_grab_status_separates_budget_omission_failures_no_history_and_usable(tmp_path):
    board = B.pages(B.board_rows(420))
    picks = B.profile([B.pred(1000, 1, [(1, 5, 3, "hit")])])
    grab = B.make_grab(tmp_path, board_pages=board, cohort_a=2, cohort_b=3,
                       early=[(1001, "alpha"), (2001, "b1"), (2002, "b2"), (2003, "b3"), (2004, "b4")],
                       profiles={1000: (200, picks), 1001: (200, picks), 2001: (200, picks),
                                 2002: (200, B.profile([]))})
    m = K.load_manifest(tmp_path / "early.json")
    cj = json.loads((grab / "cohort.json").read_text())
    ident = json.loads((grab / "identity.json").read_text())
    labels, _ = K.allocation(m, cj)
    st = K.final_grab_status(labels, ident, grab).set_index("user_id")
    assert st.loc[1001, "allocation"] == "E_in_A" and st.loc[1001, "usable"]
    assert st.loc[2001, "usable"] and st.loc[2001, "n_picks"] == 1 and st.loc[2001, "first_pick_date"] == "2026-09-18"
    assert st.loc[2002, "history"] == "no_history" and not st.loc[2002, "usable"]
    assert st.loc[2003, "history"] == "fetch_or_parse_failure" and st.loc[2003, "fetch_status"] == "http_error"
    assert st.loc[2004, "history"] == "budget_omission" and not st.loc[2004, "fetched"]


def test_parsed_file_whose_hash_differs_from_identity_is_not_usable(tmp_path):
    board = B.pages(B.board_rows(420))
    picks = B.profile([B.pred(1000, 1, [(1, 5, 3, "hit")])])
    grab = B.make_grab(tmp_path, board_pages=board, cohort_a=2, cohort_b=1, early=[(2001, "b1")],
                       profiles={1000: (200, picks), 1001: (200, picks), 2001: (200, picks)})
    (grab / "user_picks" / "2001.parquet").write_bytes(b"tampered")
    m = K.load_manifest(tmp_path / "early.json")
    labels, _ = K.allocation(m, json.loads((grab / "cohort.json").read_text()))
    st = K.final_grab_status(labels, json.loads((grab / "identity.json").read_text()), grab).set_index("user_id")
    assert not st.loc[2001, "usable"] and st.loc[2001, "history"] == "parsed_hash_mismatch"


def _members(tmp_path, pairs):
    return K.load_manifest(B.manifest(tmp_path / "e.json", pairs))


def test_unique_sanitized_name_binds_both_the_raw_and_sanitized_files(tmp_path):
    m = _members(tmp_path, [(1, "joe smith"), (2, "ann")])
    b, counts = K.bind_daily_files(m, ["joe smith", "joe_smith", "ann", "stranger"])
    b = b.set_index("user_id")
    assert b.loc[1, "binding"] == "bound" and b.loc[1, "files"] == ["joe smith", "joe_smith"]
    assert b.loc[2, "binding"] == "bound" and counts["files_not_E"] == 1 and counts["files_bound"] == 3


def test_username_collision_inside_the_manifest_quarantines_every_claimant(tmp_path):
    """Two distinct ids named jordan on 5/01 (and a b / a_b after sanitization): no file is assigned to either."""
    m = _members(tmp_path, [(1, "jordan"), (2, "jordan"), (3, "a b"), (4, "a_b"), (5, "kim")])
    b, counts = K.bind_daily_files(m, ["jordan", "a_b", "kim"])
    b = b.set_index("user_id")
    assert set(b.loc[[1, 2, 3, 4], "binding"]) == {"quarantined_manifest_collision"}
    assert b.loc[5, "binding"] == "bound" and counts["files_quarantined"] == 2
    assert counts["members"] == {"bound": 1, "quarantined_manifest_collision": 4}


def test_a_foreign_name_that_sanitizes_to_a_members_file_quarantines_the_member(tmp_path):
    """joe!smith (not in E) wrote joe!smith.parquet before 6/09 and joe_smith.parquet after: the member's file mixes
    two accounts, so it is not attributed."""
    m = _members(tmp_path, [(1, "joe smith")])
    b, _ = K.bind_daily_files(m, ["joe_smith", "joe!smith"])
    assert b.set_index("user_id").loc[1, "binding"] == "quarantined_foreign_name_collision"


def test_member_without_a_file_is_listed_not_dropped(tmp_path):
    m = _members(tmp_path, [(1, "x"), (2, "y")])
    b, counts = K.bind_daily_files(m, ["x"])
    assert len(b) == 2 and b.set_index("user_id").loc[2, "binding"] == "no_daily_file"


def test_a_pre_sanitizer_name_containing_the_old_delimiter_stays_one_path(tmp_path):
    """Review F9 probe: manifest name a;b with only its legitimate old a;b.parquet file binds to ONE path."""
    m = _members(tmp_path, [(1, "a;b")])
    b, _ = K.bind_daily_files(m, ["a;b"])
    row = b.set_index("user_id").loc[1]
    assert row["binding"] == "bound" and row["files"] == ["a;b"] and row["n_files"] == 1
