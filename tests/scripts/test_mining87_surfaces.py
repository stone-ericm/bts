"""#87 adapter: served-slate JSON adaptation and independent-witness admission (review r1 findings 1, 9; change 5)."""
from __future__ import annotations

import hashlib
import json

import pytest

from scripts.audit.mining87 import surfaces as s

DATE = "2026-06-20"
PRIMARY = {"batter_id": 1, "game_pk": 100, "p_game_hit": 0.81234567891, "locked_at": "2026-06-20T17:00:00.000000Z"}


def slate_bytes(rows=None, date=DATE, schema="bts_slate_v1"):
    rows = rows if rows is not None else [
        {"batter_id": 1, "batter_name": "A", "game_pk": 100, "p_game_hit": 0.8123456789},   # writer precision
        {"batter_id": 2, "batter_name": "B", "game_pk": 101, "p_game_hit": 0.8},
        {"batter_id": 3, "batter_name": "C", "game_pk": 102, "p_game_hit": 0.79},
        {"batter_id": 3, "batter_name": "C", "game_pk": 103, "p_game_hit": 0.75},   # doubleheader second game
        {"batter_id": 4, "batter_name": "D", "game_pk": 104, "p_game_hit": None}]
    doc = {"schema_version": schema, "date": date, "tier": "x", "written_at": "2026-06-20T15:00:00+00:00",
           "n_rows": len(rows), "rows": rows}
    return json.dumps(doc).encode()


def witness(raw, **kw):
    w = {"date": DATE, "surface_sha256": hashlib.sha256(raw).hexdigest(), "independent_of_served_slate": True,
         "source": "pre-lock prediction log", "candidate_universe": "ref:universe",
         "lineup_assumptions": "ref:lineups", "feature_computation": "ref:features",
         "prediction_timestamp_utc": "2026-06-20T15:00:00+00:00"}
    w.update(kw)
    return w


def test_json_slate_adaptation_keeps_full_row_order_as_rank_without_any_outcome_column():
    parsed = s.parse_slate(slate_bytes(), expected_date=DATE)
    assert [r["rank"] for r in parsed["rows"]] == [1, 2, 3, 4, 5] and parsed["n_rows"] == 5
    batters = s.batter_table(parsed["rows"])
    assert batters[3]["rank"] == 3 and [g for _r, g, _p in batters[3]["rows"]] == [102, 103]
    assert parsed["order_matches_probability"] is True


@pytest.mark.parametrize("raw, match", [
    (slate_bytes(schema="bts_slate_v2"), "schema"),
    (slate_bytes(date="2026-06-21"), "date"),
    (slate_bytes(rows=[{"batter_id": 1, "game_pk": 100, "p_game_hit": 0.8}] * 2), "duplicate"),
    (slate_bytes(rows=[{"batter_id": "x", "game_pk": 100, "p_game_hit": 0.8}]), "batter_id"),
    (slate_bytes(rows=[{"batter_id": 1, "game_pk": 100, "p_game_hit": 1.4}]), "p_game_hit"),
    (b"not json", "json"),
])
def test_malformed_slates_are_rejected(raw, match):
    with pytest.raises(s.SlateFormatError, match=match):
        s.parse_slate(raw, expected_date=DATE)


def test_a_selection_consistent_served_slate_is_not_admitted_without_an_independent_witness():
    raw = slate_bytes()
    out = s.admit_surface(date=DATE, slate=s.parse_slate(raw, expected_date=DATE), slate_sha256=s.sha256(raw),
                          witnesses=[], production_primary=PRIMARY)
    assert out["selection_consistency"] == "selection_consistent"
    assert out["admitted"] is False and out["reason"] == "no_independent_witness"


@pytest.mark.parametrize("mutate, primary, reason", [
    (lambda raw: [witness(raw, surface_sha256="0" * 64)], PRIMARY, "witness_surface_hash_mismatch"),
    (lambda raw: [witness(raw, lineup_assumptions="")], PRIMARY, "witness_missing_lineup_assumptions"),
    (lambda raw: [witness(raw, feature_computation=None)], PRIMARY, "witness_missing_feature_computation"),
    (lambda raw: [witness(raw, independent_of_served_slate=False)], PRIMARY, "witness_not_independent"),
    (lambda raw: [witness(raw), witness(raw, source="other")], PRIMARY, "conflicting_witnesses"),
    (lambda raw: [witness(raw, prediction_timestamp_utc="2026-06-20T18:00:00+00:00")], PRIMARY,
     "prediction_after_lock"),
    (lambda raw: [witness(raw, prediction_timestamp_utc="2026-06-20T15:00:00")], PRIMARY,
     "witness_prediction_timestamp_invalid"),
    (lambda raw: [witness(raw)], {**PRIMARY, "locked_at": None}, "lock_time_unknown"),
    (lambda raw: [witness(raw)], None, "no_locked_production_primary"),
    (lambda raw: [witness(raw)], {**PRIMARY, "p_game_hit": 0.8123}, "selection_inconsistent"),
    (lambda raw: [witness(raw)], PRIMARY, "witness_admitted"),
])
def test_admission_needs_every_original_rule_component_bound_to_this_file_and_before_lock(mutate, primary, reason):
    raw = slate_bytes()
    out = s.admit_surface(date=DATE, slate=s.parse_slate(raw, expected_date=DATE), slate_sha256=s.sha256(raw),
                          witnesses=mutate(raw), production_primary=primary)
    assert out["reason"] == reason and out["admitted"] is (reason == "witness_admitted")


def test_missing_or_invalid_slate_files_are_explicit_non_admissions():
    assert s.admit_surface(date=DATE, slate=None, slate_sha256=None, witnesses=[], production_primary=PRIMARY)[
        "reason"] == "no_served_slate"
    bad = s.admit_surface(date=DATE, slate=s.SlateFormatError("schema x"), slate_sha256="ab", witnesses=[],
                          production_primary=PRIMARY)
    assert bad["admitted"] is False and bad["reason"].startswith("slate_format_invalid")


def test_consensus_fields_distinguish_missing_surface_off_ranking_and_ambiguous_probability_targets():
    batters = s.batter_table(s.parse_slate(slate_bytes(), expected_date=DATE)["rows"])
    assert s.consensus_surface(2, batters, admitted=False, unit_game=None)["status"] == "no_admitted_surface"
    off = s.consensus_surface(99, batters, admitted=True, unit_game=None)
    assert off["status"] == "off_surface" and off["rank"] is None and off["p_game_hit"] is None
    dh = s.consensus_surface(3, batters, admitted=True, unit_game=102)
    assert dh["status"] == "ambiguous_multiple_rows" and dh["rank"] == 3 and dh["p_game_hit"] is None
    assert dh["probability_target_bound"] is False
    unbound = s.consensus_surface(2, batters, admitted=True, unit_game=None)
    assert unbound["status"] == "unique_row" and unbound["p_game_hit"] == 0.8
    assert unbound["probability_target_bound"] is False
    bound = s.consensus_surface(2, batters, admitted=True, unit_game=101)
    assert bound["probability_target_bound"] is True
    other = s.consensus_surface(2, batters, admitted=True, unit_game=555)
    assert other["status"] == "row_game_differs_from_consensus_game" and other["p_game_hit"] is None


def test_witness_file_loader_groups_by_date_and_checks_the_schema(tmp_path):
    raw = slate_bytes()
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"schema": s.WITNESS_SCHEMA, "witnesses": [witness(raw)]}))
    got, meta = s.load_witnesses(path)
    assert list(got) == [DATE] and meta["n_records"] == 1
    assert s.load_witnesses(None) == ({}, {"source": "none_supplied", "n_records": 0})
    path.write_text(json.dumps({"schema": "other", "witnesses": []}))
    with pytest.raises(ValueError, match="schema"):
        s.load_witnesses(path)
