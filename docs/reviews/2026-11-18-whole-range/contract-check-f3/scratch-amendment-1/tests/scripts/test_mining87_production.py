"""#87 adapter: the W1.1 ledger as the audited locked-slot projection (review r1 finding 2, required changes 2, 4, 11)."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.audit.mining87 import production as p
from scripts.audit.season_ledger.compile import CONTEST_SCHEMA, LEDGER_SCHEMA
from scripts.audit.season_ledger.io import build_table, write_table

FP = "5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f"


def sel(date, slot, batter, game, **kw):
    """A committed, contest-graded selection row (the defaults are the settled-hit happy path)."""
    sid = f"{date}|{slot}|{batter}|{game}"
    row = {"row_id": sid, "row_kind": "selection", "date": date, "slot": slot, "selection_id": sid,
           "batter_id": batter, "batter_name": f"B{batter}", "game_pk": game, "p_stated": 0.77,
           "projected_lineup": False, "finalization": "decision", "commit_status": "committed_evidenced",
           "commit_basis": "decision:delivered", "predicted_at": f"{date}T15:00:00.000000Z",
           "locked_at": f"{date}T16:00:00.000000Z", "delivery_confirmed": True, "entry_status": "confirmed",
           "match": "evidenced", "match_reason": "unit_capture", "round_id": 1,
           "unit_id": None if game is None else game * 10 + batter,
           "player_id": batter + 9000, "bts_outcome": "hit", "bts_outcome_status": "graded",
           "contest_slot_grade_raw": "hit", "game_eligibility": "unknown"}
    row.update(kw)
    return row


def day(date, kind, reason):
    return {"row_id": f"day|{date}|{kind}", "row_kind": kind, "date": date, "reason": reason}


def contest_for(rows, **overrides):
    out = []
    for r in rows:
        if r.get("row_kind") == "selection" and r.get("entry_status") == "confirmed":
            c = {"round_id": r["round_id"], "unit_id": r["unit_id"], "player_id": r["player_id"], "date": r["date"],
                 "batter_id": r["batter_id"], "game_pk": r["game_pk"], "selection_id": r["selection_id"],
                 "match": r["match"], "match_reason": r["match_reason"], "slot_result": r.get("bts_outcome")}
            c.update(overrides.get(r["selection_id"], {}))
            out.append(c)
    return out


def frames(rows, contest=None):
    ledger = build_table(rows, LEDGER_SCHEMA, sort_keys=["row_id"], name="ledger").to_pandas()
    slots = build_table(contest if contest is not None else contest_for(rows), CONTEST_SCHEMA,
                        sort_keys=["round_id", "unit_id", "player_id"], name="contest").to_pandas()
    return ledger, slots


def project(rows, contest=None, start="2026-03-26", end="2026-07-03"):
    ledger, slots = frames(rows, contest)
    return p.project_locked_slots(ledger, slots, window_start=start, window_end=end)


def test_only_committed_finalized_selections_are_locked_units_and_the_rest_are_counted():
    rows = [sel("2026-04-01", "primary", 1, 100),
            sel("2026-04-01", "double_down", 2, 101, commit_status="unconfirmed", locked_at=None,
                entry_status="unknown", bts_outcome=None, bts_outcome_status="unknown"),            # preview
            sel("2026-04-02", "primary", 3, 102, commit_status="conflicted", locked_at=None),
            sel("2026-04-03", "primary", 4, 103, finalization="unresolved", locked_at=None),
            sel("2026-04-04", "primary", 5, 104, finalization="pick_file_only", locked_at=None),   # lock time unknown
            day("2026-04-05", "skip_day", "decision_skip"),
            day("2026-04-06", "unfinalized_day", "commit_flag_without_record")]
    locked, inv = project(rows)
    assert list(zip(locked["date"], locked["pick_number"])) == [("2026-04-01", 1), ("2026-04-04", 1)]
    assert inv["lock_unsupported_selections"] == {"commit_unconfirmed": 1, "commit_conflicted": 1,
                                                  "finalization_unresolved": 1}
    assert inv["day_rows_without_units"] == {"skip_day:decision_skip": 1,
                                             "unfinalized_day:commit_flag_without_record": 1}
    assert inv["locked_units"] == 2 and inv["lock_time_known"] == 1


def test_void_with_a_populated_hit_field_is_void_and_unresolved_slots_stay_units():
    rows = [sel("2026-04-01", "primary", 1, 100, bts_outcome="void", contest_slot_grade_raw="void",
                local_slot_result_raw="hit", local_norm="HIT", contest_norm="HOLD",
                local_vs_contest_disagreement=True),
            sel("2026-04-02", "primary", 2, 101, bts_outcome=None, bts_outcome_status="matched_ungraded"),
            sel("2026-04-03", "primary", 3, 102, entry_status="unknown", match=None, match_reason=None,
                bts_outcome=None, bts_outcome_status="unknown"),
            sel("2026-04-04", "primary", 4, 103, entry_status="unknown", match=None, match_reason=None,
                bts_outcome=None, bts_outcome_status="match_ambiguous"),
            sel("2026-04-05", "primary", 5, 104, bts_outcome="weird", contest_slot_grade_raw="weird"),
            sel("2026-04-06", "primary", 6, 105, bts_outcome="not_hit")]
    locked, inv = project(rows)
    got = dict(zip(locked["date"], zip(locked["production_settlement"], locked["production_settlement_reason"])))
    assert got == {"2026-04-01": ("void", "contest_void"), "2026-04-02": ("ungraded", "contest_slot_not_graded"),
                   "2026-04-03": ("unknown", "no_linked_contest_slot"),
                   "2026-04-04": ("unknown", "no_linked_contest_slot"),
                   "2026-04-05": ("unknown", "unrecognized_grade"), "2026-04-06": ("not_hit", "contest_graded")}
    assert list(locked["production_resolved"]) == [False, False, False, False, False, True]
    assert locked["production_hit"].isna().sum() == 5 and locked["production_hit"].dropna().tolist() == [0]
    assert inv["settlement_status"] == {"resolved": 1, "ungraded": 1, "unknown": 3, "void": 1}


def test_a_contest_slot_bound_to_another_game_never_settles_the_selected_game():
    rows = [sel("2026-04-01", "primary", 1, 100), sel("2026-04-02", "primary", 1, 200),
            sel("2026-04-03", "primary", 1, 300, match_reason="unit_capture_other_game")]
    contest = contest_for(rows, **{"2026-04-01|primary|1|100": {"game_pk": 101}})        # doubleheader game 2
    contest = [c for c in contest if c["selection_id"] != "2026-04-02|primary|1|200"]     # slot row missing
    locked, _ = project(rows, contest)
    reasons = dict(zip(locked["date"], locked["production_settlement_reason"]))
    assert reasons == {"2026-04-01": "contest_slot_game_mismatch", "2026-04-02": "contest_slot_missing",
                       "2026-04-03": "link_without_game_identity"}
    assert not locked["production_resolved"].any()


def test_inferred_single_scheduled_game_links_settle_and_no_dd_unit_exists_without_a_production_dd():
    rows = [sel("2026-04-01", "primary", 1, 100, match="inferred",
                match_reason="pick_time_team_single_scheduled_game")]
    locked, _ = project(rows)
    assert len(locked) == 1 and locked.iloc[0]["production_settlement"] == "hit"
    assert locked.iloc[0]["production_hit"] == 1


def test_window_bounds_are_inclusive_and_duplicate_locked_slots_raise():
    rows = [sel("2026-03-25", "primary", 1, 100), sel("2026-03-26", "primary", 2, 101),
            sel("2026-07-03", "double_down", 3, 102), sel("2026-07-04", "primary", 4, 103)]
    locked, inv = project(rows)
    assert list(locked["date"]) == ["2026-03-26", "2026-07-03"] and inv["ledger_rows_outside_window"] == 2
    dup = [sel("2026-04-01", "primary", 1, 100), sel("2026-04-01", "primary", 2, 101)]
    with pytest.raises(ValueError, match="duplicate locked production slot"):
        project(dup)


def test_unexpected_structural_vocabulary_raises():
    with pytest.raises(ValueError, match="commit_status"):
        project([sel("2026-04-01", "primary", 1, 100, commit_status="maybe")])
    with pytest.raises(ValueError, match="bts_outcome_status"):
        project([sel("2026-04-01", "primary", 1, 100, bts_outcome_status="graded_twice")])


def test_regime_is_assigned_from_the_prediction_time():
    assert p.production_regime("2026-04-30T16:27:00.000000Z") == "post_bpm"
    assert p.production_regime("2026-04-30T16:26:59.999999Z") == "post_pooled_mdp_pre_bpm"
    assert p.production_regime("2026-04-01T12:00:00.000000Z") == "pre_pooled_mdp"
    assert p.production_regime(None) is None


def test_context_attaches_only_on_the_exact_selection_identity_and_duplicates_raise():
    locked, _ = project([sel("2026-04-01", "primary", 1, 100), sel("2026-04-01", "double_down", 2, 101)])
    ctx = pd.DataFrame([{"date": "2026-04-01", "slot": "primary", "batter_id": 1, "game_pk": 100,
                         "batter_skill_prior_pa": 350, "batter_skill_quartile": 2, "pick_weather_temp": 71.0,
                         "pick_is_indoor": False, "is_park_driven": False, "actual_hit": True},
                        {"date": "2026-04-01", "slot": "double_down", "batter_id": 2, "game_pk": 999,
                         "batter_skill_prior_pa": 90, "batter_skill_quartile": 4, "pick_weather_temp": None,
                         "pick_is_indoor": True, "is_park_driven": False, "actual_hit": False}])
    out, meta = p.attach_production_context(locked, ctx)
    first, second = out.iloc[0], out.iloc[1]
    assert first["production_batter_skill_prior_pa"] == 350 and first["production_weather_temp"] == 71.0
    assert pd.isna(second["production_batter_skill_prior_pa"]) and meta["rows_matched"] == 1
    assert "actual_hit" not in out.columns
    with pytest.raises(ValueError, match="duplicate"):
        p.attach_production_context(locked, pd.concat([ctx, ctx]))
    null_game, _ = project([sel("2026-04-01", "primary", 1, None, selection_id="2026-04-01|primary|1|None",
                                entry_status="unknown", match=None, match_reason=None, bts_outcome=None,
                                bts_outcome_status="unknown")])
    ctx_null = ctx.iloc[:1].assign(game_pk=None)
    joined, meta3 = p.attach_production_context(null_game, ctx_null)            # a null key is not an identity
    assert meta3["rows_matched"] == 0 and meta3["rows_dropped_incomplete_key"] == 1
    assert pd.isna(joined.iloc[0]["production_batter_skill_prior_pa"])
    bare, meta2 = p.attach_production_context(locked, None)
    assert bare["production_batter_skill_quartile"].isna().all() and meta2["source"] == "unavailable"


def _write_build(tmp_path, rows, *, fingerprint=FP, files=None, drop_col=None):
    d = tmp_path / "ledger"
    d.mkdir(parents=True)
    table = build_table(rows, LEDGER_SCHEMA, sort_keys=["row_id"], name="l")
    if drop_col:
        table = table.select([n for n in table.column_names if n != drop_col])
    write_table(table, d / p.LEDGER_FILE)
    write_table(build_table(contest_for(rows), CONTEST_SCHEMA, sort_keys=["round_id", "unit_id", "player_id"],
                            name="c"), d / p.CONTEST_FILE)
    receipt = {"run": "r", "files": files or sorted([p.LEDGER_FILE, p.CONTEST_FILE]), "rules_fingerprint": fingerprint}
    (d / "ACCEPTED.json").write_text(json.dumps(receipt))
    return d


def test_accepted_ledger_loader_requires_the_receipt_fingerprint_and_exact_schema(tmp_path):
    rows = [sel("2026-04-01", "primary", 1, 100)]
    ledger, slots, meta = p.load_accepted_ledger(_write_build(tmp_path / "ok", rows))
    assert len(ledger) == 1 and len(slots) == 1 and meta["rules_fingerprint"] == FP
    assert "streak_after" not in ledger.columns and "local_slot_result_raw" not in ledger.columns
    with pytest.raises(ValueError, match="fingerprint"):
        p.load_accepted_ledger(_write_build(tmp_path / "fp", rows, fingerprint="0" * 64))
    with pytest.raises(ValueError, match="names"):
        p.load_accepted_ledger(_write_build(tmp_path / "files", rows, files=["other.parquet"]))
    with pytest.raises(ValueError, match="schema"):
        p.load_accepted_ledger(_write_build(tmp_path / "schema", rows, drop_col="match_reason"))
    missing = tmp_path / "missing"
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match="ACCEPTED"):
        p.load_accepted_ledger(missing)


def test_the_upstream_build_manifest_and_the_projection_hash_are_recorded(tmp_path):
    rows = [sel("2026-04-01", "primary", 1, 100), sel("2026-04-02", "primary", 2, 101, bts_outcome="not_hit")]
    d = _write_build(tmp_path, rows)
    assert p.load_accepted_ledger(d)[2]["upstream_build"] == {"status": "missing"}
    (d / p.BUILD_FILE).write_text(json.dumps({"builder_version": "season-ledger-phase1/3", "code_sha": "abc",
                                              "bundle_manifest_sha256": "f" * 64, "rules_fingerprint": FP,
                                              "bundle_acquired_at_utc": "2026-09-29T02:27:33Z", "recipes": []}))
    up = p.load_accepted_ledger(d)[2]["upstream_build"]
    assert up["bundle_manifest_sha256"] == "f" * 64 and up["code_sha"] == "abc" and "recipes" not in up
    (d / p.BUILD_FILE).write_text(json.dumps({"rules_fingerprint": "0" * 64}))
    with pytest.raises(ValueError, match="build.json"):
        p.load_accepted_ledger(d)
    _, inv = project(rows)
    _, again = project(list(reversed(rows)))
    assert len(inv["projection_sha256"]) == 64 and inv["projection_sha256"] == again["projection_sha256"]
    _, changed = project([rows[0], sel("2026-04-02", "primary", 2, 101, bts_outcome="hit")])
    assert changed["projection_sha256"] != inv["projection_sha256"]
