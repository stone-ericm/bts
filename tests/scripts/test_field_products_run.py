"""W2.1/W2.2 driver: the X-24/X-25 gate and one end-to-end run over synthetic inputs in the real on-disk formats."""
from __future__ import annotations

import json
from datetime import date, datetime

import pandas as pd
import pytest

from scripts.audit.field_products import run
from tests.scripts import test_field_products_builders as B


def test_gate_refuses_while_the_commits_are_unset(monkeypatch):
    monkeypatch.setattr(run, "X24_COMMIT", None)
    monkeypatch.setattr(run, "X25_COMMIT", "abc")
    with pytest.raises(SystemExit, match="X-24 gate"):
        run.gate()


def test_gate_refuses_a_commit_that_is_not_in_head(monkeypatch):
    monkeypatch.setattr(run, "X24_COMMIT", run.git("rev-parse", "HEAD"))
    monkeypatch.setattr(run, "X25_COMMIT", "0" * 40)
    with pytest.raises(SystemExit, match="X-25 gate: .* not an ancestor"):
        run.gate()


def test_gate_refuses_without_both_register_rows(monkeypatch, tmp_path):
    head = run.git("rev-parse", "HEAD")
    monkeypatch.setattr(run, "X24_COMMIT", head)
    monkeypatch.setattr(run, "X25_COMMIT", head)
    reg = tmp_path / "register.md"
    reg.write_text("| X-24 | something |\n")
    monkeypatch.setattr(run, "REGISTER", reg)
    with pytest.raises(SystemExit, match="X-25"):
        run.gate()
    reg.write_text("| X-24 | a |\n| X-25 | b |\n")
    assert run.gate() == head


def test_main_refuses_before_reading_anything_when_the_gate_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "X24_COMMIT", None)
    with pytest.raises(SystemExit):
        run.main(["--data-root", str(tmp_path / "nope"), "--ledger-dir", str(tmp_path / "nope"),
                  "--out", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


EARLY = [(1001, "alpha"), (2001, "b one"), (2002, "jordan"), (2003, "jordan"), (2004, "dee"), (2005, "eve")]
CAP1, CAP2 = datetime(2026, 5, 2, 12), datetime(2026, 7, 4, 12)


def _inputs(tmp_path):
    hits = B.profile([B.pred(1000, 1, [(1, 5, 3, "hit")]), B.pred(1001, 2, [(1, 5, 4, "hit"), (2, 5, 6, "hit")]),
                      B.pred(1002, 0, [(1, 5, 7, "not_hit")])])
    grab = B.make_grab(tmp_path, board_pages=B.pages(B.board_rows(420)), early=EARLY, cohort_a=2, cohort_b=3,
                       profiles={1000: (200, hits), 1001: (200, hits), 2001: (200, hits),
                                 2002: (200, B.profile([]))})
    up = tmp_path / "data" / "leaderboard" / "user_picks"
    up.mkdir(parents=True)
    d1, d2, d3 = date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3)
    p = B.daily_pick
    B.write_daily(up, "alpha", [[p(10, 1, cap=CAP1, pick_date=d1, streak=1)],
                                [p(10, 1, cap=CAP2, pick_date=d1, streak=1),
                                 p(11, 1, cap=CAP2, pick_date=d2, streak=3, player=21),
                                 p(11, 2, cap=CAP2, pick_date=d2, streak=3, player=22)]])
    B.write_daily(up, "b one", [[p(10, 1, cap=CAP1, pick_date=d1, result="not_hit", streak=0)]])
    B.write_daily(up, "b_one", [[p(10, 1, cap=CAP2, pick_date=d1, result="not_hit", streak=0),
                                 p(12, 1, cap=CAP2, pick_date=d3, streak=1)]])
    B.write_daily(up, "jordan", [[p(10, 1, cap=CAP2, pick_date=d1)]])
    B.write_daily(up, "dee", [[p(10, 1, cap=CAP1, pick_date=d1, player=20)], [p(10, 1, cap=CAP2, pick_date=d1,
                                                                                player=30)]])
    B.write_daily(up, "stranger", [[p(10, 1, cap=CAP2, pick_date=d1)]])
    led = tmp_path / "ledger"
    led.mkdir()
    sel = lambda sid, d, slot, g: {"row_id": sid, "row_kind": "selection", "date": d, "round_id": 1, "slot": slot,
                                   "selection_id": sid, "batter_id": 5, "game_pk": 9,
                                   "commit_status": "committed_evidenced", "entry_status": "confirmed",
                                   "match": "evidenced", "match_reason": "unit_capture", "unit_id": 1,
                                   "contest_slot_grade_raw": g, "contest_round_result": None,
                                   "streak_before": None, "streak_after": None}
    rows = [sel("s1", "2026-05-01", "primary", "hit"), sel("s2", "2026-05-02", "primary", "hit"),
            sel("s3", "2026-05-02", "double_down", "not_hit"), sel("s4", "2026-05-04", "primary", "hit")]
    pd.DataFrame(rows).to_parquet(led / "season_2026_ledger.parquet")
    pd.DataFrame([{"selection_id": r["selection_id"], "match": "evidenced", "slot_result": r["contest_slot_grade_raw"]}
                  for r in rows]).to_parquet(led / "season_2026_ledger_contest_slots.parquet")
    (led / "ACCEPTED.json").write_text("{}")
    return grab, led


def test_main_runs_end_to_end_on_synthetic_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "gate", lambda: "f" * 40)
    grab, led = _inputs(tmp_path)
    assert run.main(["--data-root", str(tmp_path / "data"), "--ledger-dir", str(led), "--out", str(tmp_path / "out"),
                     "--our-user-id", "1150", "--n-resamples", "60",
                     "--early-manifest", str(tmp_path / "early.json")]) == 0
    out = next((tmp_path / "out").glob("fffffff-*"))
    res = json.loads((out / "results.json").read_text())
    assert res["w21"]["census_gate"]["census"] is True
    sb = res["w21"]["season_best"]
    assert sb["basis"] == "census" and sb["own"]["stored_rank"] == 151 and sb["thresholds"]["ge_20"] == 110
    assert res["w21"]["case_series_A"]["n_A"] == 2 and res["w21"]["case_series_A"]["survivor_selected"] is True
    w22 = res["w22"]
    assert w22["manifest_E"]["n"] == 6 and w22["manifest_E"]["checks"]["manifest_sha_matches_grab"] is True
    assert w22["identity"]["members"] == {"bound": 3, "quarantined_manifest_collision": 2, "no_daily_file": 1}
    assert w22["identity"]["ownership_quarantined"] == [2004]
    assert w22["availability"]["daily_history"] == {"available": 2, "quarantined_manifest_collision": 2,
                                                    "quarantined_ownership_conflict": 1, "no_daily_file": 1}
    prim = w22["primary"]
    a, s = prim["tables"]["all_observed_dates"], prim["tables"]["shared_dates"]
    # E: alpha 5/01 hit + 5/02 DD hit,hit ; b one 5/01 miss + 5/03 hit -> 5 slots, 4 hits; ours 4 slots, 3 hits
    assert a["E"] == {"dates_with_slots": 3, "users": 2, "rounds": 4, "slots": 5, "hits": 4, "ratio": 0.8}
    assert a["ours"]["slots"] == 4 and a["dates"] == 4
    assert s["date_list"] == ["2026-05-01", "2026-05-02"] and s["E"]["slots"] == 4 and s["ours"]["slots"] == 3
    assert s["intervals"]["diff_ours_minus_E"]["n_resamples"] == 60
    assert prim["ours"]["included_by_slot"] == {"primary": 3, "double_down": 1}
    av = pd.read_parquet(out / "w22_availability.parquet").set_index("user_id")
    assert av.loc[1001, "allocation"] == "E_in_A" and av.loc[1001, "window_graded_slots"] == 3
    assert av.loc[1001, "window_streak_status"] in ("exact", "lower_bound_only")
    ext = w22["extension"]
    assert ext["coverage"]["usable"] == 2 and ext["coverage"]["budget_omissions"] == 2
    assert ext["pooled"] == {"users": 2, "dates": 3, "rounds": 6, "slots": 8, "hits": 6, "ratio": 0.75}  # 1001+2001
    assert prim["E_exclusions"] == {"window_slots": 5, "usable_graded": 5, "excluded_by_label": {},
                                    "excluded_by_status": {}}
    assert a["coverage"]["E_members"] == 6 and a["coverage"]["window_calendar_dates"] == 64
    man = json.loads((out / "manifest.json").read_text())
    assert any(k.endswith("alpha.parquet") for k in man["inputs"]) and "ledger/ACCEPTED.json" in man["inputs"]
    assert "jordan" not in " ".join(man["inputs"])          # quarantined files are listed, never read
    assert man["code"]["head"] == "f" * 40 and "census.py" in man["code"]["files"]
    freeze = json.loads((out / "freeze.json").read_text())
    assert freeze["written_before_outcome_reads"] is True and freeze["binding_sha256"]
    assert (out / "w22_daily_slots.parquet").exists() and (out / "w21_distribution.parquet").exists()


def test_freeze_is_written_before_any_outcome_bearing_read(tmp_path, monkeypatch):
    """The board receipts are the first outcome-bearing read: when that step fails, freeze.json already exists and no
    pick file, ledger row or result has been read or written."""
    monkeypatch.setattr(run, "gate", lambda: "e" * 40)
    grab, led = _inputs(tmp_path)

    def boom(*a, **k):
        raise RuntimeError("outcome read attempted")
    monkeypatch.setattr(run.C, "load_board_receipts", boom)
    monkeypatch.setattr(run.P, "read_observations", boom)
    with pytest.raises(RuntimeError, match="outcome read attempted"):
        run.main(["--data-root", str(tmp_path / "data"), "--ledger-dir", str(led), "--out", str(tmp_path / "out"),
                  "--early-manifest", str(tmp_path / "early.json")])
    out = next((tmp_path / "out").glob("eeeeeee-*"))
    freeze = json.loads((out / "freeze.json").read_text())
    assert freeze["binding_counts"]["members"]["bound"] == 3 and not (out / "results.json").exists()


def test_manifest_hashes_are_the_bytes_on_disk_and_nested_files_are_counted(tmp_path, monkeypatch):
    import hashlib
    monkeypatch.setattr(run, "gate", lambda: "d" * 40)
    grab, led = _inputs(tmp_path)
    nested = tmp_path / "data" / "leaderboard" / "user_picks" / "a"
    nested.mkdir()
    B.write_daily(nested, "b", [[B.daily_pick(10, 1, cap=CAP1, pick_date=date(2026, 5, 1))]])
    assert run.main(["--data-root", str(tmp_path / "data"), "--ledger-dir", str(led), "--out", str(tmp_path / "out"),
                     "--n-resamples", "20", "--early-manifest", str(tmp_path / "early.json")]) == 0
    out = next((tmp_path / "out").glob("ddddddd-*"))
    man = json.loads((out / "manifest.json").read_text())
    assert man["inputs"]["ledger/season_2026_ledger.parquet"] == hashlib.sha256(
        (led / "season_2026_ledger.parquet").read_bytes()).hexdigest()
    assert man["inputs"]["leaderboard/user_picks/alpha.parquet"] == hashlib.sha256(
        (tmp_path / "data" / "leaderboard" / "user_picks" / "alpha.parquet").read_bytes()).hexdigest()
    freeze = json.loads((out / "freeze.json").read_text())
    assert freeze["daily_nested_files_ignored"] == ["a/b.parquet"]
    res = json.loads((out / "results.json").read_text())
    assert res["w21"]["season_best"]["own"]["matched_by_user_id"] is False      # no --our-user-id given
