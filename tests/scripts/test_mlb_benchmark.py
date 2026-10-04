"""W2.3 MLB forecast benchmark: as-of join core (design rev 2, gate 1)."""
import numpy as np
import pandas as pd

from scripts.audit.mlb_benchmark import core
from scripts.audit.mlb_benchmark import metrics as m


def _caps():
    return [
        ("2026-07-05T14:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.70, "numberSelections": 5},
                                  {"roundId": 11, "playerId": 1, "probabilityStarter": 0.60, "numberSelections": 1}]),
        ("2026-07-05T17:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.72, "numberSelections": 9},
                                  {"roundId": 10, "playerId": 2, "probabilityStarter": 0.65, "numberSelections": 3}]),
        ("2026-07-05T20:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.75, "numberSelections": 12}]),
    ]


ROUNDS = {10: "2026-07-05", 11: "2026-07-06"}
PLAYERS = {1: 701, 2: 702}


def test_asof_uses_the_latest_whole_sheet_at_or_before_the_cutoff_and_never_tomorrows_round():
    fc = core.forecasts_asof(_caps(), ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T18:00:00Z"))
    assert fc[701]["p"] == 0.72 and fc[701]["captured_at"] == "2026-07-05T17:00:00Z" and fc[702]["p"] == 0.65
    early = core.forecasts_asof(_caps(), ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T15:00:00Z"))
    assert early == {701: {"p": 0.70, "n_sel": 5, "captured_at": "2026-07-05T14:00:00Z", "round_id": 10, "player_id": 1}}


def test_a_player_absent_from_a_newer_stored_sheet_is_absent_not_resurrected():
    fc = core.forecasts_asof(_caps(), ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T21:00:00Z"))
    assert set(fc) == {701} and fc[701]["p"] == 0.75


def test_games_come_from_mlbs_own_sheets_never_from_our_slate():
    fc = {701: {"player_id": 1}, 702: {"player_id": 2}, 703: {"player_id": 3}, 704: {"player_id": 4}}
    squads = {1: 11, 2: 12, 3: None, 4: 14}
    units = [{"feedId": 100, "roundId": 10, "homeSquadId": 11, "awaySquadId": 21},
             {"feedId": 200, "roundId": 10, "homeSquadId": 12, "awaySquadId": 22},
             {"feedId": 201, "roundId": 10, "homeSquadId": 22, "awaySquadId": 12},
             {"feedId": 300, "roundId": 10, "homeSquadId": 13, "awaySquadId": 23}]
    assert core.games_by_batter(fc, squads, units, round_id=10) == {701: {100}, 702: {200, 201}, 703: set(), 704: set()}


def test_join_links_only_a_unique_game_and_labels_it_inferred():
    slate = pd.DataFrame({"batter_id": [701, 702, 703, 704, 705], "game_pk": [100, 200, 300, 400, 500],
                          "D": [0.8, 0.7, 0.6, 0.5, 0.4]})
    fc = {b: {"p": 0.5, "captured_at": "x"} for b in (701, 702, 704, 705, 799)}
    games = {701: {100}, 702: {200, 201}, 704: {401}, 705: set(), 799: {900}}   # 702 doubleheader; 704 mismatch
    joined, cov = core.join_to_slate(slate, fc, games)
    assert list(joined["link_status"]) == ["inferred_unique_game", "multi_or_no_game", "not_listed",
                                           "game_mismatch", "multi_or_no_game"]
    assert joined["mlb_p"].notna().sum() == 1 and cov["mlb_not_in_slate"] == 1


def test_a_newer_sheet_without_the_round_or_empty_is_no_support_never_a_fallback():
    caps = _caps()[:2] + [("2026-07-05T20:00:00Z", [{"roundId": 11, "playerId": 1, "probabilityStarter": 0.6}])]
    stamp, rows = core.latest_sheet(caps, ROUNDS, "2026-07-05", pd.Timestamp("2026-07-05T21:00:00Z"))
    assert stamp == "2026-07-05T20:00:00Z" and rows == []
    assert core.forecasts_asof(caps, ROUNDS, PLAYERS, "2026-07-05", pd.Timestamp("2026-07-05T21:00:00Z")) == {}
    empty = _caps()[:2] + [("2026-07-05T20:00:00Z", [])]
    assert core.latest_sheet(empty, ROUNDS, "2026-07-05", pd.Timestamp("2026-07-05T21:00:00Z")) == ("2026-07-05T20:00:00Z", [])
    assert core.latest_sheet(_caps(), ROUNDS, "2026-07-05", pd.Timestamp("2026-07-05T13:00:00Z")) == (None, [])


def test_an_unresolved_unit_makes_multiplicity_unknown_and_other_rounds_are_ignored():
    fc = {701: {"player_id": 1}, 702: {"player_id": 2}}
    squads = {1: 11, 2: 12}
    units = [{"feedId": 100, "roundId": 10, "homeSquadId": 11, "awaySquadId": 21},
             {"feedId": None, "roundId": 10, "homeSquadId": 11, "awaySquadId": 22},   # 701's possible second game
             {"feedId": 200, "roundId": 10, "homeSquadId": 12, "awaySquadId": 23},
             {"feedId": 201, "roundId": 11, "homeSquadId": 12, "awaySquadId": 24}]   # tomorrow: not today's game
    assert core.games_by_batter(fc, squads, units, round_id=10) == {701: None, 702: {200}}
    unknown_squad = units[:1] + [{"feedId": 300, "roundId": 10, "homeSquadId": None, "awaySquadId": 25}]
    assert core.games_by_batter(fc, squads, unknown_squad, round_id=10) == {701: None, 702: None}
    no_round = units[:1] + [{"feedId": 400, "roundId": None, "homeSquadId": 30, "awaySquadId": 31}]
    assert core.games_by_batter(fc, squads, no_round, round_id=10) == {701: None, 702: None}
    slate = pd.DataFrame({"batter_id": [701, 702], "game_pk": [100, 200], "D": [0.8, 0.7]})
    fc2 = {701: {"p": 0.5, "captured_at": "x"}, 702: {"p": 0.6, "captured_at": "x"}}
    joined, _ = core.join_to_slate(slate, fc2, {701: None, 702: {200}})
    assert list(joined["link_status"]) == ["multiplicity_unknown", "inferred_unique_game"]


def test_round_resolution_needs_exactly_one_round_for_the_date():
    rounds = [{"id": 10, "date": "2026-07-05T00:00:00Z"}, {"id": 11, "date": "2026-07-06T00:00:00Z"},
              {"id": 12, "date": "2026-07-06T00:00:00Z"}]
    assert core.round_for_date(rounds, "2026-07-05") == (10, None)
    assert core.round_for_date(rounds, "2026-07-06") == (None, "multiple_rounds")
    assert core.round_for_date(rounds, "2026-07-07") == (None, "no_round")


def test_player_lookup_excludes_ids_listed_twice_with_conflicting_values():
    players = [{"id": 1, "feedId": 701, "squadId": 11}, {"id": 2, "feedId": 702, "squadId": 12},
               {"id": 2, "feedId": 799, "squadId": 12}, {"id": 3, "feedId": 703, "squadId": 13},
               {"id": 3, "feedId": 703, "squadId": 13}]
    feed, squad, conflicts = core.player_lookup(players)
    assert feed == {1: 701, 3: 703} and squad == {1: 11, 3: 13} and conflicts == [2]


def test_units_are_complete_only_when_every_scheduled_game_has_a_unit_in_the_round():
    units = [{"feedId": 100, "roundId": 10}, {"feedId": 200, "roundId": 10}, {"feedId": 300, "roundId": 11}]
    assert core.units_complete(units, 10, {100, 200}) is True
    assert core.units_complete(units, 10, {100, 200, 250}) is False
    assert core.units_complete(units, 10, set()) is False        # no schedule evidence: completeness not established


def test_unchanged_since_walks_back_through_stored_sheets_while_the_value_is_identical():
    caps = [("2026-07-05T10:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.70}]),
            ("2026-07-05T12:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.72}]),
            ("2026-07-05T14:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.72}]),
            ("2026-07-05T16:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.72}])]
    assert core.unchanged_since(caps, "2026-07-05T16:00:00Z", 10, 1) == "2026-07-05T12:00:00Z"
    gap = caps[:2] + [("2026-07-05T14:00:00Z", [])] + caps[3:]   # absent in between: the run stops there
    assert core.unchanged_since(gap, "2026-07-05T16:00:00Z", 10, 1) == "2026-07-05T16:00:00Z"


def test_prepare_date_links_through_the_sheets_at_the_forecast_stamp_and_counts_every_exclusion():
    from scripts.audit.mlb_benchmark import run
    msp = [("2026-07-05T14:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.70, "numberSelections": 9},
                                     {"roundId": 10, "playerId": 2, "probabilityStarter": 1.70, "numberSelections": 1},
                                     {"roundId": 10, "playerId": 3, "probabilityStarter": 0.60, "numberSelections": 2}])]
    rounds = {"2026-07-05T13:00:00Z": [{"id": 10, "date": "2026-07-05T00:00:00Z"}]}
    players = {"2026-07-05T13:30:00Z": [{"id": 1, "feedId": 701, "squadId": 11}, {"id": 2, "feedId": 702, "squadId": 12},
                                        {"id": 3, "feedId": 703, "squadId": 13}],
               "2026-07-05T15:00:00Z": [{"id": 1, "feedId": 701, "squadId": 99}]}   # after the stamp: never used
    units = {"2026-07-05T13:45:00Z": [{"feedId": 100, "roundId": 10, "homeSquadId": 11, "awaySquadId": 21},
                                      {"feedId": 300, "roundId": 10, "homeSquadId": 13, "awaySquadId": 23}]}
    slate = pd.DataFrame({"date": ["2026-07-05"] * 3, "row_order": [0, 1, 2], "batter_id": [701, 703, 704],
                          "game_pk": [100, 300, 400], "D": [0.8, 0.7, 0.6]})
    out, cov = run.prepare_date("2026-07-05", pd.Timestamp("2026-07-05T16:00:00Z"), slate, msp,
                                lambda feed, stamp: (*run.latest_at(
                                    {"rounds": rounds, "players": players, "units": units}[feed], stamp), "sha"),
                                sched_pks={100, 300})
    assert list(out["link_status"]) == ["inferred_unique_game", "inferred_unique_game", "not_listed"]
    assert cov["forecast_counts"]["invalid_probability"] == 1 and cov["forecast_stamp"] == "2026-07-05T14:00:00Z"
    assert out.loc[0, "mlb_player_id"] == 1 and out.loc[0, "mlb_squad_id"] == 11 and out.loc[0, "mlb_round_id"] == 10
    assert out.loc[0, "src_players"] == "2026-07-05T13:30:00Z:sha" and pd.isna(out.loc[2, "mlb_player_id"])
    assert list(out["mlb_n_sel"].fillna(-1)) == [9, 2, -1]
    assert out.loc[0, "mlb_unchanged_since"] == "2026-07-05T14:00:00Z"
    incomplete, cov2 = run.prepare_date("2026-07-05", pd.Timestamp("2026-07-05T16:00:00Z"), slate, msp,
                                        lambda feed, stamp: (*run.latest_at(
                                            {"rounds": rounds, "players": players, "units": units}[feed], stamp), "sha"),
                                        sched_pks={100, 300, 500})
    assert set(incomplete["link_status"]) == {"multiplicity_unknown", "not_listed"} and cov2["units_complete"] is False
    none, cov3 = run.prepare_date("2026-07-05", pd.Timestamp("2026-07-05T13:00:00Z"), slate, msp,
                                  lambda feed, stamp: (*run.latest_at(
                                      {"rounds": rounds, "players": players, "units": units}[feed], stamp), "sha"),
                                  sched_pks={100, 300})
    assert cov3["excluded"] == "no_forecast_sheet" and none.empty


def _accept(w12):
    import hashlib
    import json as _json
    if not (w12 / "manifest.json").exists():
        (w12 / "manifest.json").write_text("{}")
    files = {n: hashlib.sha256((w12 / n).read_bytes()).hexdigest() for n in ("table.parquet", "summary.json", "manifest.json")}
    (w12 / "ACCEPTED.json").write_text(_json.dumps({"files": files, "memo": "synthetic"}))


def test_main_runs_end_to_end_on_synthetic_inputs(tmp_path, monkeypatch):
    import gzip
    import json as _json
    from scripts.audit.mlb_benchmark import run
    monkeypatch.setattr(run, "x23_gate", lambda: "f" * 40)
    rng = np.random.default_rng(5)
    dates = [f"2026-07-{d:02d}" for d in range(5, 15)]
    rows, day_meta, msp, units, players = [], [], {}, {}, []
    for k, d in enumerate(dates):
        rid, stamp = 100 + k, f"{d.replace('-', '')}T150000Z"
        day_meta.append({"date": d, "written_at": f"{d}T16:00:00+00:00"})
        sheet = []
        for j in range(6):
            bid, pid, gpk = 700 + j, j + 1, 9000 + 10 * k + j
            rows.append({"date": d, "row_order": j, "batter_id": bid, "game_pk": gpk, "D": float(rng.uniform(0.6, 0.85)),
                         "sel_state": "selection_consistent", "pool_verified": True, "pool_surrogate": True,
                         "pool_all": True, "outcome": ["hit", "no_hit", "no_pa", "hit", "hit", "no_hit"][(j + k) % 6]})
            sheet.append({"roundId": rid, "playerId": pid, "probabilityStarter": float(rng.uniform(0.5, 0.8)),
                          "numberSelections": 10 + j})
        msp[stamp] = sheet
        units[stamp] = [{"feedId": 9000 + 10 * k + j, "roundId": rid, "homeSquadId": j + 1, "awaySquadId": 50 + j}
                        for j in range(6)]
        sched = {"dates": [{"date": d, "games": [{"gamePk": 9000 + 10 * k + j, "status": {},
                                                   "teams": {"home": {"team": {"abbreviation": f"H{j}"}},
                                                             "away": {"team": {"abbreviation": f"A{j}"}}}}
                                                  for j in range(6)]}]}
        (tmp_path / "sched").mkdir(exist_ok=True)
        (tmp_path / "sched" / f"{d}.json").write_text(_json.dumps(sched))
    w12 = tmp_path / "w12"; w12.mkdir()
    pd.DataFrame(rows).to_parquet(w12 / "table.parquet")
    (w12 / "summary.json").write_text(_json.dumps({"day_meta": day_meta}))
    _accept(w12)
    snaps = tmp_path / "data" / "leaderboard" / "static_snapshots"
    feeds = {"most_selected_players": ("mostSelectedPlayers", msp), "units": ("units", units),
             "rounds": ("rounds", {"20260701T000000Z": [{"id": 100 + k, "date": f"{d}T00:00:00Z"}
                                                         for k, d in enumerate(dates)]}),
             "players": ("players", {"20260701T000000Z": [{"id": j + 1, "feedId": 700 + j, "squadId": j + 1}
                                                          for j in range(6)]})}
    for feed, (key, sheets) in feeds.items():
        (snaps / feed).mkdir(parents=True)
        for stamp, items in sheets.items():
            (snaps / feed / f"{stamp}.json.gz").write_bytes(gzip.compress(_json.dumps({key: items}).encode()))
    assert run.main(["--w12-run", str(w12), "--data-root", str(tmp_path / "data"), "--schedules",
                     str(tmp_path / "sched"), "--out", str(tmp_path / "out"), "--n-resamples", "50"]) == 0
    res = _json.loads(next((tmp_path / "out").glob("*/results.json")).read_text())
    prim = res["strata"]["selection_consistent/pool_verified"]
    assert prim["T2"]["available"] and prim["T2"]["n_dates"] == 10
    assert res["coverage"]["link_status"] == {"inferred_unique_game": 60}
    assert prim["encompassing_T2"]["point"]["available"] in (True, False)
    assert res["recency"]["sheet_to_written_at_minutes"]["mean"] == 60.0


# --- code review r1 (F1, F3, F9) -------------------------------------------------------------------------------------

def test_a_round_id_mapped_to_two_dates_is_a_conflict_not_a_resolution():
    rounds = [{"id": 100, "date": "2026-07-05T00:00:00Z"}, {"id": 100, "date": "2026-07-06T00:00:00Z"}]
    assert core.round_for_date(rounds, "2026-07-05") == (None, "conflicting_round")


def test_contradictory_unit_rows_make_the_round_conflicted():
    units = [{"id": 99, "feedId": 9000, "roundId": 10, "homeSquadId": 1, "awaySquadId": 2},
             {"id": 99, "feedId": 9001, "roundId": 10, "homeSquadId": 3, "awaySquadId": 4}]
    assert core.unit_conflicts(units, 10) == 1
    same = [units[0], dict(units[0])]
    assert core.unit_conflicts(same, 10) == 0
    game_twice = [{"id": 1, "feedId": 9000, "roundId": 10, "homeSquadId": 1, "awaySquadId": 2},
                  {"id": 2, "feedId": 9000, "roundId": 10, "homeSquadId": 5, "awaySquadId": 6}]
    assert core.unit_conflicts(game_twice, 10) == 1


def test_duplicates_are_resolved_on_the_raw_sheet_before_probability_validity():
    caps = [("2026-07-05T14:00:00Z", [{"roundId": 10, "playerId": 1, "probabilityStarter": 0.6},
                                      {"roundId": 10, "playerId": 1, "probabilityStarter": 1.2},
                                      {"roundId": 10, "playerId": 2, "probabilityStarter": 1.5},
                                      {"roundId": 10, "playerId": 3, "probabilityStarter": 0.7}])]
    fc, counts = core.forecasts_counted(caps, ROUNDS, {1: 701, 2: 702, 3: 703}, "2026-07-05",
                                        pd.Timestamp("2026-07-05T15:00:00Z"))
    assert set(fc) == {703} and counts["duplicate_player"] == 1 and counts["invalid_probability"] == 1


def test_schema_validation_separates_a_valid_empty_sheet_from_an_invalid_one():
    from scripts.audit.mlb_benchmark import run
    assert run.validate_sheet({"mostSelectedPlayers": []}, "mostSelectedPlayers") == []
    assert run.validate_sheet({"unexpected": []}, "mostSelectedPlayers") is None
    assert run.validate_sheet({"mostSelectedPlayers": {"a": 1}}, "mostSelectedPlayers") is None
    assert run.validate_sheet({"mostSelectedPlayers": [1, 2]}, "mostSelectedPlayers") is None


def _synthetic_inputs(tmp_path, msp_items=None, extra_msp=None):
    import gzip
    import json as _json
    d = "2026-07-05"
    rows = [{"date": d, "row_order": j, "batter_id": 700 + j, "game_pk": 9000 + j, "D": 0.7,
             "sel_state": "selection_consistent", "pool_verified": True, "pool_surrogate": True, "pool_all": True,
             "outcome": "no_pa"} for j in range(2)]
    w12 = tmp_path / "w12"; w12.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(w12 / "table.parquet")
    (w12 / "summary.json").write_text(_json.dumps({"day_meta": [{"date": d, "written_at": f"{d}T16:00:00+00:00"}]}))
    _accept(w12)
    snaps = tmp_path / "data" / "leaderboard" / "static_snapshots"
    msp = {"20260705T140000Z": {"mostSelectedPlayers": msp_items if msp_items is not None else
                                [{"roundId": 10, "playerId": j + 1, "probabilityStarter": 0.6, "numberSelections": 1}
                                 for j in range(2)]}}
    msp.update(extra_msp or {})
    sheets = {"most_selected_players": msp,
              "rounds": {"20260701T000000Z": {"rounds": [{"id": 10, "date": f"{d}T00:00:00Z"}]}},
              "players": {"20260701T000000Z": {"players": [{"id": j + 1, "feedId": 700 + j, "squadId": j + 1} for j in range(2)]}},
              "units": {"20260701T000000Z": {"units": [{"id": 50 + j, "feedId": 9000 + j, "roundId": 10,
                                                         "homeSquadId": j + 1, "awaySquadId": 20 + j} for j in range(2)]}}}
    for feed, by_stamp in sheets.items():
        (snaps / feed).mkdir(parents=True)
        for stamp, doc in by_stamp.items():
            (snaps / feed / f"{stamp}.json.gz").write_bytes(gzip.compress(_json.dumps(doc).encode()))
    sched = tmp_path / "sched"; sched.mkdir()
    (sched / f"{d}.json").write_text(_json.dumps({"dates": [{"date": d, "games": [
        {"gamePk": 9000 + j, "status": {}, "teams": {"home": {"team": {"abbreviation": f"H{j}"}},
                                                      "away": {"team": {"abbreviation": f"A{j}"}}}} for j in range(2)]}]}))
    return w12, tmp_path / "data", sched


def _run(tmp_path, monkeypatch, **kw):
    import json as _json
    from scripts.audit.mlb_benchmark import run
    monkeypatch.setattr(run, "x23_gate", lambda: "f" * 40)
    w12, data, sched = _synthetic_inputs(tmp_path, **kw)
    assert run.main(["--w12-run", str(w12), "--data-root", str(data), "--schedules", str(sched),
                     "--out", str(tmp_path / "out"), "--n-resamples", "20"]) == 0
    out = next((tmp_path / "out").glob("*"))
    return _json.loads((out / "results.json").read_text()), _json.loads((out / "manifest.json").read_text())


def test_an_all_no_pa_stratum_and_empty_forecast_support_still_write_a_report(tmp_path, monkeypatch):
    res, _ = _run(tmp_path, monkeypatch)
    prim = res["strata"]["selection_consistent/pool_verified"]
    assert prim["T2"]["available"] is False and prim["encompassing_T2"]["point"]["available"] is False
    res2, _ = _run(tmp_path / "b", monkeypatch, msp_items=[])
    assert res2["coverage"]["per_date"][0]["excluded"] == "no_target_round_rows"
    assert res2["strata"]["selection_consistent/pool_verified"]["reason"] == "no_joined_rows"


def test_a_newer_schema_invalid_forecast_sheet_is_skipped_and_every_consumed_file_is_hashed(tmp_path, monkeypatch):
    res, man = _run(tmp_path, monkeypatch, extra_msp={"20260705T150000Z": {"unexpected": []}})
    assert res["coverage"]["invalid_forecast_sheets"] == ["20260705T150000Z.json.gz"]
    assert pd.Timestamp(res["coverage"]["per_date"][0]["forecast_stamp"]) == pd.Timestamp("2026-07-05T14:00:00Z")
    assert set(man["feeds"]["players"]) == {"20260701T000000Z.json.gz"}
    assert len(man["feeds"]["players"]["20260701T000000Z.json.gz"]["sha256"]) == 64
    assert set(man["w12_accepted_files"]) == {"table.parquet", "summary.json", "manifest.json", "ACCEPTED.json"}


def test_encompassing_draw_failures_make_the_interval_unavailable(monkeypatch):
    from scripts.audit.mlb_benchmark import run
    df = pd.DataFrame({"date": ["d1"] * 3 + ["d2"] * 3, "ours": [0.6, 0.7, 0.8] * 2, "mlb": [0.65, 0.6, 0.85] * 2,
                       "outcome": ["no_hit"] * 3 + ["hit"] * 3, "row_order": [0, 1, 2] * 2,
                       "batter_id": range(6), "game_pk": range(6)})
    monkeypatch.setattr(m, "encompassing", lambda x: {"available": True, "coef_mlb": 0.1, "delta_log_loss": -0.01}
                        if x["outcome"].nunique() > 1 else {"available": False, "reason": "one_class"})
    out = run.score_encompassing(df, n_resamples=200)
    iv = out["intervals"]["coef_mlb"]
    assert iv["n_failed"] > 0 and iv["lo"] is None and iv["failure_reasons"] == {"one_class": iv["n_failed"]}
