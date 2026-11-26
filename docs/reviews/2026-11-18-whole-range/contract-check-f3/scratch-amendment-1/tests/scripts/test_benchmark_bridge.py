"""W1.2 benchmark bridge: the pure core (design docs/superpowers/specs/2026-10-04-benchmark-bridge-design.md)."""
import math

import numpy as np
import pandas as pd
import pytest

from scripts.audit.benchmark_bridge import core


def _slate_p(p):
    return core.serialized_probability(p)


# --- selection consistency (design §2, Codex r1 F1) ---

def test_serialized_probability_replays_the_slate_writers_rounding():
    raw = 0.7654321098765432
    assert _slate_p(raw) != raw                      # pandas to_json rounds
    assert _slate_p(raw) == pytest.approx(raw, abs=1e-9)


def test_a_selection_matches_its_slate_row_after_the_writers_serialization():
    rows = [{"batter_id": 7, "game_pk": 100, "p_game_hit": _slate_p(0.7654321098765432)},
            {"batter_id": 8, "game_pk": 101, "p_game_hit": _slate_p(0.70)}]
    sel = core.selection_consistency(rows, decision_primary=None,
                                     pick_primary={"batter_id": "7", "game_pk": 100,
                                                   "p_game_hit": 0.7654321098765432})
    assert sel["state"] == "selection_consistent" and sel["source"] == "pick"


def test_decision_first_and_a_conflict_is_recorded_not_resolved_favourably():
    rows = [{"batter_id": 7, "game_pk": 100, "p_game_hit": _slate_p(0.76)},
            {"batter_id": 8, "game_pk": 101, "p_game_hit": _slate_p(0.70)}]
    sel = core.selection_consistency(rows,
                                     decision_primary={"batter_id": 8, "game_pk": 101, "p_game_hit": 0.70},
                                     pick_primary={"batter_id": 7, "game_pk": 100, "p_game_hit": 0.76})
    assert sel["source"] == "decision" and sel["state"] == "selection_consistent"
    assert sel["conflict"] is True


def test_no_matching_row_is_inconsistent_and_no_selection_is_its_own_stratum():
    rows = [{"batter_id": 7, "game_pk": 100, "p_game_hit": _slate_p(0.76)}]
    off = core.selection_consistency(rows, None, {"batter_id": 7, "game_pk": 100, "p_game_hit": 0.75})
    assert off["state"] == "inconsistent"
    none = core.selection_consistency(rows, None, None)
    assert none["state"] == "no_selection"


# --- the count normalization (design §3, Codex r1 F2) ---

def test_count_normalization_is_the_identity_when_the_counts_agree():
    a26 = 1 - (1 - 0.1) * (1 - 0.5)                  # two PAs with different p
    assert core.count_normalized(a26, n_actual=2, lineup_slot=None, n_est_override=2.0) == pytest.approx(a26)


def test_count_normalization_uses_the_lineup_slot_map_with_the_4_0_default():
    a26 = 1 - 0.8 ** 5
    got = core.count_normalized(a26, n_actual=5, lineup_slot=1)
    assert got == pytest.approx(1 - 0.8 ** 4.5)      # slot 1 -> 4.5 PAs
    assert core.count_normalized(a26, n_actual=5, lineup_slot=None) == pytest.approx(1 - 0.8 ** 4.0)
    assert math.isnan(core.count_normalized(a26, n_actual=0, lineup_slot=1))


# --- outcomes (design §4, Codex r1 F6) ---

def test_outcome_labels_need_a_complete_source_to_assert_no_pa():
    assert core.outcome_label(n_pa=3, n_hits=1, source_complete=True) == "hit"
    assert core.outcome_label(n_pa=3, n_hits=0, source_complete=True) == "no_hit"
    assert core.outcome_label(n_pa=0, n_hits=0, source_complete=True) == "no_pa"
    assert core.outcome_label(n_pa=0, n_hits=0, source_complete=False) == "unknown"


# --- eligibility surrogate and rank-1 (design §5, Codex r1 F5) ---

def test_eligibility_surrogate_needs_first_pitch_more_than_five_minutes_after_written_at():
    w = pd.Timestamp("2026-07-01T16:00:00Z")
    assert core.eligible_surrogate(pd.Timestamp("2026-07-01T16:05:01Z"), w) is True
    assert core.eligible_surrogate(pd.Timestamp("2026-07-01T16:05:00Z"), w) is False
    assert core.eligible_surrogate(None, w) is None


def test_rank1_breaks_exact_ties_by_the_archived_row_order_and_skips_ineligible():
    df = pd.DataFrame({"row_order": [0, 1, 2], "score": [0.8, 0.8, 0.9], "eligible": [True, True, False]})
    assert core.rank1(df, "score") == 0


def test_a_void_or_unknown_winner_is_never_replaced_by_the_next_batter():
    df = pd.DataFrame({"row_order": [0, 1], "score": [0.9, 0.8], "eligible": [True, True],
                       "outcome": ["no_pa", "hit"]})
    winner = core.rank1(df, "score")
    assert df.loc[winner, "outcome"] == "no_pa"     # ranked before outcomes; not reselected


# --- shared comparison pools (design §6, Codex r1 F6) ---

def test_a_paired_pool_keeps_only_rows_scored_and_eligible_in_both_arms():
    df = pd.DataFrame({"a": [0.7, np.nan, 0.6, 0.5], "b": [0.71, 0.6, np.nan, 0.52],
                       "eligible": [True, True, True, False]})
    pool, excl = core.paired_pool(df, "a", "b")
    assert list(pool.index) == [0]
    assert excl == {"missing_a": 1, "missing_b": 1, "ineligible": 1}


# --- metrics and the date-cluster bootstrap (design §6, Codex r1 F8) ---

def test_auc_is_an_equal_date_mean_and_dates_without_both_classes_are_counted():
    df = pd.DataFrame({"date": ["d1"] * 4 + ["d2"] * 2,
                       "s": [0.9, 0.8, 0.2, 0.1, 0.5, 0.4],
                       "y": [1, 0, 1, 0, 1, 1]})
    auc, n_used, n_omitted = core.equal_date_auc(df, "s", "y")
    assert (n_used, n_omitted) == (1, 1)
    assert auc == pytest.approx(0.75)                # d1: 3 of 4 positive/negative pairs ordered


def test_bootstrap_resamples_whole_dates_and_keeps_repeated_dates():
    df = pd.DataFrame({"date": ["d1", "d1", "d2"], "x": [1.0, 1.0, 0.0]})
    stat = lambda frame: float(frame["x"].mean())
    draws = core.date_block_bootstrap(df, stat, n_resamples=200, seed=7, return_draws=True)
    # whole-date resampling can only produce means from {d1,d1},{d1,d2},{d2,d2} multisets:
    allowed = {1.0, 2 / 3, 0.0}
    assert set(np.round(draws, 9)) <= {round(a, 9) for a in allowed}
    lo, hi = core.date_block_bootstrap(df, stat, n_resamples=200, seed=7)
    assert lo <= hi


# --- C: slot reconstruction from a D slate row plus the archived feed (design §3, Codex r1 F4) ---

def _feed(home="NYM", away="ATL", home_id=121, away_id=144, temp="72", wind="8 mph, Out To CF",
          roof="Open", ump=555):
    return {"gameData": {
        "venue": {"id": 3289, "fieldInfo": {"roofType": roof}},
        "weather": {"temp": temp, "wind": wind},
        "teams": {"home": {"abbreviation": home, "id": home_id}, "away": {"abbreviation": away, "id": away_id}},
        "officials": [{"officialType": "Home Plate", "official": {"id": ump}}],
    }}


def test_slots_keep_the_slate_rows_batter_inputs_and_take_game_fields_from_the_feed():
    row = {"batter_id": 7, "batter_name": "A", "team": "ATL", "game_pk": 100, "lineup": 2,
           "pitcher_id": 900, "pitcher_name": "P", "projected": True, "p_game_hit": 0.7}
    slot, src = core.reconstruct_slot(row, _feed())
    assert slot["batter_id"] == 7 and slot["lineup"] == 2 and slot["pitcher_id"] == 900 and slot["projected"] is True
    assert slot["opp_team_id"] == 121 and slot["pitcher_team"] == "NYM" and slot["venue_id"] == 3289
    assert slot["weather_temp"] == 72 and slot["weather_wind_speed"] == 8.0 and slot["weather_wind_dir"] == "Out To CF"
    assert slot["roof_type"] == "Open" and slot["hp_umpire_id"] == 555
    assert slot["pitcher_hand"] is None            # pre-game serving had no plays: the lookup fallback applies
    assert src["game_fields"] == "final_feed" and src["pitcher_hand"] == "serving_none"


def test_a_slim_feed_drops_live_data_and_players_and_changes_no_slot_start_or_status():
    full = _feed()
    full["gameData"].update(datetime={"dateTime": "2026-07-05T23:10:00Z"}, status={"abstractGameState": "Final"},
                            players={"ID1": {"fullName": "x" * 1000}})
    full["liveData"] = {"plays": {"allPlays": [{"x": "y" * 1000}] * 50}}
    slim = core.slim_feed(full)
    assert set(slim) == {"gameData"} and "players" not in slim["gameData"]
    for row in ({"batter_id": 7, "team": "ATL", "game_pk": 100, "lineup": 2, "pitcher_id": 900, "projected": True},
                {"batter_id": 8, "team": "LAD", "game_pk": 100, "lineup": 3, "pitcher_id": 901}):
        assert core.reconstruct_slot(row, slim) == core.reconstruct_slot(row, full)
    assert slim["gameData"]["datetime"] == full["gameData"]["datetime"]
    assert slim["gameData"]["status"] == full["gameData"]["status"]
    assert core.slim_feed(None) is None


def test_a_slate_row_whose_team_is_not_in_the_feed_is_unreconstructable():
    row = {"batter_id": 7, "team": "LAD", "game_pk": 100, "lineup": 2, "pitcher_id": 900}
    slot, src = core.reconstruct_slot(row, _feed())
    assert slot is None and src["reason"] == "team_not_in_feed"


def test_a_missing_projected_flag_is_confirmed_not_projected():
    row = {"batter_id": 7, "team": "NYM", "game_pk": 100, "lineup": 1, "pitcher_id": 901, "projected": None}
    slot, _ = core.reconstruct_slot(row, _feed())
    assert "projected" not in slot and slot["opp_team_id"] == 144


def test_score_c_injects_the_slots_truncates_history_and_never_mutates_the_artifact(monkeypatch):
    import bts.model.predict as pm
    seen = {}

    def fake_lookups(df):
        seen["lookup_max_date"] = df["date"].max()
        return {"L": 1}

    def fake_predict(date, df, model, lookups, check_openers=True, blend=None, feature_cols=None):
        seen.update(slots=pm._fetch_game_slots(date), hist_max=df["date"].max(), model=model,
                    blend_keys=sorted(blend), lookups=lookups)
        return pd.DataFrame({"batter_id": [7], "game_pk": [100], "p_game_hit": [0.71]})

    monkeypatch.setattr(pm, "_build_feature_lookups", fake_lookups)
    monkeypatch.setattr(pm, "predict", fake_predict)
    real_fetch = pm._fetch_game_slots
    hist = pd.DataFrame({"date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03"])})
    artifact = {"_model": "single", "baseline": "m0", "statcast_a": "m1"}
    out = core.score_c("2026-07-03", [{"batter_id": 7, "game_pk": 100}], hist, artifact)
    assert seen["slots"] == [{"batter_id": 7, "game_pk": 100}]
    assert seen["hist_max"] == pd.Timestamp("2026-07-02") == seen["lookup_max_date"]
    assert seen["model"] == "single" and seen["blend_keys"] == ["baseline", "statcast_a"]
    assert artifact == {"_model": "single", "baseline": "m0", "statcast_a": "m1"}      # not mutated
    assert pm._fetch_game_slots is real_fetch                                            # restored
    assert list(out.columns) == ["batter_id", "game_pk", "p_game_hit"]


# --- verified eligibility from BTS unit captures (design §5) ---

def test_verified_eligibility_uses_the_latest_unit_capture_before_written_at():
    hist = [("2026-07-05T15:00:00Z", "scheduled"), ("2026-07-05T23:30:00Z", "inprogress")]
    w = pd.Timestamp("2026-07-05T16:00:00Z")
    assert core.verified_eligibility(hist, w, pregame={"scheduled"}, not_pregame={"inprogress", "final", "postponed"}) is True
    late = pd.Timestamp("2026-07-06T00:00:00Z")
    assert core.verified_eligibility(hist, late, pregame={"scheduled"}, not_pregame={"inprogress", "final", "postponed"}) is False


def test_verified_eligibility_is_unknown_without_a_prior_capture_or_for_an_unclassified_status():
    w = pd.Timestamp("2026-07-05T16:00:00Z")
    assert core.verified_eligibility([("2026-07-05T17:00:00Z", "scheduled")], w, {"scheduled"}, {"final"}) is None
    assert core.verified_eligibility([("2026-07-05T15:00:00Z", "weird")], w, {"scheduled"}, {"final"}) is None
    assert core.verified_eligibility([], w, {"scheduled"}, {"final"}) is None


# --- outcomes from PA rows (design §4) ---

def test_outcome_table_labels_rows_and_needs_a_final_game_for_no_pa():
    pa = pd.DataFrame({"date": ["2026-07-05"] * 3, "game_pk": [100, 100, 100], "batter_id": [7, 7, 8],
                       "is_hit": [0, 1, 0]})
    cands = pd.DataFrame({"date": ["2026-07-05"] * 4, "game_pk": [100, 100, 100, 200],
                          "batter_id": [7, 8, 9, 10]})
    out = core.outcome_table(cands, pa, final_games={100})
    lab = dict(zip(out["batter_id"], out["outcome"]))
    assert lab == {7: "hit", 8: "no_hit", 9: "no_pa", 10: "unknown"}


# --- report (design §6) ---

def _table():
    return pd.DataFrame({
        "date": ["d1", "d1", "d1", "d2", "d2", "d2"],
        "row_order": [0, 1, 2, 0, 1, 2],
        "outcome": ["hit", "no_hit", "no_pa", "no_hit", "hit", "unknown"],
        "A": [0.9, 0.8, 0.7, 0.9, 0.6, 0.5],
        "B": [0.7, 0.9, 0.6, 0.6, 0.95, 0.5],
        "pool_all": [True] * 6,
    })


def test_surface_metrics_rank_before_outcomes_and_count_void_winners():
    from scripts.audit.benchmark_bridge import report
    m = report.surface_metrics(_table(), "A", "pool_all")
    assert m["top1"] == {"hit": 1, "no_hit": 1, "no_pa": 0, "unknown": 0, "hit_rate": 0.5}
    assert m["n_known_rows"] == 4 and m["auc_dates_used"] == 2


def test_pair_metrics_use_the_identical_pool_and_report_discordance():
    from scripts.audit.benchmark_bridge import report
    m = report.pair_metrics(_table(), "A", "B", "pool_all", n_resamples=200)
    assert m["n_rows"] == 6 and m["rank1_changed_dates"] == 2
    assert m["paired_top1_dates"] == 2
    assert m["discordant_dates"] == {"a_hit_b_miss": 1, "a_miss_b_hit": 1}
    assert m["top1_diff_b_minus_a"] == 0.0
    lo, hi = m["top1_diff_ci95"]
    assert lo <= 0.0 <= hi


def test_the_opener_check_runs_once_per_pitcher_per_date_across_scorings(monkeypatch):
    import bts.model.predict as pm
    calls = []
    monkeypatch.setattr(pm, "_build_feature_lookups", lambda df: {})
    monkeypatch.setattr(pm, "_check_opener", lambda pid, df: calls.append(pid) or {"is_opener": False})

    def fake_predict(date, df, model, lookups, check_openers=True, blend=None, feature_cols=None):
        for s in pm._fetch_game_slots(date):
            pm._check_opener(s["pitcher_id"], df)
        return pd.DataFrame({"batter_id": [1], "game_pk": [1], "p_game_hit": [0.5]})

    monkeypatch.setattr(pm, "predict", fake_predict)
    hist = pd.DataFrame({"date": pd.to_datetime(["2026-07-01"])})
    prepared = core.history_and_lookups("2026-07-02", hist)
    slots = [{"pitcher_id": 9}, {"pitcher_id": 9}, {"pitcher_id": 8}]
    core.score_c("2026-07-02", slots, hist, {"_model": "m"}, prepared=prepared)
    core.score_c("2026-07-02", slots, hist, {"_model": "m"}, prepared=prepared)
    assert sorted(calls) == [8, 9]


def test_capture_files_lists_plain_and_gzipped_captures_in_stamp_order(tmp_path):
    import gzip
    (tmp_path / "20260709T120000Z.json").write_text('{"units": []}')
    (tmp_path / "20260710T160001Z.json.gz").write_bytes(gzip.compress(b'{"units": []}'))
    (tmp_path / "20260710T153000Z.json").write_text('{"units": []}')
    (tmp_path / "notes.txt").write_text("x")
    names = [p.name for p in core.capture_files(tmp_path)]
    assert names == ["20260709T120000Z.json", "20260710T153000Z.json", "20260710T160001Z.json.gz"]


def test_history_slice_on_a_date_sorted_frame_equals_the_masked_copy(monkeypatch):
    import bts.model.predict as pm
    monkeypatch.setattr(pm, "_build_feature_lookups", lambda df: {"n": len(df), "last": df["x"].iloc[-1]})
    df = pd.DataFrame({"date": pd.to_datetime(["2026-06-01", "2026-06-01", "2026-06-02", "2026-06-03"]),
                       "x": [1, 2, 3, 4]})
    hist, lookups, cache = core.history_and_lookups("2026-06-03", df)
    pd.testing.assert_frame_equal(hist, df[df["date"] < pd.Timestamp("2026-06-03")])
    assert lookups == {"n": 3, "last": 3} and cache == {}
    shuffled = df.iloc[[2, 0, 3, 1]]                 # not date-sorted: the masked copy path, same rows in frame order
    h2, _, _ = core.history_and_lookups("2026-06-03", shuffled)
    pd.testing.assert_frame_equal(h2, shuffled[shuffled["date"] < pd.Timestamp("2026-06-03")])


def test_read_accepted_run_returns_the_verified_bytes_and_refuses_a_mismatch(tmp_path):
    import hashlib
    import json
    import pytest
    for name, body in (("table.parquet", b"T"), ("summary.json", b"{}"), ("manifest.json", b"{}")):
        (tmp_path / name).write_bytes(body)
    files = {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest()
             for n in ("table.parquet", "summary.json", "manifest.json")}
    with pytest.raises(SystemExit):
        core.read_accepted_run(tmp_path)                      # no ACCEPTED.json: not an accepted run
    (tmp_path / "ACCEPTED.json").write_text(json.dumps({"files": files, "memo": "m"}))
    got, acc = core.read_accepted_run(tmp_path)
    assert got["table.parquet"] == b"T" and acc["memo"] == "m" and "ACCEPTED.json" in got
    (tmp_path / "summary.json").write_bytes(b'{"x": 1}')
    with pytest.raises(SystemExit):
        core.read_accepted_run(tmp_path)
