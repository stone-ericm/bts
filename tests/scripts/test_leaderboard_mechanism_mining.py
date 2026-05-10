from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.leaderboard_mechanism_mining import (
    DECOMPOSITION_VARIABLES,
    build_audit,
    decomposition_fdr_table,
    load_realized_production_picks,
)


def _write_user_picks(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _pick(
    *,
    captured_at: str,
    pick_date: str = "2026-05-10",
    pick_number: int,
    batter_id: int,
    batter_name: str,
    result: str,
):
    return {
        "captured_at": pd.Timestamp(captured_at),
        "round_id": 1,
        "pick_date": pd.Timestamp(pick_date).date(),
        "pick_number": pick_number,
        "unit_id": 1,
        "bts_player_id": batter_id,
        "result": result,
        "at_bats": 4,
        "hits": 1 if result == "hit" else 0,
        "streak_after": 1,
        "batter_id": batter_id,
        "batter_name": batter_name,
        "batter_team": "BOS",
        "opponent_team": "NYY",
        "home_or_away": "home",
    }


def _write_snapshot(path, users):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "captured_at": pd.Timestamp("2026-05-10T09:00:00"),
            "tab": "active_streak",
            "rank": i + 1,
            "username": user,
            "streak": 10 - i,
            "hits_today": 0,
        }
        for i, user in enumerate(users)
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_realized(path, *, void_primary: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source_file": "2026-05-10.json",
            "date": "2026-05-10",
            "run_time": "2026-05-10T10:00:00+00:00",
            "slot": "primary",
            "batter_id": 11,
            "batter_name": "Production One",
            "pitcher_id": 100,
            "game_pk": 1001,
            "p_game_hit": 0.72,
            "actual_hit": None if void_primary else False,
            "result_status": "void" if void_primary else "resolved",
            "projected_lineup": True,
            "pick_file_result": "void" if void_primary else "not_hit",
            "regime": "normal",
            "model_cutoff_label": "lock",
            "cutoff_iso": "2026-05-10T14:00:00+00:00",
            "attribution_source": "fixture",
            "pick_venue_id": 1,
            "pick_roof_type": "open",
            "pick_weather_temp": 72,
            "pick_is_indoor": False,
            "is_park_driven": True,
            "batter_skill_prior_pa": 350,
            "batter_skill_prior_hit_rate": 0.31,
            "batter_skill_quartile": 2,
        },
        {
            "source_file": "2026-05-10.json",
            "date": "2026-05-10",
            "run_time": "2026-05-10T10:00:00+00:00",
            "slot": "double_down",
            "batter_id": 20,
            "batter_name": "Consensus Two",
            "pitcher_id": 200,
            "game_pk": 1002,
            "p_game_hit": 0.81,
            "actual_hit": True,
            "result_status": "resolved",
            "projected_lineup": True,
            "pick_file_result": "hit",
            "regime": "normal",
            "model_cutoff_label": "lock",
            "cutoff_iso": "2026-05-10T14:00:00+00:00",
            "attribution_source": "fixture",
            "pick_venue_id": 2,
            "pick_roof_type": "dome",
            "pick_weather_temp": None,
            "pick_is_indoor": True,
            "is_park_driven": False,
            "batter_skill_prior_pa": 90,
            "batter_skill_prior_hit_rate": 0.34,
            "batter_skill_quartile": 4,
        },
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_ranked_surface(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "date": "2026-05-10",
            "rank": 1,
            "batter_id": 11,
            "batter_name": "Production One",
            "game_pk": 1001,
            "p_game_hit": 0.72,
            "actual_hit": 0,
            "n_pas": 4,
        },
        {
            "date": "2026-05-10",
            "rank": 3,
            "batter_id": 10,
            "batter_name": "Consensus One",
            "game_pk": 1003,
            "p_game_hit": 0.69,
            "actual_hit": 1,
            "n_pas": 4,
        },
        {
            "date": "2026-05-10",
            "rank": 6,
            "batter_id": 20,
            "batter_name": "Consensus Two",
            "game_pk": 1002,
            "p_game_hit": 0.66,
            "actual_hit": 1,
            "n_pas": 4,
        },
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _fixture_tree(tmp_path):
    leaderboard = tmp_path / "leaderboard"
    _write_user_picks(
        leaderboard / "user_picks" / "alice.parquet",
        [
            _pick(
                captured_at="2026-05-10T10:00:00",
                pick_number=1,
                batter_id=10,
                batter_name="Consensus One",
                result="hit",
            ),
            _pick(
                captured_at="2026-05-10T10:00:00",
                pick_number=2,
                batter_id=20,
                batter_name="Consensus Two",
                result="hit",
            ),
        ],
    )
    _write_user_picks(
        leaderboard / "user_picks" / "bob.parquet",
        [
            _pick(
                captured_at="2026-05-10T10:05:00",
                pick_number=1,
                batter_id=10,
                batter_name="Consensus One",
                result="hit",
            ),
            _pick(
                captured_at="2026-05-10T10:05:00",
                pick_number=2,
                batter_id=20,
                batter_name="Consensus Two",
                result="hit",
            ),
        ],
    )
    _write_user_picks(
        leaderboard / "user_picks" / "eve.parquet",
        [
            _pick(
                captured_at="2026-05-10T10:05:00",
                pick_number=1,
                batter_id=99,
                batter_name="All Tracked Only",
                result="not_hit",
            ),
            _pick(
                captured_at="2026-05-10T10:05:00",
                pick_number=2,
                batter_id=98,
                batter_name="All Tracked Two",
                result="not_hit",
            ),
        ],
    )
    _write_snapshot(
        leaderboard / "leaderboard_snapshots" / "2026-05-10.parquet",
        ["alice", "bob"],
    )
    realized = tmp_path / "surfaces" / "realized.parquet"
    ranked = tmp_path / "surfaces" / "ranked.parquet"
    _write_realized(realized)
    _write_ranked_surface(ranked)
    return leaderboard, realized, ranked


def test_build_audit_constructs_units_and_preserves_decomposition_order(tmp_path):
    leaderboard, realized, ranked = _fixture_tree(tmp_path)

    report = build_audit(
        leaderboard_dir=leaderboard,
        realized_production_surface=realized,
        output_path=tmp_path / "mechanism.json",
        units_output_path=tmp_path / "mechanism.units.parquet",
        surface_specs={"ranked_fixture": ranked},
        decision_cutoff_iso=None,
        cohort_as_of_iso=None,
        cohort_users_json=None,
        dates={"2026-05-10"},
        min_date=None,
        max_date=None,
        top_k=(1, 2, 5, 10),
        n_bootstrap=0,
        expected_block_length=7,
        seed=20260510,
        fdr_min_n=15,
        generated_at="2026-05-10T11:01:00+00:00",
    )

    assert report["schema_version"] == "leaderboard_mechanism_mining_v1"
    assert report["research_only"] is True
    assert report["production_deploy_claim"] is False
    assert report["no_policy_edit_supported"] is True
    assert report["decomposition_variables"] == DECOMPOSITION_VARIABLES
    assert report["fdr_method"]["day_block_bootstrap_seed"] == 20260510

    units = pd.read_parquet(report["outputs"]["units_parquet"])
    assert set(units["cohort"]) == {"all_tracked", "fixed_cohort"}
    fixed = units[units["cohort"] == "fixed_cohort"].sort_values("pick_number")
    assert fixed["resolved_for_outcome"].tolist() == [True, True]
    assert fixed["agreement_state"].tolist() == ["different_batter", "same_batter"]
    assert fixed["delta"].tolist() == [1.0, 0.0]
    assert fixed["consensus_model_rank_bin"].tolist() == ["rank3_5", "rank6_10"]
    assert fixed["consensus_model_probability_bin"].tolist() == ["0.68-0.74", "<0.68"]
    assert fixed["production_batter_skill_prior_pa_bin"].tolist() == ["300-599", "<100"]
    assert fixed["production_weather_temp_bin"].tolist() == ["60-74", "indoor_or_missing"]

    summary = report["summary"]["fixed_cohort"]["ranked_fixture"]
    assert summary["n_resolved_units"] == 2
    assert summary["n_disagreement_units"] == 1
    assert summary["production_hit_rate"] == pytest.approx(0.5)
    assert summary["consensus_hit_rate"] == pytest.approx(1.0)
    assert summary["top_k_coverage_on_surface_dates"]["5"] == pytest.approx(0.5)
    assert summary["top_k_coverage_on_surface_dates"]["10"] == pytest.approx(1.0)


def test_voided_production_slot_is_kept_but_excluded_from_denominators(tmp_path):
    leaderboard, realized, _ranked = _fixture_tree(tmp_path)
    _write_realized(realized, void_primary=True)

    report = build_audit(
        leaderboard_dir=leaderboard,
        realized_production_surface=realized,
        output_path=tmp_path / "mechanism.json",
        units_output_path=tmp_path / "mechanism.units.parquet",
        surface_specs={},
        decision_cutoff_iso=None,
        cohort_as_of_iso=None,
        cohort_users_json=None,
        dates={"2026-05-10"},
        min_date=None,
        max_date=None,
        top_k=(1, 2),
        n_bootstrap=0,
        expected_block_length=7,
        seed=20260510,
        fdr_min_n=15,
        generated_at="2026-05-10T11:01:00+00:00",
    )

    units = pd.read_parquet(report["outputs"]["units_parquet"])
    fixed = units[units["cohort"] == "fixed_cohort"].sort_values("pick_number")
    assert fixed["resolved_for_outcome"].tolist() == [False, True]
    assert fixed["consensus_model_rank_bin"].tolist() == [
        "missing_surface",
        "missing_surface",
    ]
    assert pd.isna(fixed.iloc[0]["production_hit"])
    assert pd.isna(fixed.iloc[0]["delta"])

    summary = report["summary"]["fixed_cohort"]["realized_production_only"]
    assert summary["n_units_total"] == 2
    assert summary["n_resolved_units"] == 1
    assert summary["n_unresolved_or_void_units"] == 1
    assert summary["production_hit_rate"] == pytest.approx(1.0)
    assert summary["consensus_hit_rate"] == pytest.approx(1.0)


def test_realized_production_loader_raises_on_missing_required_columns(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame([{"date": "2026-05-10", "slot": "primary"}]).to_parquet(
        path,
        index=False,
    )

    with pytest.raises(ValueError, match="missing realized-production columns"):
        load_realized_production_picks(
            path,
            dates=None,
            min_date=None,
            max_date=None,
        )


def test_decomposition_fdr_applies_bh_by_and_mechanism_threshold():
    rows = []
    for i in range(30):
        rows.append({
            "surface": "ranked_fixture",
            "cohort": "fixed_cohort",
            "pick_number": 1,
            "consensus_pick_share_bin": ">=0.25",
            "production_p_game_hit_bin": "0.68-0.74",
            "agreement_state": "different_batter",
            "production_batter_skill_quartile": "2",
            "production_batter_skill_prior_pa_bin": "300-599",
            "production_projected_lineup": "true",
            "production_regime": "normal",
            "production_is_park_driven": "true",
            "production_is_indoor": "false",
            "production_weather_temp_bin": "60-74",
            "consensus_model_rank_bin": "rank3_5",
            "consensus_model_probability_bin": "0.68-0.74",
            "resolved_for_outcome": True,
            "production_hit": 0,
            "consensus_hit": 1,
            "delta": 1,
        })
    # Same cell in all_tracked has non-negative direction, satisfying condition 4.
    for i in range(15):
        rows.append({
            **rows[0],
            "cohort": "all_tracked",
            "delta": 1 if i < 10 else 0,
            "production_hit": 0 if i < 10 else 1,
            "consensus_hit": 1,
        })
    # A second weak testable cell checks q-value ordering without nomination.
    for i in range(15):
        rows.append({
            **rows[0],
            "production_batter_skill_quartile": "4",
            "production_batter_skill_prior_pa_bin": "<100",
            "delta": 0,
            "production_hit": 1,
            "consensus_hit": 1,
        })
    units = pd.DataFrame(rows)

    result = decomposition_fdr_table(units, min_n=15)

    assert result["method"] == "exact_positive_sign_test_plus_BH_BY"
    assert result["n_testable_cells"] == 3
    assert result["n_survive_BH_0_10"] >= 1
    assert result["actionable_mechanism_found"] is True
    winner = next(row for row in result["rows"] if row["mechanism_candidate"])
    assert winner["cohort"] == "fixed_cohort"
    assert winner["q_BH"] <= 0.10
    assert winner["q_BY"] <= 0.10


def test_cli_report_json_is_written(tmp_path):
    leaderboard, realized, _ranked = _fixture_tree(tmp_path)

    report = build_audit(
        leaderboard_dir=leaderboard,
        realized_production_surface=realized,
        output_path=tmp_path / "mechanism.json",
        units_output_path=None,
        surface_specs={},
        decision_cutoff_iso=None,
        cohort_as_of_iso=None,
        cohort_users_json=None,
        dates={"2026-05-10"},
        min_date=None,
        max_date=None,
        top_k=(1, 2),
        n_bootstrap=0,
        expected_block_length=7,
        seed=20260510,
        fdr_min_n=15,
        generated_at="2026-05-10T11:01:00+00:00",
    )

    saved = json.loads((tmp_path / "mechanism.json").read_text())
    assert saved["outputs"]["units_parquet"] == str(tmp_path / "mechanism.units.parquet")
    assert report["outputs"] == saved["outputs"]
