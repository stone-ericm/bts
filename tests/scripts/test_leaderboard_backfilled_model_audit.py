from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.leaderboard_backfilled_model_audit import build_audit
from scripts.leaderboard_backfilled_model_audit import load_ranked_surfaces
from scripts.leaderboard_backfilled_model_audit import load_realized_pick_surfaces


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


def _write_surface(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "date": "2026-05-10",
            "rank": 1,
            "batter_id": 11,
            "game_pk": 1001,
            "p_game_hit": 0.72,
            "actual_hit": 0,
            "n_pas": 4,
        },
        {
            "date": "2026-05-10",
            "rank": 2,
            "batter_id": 20,
            "game_pk": 1002,
            "p_game_hit": 0.69,
            "actual_hit": 1,
            "n_pas": 4,
        },
        {
            "date": "2026-05-10",
            "rank": 3,
            "batter_id": 10,
            "game_pk": 1003,
            "p_game_hit": 0.68,
            "actual_hit": 1,
            "n_pas": 4,
        },
        {
            "date": "2026-05-10",
            "rank": 4,
            "batter_id": 40,
            "game_pk": 1004,
            "p_game_hit": 0.67,
            "actual_hit": 0,
            "n_pas": 4,
        },
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_realized_surface(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source_file": "2026-05-10.json",
            "date": "2026-05-10",
            "run_time": "2026-05-10T10:00:00+00:00",
            "slot": "primary",
            "batter_id": 10,
            "batter_name": "Consensus One",
            "game_pk": 1003,
            "p_game_hit": 0.70,
            "actual_hit": True,
            "result_status": "resolved",
        },
        {
            "source_file": "2026-05-10.json",
            "date": "2026-05-10",
            "run_time": "2026-05-10T10:00:00+00:00",
            "slot": "double_down",
            "batter_id": 99,
            "batter_name": "Production Two",
            "game_pk": 1099,
            "p_game_hit": 0.68,
            "actual_hit": False,
            "result_status": "resolved",
        },
    ]
    pd.DataFrame(rows).to_parquet(path, index=False)


def _write_realized_surface_with_voided_primary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "source_file": "2026-05-10.json",
            "date": "2026-05-10",
            "run_time": "2026-05-10T10:00:00+00:00",
            "slot": "primary",
            "batter_id": 10,
            "batter_name": "Consensus One",
            "game_pk": 1003,
            "p_game_hit": 0.70,
            "actual_hit": None,
            "result_status": "void",
        },
        {
            "source_file": "2026-05-10.json",
            "date": "2026-05-10",
            "run_time": "2026-05-10T10:00:00+00:00",
            "slot": "double_down",
            "batter_id": 20,
            "batter_name": "Consensus Two",
            "game_pk": 1002,
            "p_game_hit": 0.68,
            "actual_hit": True,
            "result_status": "resolved",
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
                captured_at="2026-05-10T12:30:00",
                pick_number=1,
                batter_id=99,
                batter_name="Off Cohort",
                result="hit",
            ),
        ],
    )
    _write_snapshot(leaderboard / "leaderboard_snapshots" / "2026-05-10.parquet", ["alice", "bob"])
    surface = tmp_path / "surfaces" / "production_backfill.parquet"
    _write_surface(surface)
    return leaderboard, surface


def test_build_audit_compares_fixed_consensus_to_backfilled_surface(tmp_path):
    leaderboard, surface = _fixture_tree(tmp_path)
    output = tmp_path / "leaderboard_backfilled_model_audit.json"

    report = build_audit(
        leaderboard_dir=leaderboard,
        surface_specs={"production_backfill": surface},
        output_path=output,
        joined_output_path=None,
        consensus_units_output_path=None,
        cohort_as_of_iso=None,
        cohort_users_json=None,
        dates={"2026-05-10"},
        min_date=None,
        max_date=None,
        top_k=(1, 2, 3),
        n_bootstrap=0,
        expected_block_length=7,
        seed=7,
        generated_at="2026-05-10T11:01:00+00:00",
    )

    assert output.exists()
    assert report["schema_version"] == "leaderboard_backfilled_model_audit_v1"
    assert report["no_policy_edit_supported"] is True
    assert "historical_backtest_oracle_exposure_caveat" in report["methodology_constraints"]
    assert "realized_production_surface_at_lock_anchor" in report["methodology_constraints"]
    assert report["pre_registered_primary_comparison"]["primary_cohort"] == "fixed_cohort"
    assert report["inventory"]["leaderboard"]["dedup_rows_after_date_filter"] == 5
    assert report["cohorts"]["fixed_cohort"]["n_users"] == 2

    fixed_vs_model = report["comparison"]["consensus_vs_model"]["fixed_cohort"][
        "production_backfill"
    ]
    assert fixed_vs_model["n_units"] == 2
    assert fixed_vs_model["n_disagreements"] == 1
    assert fixed_vs_model["model_hit_rate"] == pytest.approx(0.5)
    assert fixed_vs_model["consensus_hit_rate"] == pytest.approx(1.0)
    assert fixed_vs_model["disagreement_mean_delta"] == pytest.approx(1.0)

    coverage = report["comparison"]["consensus_rank_coverage"]["fixed_cohort"][
        "production_backfill"
    ]
    assert coverage["n_units_on_surface_dates"] == 2
    assert coverage["top_k_coverage"]["2"] == pytest.approx(0.5)
    assert coverage["top_k_coverage"]["3"] == pytest.approx(1.0)
    assert coverage["top_k_coverage_on_surface_dates"]["3"] == pytest.approx(1.0)

    individual = report["comparison"]["individual_pick_overlap"]["fixed_cohort"][
        "production_backfill"
    ]
    assert individual["n_picks"] == 4
    assert individual["n_picks_on_surface_dates"] == 4
    assert individual["top_k_share"]["2"] == pytest.approx(0.5)
    assert individual["top_k_share"]["3"] == pytest.approx(1.0)
    assert individual["top_k_share_on_surface_dates"]["3"] == pytest.approx(1.0)

    joined = pd.read_parquet(report["joined_individual_picks_path"])
    assert set(joined["cohort"]) == {"all_tracked", "fixed_cohort"}
    assert 99 not in set(joined[joined["cohort"] == "fixed_cohort"]["batter_id"])

    units = pd.read_parquet(report["joined_consensus_units_path"])
    fixed_slot1 = units[
        (units["cohort"] == "fixed_cohort") & (units["pick_number"] == 1)
    ].iloc[0]
    assert fixed_slot1["consensus_batter_id"] == 10
    assert fixed_slot1["model_batter_id"] == 11
    assert fixed_slot1["delta"] == 1


def test_build_audit_can_use_explicit_fixed_cohort_json(tmp_path):
    leaderboard, surface = _fixture_tree(tmp_path)
    cohort_path = tmp_path / "cohort.json"
    cohort_path.write_text(json.dumps({"users": ["alice"]}))

    report = build_audit(
        leaderboard_dir=leaderboard,
        surface_specs={"production_backfill": surface},
        output_path=tmp_path / "audit.json",
        joined_output_path=tmp_path / "joined.parquet",
        consensus_units_output_path=tmp_path / "units.parquet",
        cohort_as_of_iso=None,
        cohort_users_json=cohort_path,
        dates={"2026-05-10"},
        min_date=None,
        max_date=None,
        top_k=(1, 2, 3),
        n_bootstrap=0,
        expected_block_length=7,
        seed=7,
        generated_at="2026-05-10T11:01:00+00:00",
    )

    assert report["cohorts"]["fixed_cohort"]["source"] == "cohort_users_json"
    assert report["cohorts"]["fixed_cohort"]["n_users"] == 1
    individual = report["comparison"]["individual_pick_overlap"]["fixed_cohort"][
        "production_backfill"
    ]
    assert individual["n_picks"] == 2


def test_build_audit_accepts_realized_production_surface(tmp_path):
    leaderboard, _surface = _fixture_tree(tmp_path)
    realized = tmp_path / "surfaces" / "realized.parquet"
    _write_realized_surface(realized)

    report = build_audit(
        leaderboard_dir=leaderboard,
        surface_specs={},
        realized_surface_specs={"realized_prod": realized},
        output_path=tmp_path / "audit.json",
        joined_output_path=None,
        consensus_units_output_path=None,
        cohort_as_of_iso=None,
        cohort_users_json=None,
        dates={"2026-05-10"},
        min_date=None,
        max_date=None,
        top_k=(1, 2),
        n_bootstrap=0,
        expected_block_length=7,
        seed=7,
        generated_at="2026-05-10T11:01:00+00:00",
    )

    assert report["realized_production_surface_specs"] == {
        "realized_prod": str(realized)
    }
    assert report["inventory"]["surfaces"]["realized_prod"]["surface_type"] == (
        "realized_production_pick"
    )
    fixed_vs_model = report["comparison"]["consensus_vs_model"]["fixed_cohort"][
        "realized_prod"
    ]
    assert fixed_vs_model["n_units"] == 2
    assert fixed_vs_model["n_disagreements"] == 1
    assert fixed_vs_model["model_hit_rate"] == pytest.approx(0.5)
    assert fixed_vs_model["consensus_hit_rate"] == pytest.approx(1.0)
    assert fixed_vs_model["disagreement_mean_delta"] == pytest.approx(1.0)

    units = pd.read_parquet(report["joined_consensus_units_path"])
    fixed = units[units["cohort"] == "fixed_cohort"].sort_values("pick_number")
    assert fixed["model_batter_id"].tolist() == [10, 99]
    assert fixed["model_hit"].tolist() == [1, 0]


def test_realized_production_surface_preserves_voided_slots_as_null(tmp_path):
    leaderboard, _surface = _fixture_tree(tmp_path)
    realized = tmp_path / "surfaces" / "realized_void.parquet"
    _write_realized_surface_with_voided_primary(realized)

    report = build_audit(
        leaderboard_dir=leaderboard,
        surface_specs={},
        realized_surface_specs={"realized_prod": realized},
        output_path=tmp_path / "audit.json",
        joined_output_path=None,
        consensus_units_output_path=None,
        cohort_as_of_iso=None,
        cohort_users_json=None,
        dates={"2026-05-10"},
        min_date=None,
        max_date=None,
        top_k=(1, 2),
        n_bootstrap=0,
        expected_block_length=7,
        seed=7,
        generated_at="2026-05-10T11:01:00+00:00",
    )

    inventory = report["inventory"]["surfaces"]["realized_prod"]
    assert inventory["actual_hit_null_rows"] == 1
    assert inventory["result_status_counts"] == {"resolved": 1, "void": 1}

    fixed_vs_model = report["comparison"]["consensus_vs_model"]["fixed_cohort"][
        "realized_prod"
    ]
    assert fixed_vs_model["n_units"] == 1
    assert fixed_vs_model["model_hit_rate"] == pytest.approx(1.0)
    assert fixed_vs_model["consensus_hit_rate"] == pytest.approx(1.0)

    units = pd.read_parquet(report["joined_consensus_units_path"])
    fixed = units[units["cohort"] == "fixed_cohort"].sort_values("pick_number")
    slot1 = fixed.iloc[0]
    slot2 = fixed.iloc[1]
    assert slot1["model_batter_id"] == 10
    assert pd.isna(slot1["model_hit"])
    assert pd.isna(slot1["delta"])
    assert slot2["model_batter_id"] == 20
    assert slot2["model_hit"] == 1


def test_build_audit_reports_surface_date_denominators_separately(tmp_path):
    leaderboard, surface = _fixture_tree(tmp_path)
    _write_user_picks(
        leaderboard / "user_picks" / "carol.parquet",
        [
            _pick(
                captured_at="2026-05-11T10:00:00",
                pick_date="2026-05-11",
                pick_number=1,
                batter_id=10,
                batter_name="Consensus One",
                result="hit",
            ),
            _pick(
                captured_at="2026-05-11T10:00:00",
                pick_date="2026-05-11",
                pick_number=2,
                batter_id=20,
                batter_name="Consensus Two",
                result="hit",
            ),
        ],
    )
    _write_snapshot(
        leaderboard / "leaderboard_snapshots" / "2026-05-11.parquet",
        ["alice", "bob", "carol"],
    )

    report = build_audit(
        leaderboard_dir=leaderboard,
        surface_specs={"production_backfill": surface},
        output_path=tmp_path / "audit.json",
        joined_output_path=None,
        consensus_units_output_path=None,
        cohort_as_of_iso=None,
        cohort_users_json=None,
        dates=None,
        min_date=None,
        max_date=None,
        top_k=(1, 2, 3),
        n_bootstrap=0,
        expected_block_length=7,
        seed=7,
        generated_at="2026-05-10T11:01:00+00:00",
    )

    coverage = report["comparison"]["consensus_rank_coverage"]["all_tracked"][
        "production_backfill"
    ]
    assert coverage["n_units"] == 4
    assert coverage["n_units_on_surface_dates"] == 2
    assert coverage["top_k_coverage"]["3"] == pytest.approx(0.5)
    assert coverage["top_k_coverage_on_surface_dates"]["3"] == pytest.approx(1.0)

    individual = report["comparison"]["individual_pick_overlap"]["all_tracked"][
        "production_backfill"
    ]
    assert individual["n_picks"] == 7
    assert individual["n_picks_on_surface_dates"] == 5
    assert individual["top_k_share"]["3"] == pytest.approx(4 / 7)
    assert individual["top_k_share_on_surface_dates"]["3"] == pytest.approx(4 / 5)


def test_load_ranked_surfaces_fails_loud_on_missing_required_columns(tmp_path):
    surface = tmp_path / "broken.parquet"
    pd.DataFrame(
        [
            {
                "date": "2026-05-10",
                "rank": 1,
                "batter_id": 10,
                "p_game_hit": 0.70,
            }
        ]
    ).to_parquet(surface, index=False)

    with pytest.raises(ValueError, match="missing ranked-surface columns"):
        load_ranked_surfaces({"broken": surface})


def test_load_ranked_surfaces_fails_loud_on_duplicate_date_rank(tmp_path):
    surface = tmp_path / "duplicate.parquet"
    pd.DataFrame(
        [
            {
                "date": "2026-05-10",
                "rank": 1,
                "batter_id": 10,
                "p_game_hit": 0.70,
                "actual_hit": 1,
            },
            {
                "date": "2026-05-10",
                "rank": 1,
                "batter_id": 11,
                "p_game_hit": 0.69,
                "actual_hit": 0,
            },
        ]
    ).to_parquet(surface, index=False)

    with pytest.raises(ValueError, match="duplicate date/rank rows"):
        load_ranked_surfaces({"duplicate": surface})


def test_load_realized_pick_surfaces_fails_loud_on_missing_columns(tmp_path):
    surface = tmp_path / "broken_realized.parquet"
    pd.DataFrame(
        [
            {
                "date": "2026-05-10",
                "slot": "primary",
                "batter_id": 10,
                "p_game_hit": 0.70,
            }
        ]
    ).to_parquet(surface, index=False)

    with pytest.raises(ValueError, match="missing realized-pick columns"):
        load_realized_pick_surfaces({"broken": surface})
