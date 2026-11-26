import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bts.simulate.mdp import ACTIONS
from scripts.measure_raw_rebin_gate import (
    load_resolved_pair_rows,
    project_policy_to_candidate_bins,
    quality_bins_from_pair_rows,
    run_measurement,
)


def _write_policy(path, *, n_days=12):
    policy = np.ones((58, n_days + 1, 2, 5), dtype=np.int8)
    policy[:, :, :, 0] = ACTIONS.index("double")
    np.savez_compressed(
        path,
        policy_table=policy,
        boundaries=np.array([0.80, 0.82, 0.84, 0.86]),
        season_length=np.array(n_days),
        optimal_p57=np.array(0.0),
    )


def _write_pick(picks_dir, d, p1, p2, b1, b2, y1, y2):
    body = {
        "date": d.isoformat(),
        "result": "hit" if y1 else "miss",
        "pick": {"batter_id": b1, "p_game_hit": p1},
        "double_down": {"batter_id": b2, "p_game_hit": p2},
    }
    (picks_dir / f"{d.isoformat()}.json").write_text(json.dumps(body))


def _fixture_files(tmp_path, *, n_days=12):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    pa_rows = []
    start = date(2026, 4, 1)
    for i in range(n_days):
        d = start + timedelta(days=i)
        b1 = 1000 + i
        b2 = 2000 + i
        y1 = 1 if i % 3 else 0
        y2 = 1 if i % 4 else 0
        _write_pick(
            picks_dir,
            d,
            p1=0.70 + 0.005 * i,
            p2=0.68 + 0.004 * i,
            b1=b1,
            b2=b2,
            y1=y1,
            y2=y2,
        )
        pa_rows.append({"batter_id": b1, "date": pd.Timestamp(d), "is_hit": y1})
        pa_rows.append({"batter_id": b2, "date": pd.Timestamp(d), "is_hit": y2})

    pa_parquet = tmp_path / "pa_2026.parquet"
    pd.DataFrame(pa_rows).to_parquet(pa_parquet)
    policy_path = tmp_path / "mdp_policy.npz"
    _write_policy(policy_path)
    return picks_dir, pa_parquet, policy_path


def test_load_resolved_pair_rows_joins_day_hits(tmp_path):
    picks_dir, pa_parquet, _policy_path = _fixture_files(tmp_path, n_days=3)

    rows = load_resolved_pair_rows(picks_dir, pa_parquet, date(2026, 4, 10))

    assert len(rows) == 3
    assert rows[0].date == "2026-04-01"
    assert rows[0].p1 == pytest.approx(0.70)
    assert rows[0].y1 == 0
    assert rows[1].y1 == 1


def test_project_policy_to_candidate_bins_maps_to_old_quality_bin(tmp_path):
    picks_dir, pa_parquet, policy_path = _fixture_files(tmp_path, n_days=10)
    rows = load_resolved_pair_rows(picks_dir, pa_parquet, date(2026, 4, 20))
    bins = quality_bins_from_pair_rows(rows, 2)
    policy = np.load(policy_path)["policy_table"]
    boundaries = [0.80, 0.82, 0.84, 0.86]

    projected, mapping = project_policy_to_candidate_bins(policy, boundaries, bins)

    assert projected.shape[-1] == 2
    assert mapping == [0, 0]
    assert set(projected[:, :, :, 0].ravel()) == {ACTIONS.index("double")}


def test_run_measurement_reports_insufficient_support_and_profile_mismatch(tmp_path):
    picks_dir, pa_parquet, policy_path = _fixture_files(tmp_path, n_days=12)
    profiles_dir = tmp_path / "simulation"
    profiles_dir.mkdir()
    profile_rows = []
    for i in range(20):
        d = pd.Timestamp("2025-04-01") + pd.Timedelta(days=i)
        profile_rows.append({"date": d, "rank": 1, "p_game_hit": 0.82 + i * 0.001})
        profile_rows.append({"date": d, "rank": 2, "p_game_hit": 0.79 + i * 0.001})
    pd.DataFrame(profile_rows).to_parquet(profiles_dir / "backtest_2025.parquet")

    result = run_measurement(
        picks_dir=picks_dir,
        pa_parquet=pa_parquet,
        today=date(2026, 4, 20),
        policy_path=policy_path,
        profiles_dir=profiles_dir,
        n_bins_values=(2, 3),
        min_n=50,
        min_per_bin=10,
        season_length=12,
    )

    assert result["production_deploy_claim"] is False
    assert result["heavy_compute"] is False
    assert result["row_summary"]["n"] == 12
    assert result["gate_b"]["decision"] == "INSUFFICIENT_SUPPORT"
    assert len(result["gate_b"]["evaluations"]) == 2
    assert result["distribution_mismatch"]["current_max_below_profile_q20"] is True
    json.dumps(result)
