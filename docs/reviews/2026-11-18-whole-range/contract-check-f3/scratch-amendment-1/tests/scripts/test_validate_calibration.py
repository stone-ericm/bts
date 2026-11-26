import json
from datetime import date, timedelta

import pandas as pd
import pytest

from scripts.validate_calibration import (
    cross_validate_isotonic,
    run_validation,
    sample_summary,
)


pytest.importorskip("sklearn.isotonic")


def test_sample_summary_reports_fixed_probability_buckets():
    samples = [(0.68, 1), (0.72, 0), (0.77, 1), (0.81, 1)]

    summary = sample_summary(samples)

    assert summary["n"] == 4
    assert summary["raw_mean_p"] == pytest.approx(0.745)
    assert summary["hit_rate"] == pytest.approx(0.75)
    assert [b["n"] for b in summary["buckets"]] == [1, 1, 1, 1]


def test_cross_validate_waits_for_min_n_after_running_cv():
    samples = [
        (0.60 + 0.01 * (i % 5), 1 if i % 3 else 0)
        for i in range(30)
    ]

    result = cross_validate_isotonic(
        samples,
        n_folds=3,
        bootstrap_reps=20,
        seed=7,
        min_n=40,
    )

    assert result["decision"] == "WAIT_FOR_N"
    assert result["n_samples"] == 30
    assert result["min_n"] == 40
    assert "improvement" in result
    assert len(result["fold_results"]) == 3


def test_cross_validate_rejects_too_few_samples_for_folds():
    result = cross_validate_isotonic(
        [(0.7, 1)] * 9,
        n_folds=2,
        bootstrap_reps=20,
        min_n=10,
    )

    assert result["decision"] == "INSUFFICIENT_FOLDS"
    assert result["n_samples"] == 9


def test_run_validation_loads_pick_and_pa_files(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    pa_rows = []
    start = date(2026, 4, 1)
    for i in range(6):
        d = start + timedelta(days=i)
        primary_id = 1000 + i
        dd_id = 2000 + i
        primary_hit = 1 if i % 2 else 0
        dd_hit = 1 if i % 3 else 0
        body = {
            "date": d.isoformat(),
            "result": "hit" if primary_hit else "miss",
            "pick": {"batter_id": primary_id, "p_game_hit": 0.74},
            "double_down": {"batter_id": dd_id, "p_game_hit": 0.71},
        }
        (picks_dir / f"{d.isoformat()}.json").write_text(json.dumps(body))
        pa_rows.append({"batter_id": primary_id, "date": pd.Timestamp(d), "is_hit": primary_hit})
        pa_rows.append({"batter_id": dd_id, "date": pd.Timestamp(d), "is_hit": dd_hit})

    pa_parquet = tmp_path / "pa_2026.parquet"
    pd.DataFrame(pa_rows).to_parquet(pa_parquet)

    result = run_validation(
        picks_dir=picks_dir,
        pa_parquet=pa_parquet,
        today=date(2026, 4, 10),
        lookback_days=30,
        n_folds=2,
        bootstrap_reps=20,
        min_n=20,
    )

    assert result["sample_summary"]["n"] == 12
    assert result["cross_validation"]["decision"] == "WAIT_FOR_N"
    json.dumps(result)
