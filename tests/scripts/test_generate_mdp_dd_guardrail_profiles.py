from __future__ import annotations

import pandas as pd

from scripts.generate_mdp_dd_guardrail_profiles import profile_output_manifest


def test_profile_output_manifest_reports_hashes_schema_and_ambiguity(tmp_path):
    output_dir = tmp_path / "profiles"
    output_dir.mkdir()
    rows = [
        {
            "date": "2021-04-01",
            "season": 2021,
            "rank": 1,
            "batter_id": 1,
            "game_pk": 100,
            "p_game_hit": 0.70,
            "actual_hit": 1,
        },
        {
            "date": "2021-04-01",
            "season": 2021,
            "rank": 2,
            "batter_id": 1,
            "game_pk": 101,
            "p_game_hit": 0.69,
            "actual_hit": 0,
        },
        {
            "date": "2021-04-01",
            "season": 2021,
            "rank": 3,
            "batter_id": 2,
            "game_pk": 102,
            "p_game_hit": 0.68,
            "actual_hit": 1,
        },
    ]
    pd.DataFrame(rows).to_parquet(output_dir / "backtest_2021.parquet")

    manifest = profile_output_manifest(output_dir, [2021])
    season = manifest["seasons"]["2021"]

    assert manifest["valid"] is True
    assert season["sha256"]
    assert season["row_count"] == 3
    assert season["date_count"] == 1
    assert season["null_counts"]["game_pk"] == 0
    assert season["duplicate_date_batter_rows"] == 2
    assert season["ambiguous_date_batter_rows"] == 2
