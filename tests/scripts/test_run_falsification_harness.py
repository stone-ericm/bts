"""Smoke test for the falsification-harness driver."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_falsification_harness import run_harness


class TestHarnessSmokeTest:
    def test_emits_expected_verdict_json(self, tmp_path: Path):
        """Run on synthetic data; verify JSON has all expected fields."""
        rng = np.random.default_rng(0)
        rows = []
        for season in [2023, 2024]:
            for d in range(50):
                date = pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=d)
                for seed in range(5):
                    rows.append({
                        "season": season,
                        "date": date,
                        "seed": seed,
                        "top1_p": rng.uniform(0.65, 0.90),
                        "top1_hit": int(rng.random() < 0.78),
                        "top2_p": rng.uniform(0.65, 0.85),
                        "top2_hit": int(rng.random() < 0.75),
                    })
        profiles = pd.DataFrame(rows)
        # Synthetic PA-level data shaped for the dependence diagnostics.
        pa_rows = []
        for season in [2023, 2024]:
            for d in range(50):
                for batter in range(8):
                    bg_id = f"{season}_{d}_{batter}"
                    for pa in range(5):
                        pa_rows.append({
                            "season": season,
                            "date": pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=d),
                            "batter_game_id": bg_id,
                            "pa_index": pa,
                            "p_pa": rng.uniform(0.20, 0.30),
                            "actual_hit": int(rng.random() < 0.25),
                        })
        pa_df = pd.DataFrame(pa_rows)

        out_json = tmp_path / "falsification_harness.json"
        result = run_harness(
            profiles, pa_df,
            output_path=out_json,
            headline_p57_in_sample=0.0817,
            n_bootstrap=200,
            n_final=2000,
        )
        assert out_json.exists()
        with open(out_json) as f:
            data = json.load(f)
        for key in (
            "headline_p57_in_sample",
            "fixed_policy_terminal_r_mc_p57",
            "pipeline_terminal_r_mc_p57",
            "rare_event_ce_p57",
            "rho_PA_within_game",
            "rho_pair_cross_game",
            "corrected_pipeline_p57",
            "verdict",
            "verdict_rationale",
        ):
            assert key in data, f"missing key {key} in verdict JSON"
        assert data["verdict"] in ("HEADLINE_DEFENDED", "HEADLINE_REDUCED", "HEADLINE_BROKEN", "HEADLINE_INCONCLUSIVE")
