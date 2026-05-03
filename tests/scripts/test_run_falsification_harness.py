"""Smoke test for the falsification-harness driver."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_falsification_harness import run_harness


def _make_smoke_profiles(rng: np.random.Generator, seasons=(2023, 2024)) -> pd.DataFrame:
    """Synthetic backtest profiles for smoke tests."""
    rows = []
    for season in seasons:
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
    return pd.DataFrame(rows)


def _make_smoke_pa_df(rng: np.random.Generator, seasons=(2023, 2024)) -> pd.DataFrame:
    """Synthetic PA-level data for the dependence diagnostics."""
    pa_rows = []
    for season in seasons:
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
    return pd.DataFrame(pa_rows)


class TestHarnessSmokeTest:
    def test_emits_expected_verdict_json(self, tmp_path: Path):
        """Run on synthetic data; verify JSON has all expected v2 fields."""
        rng = np.random.default_rng(0)
        profiles = _make_smoke_profiles(rng)
        pa_df = _make_smoke_pa_df(rng)

        out_json = tmp_path / "falsification_harness.json"
        result = run_harness(
            profiles, pa_df,
            output_path=out_json,
            headline_p57_in_sample=0.0817,
            n_bootstrap=50,
            n_final=500,
            n_permutations=50,
        )
        assert out_json.exists()
        with open(out_json) as f:
            data = json.load(f)

        # Core keys (v1 + v2 both present).
        for key in (
            "headline_p57_in_sample",
            "fixed_policy_terminal_r_mc_p57",
            "pipeline_terminal_r_mc_p57",
            "rare_event_ce_p57",
            "rho_PA_within_game",
            "rho_pair_cross_game",
            "corrected_fixed_policy_p57",
            "corrected_pipeline_p57",
            "verdict_basis",
            "verdict",
            "verdict_rationale",
        ):
            assert key in data, f"missing key {key} in verdict JSON"

        # v2-specific keys.
        assert "fold_metadata" in data, "missing fold_metadata (v2 key)"
        assert "v1_reference_p57" in data, "missing v1_reference_p57 (v2 key)"
        assert "v1_reference_path" in data, "missing v1_reference_path (v2 key)"
        assert "diagnostic_heatmap_path" in data, "missing diagnostic_heatmap_path (v2 key)"

        # fold_metadata structure.
        assert isinstance(data["fold_metadata"], list)
        assert len(data["fold_metadata"]) == 2  # one per season (2 seasons in smoke data)
        fm0 = data["fold_metadata"][0]
        for fm_key in ("held_out_season", "rho_PA", "tau", "rho_pair_per_bin",
                       "rho_pair_global", "stability", "fold_p57"):
            assert fm_key in fm0, f"fold_metadata missing key {fm_key}"
        assert isinstance(fm0["stability"], dict)
        assert "small_sample_warning" in fm0["stability"]

        # Heatmap file is written alongside the verdict JSON.
        heatmap_file = out_json.parent / data["diagnostic_heatmap_path"]
        assert heatmap_file.exists(), "heatmap JSON not written"
        with open(heatmap_file) as f:
            heatmap_data = json.load(f)
        for hm_key in ("rho_matrix", "n_matrix", "bin_labels"):
            assert hm_key in heatmap_data, f"heatmap JSON missing key {hm_key}"

        # Verdict is a valid category.
        assert data["verdict"] in (
            "HEADLINE_DEFENDED", "HEADLINE_REDUCED",
            "HEADLINE_BROKEN", "HEADLINE_INCONCLUSIVE"
        )

        # rho_pair_per_bin in fold_metadata must be JSON-safe (no NaN — only None).
        raw = json.dumps(data)  # re-serialize; would raise on NaN literals
        assert "NaN" not in raw, "NaN literal found in verdict JSON (use None instead)"


def test_run_harness_forwards_block_bootstrap_kwargs(monkeypatch, tmp_path: Path):
    """run_harness must forward n_block_bootstrap and expected_block_length to corrected_audit_pipeline."""
    import scripts.run_falsification_harness as rfh

    captured: dict = {}
    real_corrected = rfh.corrected_audit_pipeline

    def capturing_corrected(*args, **kwargs):
        captured.update(kwargs)
        return real_corrected(*args, **kwargs)

    monkeypatch.setattr(rfh, "corrected_audit_pipeline", capturing_corrected)

    rng = np.random.default_rng(1)
    profiles = _make_smoke_profiles(rng)
    pa_df = _make_smoke_pa_df(rng)

    rfh.run_harness(
        profiles, pa_df,
        output_path=tmp_path / "verdict.json",
        n_bootstrap=50,
        n_permutations=10,
        pa_n_bootstrap=10,
        n_final=100,
        n_block_bootstrap=5,
        expected_block_length=14,
    )

    assert captured.get("n_block_bootstrap") == 5, (
        f"n_block_bootstrap not forwarded; captured kwargs: {list(captured)}"
    )
    assert captured.get("expected_block_length") == 14, (
        f"expected_block_length not forwarded; captured kwargs: {list(captured)}"
    )
