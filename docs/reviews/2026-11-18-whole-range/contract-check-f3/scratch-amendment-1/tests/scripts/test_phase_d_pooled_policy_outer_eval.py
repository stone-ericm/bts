import json
from pathlib import Path

import pytest

from scripts.phase_d_pooled_policy_outer_eval import (
    derive_verdict,
    discover_seed_records,
    parse_seasons,
    summarize_p_at_1,
    summarize_policy_gaps,
)


def _write_root_metadata(root: Path, provider: str) -> None:
    root.mkdir(parents=True)
    (root / "audit_validation_split.json").write_text(json.dumps({
        "artifact_role": "raw_backtest_profile_surface",
        "split_mode": "season_level_selection_outer_eval",
        "selection_seasons": [2021, 2022, 2023, 2024],
        "outer_eval_seasons": [2025],
        "production_deploy_claim": False,
        "audit_driver": {
            "provider": provider,
            "run_kind": "profiles",
            "queue_mode": "backtest",
        },
    }))


def _write_seed_dir(root: Path, box: str, seed: int, provider: str) -> Path:
    seed_dir = root / box / f"simulation_seed{seed}"
    seed_dir.mkdir(parents=True)
    (seed_dir / "audit_validation_split.json").write_text(json.dumps({
        "audit_driver": {
            "provider": provider,
            "box_name": box,
            "box_region": "test-region",
        }
    }))
    for season in [2021, 2022, 2023, 2024, 2025]:
        (seed_dir / f"backtest_{season}.parquet").write_text("placeholder")
    return seed_dir


def test_parse_seasons_rejects_empty():
    assert parse_seasons("2021, 2022,2023") == [2021, 2022, 2023]
    with pytest.raises(ValueError, match="must not be empty"):
        parse_seasons(" , ")


def test_discover_seed_records_preserves_provider_and_rejects_duplicates(tmp_path: Path):
    h_root = tmp_path / "hetzner"
    o_root = tmp_path / "oci"
    _write_root_metadata(h_root, "hetzner")
    _write_root_metadata(o_root, "oci")
    _write_seed_dir(h_root, "h-box", 11, "hetzner")
    _write_seed_dir(o_root, "o-box", 12, "oci")

    records, metadata = discover_seed_records(
        [h_root, o_root],
        selection_seasons=[2021, 2022, 2023, 2024],
        outer_eval_seasons=[2025],
        expect_seeds=2,
    )

    assert [(r.provider, r.seed, r.box) for r in records] == [
        ("hetzner", 11, "h-box"),
        ("oci", 12, "o-box"),
    ]
    assert [item["metadata"]["audit_driver"]["provider"] for item in metadata] == [
        "hetzner",
        "oci",
    ]

    _write_seed_dir(o_root, "o-box-duplicate", 11, "oci")
    with pytest.raises(ValueError, match="duplicate seed 11"):
        discover_seed_records(
            [h_root, o_root],
            selection_seasons=[2021, 2022, 2023, 2024],
            outer_eval_seasons=[2025],
        )


def test_summarize_policy_gaps_supports_provider_stratified_bootstrap():
    rows = [
        {"provider": "hetzner", "seed": 1, "v_prod": 0.01, "v_pooled": 0.03, "gap": 0.02},
        {"provider": "hetzner", "seed": 2, "v_prod": 0.02, "v_pooled": 0.04, "gap": 0.02},
        {"provider": "oci", "seed": 3, "v_prod": 0.03, "v_pooled": 0.06, "gap": 0.03},
        {"provider": "oci", "seed": 4, "v_prod": 0.04, "v_pooled": 0.08, "gap": 0.04},
    ]

    result = summarize_policy_gaps(
        rows,
        n_bootstrap=300,
        seed=7,
        stratify_by_provider=True,
    )

    assert result["n"] == 4
    assert result["mean_gap"] == pytest.approx(0.0275)
    assert result["bootstrap"]["kind"] == "provider_stratified_seed_bootstrap"
    assert result["bootstrap"]["ci_lower"] > 0
    assert result["n_positive"] == 4
    assert result["exact_sign_p_two_sided"] == pytest.approx(0.125)


def test_derive_verdict_requires_positive_ci_and_provider_agreement():
    overall = {
        "mean_gap": 0.02,
        "bootstrap": {"ci_lower": 0.001, "ci_upper": 0.04},
    }
    providers = {
        "hetzner": {"mean_gap": 0.01},
        "oci": {"mean_gap": 0.03},
    }
    assert derive_verdict(overall, providers)["verdict"] == "survives_outer_eval"
    assert derive_verdict(overall, providers)["production_deploy_ready"] is False

    uncertain = {
        "mean_gap": 0.02,
        "bootstrap": {"ci_lower": -0.001, "ci_upper": 0.04},
    }
    assert derive_verdict(uncertain, providers)["verdict"] == "inconclusive"

    split_provider = {
        "hetzner": {"mean_gap": 0.01},
        "oci": {"mean_gap": -0.001},
    }
    assert derive_verdict(overall, split_provider)["verdict"] == "inconclusive"

    negative = {
        "mean_gap": -0.001,
        "bootstrap": {"ci_lower": -0.01, "ci_upper": 0.01},
    }
    assert derive_verdict(negative, providers)["verdict"] == "falsified"


def test_summarize_p_at_1_marks_policy_gap_not_applicable():
    rows = [
        {"outer_p_at_1": 0.6},
        {"outer_p_at_1": 0.8},
    ]

    result = summarize_p_at_1(rows)

    assert result["mean_seed_outer_p_at_1"] == pytest.approx(0.7)
    assert result["gap_candidate_vs_prod"] is None
    assert "does not change the rank-1 probability model" in result["gap_not_applicable_reason"]
