from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_audit_verdict_fdr import (
    audit_record_from_diff,
    build_report,
    collect_audit_records,
)


def _write_diff(root: Path, name: str, deltas: dict[str, float]) -> Path:
    path = root / name / "diff.json"
    path.parent.mkdir(parents=True)
    payload = {
        "p_at_1_by_season": {
            season: {"baseline": 0.8, "variant": 0.8 + delta, "delta": delta}
            for season, delta in deltas.items()
        },
        "p_57_mdp": {"baseline": 0.04, "variant": 0.05, "delta": 0.01},
        "p_57_exact": {"baseline": 0.02, "variant": 0.03, "delta": 0.01},
        "streak_metrics": {
            "mean_max_streak": {"baseline": 30.0, "variant": 31.0, "delta": 1.0},
        },
    }
    path.write_text(json.dumps(payload))
    (path.parent / "summary.txt").write_text("KEEP | synthetic")
    return path


def test_audit_record_from_diff_uses_sign_flip_pvalue(tmp_path: Path):
    path = _write_diff(tmp_path, "feature_a", {"2024": 0.01, "2025": 0.02})

    record = audit_record_from_diff(path)

    assert record is not None
    assert record["experiment"] == "feature_a"
    assert record["mean_p_at_1_delta"] == pytest.approx(0.015)
    assert record["p_two_sided"] == pytest.approx(0.5)
    assert record["direction"] == "positive"
    assert record["summary"] == "KEEP | synthetic"


def test_collect_audit_records_skips_untestable_diffs(tmp_path: Path):
    _write_diff(tmp_path, "feature_a", {"2024": 0.01, "2025": 0.02})
    skip_path = tmp_path / "feature_b" / "diff.json"
    skip_path.parent.mkdir()
    skip_path.write_text(json.dumps({"p_57_mdp": {"delta": 0.1}}))

    records, skipped = collect_audit_records([str(tmp_path / "*" / "diff.json")])

    assert [r["experiment"] for r in records] == ["feature_a"]
    assert skipped == [skip_path.resolve().as_posix()]


def test_build_report_applies_bh_by_and_positive_survival_flags(tmp_path: Path):
    _write_diff(tmp_path, "positive", {"2024": 0.01, "2025": 0.02})
    _write_diff(tmp_path, "mixed", {"2024": 0.01, "2025": -0.02})

    report = build_report([str(tmp_path / "*" / "diff.json")], q=0.05)

    assert report["schema_version"] == "audit_verdict_fdr_v1"
    assert report["m"] == 2
    assert report["methodology"]["deploy_gate"] is None
    assert "does NOT close e-BH" in report["methodology"]["notes"]
    assert report["n_positive_survive_bh_q"] == 0
    assert report["n_positive_survive_by_q"] == 0
    assert {row["experiment"] for row in report["records"]} == {"positive", "mixed"}
    for row in report["records"]:
        assert "q_bh" in row
        assert "q_by" in row
        assert row["positive_survives_bh_q"] is False
