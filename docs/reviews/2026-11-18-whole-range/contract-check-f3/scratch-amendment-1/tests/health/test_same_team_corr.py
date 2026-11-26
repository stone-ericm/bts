"""Tests for Tier-2 same-team correlation drift check."""

import json
from datetime import date

from bts.health.same_team_corr import (
    RESIDUAL_SOURCE, SOURCE, check, compute_metrics, evaluate, CorrMetrics,
)


def _write_pair(picks_dir, date_iso, p1, pdd, result, slot_results=None):
    data = {
        "date": date_iso,
        "pick": {"batter_name": "X", "p_game_hit": p1},
        "double_down": {"batter_name": "Y", "p_game_hit": pdd},
        "result": result,
    }
    if slot_results is not None:
        data["slot_results"] = slot_results
    (picks_dir / f"{date_iso}.json").write_text(json.dumps(data))


class TestComputeMetrics:
    def test_skips_no_dd_days(self, tmp_path):
        (tmp_path / "2026-04-27.json").write_text(json.dumps({
            "date": "2026-04-27",
            "pick": {"p_game_hit": 0.75},
            "result": "hit",
        }))
        m = compute_metrics(tmp_path, today=date(2026, 4, 27))
        assert m.pair_days == []

    def test_skips_unresolved(self, tmp_path):
        _write_pair(tmp_path, "2026-04-27", 0.75, 0.72, None)
        m = compute_metrics(tmp_path, today=date(2026, 4, 27))
        assert m.pair_days == []

    def test_skips_partial_void_pair(self, tmp_path):
        data = {
            "date": "2026-04-27",
            "pick": {"batter_name": "X", "p_game_hit": 0.75},
            "double_down": {"batter_name": "Y", "p_game_hit": 0.72},
            "result": "hit",
            "slot_results": {"pick": "void", "double_down": "hit"},
        }
        (tmp_path / "2026-04-27.json").write_text(json.dumps(data))
        m = compute_metrics(tmp_path, today=date(2026, 4, 27))
        assert m.pair_days == []

    def test_computes_independence_baseline(self, tmp_path):
        # 4 pair days, all p1=0.8 pdd=0.7 → predicted 0.56 each
        # Realized: 2 hits, 2 misses → mean realized 0.5
        # Single-window gap: 0.06
        for i in range(1, 5):
            _write_pair(tmp_path, f"2026-04-{i:02d}", 0.8, 0.7,
                       "hit" if i <= 2 else "miss")
        m = compute_metrics(tmp_path, today=date(2026, 4, 5))
        assert len(m.pair_days) == 4
        # 14d and 28d both windows = same 4 days, so gap is same → drift=0
        assert m.drift == 0

    def test_computes_residual_gap_from_slot_marginals(self, tmp_path):
        # Primary 2/4, DD 2/4, both 1/4 => empirical marginal product 0.25,
        # observed both 0.25, so no residual pair shortfall.
        rows = [
            ("hit", {"pick": "hit", "double_down": "hit"}),
            ("miss", {"pick": "hit", "double_down": "miss"}),
            ("miss", {"pick": "miss", "double_down": "hit"}),
            ("miss", {"pick": "miss", "double_down": "miss"}),
        ]
        for i, (result, slot_results) in enumerate(rows, start=1):
            _write_pair(
                tmp_path,
                f"2026-04-{i:02d}",
                0.8,
                0.8,
                result,
                slot_results=slot_results,
            )

        m = compute_metrics(tmp_path, today=date(2026, 4, 5))

        assert m.rolling_14d_residual_gap == 0
        assert m.n_residual_14d == 4


class TestEvaluate:
    def _m(self, n_days, gap_14, gap_28):
        return CorrMetrics(
            pair_days=[{"date": f"2026-04-{i:02d}"} for i in range(1, n_days + 1)],
            rolling_14d_gap=gap_14,
            baseline_28d_gap=gap_28,
            drift=(gap_14 - gap_28) if gap_14 is not None and gap_28 is not None else None,
        )

    def test_no_alert_below_info(self):
        alerts = evaluate(self._m(28, 0.07, 0.05))  # drift 0.02 < info=0.05
        assert alerts == []

    def test_info_drift(self):
        alerts = evaluate(self._m(28, 0.12, 0.05))  # drift 0.07
        assert any(a.level == "INFO" for a in alerts)

    def test_warn_drift(self):
        alerts = evaluate(self._m(28, 0.18, 0.05))  # drift 0.13
        assert any(a.level == "WARN" or a.level == "CRITICAL" for a in alerts)

    def test_critical_drift(self):
        alerts = evaluate(self._m(28, 0.22, 0.05))  # drift 0.17
        assert any(a.level == "CRITICAL" for a in alerts)
        assert any(a.source == SOURCE for a in alerts)

    def test_insufficient_data(self):
        # n=5 < min_days_14d=8 → no alert
        alerts = evaluate(self._m(5, 0.20, 0.05))
        assert alerts == []

    def test_source(self):
        alerts = evaluate(self._m(28, 0.22, 0.05))
        assert all(a.source == SOURCE for a in alerts)

    def test_negative_drift_no_alert(self):
        alerts = evaluate(self._m(28, 0.05, 0.10))
        assert alerts == []

    def test_model_shortfall_does_not_imply_residual_corr(self):
        metrics = self._m(28, 0.22, 0.05)
        metrics = CorrMetrics(
            pair_days=metrics.pair_days,
            rolling_14d_gap=metrics.rolling_14d_gap,
            baseline_28d_gap=metrics.baseline_28d_gap,
            drift=metrics.drift,
            rolling_14d_residual_gap=0.0,
            baseline_28d_residual_gap=0.0,
            residual_drift=0.0,
            n_residual_14d=14,
        )

        alerts = evaluate(metrics)

        assert any(a.source == SOURCE for a in alerts)
        assert not any(a.source == RESIDUAL_SOURCE for a in alerts)

    def test_residual_pair_shortfall_gets_separate_source(self):
        metrics = self._m(28, 0.04, 0.00)
        metrics = CorrMetrics(
            pair_days=metrics.pair_days,
            rolling_14d_gap=metrics.rolling_14d_gap,
            baseline_28d_gap=metrics.baseline_28d_gap,
            drift=metrics.drift,
            rolling_14d_residual_gap=0.12,
            baseline_28d_residual_gap=0.02,
            residual_drift=0.10,
            n_residual_14d=14,
        )

        alerts = evaluate(metrics)

        assert any(a.source == RESIDUAL_SOURCE and a.level == "WARN" for a in alerts)
