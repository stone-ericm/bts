"""Tests for Tier-2 predicted-vs-realized divergence check."""

import json
from datetime import date

from bts.health.predicted_vs_realized import (
    check, compute_metrics, evaluate, SOURCE, DEFAULT_THRESHOLDS,
)


def _write_pick(picks_dir, date_iso, predicted, result):
    data = {
        "date": date_iso,
        "pick": {"batter_name": "X", "p_game_hit": predicted},
        "result": result,
    }
    (picks_dir / f"{date_iso}.json").write_text(json.dumps(data))


class TestComputeMetrics:
    def test_no_data(self, tmp_path):
        m = compute_metrics(tmp_path, today=date(2026, 4, 27))
        assert m.daily == {}
        assert m.drift is None

    def test_skips_unresolved(self, tmp_path):
        # result=null → skip
        (tmp_path / "2026-04-27.json").write_text(json.dumps({
            "date": "2026-04-27",
            "pick": {"p_game_hit": 0.75},
            "result": None,
        }))
        m = compute_metrics(tmp_path, today=date(2026, 4, 27))
        assert m.daily == {}

    def test_computes_gaps(self, tmp_path):
        # 28 days. First 14: pred=0.74, realized 8/14 (0.571). Gap 0.169.
        # Last 14: pred=0.74, realized 6/14 (0.429). Gap 0.311.
        # Drift: 0.311 - mean(both 14d) ≈ 0.071 → CRITICAL
        for i in range(1, 15):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.74, "hit" if i <= 8 else "miss")
        for i in range(15, 29):
            _write_pick(tmp_path, f"2026-04-{i:02d}", 0.74, "hit" if i <= 20 else "miss")
        m = compute_metrics(tmp_path, today=date(2026, 4, 28))
        assert len(m.daily) == 28
        assert m.rolling_14d_gap is not None
        assert m.baseline_28d_gap is not None
        assert m.drift is not None


class TestEvaluate:
    def _m(self, n_days, gap_14, gap_28):
        from bts.health.predicted_vs_realized import PredRealMetrics
        return PredRealMetrics(
            daily={f"2026-04-{i:02d}": {} for i in range(1, n_days + 1)},
            rolling_14d_gap=gap_14,
            baseline_28d_gap=gap_28,
            drift=(gap_14 - gap_28) if gap_14 is not None and gap_28 is not None else None,
        )

    def test_no_alert_drift_below_info(self):
        # 14d gap 0.10, 28d gap 0.08 → drift 0.02 < 0.03
        alerts = evaluate(self._m(28, 0.10, 0.08))
        assert alerts == []

    def test_info_drift(self):
        alerts = evaluate(self._m(28, 0.13, 0.08))  # drift 0.05
        assert len(alerts) == 1
        assert alerts[0].level in ("INFO", "WARN")

    def test_warn_drift(self):
        alerts = evaluate(self._m(28, 0.18, 0.08))  # drift 0.10 → WARN at threshold 0.08
        assert any(a.level == "WARN" or a.level == "CRITICAL" for a in alerts)

    def test_critical_drift(self):
        alerts = evaluate(self._m(28, 0.22, 0.08))  # drift 0.14 → CRITICAL at threshold 0.12
        assert any(a.level == "CRITICAL" for a in alerts)

    def test_negative_drift_no_alert(self):
        # 14d gap 0.05, 28d gap 0.10 → drift -0.05 (improvement, no alert)
        alerts = evaluate(self._m(28, 0.05, 0.10))
        assert alerts == []

    def test_insufficient_data(self):
        # n=5 days < min_days_14d=10 → no alert even with big drift
        alerts = evaluate(self._m(5, 0.20, 0.08))
        assert alerts == []

    def test_source(self):
        alerts = evaluate(self._m(28, 0.18, 0.08))
        assert all(a.source == SOURCE for a in alerts)

    def test_no_alert_when_drift_none(self):
        from bts.health.predicted_vs_realized import PredRealMetrics
        m = PredRealMetrics(daily={}, rolling_14d_gap=None, baseline_28d_gap=None, drift=None)
        assert evaluate(m) == []
