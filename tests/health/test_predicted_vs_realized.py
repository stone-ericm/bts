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

    def test_void_primary_still_counts_graded_dd_slot(self, tmp_path):
        # Per-slot basis: a void primary is skipped, but the DD leg was a real
        # delivered prediction with a real outcome — it must still be graded.
        (tmp_path / "2026-04-27.json").write_text(json.dumps({
            "date": "2026-04-27",
            "pick": {"p_game_hit": 0.75},
            "double_down": {"p_game_hit": 0.72},
            "result": "hit",
            "slot_results": {"pick": "void", "double_down": "hit"},
        }))
        m = compute_metrics(tmp_path, today=date(2026, 4, 27))
        assert list(m.daily) == ["2026-04-27"]
        # One graded slot: the DD leg (0.72 predicted, realized hit)
        assert m.rolling_14d_gap is not None
        assert abs(m.rolling_14d_gap - (0.72 - 1)) < 1e-9

    def test_all_slots_void_skips_day(self, tmp_path):
        (tmp_path / "2026-04-27.json").write_text(json.dumps({
            "date": "2026-04-27",
            "pick": {"p_game_hit": 0.75},
            "result": "hit",
            "slot_results": {"pick": "void"},
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


class TestPerSlotBasis:
    """DD-attribution fix (2026-07-12 incident).

    The check compared the PRIMARY's p_game_hit against the DAY-level result,
    but on double-down days the day result requires BOTH legs to hit — a
    DD-dense stretch (exactly what the MDP produces at streak 0) mechanically
    inflates the gap regardless of model health. Live decomposition that
    night: day-level drift +0.1737 (CRITICAL) vs primary-only +0.042 (quiet);
    the real signal was the DD legs going 1-for-6. Per-slot grading (the same
    attribution realized_calibration adopted 2026-05-01) scores every
    delivered leg against its own p, so both a primary collapse and a DD-leg
    collapse surface honestly.
    """

    def test_dd_day_grades_each_slot_against_its_own_p(self, tmp_path):
        # Primary hit, DD leg missed → day-level result is "miss", but the
        # slots are (0.78→1) and (0.74→0): gap = mean(p) - mean(realized).
        (tmp_path / "2026-04-27.json").write_text(json.dumps({
            "date": "2026-04-27",
            "pick": {"p_game_hit": 0.78},
            "double_down": {"p_game_hit": 0.74},
            "result": "miss",
            "slot_results": {"pick": "hit", "double_down": "miss"},
        }))
        m = compute_metrics(tmp_path, today=date(2026, 4, 27))
        assert list(m.daily) == ["2026-04-27"]
        expected = (0.78 + 0.74) / 2 - 0.5
        assert abs(m.rolling_14d_gap - expected) < 1e-9

    def test_dd_day_without_slot_results_miss_is_ambiguous_and_skipped(self, tmp_path):
        # A day-level "miss" on a DD day without slot_results can't be
        # attributed to a leg — excluding it beats poisoning the series.
        (tmp_path / "2026-04-27.json").write_text(json.dumps({
            "date": "2026-04-27",
            "pick": {"p_game_hit": 0.78},
            "double_down": {"p_game_hit": 0.74},
            "result": "miss",
        }))
        m = compute_metrics(tmp_path, today=date(2026, 4, 27))
        assert m.daily == {}

    def test_dd_day_without_slot_results_excluded_even_on_hit(self, tmp_path):
        # Round-2 review #7: including legacy-DD hit days while skipping the
        # unattributable miss days is outcome-dependent censoring — it
        # inflates realized and can MASK degradation. Legacy DD days are
        # excluded symmetrically.
        (tmp_path / "2026-04-27.json").write_text(json.dumps({
            "date": "2026-04-27",
            "pick": {"p_game_hit": 0.78},
            "double_down": {"p_game_hit": 0.74},
            "result": "hit",
        }))
        m = compute_metrics(tmp_path, today=date(2026, 4, 27))
        assert m.daily == {}

    def test_single_pick_day_unchanged(self, tmp_path):
        # No DD, no slot_results: day result IS the primary outcome (legacy).
        _write_pick(tmp_path, "2026-04-27", 0.80, "miss")
        m = compute_metrics(tmp_path, today=date(2026, 4, 27))
        assert abs(m.rolling_14d_gap - 0.80) < 1e-9


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

    def test_moderate_drift_is_warn_not_critical(self):
        # Round-2 review #9: the drift of overlapping 14d/28d windows has
        # SE ≈ 0.094 at day level — 0.12-0.14 is ~1.3σ, not CRITICAL signal.
        # CRITICAL is reserved for catastrophic/pipeline-scale drift (≥0.25)
        # pending a day-block bootstrap recalibration.
        alerts = evaluate(self._m(28, 0.22, 0.08))  # drift 0.14
        assert len(alerts) == 1 and alerts[0].level == "WARN"

    def test_critical_drift(self):
        alerts = evaluate(self._m(28, 0.36, 0.08))  # drift 0.28 ≥ 0.25 → CRITICAL
        assert any(a.level == "CRITICAL" for a in alerts)

    def test_negative_drift_no_alert(self):
        # 14d gap 0.05, 28d gap 0.10 → drift -0.05 (improvement, no alert)
        alerts = evaluate(self._m(28, 0.05, 0.10))
        assert alerts == []

    def test_insufficient_data(self):
        # n=5 days < min_days_14d=10 → no alert even with big drift
        alerts = evaluate(self._m(5, 0.20, 0.08))
        assert alerts == []

    def test_insufficient_28d_baseline_suppresses_drift(self):
        # 10 <= n < min_days_28d (20): the 14d gate passes but the 28d baseline is
        # too thin to be a real baseline (it overlaps the 14d window), so a big
        # drift must NOT alert. min_days_28d was previously never enforced and
        # n14 counted the whole lookback rather than the window (audit).
        alerts = evaluate(self._m(15, 0.22, 0.08))  # drift 0.14, would be CRITICAL
        assert alerts == [], alerts

    def test_source(self):
        alerts = evaluate(self._m(28, 0.18, 0.08))
        assert all(a.source == SOURCE for a in alerts)

    def test_no_alert_when_drift_none(self):
        from bts.health.predicted_vs_realized import PredRealMetrics
        m = PredRealMetrics(daily={}, rolling_14d_gap=None, baseline_28d_gap=None, drift=None)
        assert evaluate(m) == []
