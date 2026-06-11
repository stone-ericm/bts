"""Tests for the slate AUC health check (M3 revisit trigger).

The check computes a rolling realized AUC over persisted daily slates joined
to PA outcomes. It exists to make the M3 revisit trigger observable: the
serving-staleness HOLD (docs/audit/2026-06-11-m3-serving-staleness.md) is
worth revisiting once the model discriminates adjacent candidates materially
better than the ~0.59 replay baseline.
"""

import json
from datetime import date, timedelta

import pandas as pd

from bts.health.slate_auc import (
    DEFAULT_THRESHOLDS,
    SOURCE,
    _rank_auc,
    _status_path,
    check,
)


def _write_slate(picks_dir, d, rows):
    slates = picks_dir / "slates"
    slates.mkdir(parents=True, exist_ok=True)
    (slates / f"{d}.json").write_text(json.dumps({
        "schema_version": "bts_slate_v1",
        "date": str(d),
        "tier": "hetzner",
        "written_at": f"{d}T15:00:00Z",
        "n_rows": len(rows),
        "rows": rows,
    }))


def _write_outcomes(data_dir, year, records):
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(data_dir / f"pa_{year}.parquet")


def _build_window(picks_dir, data_dir, today, n_days, auc_high=False):
    """n_days of slates; outcomes arranged for high or chance-level AUC."""
    pa_records = []
    for i in range(n_days):
        d = today - timedelta(days=i + 1)
        rows = []
        for j in range(10):
            batter = 1000 + j
            game = 50000 + i
            p = 0.85 - 0.05 * j
            rows.append({"batter_id": batter, "game_pk": game, "p_game_hit": p})
            if auc_high:
                hit = 1 if j < 5 else 0  # top-scored candidates hit -> AUC 1.0
            else:
                hit = 1 if j % 2 == 0 else 0  # interleaved -> mid AUC
            pa_records.append({
                "batter_id": batter, "game_pk": game,
                "date": str(d), "is_hit": hit,
            })
        _write_slate(picks_dir, d, rows)
    _write_outcomes(data_dir, today.year, pa_records)


def test_rank_auc_known_values():
    assert _rank_auc([0.9, 0.8], [0.7, 0.6]) == 1.0
    assert _rank_auc([0.8, 0.6], [0.7, 0.5]) == 0.75
    assert _rank_auc([0.7], [0.7]) == 0.5  # tie -> 0.5 credit
    assert _rank_auc([], [0.5]) is None
    assert _rank_auc([0.5], []) is None


def test_no_alert_when_no_slates(tmp_path):
    assert check(tmp_path / "picks", data_dir=tmp_path / "data", today=date(2026, 6, 11)) == []


def test_no_alert_below_revisit_threshold(tmp_path):
    today = date(2026, 6, 11)
    _build_window(tmp_path / "picks", tmp_path / "data", today,
                  n_days=DEFAULT_THRESHOLDS["min_days"], auc_high=False)

    alerts = check(tmp_path / "picks", data_dir=tmp_path / "data", today=today)

    assert alerts == []
    status = json.loads(_status_path(tmp_path / "picks").read_text())
    assert status["auc"] is not None
    assert status["auc"] < DEFAULT_THRESHOLDS["revisit_auc"]
    assert status["n_days"] == DEFAULT_THRESHOLDS["min_days"]


def test_warn_with_m3_pointer_at_revisit_threshold(tmp_path):
    today = date(2026, 6, 11)
    _build_window(tmp_path / "picks", tmp_path / "data", today,
                  n_days=DEFAULT_THRESHOLDS["min_days"], auc_high=True)

    alerts = check(tmp_path / "picks", data_dir=tmp_path / "data", today=today)

    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert alerts[0].source == SOURCE
    assert "M3" in alerts[0].message
    assert "replay_m3_serving_parity" in alerts[0].message


def test_insufficient_days_writes_status_but_no_alert(tmp_path):
    today = date(2026, 6, 11)
    _build_window(tmp_path / "picks", tmp_path / "data", today,
                  n_days=DEFAULT_THRESHOLDS["min_days"] - 1, auc_high=True)

    alerts = check(tmp_path / "picks", data_dir=tmp_path / "data", today=today)

    assert alerts == []
    status = json.loads(_status_path(tmp_path / "picks").read_text())
    assert status["reason"] == "insufficient_days"


def test_fresh_cache_skips_recompute(tmp_path):
    today = date(2026, 6, 11)
    _build_window(tmp_path / "picks", tmp_path / "data", today,
                  n_days=DEFAULT_THRESHOLDS["min_days"], auc_high=False)

    check(tmp_path / "picks", data_dir=tmp_path / "data", today=today)
    first = json.loads(_status_path(tmp_path / "picks").read_text())

    # underlying data changes, but cache is fresh -> no recompute
    _build_window(tmp_path / "picks", tmp_path / "data", today,
                  n_days=DEFAULT_THRESHOLDS["min_days"], auc_high=True)
    alerts = check(tmp_path / "picks", data_dir=tmp_path / "data", today=today)

    second = json.loads(_status_path(tmp_path / "picks").read_text())
    assert second == first
    assert alerts == []  # cached level was no-alert


def test_cached_warn_re_emits_without_recompute(tmp_path):
    today = date(2026, 6, 11)
    _build_window(tmp_path / "picks", tmp_path / "data", today,
                  n_days=DEFAULT_THRESHOLDS["min_days"], auc_high=True)

    first = check(tmp_path / "picks", data_dir=tmp_path / "data", today=today)
    second = check(tmp_path / "picks", data_dir=tmp_path / "data", today=today)

    assert [a.level for a in first] == ["WARN"]
    assert [a.level for a in second] == ["WARN"]


def test_stale_cache_recomputes(tmp_path):
    today = date(2026, 6, 11)
    picks, data = tmp_path / "picks", tmp_path / "data"
    _build_window(picks, data, today, n_days=DEFAULT_THRESHOLDS["min_days"], auc_high=False)
    check(picks, data_dir=data, today=today)

    later = today + timedelta(days=DEFAULT_THRESHOLDS["recompute_every_days"] + 1)
    _build_window(picks, data, later, n_days=DEFAULT_THRESHOLDS["min_days"], auc_high=True)
    alerts = check(picks, data_dir=data, today=later)

    assert [a.level for a in alerts] == ["WARN"]


def test_outcomes_unjoinable_no_alert(tmp_path):
    today = date(2026, 6, 11)
    picks, data = tmp_path / "picks", tmp_path / "data"
    _build_window(picks, data, today, n_days=DEFAULT_THRESHOLDS["min_days"], auc_high=True)
    # wipe outcomes: parquet exists but matches nothing
    _write_outcomes(data, today.year, [{
        "batter_id": 1, "game_pk": 1, "date": str(today), "is_hit": 1,
    }])

    alerts = check(picks, data_dir=data, today=today)

    assert alerts == []
    status = json.loads(_status_path(picks).read_text())
    assert status["reason"] in ("insufficient_rows", "insufficient_days")
