"""Tests for the park_drag_freshness health source (arming item 3)."""
import json
from datetime import date, datetime, timedelta

from bts.health import park_drag_freshness as pdf

TODAY = date(2026, 7, 8)


def _write(root, *, manifest=None, status=None, export=True):
    root.mkdir(parents=True, exist_ok=True)
    if export:
        (root / "park_drag_export.csv").write_text("venue_id,date,park_drag_delta\n")
    if manifest is not None:
        (root / "park_drag_manifest.json").write_text(json.dumps(manifest))
    if status is not None:
        (root / "producer_status.json").write_text(json.dumps(status))


def _fresh_manifest(days_behind=1, gen_hours_ago=2.0):
    return {
        "max_source_game_date": str(TODAY - timedelta(days=days_behind)),
        "generated_at": (datetime.now() - timedelta(hours=gen_hours_ago))
        .isoformat(timespec="seconds"),
    }


def test_missing_root_silent(tmp_path):
    assert pdf.check(tmp_path / "nope", today=TODAY) == []


def test_offseason_silent(tmp_path):
    _write(tmp_path, manifest=_fresh_manifest(days_behind=90))
    assert pdf.check(tmp_path, today=date(2026, 12, 15)) == []


def test_fresh_is_quiet(tmp_path):
    _write(tmp_path, manifest=_fresh_manifest())
    assert pdf.check(tmp_path, today=TODAY) == []


def test_failed_run_warns(tmp_path):
    _write(tmp_path, manifest=_fresh_manifest(),
           status={"ok": False, "error": "savant 403"})
    alerts = pdf.check(tmp_path, today=TODAY)
    assert [a.level for a in alerts] == ["WARN"]
    assert "savant 403" in alerts[0].message


def test_data_gap_warn_then_critical(tmp_path):
    _write(tmp_path, manifest=_fresh_manifest(days_behind=4))
    assert [a.level for a in pdf.check(tmp_path, today=TODAY)] == ["WARN"]
    _write(tmp_path, manifest=_fresh_manifest(days_behind=7))
    assert [a.level for a in pdf.check(tmp_path, today=TODAY)] == ["CRITICAL"]


def test_stale_generated_at_warns(tmp_path):
    _write(tmp_path, manifest=_fresh_manifest(gen_hours_ago=40.0))
    alerts = pdf.check(tmp_path, today=TODAY)
    assert [a.level for a in alerts] == ["WARN"]
    assert "liveness" in alerts[0].message


def test_missing_manifest_with_root_warns(tmp_path):
    _write(tmp_path, export=False)
    alerts = pdf.check(tmp_path, today=TODAY)
    assert [a.level for a in alerts] == ["WARN"]
    assert "manifest/export missing" in alerts[0].message
