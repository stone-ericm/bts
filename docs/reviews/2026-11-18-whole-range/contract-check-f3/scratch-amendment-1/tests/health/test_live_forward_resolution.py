import json
from datetime import date
from pathlib import Path

from bts.health.live_forward_resolution import SOURCE, check


def _write_preoutcome(root: Path, date_iso: str) -> None:
    artifact = root / date_iso
    artifact.mkdir(parents=True)
    (artifact / "manifest.json").write_text(json.dumps({
        "run_kind": "live_forward_preoutcome",
        "date": date_iso,
        "dates": [date_iso],
    }))


def _write_status(root: Path, date_iso: str, status: str, **extra) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "date": date_iso,
        "generated_at": f"{date_iso}T11:00:00+00:00",
        "message": "status message",
    }
    payload.update(extra)
    (root / f"{date_iso}.json").write_text(json.dumps(payload))


def test_recent_pending_does_not_alert(tmp_path):
    pre = tmp_path / "pre"
    status = tmp_path / "status"
    _write_preoutcome(pre, "2026-05-23")
    _write_status(status, "2026-05-23", "pending_outcomes")

    alerts = check(preoutcome_root=pre, status_root=status, today=date(2026, 5, 24))

    assert alerts == []


def test_three_day_void_lag_grace_does_not_alert(tmp_path):
    pre = tmp_path / "pre"
    status = tmp_path / "status"
    _write_preoutcome(pre, "2026-05-09")
    _write_status(status, "2026-05-09", "pending_outcomes")

    alerts = check(preoutcome_root=pre, status_root=status, today=date(2026, 5, 12))

    assert alerts == []


def test_past_grace_pending_warns(tmp_path):
    pre = tmp_path / "pre"
    status = tmp_path / "status"
    _write_preoutcome(pre, "2026-05-09")
    _write_status(status, "2026-05-09", "pending_outcomes")

    alerts = check(preoutcome_root=pre, status_root=status, today=date(2026, 5, 13))

    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert alerts[0].source == SOURCE
    assert "2026-05-09" in alerts[0].message
    assert "pending_outcomes" in alerts[0].message


def test_sustained_stall_is_critical(tmp_path):
    pre = tmp_path / "pre"
    status = tmp_path / "status"
    _write_preoutcome(pre, "2026-05-12")
    _write_status(
        status,
        "2026-05-12",
        "pending_outcomes",
        message="missing outcomes for 2 live-forward artifact rows",
    )

    alerts = check(preoutcome_root=pre, status_root=status, today=date(2026, 5, 24))

    assert alerts[0].level == "CRITICAL"
    assert "realized-data n-growth" in alerts[0].message
    assert "missing outcomes" in alerts[0].message


def test_success_statuses_do_not_alert_after_grace(tmp_path):
    pre = tmp_path / "pre"
    status = tmp_path / "status"
    for index, ok_status in enumerate([
        "existing_verified",
        "existing_verified_with_voids",
        "resolved_verified",
        "resolved_with_voids",
    ], start=1):
        date_iso = f"2026-05-{index:02d}"
        _write_preoutcome(pre, date_iso)
        _write_status(status, date_iso, ok_status)

    alerts = check(preoutcome_root=pre, status_root=status, today=date(2026, 5, 24))

    assert alerts == []


def test_missing_status_distinguishes_no_resolver_status(tmp_path):
    pre = tmp_path / "pre"
    status = tmp_path / "status"
    _write_preoutcome(pre, "2026-05-12")

    alerts = check(preoutcome_root=pre, status_root=status, today=date(2026, 5, 17))

    assert alerts[0].level == "WARN"
    assert "missing_status_json" in alerts[0].message


def test_malformed_status_alerts_as_unusable_status(tmp_path):
    pre = tmp_path / "pre"
    status = tmp_path / "status"
    _write_preoutcome(pre, "2026-05-12")
    status.mkdir(parents=True)
    (status / "2026-05-12.json").write_text("{not json")

    alerts = check(preoutcome_root=pre, status_root=status, today=date(2026, 5, 17))

    assert alerts[0].level == "WARN"
    assert "malformed_status_json" in alerts[0].message


def test_skips_stale_snapshot_and_non_date_dirs(tmp_path):
    pre = tmp_path / "pre"
    status = tmp_path / "status"
    _write_preoutcome(pre, "2026-05-12.stale_pick_snapshot.abc123")
    _write_preoutcome(pre, "latest")

    alerts = check(preoutcome_root=pre, status_root=status, today=date(2026, 5, 24))

    assert alerts == []


def test_threshold_overrides_can_make_past_grace_critical(tmp_path):
    pre = tmp_path / "pre"
    status = tmp_path / "status"
    _write_preoutcome(pre, "2026-05-12")
    _write_status(status, "2026-05-12", "failed_verify")

    alerts = check(
        preoutcome_root=pre,
        status_root=status,
        today=date(2026, 5, 16),
        thresholds={"critical_age_days": 4},
    )

    assert alerts[0].level == "CRITICAL"
