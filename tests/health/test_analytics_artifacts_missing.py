import json
from datetime import date

from bts.health.analytics_artifacts_missing import check


def _write_state(picks_dir, *, locked=True, jobs=None):
    state_dir = picks_dir / "2026-05-21"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "scheduler_state.json").write_text(json.dumps({
        "date": "2026-05-21",
        "pick_locked": locked,
        "analytics_jobs": jobs or {},
    }))


def _write_capture_status(root, payload):
    path = root / "2026-05-21" / "capture_status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def test_no_locked_pick_no_artifact_alerts(tmp_path, monkeypatch):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    _write_state(picks_dir, locked=False)
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing.read_systemd_unit_summary",
        lambda unit: None,
    )

    alerts = check(
        picks_dir,
        today=date(2026, 5, 21),
        shadow_expected=True,
        capture_expected=True,
        capture_artifact_root=tmp_path / "validation",
    )
    assert alerts == []


def test_missing_shadow_is_warn_after_locked_pick(tmp_path, monkeypatch):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    _write_state(
        picks_dir,
        jobs={"shadow": {"status": "dispatched", "updated_at": "now"}},
    )
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing.read_systemd_unit_summary",
        lambda unit: "Result=oom-kill",
    )

    alerts = check(
        picks_dir,
        today=date(2026, 5, 21),
        shadow_expected=True,
        capture_expected=False,
        shadow_unit="bts-shadow.service",
    )

    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert alerts[0].source == "analytics_artifacts_missing"
    assert "shadow artifact missing" in alerts[0].message
    assert "Result=oom-kill" in alerts[0].message


def test_shadow_benign_abstention_is_info_not_warn(tmp_path, monkeypatch):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    _write_state(
        picks_dir,
        jobs={
            "shadow": {
                "status": "failed",
                "reason": "select_pick_returned_none",
                "updated_at": "now",
            }
        },
    )
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing.read_systemd_unit_summary",
        lambda unit: None,
    )

    alerts = check(
        picks_dir,
        today=date(2026, 5, 21),
        shadow_expected=True,
        capture_expected=False,
    )

    assert len(alerts) == 1
    assert alerts[0].level == "INFO"
    assert "shadow model abstained" in alerts[0].message


def test_shadow_prediction_failure_remains_warn(tmp_path, monkeypatch):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    _write_state(
        picks_dir,
        jobs={
            "shadow": {
                "status": "failed",
                "reason": "prediction_failed_or_none",
                "updated_at": "now",
            }
        },
    )
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing.read_systemd_unit_summary",
        lambda unit: None,
    )

    alerts = check(
        picks_dir,
        today=date(2026, 5, 21),
        shadow_expected=True,
        capture_expected=False,
    )

    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert "prediction_failed_or_none" in alerts[0].message


def test_capture_stale_snapshot_is_critical(tmp_path, monkeypatch):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    _write_state(
        picks_dir,
        jobs={"live_forward_capture": {"status": "dispatched", "unit": "cap.service"}},
    )
    artifact_root = tmp_path / "validation"
    _write_capture_status(artifact_root, {
        "status": "failed_recapture_export",
        "stale_pick_snapshot": True,
        "message": "export failed",
    })
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing.read_systemd_unit_summary",
        lambda unit: "Result=oom-kill",
    )

    alerts = check(
        picks_dir,
        today=date(2026, 5, 21),
        shadow_expected=False,
        capture_expected=True,
        capture_artifact_root=artifact_root,
        capture_unit="cap.service",
    )

    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"
    assert "stale_pick_snapshot=true" in alerts[0].message
    assert "Result=oom-kill" in alerts[0].message


def test_capture_ok_status_is_clean(tmp_path, monkeypatch):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    _write_state(picks_dir)
    artifact_root = tmp_path / "validation"
    _write_capture_status(artifact_root, {
        "status": "existing_verified",
        "stale_pick_snapshot": False,
    })
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing.read_systemd_unit_summary",
        lambda unit: None,
    )

    alerts = check(
        picks_dir,
        today=date(2026, 5, 21),
        capture_expected=True,
        capture_artifact_root=artifact_root,
    )
    assert alerts == []


def test_capture_missing_status_is_critical(tmp_path, monkeypatch):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    _write_state(picks_dir)
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing.read_systemd_unit_summary",
        lambda unit: None,
    )

    alerts = check(
        picks_dir,
        today=date(2026, 5, 21),
        capture_expected=True,
        capture_artifact_root=tmp_path / "validation",
    )

    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"
    assert "capture status missing" in alerts[0].message
