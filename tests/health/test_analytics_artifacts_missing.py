import json
from datetime import date
from types import SimpleNamespace

from bts.health import analytics_artifacts_missing
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
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing._fatal_scheduler_journal_line",
        lambda unit, since: (_ for _ in ()).throw(
            AssertionError("external shadow unit must not query scheduler journal")
        ),
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


def test_inline_shadow_missing_with_scheduler_death_is_critical(tmp_path, monkeypatch):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    _write_state(
        picks_dir,
        jobs={
            "shadow": {
                "status": "dispatched",
                "updated_at": "2026-05-21T13:53:36-04:00",
            }
        },
    )
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing.read_systemd_unit_summary",
        lambda unit: None,
    )

    calls = []

    def fake_scheduler_journal(unit, since):
        calls.append((unit, since))
        return (
            "May 21 13:56:58 host systemd[123]: bts-scheduler.service: "
            "Main process exited, code=killed, status=9/KILL"
        )

    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing._fatal_scheduler_journal_line",
        fake_scheduler_journal,
    )

    alerts = check(
        picks_dir,
        today=date(2026, 5, 21),
        shadow_expected=True,
        capture_expected=False,
        scheduler_unit="bts-scheduler.service",
    )

    assert calls == [
        ("bts-scheduler.service", "2026-05-21T13:53:36-04:00")
    ]
    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"
    assert "shadow artifact missing" in alerts[0].message
    assert "scheduler death evidence" in alerts[0].message
    assert "code=killed" in alerts[0].message


def test_inline_shadow_missing_without_scheduler_death_remains_warn(tmp_path, monkeypatch):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    _write_state(
        picks_dir,
        jobs={
            "shadow": {
                "status": "failed",
                "reason": "prior_dispatched_without_artifact",
                "updated_at": "2026-05-21T13:53:36-04:00",
            }
        },
    )
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing.read_systemd_unit_summary",
        lambda unit: None,
    )
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing._fatal_scheduler_journal_line",
        lambda unit, since: None,
    )

    alerts = check(
        picks_dir,
        today=date(2026, 5, 21),
        shadow_expected=True,
        capture_expected=False,
    )

    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert "prior_dispatched_without_artifact" in alerts[0].message


def test_terminalized_inline_shadow_uses_original_dispatch_time(
    tmp_path,
    monkeypatch,
):
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    _write_state(
        picks_dir,
        jobs={
            "shadow": {
                "status": "failed",
                "reason": "prior_dispatched_without_artifact",
                "dispatched_at": "2026-05-21T13:53:36-04:00",
                "updated_at": "2026-05-21T13:57:42-04:00",
            }
        },
    )
    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing.read_systemd_unit_summary",
        lambda unit: None,
    )

    calls = []

    def fake_scheduler_journal(unit, since):
        calls.append((unit, since))
        return (
            "May 21 13:56:58 host systemd[123]: bts-scheduler.service: "
            "Failed with result 'oom-kill'."
        )

    monkeypatch.setattr(
        "bts.health.analytics_artifacts_missing._fatal_scheduler_journal_line",
        fake_scheduler_journal,
    )

    alerts = check(
        picks_dir,
        today=date(2026, 5, 21),
        shadow_expected=True,
        capture_expected=False,
        scheduler_unit="bts-scheduler.service",
    )

    assert calls == [
        ("bts-scheduler.service", "2026-05-21T13:53:36-04:00")
    ]
    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"


def test_scheduler_journal_detects_status_137_as_fatal(monkeypatch):
    line = (
        "May 21 13:56:58 host systemd[123]: bts-scheduler.service: "
        "Main process exited, code=exited, status=137/n/a"
    )

    def fake_run(args, **kwargs):
        assert args[args.index("--since") + 1] == "2026-05-21 13:53:36"
        return SimpleNamespace(returncode=0, stdout=f"{line}\n")

    monkeypatch.setattr(analytics_artifacts_missing.subprocess, "run", fake_run)

    assert analytics_artifacts_missing._fatal_scheduler_journal_line(
        "bts-scheduler.service",
        "2026-05-21T13:53:36-04:00",
    ) == line


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
