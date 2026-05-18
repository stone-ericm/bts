from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd

from scripts.live_forward_capture_once import CaptureConfig, capture_once, file_sha256


def _config(
    tmp_path: Path,
    *,
    auto_recapture_on_snapshot_drift: bool = False,
) -> CaptureConfig:
    return CaptureConfig(
        date="2026-05-11",
        production_root=tmp_path / "prod",
        live_forward_root=tmp_path / "live",
        python=Path("/venv/bin/python"),
        candidate="decision_weighted_lgbm_v0",
        artifact_root=Path("data/validation/live"),
        data_dir=Path("data/processed"),
        picks_dir=Path("data/picks"),
        top_n=10,
        overwrite=False,
        fail_on_pending=False,
        auto_recapture_on_snapshot_drift=auto_recapture_on_snapshot_drift,
    )


def _write_pick(root: Path, *, result=None) -> Path:
    path = root / "data" / "picks" / "2026-05-11.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": "2026-05-11", "result": result}))
    return path


def _write_manifest(artifact_dir: Path, *, pick_path: Path | None = None) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {}
    if pick_path is not None:
        pick_json = json.loads(pick_path.read_text())
        manifest["production_pick_snapshot"] = {
            "source_sha256": file_sha256(pick_path),
            "production_pick_json": pick_json,
        }
    artifact_dir.joinpath("manifest.json").write_text(json.dumps(manifest))


def _fake_completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr=stderr)


def test_capture_once_waits_for_pick_file(tmp_path, monkeypatch):
    config = _config(tmp_path)

    def fake_run(args, *, cwd, env=None):
        assert args == ["git", "rev-parse", "HEAD"]
        return _fake_completed(args, stdout="abc123\n")

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 0
    assert payload["status"] == "pending_pick"
    assert not (config.production_root / "data" / "validation").exists()


def test_capture_once_refuses_resolved_pick_file(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_pick(config.production_root, result="hit")

    def fake_run(args, *, cwd, env=None):
        assert args == ["git", "rev-parse", "HEAD"]
        return _fake_completed(args, stdout="abc123\n")

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 1
    assert payload["status"] == "failed_pick_already_resolved"


def test_capture_once_treats_partial_pick_json_as_transient(tmp_path, monkeypatch):
    config = _config(tmp_path)
    pick_path = config.production_root / "data" / "picks" / "2026-05-11.json"
    pick_path.parent.mkdir(parents=True, exist_ok=True)
    pick_path.write_text("{")

    def fake_run(args, *, cwd, env=None):
        assert args == ["git", "rev-parse", "HEAD"]
        return _fake_completed(args, stdout="abc123\n")

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 0
    assert payload["status"] == "transient_pick_read_error"
    assert not (config.production_root / "data" / "validation").exists()


def test_capture_once_exports_and_verifies(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_pick(config.production_root)
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append((args, cwd, env))
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout=("prod-sha\n" if cwd == config.production_root else "live-sha\n"))
        if "export-live-candidate-artifacts" in args:
            artifact_dir = Path(args[args.index("--output-dir") + 1])
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "manifest.json").write_text("{}")
            return _fake_completed(args, stdout="export ok\n")
        if "verify-candidate-artifacts" in args:
            verification = Path(args[args.index("--save") + 1])
            verification.parent.mkdir(parents=True, exist_ok=True)
            verification.write_text("{}")
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 0
    assert payload["status"] == "exported_verified"
    assert payload["production_head"] == "prod-sha"
    assert payload["live_forward_head"] == "live-sha"
    assert payload["pa_outcome_rows_for_date"] == 0
    export_args = next(args for args, _, _ in calls if "export-live-candidate-artifacts" in args)
    verify_args = next(args for args, _, _ in calls if "verify-candidate-artifacts" in args)
    assert "--production-pick-file" in export_args
    assert "--require-production-pick-snapshot" in verify_args
    assert verify_args[verify_args.index("--expected-git-commit") + 1] == "live-sha"
    assert (
        config.production_root
        / "data"
        / "validation"
        / "live"
        / "2026-05-11"
        / "capture_status.json"
    ).exists()


def test_capture_once_refuses_initial_export_after_pa_outcomes_exist(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    _write_pick(config.production_root)
    pa_path = config.production_root / config.data_dir / "pa_2026.parquet"
    pa_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2026-05-11"]}).to_parquet(pa_path, index=False)
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append((args, cwd, env))
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="sha\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 1
    assert payload["status"] == "failed_export_post_outcomes"
    assert payload["pa_outcome_rows_for_date"] == 1
    assert not any("export-live-candidate-artifacts" in args for args, _, _ in calls)


def test_capture_once_existing_manifest_verifies_without_pick_file(tmp_path, monkeypatch):
    config = _config(tmp_path)
    artifact_dir = config.production_root / config.artifact_root / config.date
    _write_manifest(artifact_dir)
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append(args)
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="sha\n")
        if "verify-candidate-artifacts" in args:
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 0
    assert payload["status"] == "existing_verified"
    assert not any("export-live-candidate-artifacts" in args for args in calls)


def test_capture_once_existing_manifest_verifies_after_pick_resolves(tmp_path, monkeypatch):
    config = _config(tmp_path)
    pick_path = _write_pick(config.production_root, result="hit")
    artifact_dir = config.production_root / config.artifact_root / config.date
    _write_manifest(artifact_dir, pick_path=pick_path)
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append(args)
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="sha\n")
        if "verify-candidate-artifacts" in args:
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 0
    assert payload["status"] == "existing_verified"
    assert payload["snapshot_matches_current_pick"] is True
    assert not any("export-live-candidate-artifacts" in args for args in calls)


def test_capture_once_result_only_change_is_not_stale_snapshot(tmp_path, monkeypatch):
    config = _config(tmp_path, auto_recapture_on_snapshot_drift=True)
    pick_path = _write_pick(config.production_root)
    artifact_dir = config.production_root / config.artifact_root / config.date
    _write_manifest(artifact_dir, pick_path=pick_path)
    _write_pick(config.production_root, result="miss")
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append(args)
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="sha\n")
        if "verify-candidate-artifacts" in args:
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 0
    assert payload["status"] == "existing_verified"
    assert payload["current_pick_result"] == "miss"
    assert payload["snapshot_matches_current_pick"] is False
    assert payload["snapshot_decision_matches_current_pick"] is True
    assert payload["stale_pick_snapshot"] is False
    assert not any("export-live-candidate-artifacts" in args for args in calls)


def test_capture_once_detects_stale_snapshot_without_auto_recapture(
    tmp_path, monkeypatch
):
    config = _config(tmp_path)
    _write_pick(config.production_root)
    artifact_dir = config.production_root / config.artifact_root / config.date
    _write_manifest(artifact_dir)
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append(args)
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="sha\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 1
    assert payload["status"] == "stale_pick_snapshot"
    assert payload["snapshot_matches_current_pick"] is False
    assert payload["stale_pick_snapshot"] is True
    assert not any("export-live-candidate-artifacts" in args for args in calls)


def test_capture_once_recaptures_authorized_stale_snapshot_before_pick_resolves(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, auto_recapture_on_snapshot_drift=True)
    pick_path = _write_pick(config.production_root)
    artifact_dir = config.production_root / config.artifact_root / config.date
    _write_manifest(artifact_dir)
    (artifact_dir / "old.txt").write_text("old artifact")
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append(args)
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(
                args,
                stdout=("prod-sha\n" if cwd == config.production_root else "live-sha\n"),
            )
        if "export-live-candidate-artifacts" in args:
            refresh_dir = Path(args[args.index("--output-dir") + 1])
            refresh_dir.mkdir(parents=True, exist_ok=True)
            refresh_manifest = {
                "production_pick_snapshot": {
                    "source_sha256": file_sha256(pick_path),
                }
            }
            (refresh_dir / "manifest.json").write_text(json.dumps(refresh_manifest))
            return _fake_completed(args, stdout="export ok\n")
        if "verify-candidate-artifacts" in args:
            verification = Path(args[args.index("--save") + 1])
            verification.parent.mkdir(parents=True, exist_ok=True)
            verification.write_text("{}")
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 0
    assert payload["status"] == "recaptured_due_to_snapshot_drift"
    assert payload["artifact_pick_snapshot_sha256"] == file_sha256(pick_path)
    assert payload["current_pick_sha256"] == file_sha256(pick_path)
    assert payload["snapshot_matches_current_pick"] is True
    assert payload["stale_pick_snapshot"] is False
    assert payload["pa_outcome_rows_for_date"] == 0
    assert payload["previous_artifact_pick_snapshot_sha256"] is None
    assert payload["previous_snapshot_matches_current_pick"] is False
    assert payload["previous_stale_pick_snapshot"] is True
    assert Path(payload["stale_artifact_backup_dir"]).joinpath("old.txt").exists()
    assert not artifact_dir.joinpath("old.txt").exists()
    assert json.loads(artifact_dir.joinpath("manifest.json").read_text())[
        "production_pick_snapshot"
    ]["source_sha256"] == file_sha256(pick_path)
    assert any("export-live-candidate-artifacts" in args for args in calls)


def test_capture_once_refuses_stale_snapshot_after_pick_resolves(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, auto_recapture_on_snapshot_drift=True)
    pick_path = _write_pick(config.production_root, result="miss")
    artifact_dir = config.production_root / config.artifact_root / config.date
    _write_manifest(artifact_dir)
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append(args)
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="sha\n")
        if "verify-candidate-artifacts" in args:
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 1
    assert payload["status"] == "failed_recapture_post_resolution"
    assert payload["current_pick_sha256"] == file_sha256(pick_path)
    assert payload["snapshot_matches_current_pick"] is False
    assert payload["stale_pick_snapshot"] is True
    assert not any("export-live-candidate-artifacts" in args for args in calls)


def test_capture_once_refuses_recapture_after_pa_outcomes_exist(
    tmp_path, monkeypatch
):
    config = _config(tmp_path, auto_recapture_on_snapshot_drift=True)
    _write_pick(config.production_root)
    artifact_dir = config.production_root / config.artifact_root / config.date
    _write_manifest(artifact_dir)
    pa_path = config.production_root / config.data_dir / "pa_2026.parquet"
    pa_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": ["2026-05-11"]}).to_parquet(pa_path, index=False)
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append(args)
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="sha\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert code == 1
    assert payload["status"] == "failed_recapture_post_outcomes"
    assert payload["pa_outcome_rows_for_date"] == 1
    assert payload["stale_pick_snapshot"] is True
    assert not any("export-live-candidate-artifacts" in args for args in calls)
