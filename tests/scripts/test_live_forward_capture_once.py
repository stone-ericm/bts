from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.live_forward_capture_once import CaptureConfig, capture_once


def _config(tmp_path: Path) -> CaptureConfig:
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
    )


def _write_pick(root: Path, *, result=None) -> Path:
    path = root / "data" / "picks" / "2026-05-11.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": "2026-05-11", "result": result}))
    return path


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


def test_capture_once_existing_manifest_verifies_without_export(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_pick(config.production_root)
    artifact_dir = config.production_root / config.artifact_root / config.date
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text("{}")
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
