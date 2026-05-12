from __future__ import annotations

import json
import subprocess
from pathlib import Path

from scripts.live_forward_resolve_once import ResolveConfig, resolve_once


def _config(tmp_path: Path, *, dates: tuple[str, ...] = ()) -> ResolveConfig:
    return ResolveConfig(
        production_root=tmp_path / "prod",
        python=Path("/venv/bin/python"),
        candidate="decision_weighted_lgbm_v0",
        preoutcome_root=Path("data/validation/pre"),
        resolved_root=Path("data/validation/resolved"),
        status_root=Path("data/validation/resolved_status"),
        data_dir=Path("data/processed"),
        top_n=10,
        dates=dates,
        overwrite=False,
        fail_on_pending=False,
    )


def _fake_completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _write_preoutcome_manifest(config: ResolveConfig, date: str = "2026-05-11") -> Path:
    artifact_dir = config.production_root / config.preoutcome_root / date
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "manifest.json").write_text(json.dumps({
        "schema_version": "candidate_artifact_pair_v1",
        "run_kind": "live_forward_preoutcome",
        "candidate_name": config.candidate,
        "git_commit": "frozen-sha",
        "date": date,
        "dates": [date],
        "top_n": 10,
        "profile_paths": {"production": {}, "candidate": {}},
    }))
    return artifact_dir


def test_resolve_once_no_preoutcome_artifacts_is_noop(tmp_path, monkeypatch):
    config = _config(tmp_path)

    def fake_run(args, *, cwd, env=None):
        assert args == ["git", "rev-parse", "HEAD"]
        return _fake_completed(args, stdout="prod-sha\n")

    monkeypatch.setattr("scripts.live_forward_resolve_once.run", fake_run)

    code, payload = resolve_once(config)

    assert code == 0
    assert payload["status"] == "no_preoutcome_artifacts"


def test_resolve_once_explicit_missing_preoutcome_is_pending(tmp_path, monkeypatch):
    config = _config(tmp_path, dates=("2026-05-11",))

    def fake_run(args, *, cwd, env=None):
        assert args == ["git", "rev-parse", "HEAD"]
        return _fake_completed(args, stdout="prod-sha\n")

    monkeypatch.setattr("scripts.live_forward_resolve_once.run", fake_run)

    code, payload = resolve_once(config)

    assert code == 0
    assert payload["status_counts"] == {"pending_preoutcome_artifact": 1}
    status_path = (
        config.production_root
        / config.status_root
        / "2026-05-11.json"
    )
    assert json.loads(status_path.read_text())["status"] == "pending_preoutcome_artifact"


def test_resolve_once_pending_when_pa_rows_absent(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_preoutcome_manifest(config)

    def fake_run(args, *, cwd, env=None):
        assert args == ["git", "rev-parse", "HEAD"]
        return _fake_completed(args, stdout="prod-sha\n")

    monkeypatch.setattr("scripts.live_forward_resolve_once.run", fake_run)
    monkeypatch.setattr(
        "scripts.live_forward_resolve_once.pa_rows_for_dates",
        lambda data_dir, dates: ([], [], 0),
    )

    code, payload = resolve_once(config)

    assert code == 0
    assert payload["status_counts"] == {"pending_outcomes": 1}
    assert not (config.production_root / config.resolved_root).exists()


def test_resolve_once_shells_to_resolver_and_verifier(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_preoutcome_manifest(config)
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append((args, cwd, env))
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="prod-sha\n")
        if "resolve-live-candidate-artifacts" in args:
            output_dir = Path(args[args.index("--output-dir") + 1])
            output_dir.mkdir(parents=True)
            (output_dir / "manifest.json").write_text("{}")
            (output_dir / "resolution.json").write_text(json.dumps({
                "missing_count": 0,
                "terminal_void_count": 2,
            }))
            return _fake_completed(args, stdout="resolve ok\n")
        if "verify-candidate-artifacts" in args:
            verification = Path(args[args.index("--save") + 1])
            verification.write_text("{}")
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_resolve_once.run", fake_run)
    monkeypatch.setattr(
        "scripts.live_forward_resolve_once.pa_rows_for_dates",
        lambda data_dir, dates: ([data_dir / "pa_2026.parquet"], [], 42),
    )

    code, payload = resolve_once(config)

    assert code == 0
    assert payload["status_counts"] == {"resolved_with_voids": 1}
    assert payload["statuses"][0]["live_forward_head"] == "frozen-sha"
    assert payload["statuses"][0]["missing_count"] == 0
    assert payload["statuses"][0]["terminal_void_count"] == 2
    resolve_args = next(
        args for args, _, _ in calls if "resolve-live-candidate-artifacts" in args
    )
    verify_args = next(
        args for args, _, _ in calls if "verify-candidate-artifacts" in args
    )
    assert resolve_args[resolve_args.index("--artifact-dir") + 1].endswith(
        "/data/validation/pre/2026-05-11"
    )
    assert "--allow-partial" not in resolve_args
    assert "--treat-void-games-as-terminal" in resolve_args
    assert (
        verify_args[verify_args.index("--expected-run-kind") + 1]
        == "live_forward_resolved"
    )
    assert "--require-production-pick-snapshot" in verify_args
    assert verify_args[verify_args.index("--expected-git-commit") + 1] == "frozen-sha"
    assert (
        config.production_root
        / config.resolved_root
        / "2026-05-11"
        / "resolve_status.json"
    ).exists()


def test_resolve_once_existing_manifest_verifies_without_resolving(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_preoutcome_manifest(config)
    resolved_dir = config.production_root / config.resolved_root / "2026-05-11"
    resolved_dir.mkdir(parents=True)
    (resolved_dir / "manifest.json").write_text("{}")
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append(args)
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="prod-sha\n")
        if "verify-candidate-artifacts" in args:
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_resolve_once.run", fake_run)
    monkeypatch.setattr(
        "scripts.live_forward_resolve_once.pa_rows_for_dates",
        lambda data_dir, dates: (_ for _ in ()).throw(
            AssertionError("existing resolved manifests should not read PA data")
        ),
    )

    code, payload = resolve_once(config)

    assert code == 0
    assert payload["status_counts"] == {"existing_verified": 1}
    assert not any("resolve-live-candidate-artifacts" in args for args in calls)


def test_resolve_once_legacy_artifact_skips_pick_snapshot_gate(
    tmp_path,
    monkeypatch,
):
    config = _config(tmp_path, dates=("2026-05-09",))
    _write_preoutcome_manifest(config, date="2026-05-09")
    resolved_dir = config.production_root / config.resolved_root / "2026-05-09"
    resolved_dir.mkdir(parents=True)
    (resolved_dir / "manifest.json").write_text("{}")
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append(args)
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="prod-sha\n")
        if "verify-candidate-artifacts" in args:
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_resolve_once.run", fake_run)
    monkeypatch.setattr(
        "scripts.live_forward_resolve_once.pa_rows_for_dates",
        lambda data_dir, dates: (_ for _ in ()).throw(
            AssertionError("existing resolved manifests should not read PA data")
        ),
    )

    code, payload = resolve_once(config)

    verify_args = next(args for args in calls if "verify-candidate-artifacts" in args)
    assert code == 0
    assert payload["status_counts"] == {"existing_verified": 1}
    assert "--require-production-pick-snapshot" not in verify_args


def test_resolve_once_pa_read_error_writes_status(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_preoutcome_manifest(config)

    def fake_run(args, *, cwd, env=None):
        assert args == ["git", "rev-parse", "HEAD"]
        return _fake_completed(args, stdout="prod-sha\n")

    monkeypatch.setattr("scripts.live_forward_resolve_once.run", fake_run)
    monkeypatch.setattr(
        "scripts.live_forward_resolve_once.pa_rows_for_dates",
        lambda data_dir, dates: (_ for _ in ()).throw(ValueError("bad parquet")),
    )

    code, payload = resolve_once(config)

    assert code == 1
    assert payload["status_counts"] == {"failed_pa_data_read": 1}
    status_path = (
        config.production_root
        / config.status_root
        / "2026-05-11.json"
    )
    status = json.loads(status_path.read_text())
    assert status["status"] == "failed_pa_data_read"
    assert "bad parquet" in status["message"]


def test_resolve_once_missing_outcomes_error_is_pending(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_preoutcome_manifest(config)

    def fake_run(args, *, cwd, env=None):
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="prod-sha\n")
        if "resolve-live-candidate-artifacts" in args:
            return _fake_completed(
                args,
                returncode=1,
                stderr="Error: missing outcomes for 2 live-forward artifact rows\n",
            )
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_resolve_once.run", fake_run)
    monkeypatch.setattr(
        "scripts.live_forward_resolve_once.pa_rows_for_dates",
        lambda data_dir, dates: ([data_dir / "pa_2026.parquet"], [], 2),
    )

    code, payload = resolve_once(config)

    assert code == 0
    assert payload["status_counts"] == {"pending_outcomes": 1}
    assert not (config.production_root / config.resolved_root).exists()


def test_resolve_once_refuses_partial_resolved_dir(tmp_path, monkeypatch):
    config = _config(tmp_path)
    _write_preoutcome_manifest(config)
    resolved_dir = config.production_root / config.resolved_root / "2026-05-11"
    resolved_dir.mkdir(parents=True)
    (resolved_dir / "orphan.json").write_text("{}")

    def fake_run(args, *, cwd, env=None):
        assert args == ["git", "rev-parse", "HEAD"]
        return _fake_completed(args, stdout="prod-sha\n")

    monkeypatch.setattr("scripts.live_forward_resolve_once.run", fake_run)
    monkeypatch.setattr(
        "scripts.live_forward_resolve_once.pa_rows_for_dates",
        lambda data_dir, dates: ([data_dir / "pa_2026.parquet"], [], 42),
    )

    code, payload = resolve_once(config)

    assert code == 1
    assert payload["status_counts"] == {"failed_partial_resolved_dir": 1}
