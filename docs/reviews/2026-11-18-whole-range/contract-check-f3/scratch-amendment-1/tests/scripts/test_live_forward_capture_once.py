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


def _write_pick(root: Path, *, result=None, extra: dict | None = None) -> Path:
    path = root / "data" / "picks" / "2026-05-11.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"date": "2026-05-11", "result": result}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload))
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


def test_capture_once_null_feature_env_backfill_is_not_stale_snapshot(tmp_path, monkeypatch):
    config = _config(tmp_path, auto_recapture_on_snapshot_drift=True)
    pick_path = _write_pick(config.production_root)
    artifact_dir = config.production_root / config.artifact_root / config.date
    _write_manifest(artifact_dir, pick_path=pick_path)
    _write_pick(
        config.production_root,
        result="hit",
        extra={
            "feature_env_schema_version": None,
            "feature_env": None,
            "feature_env_hash": None,
        },
    )
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
    assert payload["current_pick_result"] == "hit"
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


# ---------------------------------------------------------------------------
# D8 (2026-09-14): research-only forecast capture on no-pick (skip) days.
# Separate artifact root, opt-in flag, pre-first-pitch deadline, completion
# contract (sidecar written last with hashes), no scheduler-state side effects.
# ---------------------------------------------------------------------------
from datetime import datetime, timedelta, timezone  # noqa: E402

from scripts.live_forward_capture_once import RESEARCH_ACCEPTED_NAME, RESEARCH_SIDECAR_NAME  # noqa: E402

_DATE = "2026-05-11"
_FIRST_PITCH = datetime(2026, 5, 11, 17, 10, tzinfo=timezone.utc)  # 13:10 ET


def _research_config(tmp_path: Path, *, on: bool = True, now: datetime | None = None,
                     research_root: Path | None = None) -> CaptureConfig:
    cfg = _config(tmp_path)
    return CaptureConfig(
        **{
            **cfg.__dict__,
            "capture_research_on_skip": on,
            "research_root": research_root or Path("data/validation/live_research"),
            "now": now or (_FIRST_PITCH - timedelta(hours=2)),
        }
    )


def _write_state(root: Path, *, final_skip_candidate=None, skip_notified_at=None,
                 games=None, date=_DATE, raw: str | None = None) -> Path:
    path = root / "data" / "picks" / _DATE / "scheduler_state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        path.write_text(raw)
        return path
    if games is None:
        games = [
            {"game_pk": 1, "game_time_et": "2026-05-11T13:10:00-04:00", "lineup_confirmed": True,
             "is_doubleheader_game2": False},
            {"game_pk": 2, "game_time_et": "2026-05-11T19:05:00-04:00", "lineup_confirmed": False,
             "is_doubleheader_game2": False},
        ]
    path.write_text(json.dumps({
        "date": date, "games": games,
        "final_skip_candidate": final_skip_candidate,
        "skip_notified_at": skip_notified_at,
    }))
    return path


_FSC = {"primary": {"batter_id": 518692, "batter_name": "F. Freeman", "game_pk": 2,
                    "p_game_hit": 0.689}, "streak": 0, "saver_available": False}


def _write_decision(root: Path, *, scoreable: bool, action: str = "skip") -> Path:
    path = root / "data" / "picks" / _DATE / "decision.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"date": _DATE, "action": action, "source": "mdp",
                                "scoreable": scoreable}))
    return path


def _fake_run_research(config: CaptureConfig, calls: list, *, export_ok: bool = True,
                       verify_ok: bool = True):
    def fake_run(args, *, cwd, env=None):
        calls.append((list(args), cwd, env))
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout=("prod-sha\n" if cwd == config.production_root
                                                 else "live-sha\n"))
        if "export-live-candidate-artifacts" in args:
            if not export_ok:
                return _fake_completed(args, returncode=1, stderr="export boom\n")
            out = Path(args[args.index("--output-dir") + 1])
            for variant in ("production", "candidate"):
                p = out / "profiles" / variant / f"live_{_DATE}.parquet"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(f"parquet-{variant}".encode())
            (out / "manifest.json").write_text(json.dumps({
                "schema_version": "bts_candidate_ranked_slate_pair_v1",
                "run_kind": "live_forward_preoutcome",
                "candidate_name": args[args.index("--candidate") + 1],
                "date": _DATE, "dates": [_DATE], "production_pick_snapshot": None,
                "git_commit": "live-sha",
            }))
            return _fake_completed(args, stdout="export ok\n")
        if "verify-candidate-artifacts" in args:
            if not verify_ok:
                return _fake_completed(args, returncode=1, stderr="verify boom\n")
            verification = Path(args[args.index("--save") + 1])
            verification.parent.mkdir(parents=True, exist_ok=True)
            verification.write_text(json.dumps({"ok": True, "failure_count": 0,
                                                "manifest": {"git_commit": "live-sha"}}))
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)
    return fake_run


def _research_dir(config: CaptureConfig) -> Path:
    return config.production_root / config.research_root / _DATE


def _official_dir(config: CaptureConfig) -> Path:
    return config.production_root / config.artifact_root / _DATE


def test_research_flag_off_keeps_pending_pick_even_with_skip_signal(tmp_path, monkeypatch):
    config = _research_config(tmp_path, on=False)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "pending_pick")
    assert not _research_dir(config).exists()
    assert not any("export-live-candidate-artifacts" in a for a, _, _ in calls)


def test_research_no_state_file_is_pending_pick(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "pending_pick")
    assert not _research_dir(config).exists()


def test_research_captures_to_separate_root_without_pick_snapshot(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert code == 0
    assert payload["status"] == "captured_research_no_production_pick"
    export_args = next(a for a, _, _ in calls if "export-live-candidate-artifacts" in a)
    verify_args = next(a for a, _, _ in calls if "verify-candidate-artifacts" in a)
    assert "--production-pick-file" not in export_args
    assert Path(export_args[export_args.index("--output-dir") + 1]) == _research_dir(config)
    assert "--require-production-pick-snapshot" not in verify_args
    assert "--require-live-preoutcome" in verify_args
    assert not _official_dir(config).exists()
    sidecar = json.loads((_research_dir(config) / RESEARCH_SIDECAR_NAME).read_text())
    assert sidecar["research_only"] is True
    assert sidecar["eligible_for_official_read"] is False
    assert sidecar["date"] == _DATE
    assert sidecar["reason"] == "no_production_pick_skip_day"
    assert sidecar["trigger"]["final_skip_candidate"]["streak"] == 0
    assert sidecar["earliest_first_pitch_utc"] == _FIRST_PITCH.isoformat()
    for name in ("manifest.json", "verification.json",
                 f"profiles/production/live_{_DATE}.parquet",
                 f"profiles/candidate/live_{_DATE}.parquet"):
        assert sidecar["file_sha256"][name] == file_sha256(_research_dir(config) / name)
    assert (_research_dir(config) / "capture_status.json").exists()


def test_research_second_run_is_idempotent_and_verifies_sidecar(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))
    capture_once(config)
    n_exports = sum("export-live-candidate-artifacts" in a for a, _, _ in calls)

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "research_existing_verified")
    assert sum("export-live-candidate-artifacts" in a for a, _, _ in calls) == n_exports


def test_research_partial_capture_before_deadline_is_quarantined_and_redone(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    partial = _research_dir(config)
    partial.mkdir(parents=True)
    (partial / "manifest.json").write_text("{}")  # exporter finished, sidecar never written
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "captured_research_no_production_pick")
    quarantined = [p for p in partial.parent.iterdir() if p.name.startswith(f"{_DATE}.partial.")]
    assert len(quarantined) == 1
    assert (partial / RESEARCH_SIDECAR_NAME).exists()


def test_research_partial_capture_after_deadline_is_left_alone(tmp_path, monkeypatch):
    config = _research_config(tmp_path, now=_FIRST_PITCH + timedelta(minutes=30))
    _write_state(config.production_root, final_skip_candidate=_FSC)
    partial = _research_dir(config)
    partial.mkdir(parents=True)
    (partial / "manifest.json").write_text("{}")
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (1, "research_partial_after_deadline")
    assert not any("export-live-candidate-artifacts" in a for a, _, _ in calls)
    assert (partial / "manifest.json").exists()


def test_research_tampered_sidecar_hash_is_not_accepted(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))
    capture_once(config)
    (_research_dir(config) / "manifest.json").write_text(json.dumps({"date": _DATE, "ok": False}))

    code, payload = capture_once(config)

    # Hash mismatch = incomplete/corrupt contract; still pregame -> quarantine + recapture.
    assert (code, payload["status"]) == (0, "captured_research_no_production_pick")
    assert any(p.name.startswith(f"{_DATE}.partial.") for p in _research_dir(config).parent.iterdir())


def test_research_skip_notified_but_candidate_cleared_is_not_a_skip(tmp_path, monkeypatch):
    # A genuine pick attempt clears final_skip_candidate but keeps skip_notified_at.
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=None,
                 skip_notified_at="2026-05-11T11:00:00-04:00")
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "pending_pick")
    assert not _research_dir(config).exists()


def test_research_scoreable_decision_defers_to_official_path(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    _write_decision(config.production_root, scoreable=True, action="single")
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "pending_pick")
    assert not _research_dir(config).exists()


def test_research_deadline_passed_captures_nothing(tmp_path, monkeypatch):
    config = _research_config(tmp_path, now=_FIRST_PITCH - timedelta(minutes=4))
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "research_deadline_passed")
    assert not _research_dir(config).exists()
    assert not any("export-live-candidate-artifacts" in a for a, _, _ in calls)


def test_research_completion_after_first_pitch_is_discarded(tmp_path, monkeypatch):
    config = _research_config(tmp_path, now=_FIRST_PITCH - timedelta(minutes=10))
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    inner = _fake_run_research(config, calls)
    clock = {"t": config.now}

    def slow_run(args, *, cwd, env=None):
        if "export-live-candidate-artifacts" in args:
            clock["t"] = _FIRST_PITCH + timedelta(minutes=1)  # export straddled first pitch
        return inner(args, cwd=cwd, env=env)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", slow_run)
    monkeypatch.setattr("scripts.live_forward_capture_once.utc_now_dt", lambda: clock["t"])
    config = CaptureConfig(**{**config.__dict__, "now": None})

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (1, "discarded_research_after_first_pitch")
    assert not (_research_dir(config) / RESEARCH_SIDECAR_NAME).exists()
    assert any(p.name.startswith(f"{_DATE}.late.") for p in _research_dir(config).parent.iterdir())


def test_research_no_game_times_fails_closed(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC, games=[])
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "research_pending_no_game_times")
    assert not _research_dir(config).exists()


def test_research_corrupt_state_is_pending_and_not_quarantined(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    state_path = _write_state(config.production_root, raw="{not json")
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "research_pending_state_unreadable")
    assert state_path.exists() and state_path.read_text() == "{not json"
    assert not any(p.name.endswith("corrupt") or ".corrupt-" in p.name
                   for p in state_path.parent.iterdir())


def test_research_state_date_mismatch_is_pending_pick(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC, date="2026-05-10")
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "pending_pick")


def test_research_root_overlapping_official_root_is_refused(tmp_path, monkeypatch):
    config = _research_config(tmp_path, research_root=Path("data/validation/live/research"))
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (1, "failed_research_root_overlap")
    assert not (config.production_root / "data/validation/live").exists()


def test_research_export_failure_reports_and_leaves_partial_for_next_run(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run",
                        _fake_run_research(config, calls, export_ok=False))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (1, "failed_research_export")
    assert not (_research_dir(config) / RESEARCH_SIDECAR_NAME).exists()


def test_research_pick_file_present_uses_official_path_unchanged(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_pick(config.production_root)
    _write_state(config.production_root, final_skip_candidate=_FSC)  # stale skip signal
    calls = []

    def fake_run(args, *, cwd, env=None):
        calls.append((args, cwd, env))
        if args == ["git", "rev-parse", "HEAD"]:
            return _fake_completed(args, stdout="sha\n")
        if "export-live-candidate-artifacts" in args:
            out = Path(args[args.index("--output-dir") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / "manifest.json").write_text("{}")
            return _fake_completed(args, stdout="export ok\n")
        if "verify-candidate-artifacts" in args:
            v = Path(args[args.index("--save") + 1])
            v.parent.mkdir(parents=True, exist_ok=True)
            v.write_text("{}")
            return _fake_completed(args, stdout="verify ok\n")
        raise AssertionError(args)

    monkeypatch.setattr("scripts.live_forward_capture_once.run", fake_run)

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "exported_verified")
    export_args = next(a for a, _, _ in calls if "export-live-candidate-artifacts" in a)
    assert "--production-pick-file" in export_args
    assert not _research_dir(config).exists()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def test_research_root_equal_to_picks_dir_is_refused_and_state_untouched(tmp_path, monkeypatch):
    config = _research_config(tmp_path, research_root=Path("data/picks"))
    _write_state(config.production_root, final_skip_candidate=_FSC)
    before = _tree_hashes(config.production_root / "data" / "picks")
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (1, "failed_research_root_overlap")
    assert _tree_hashes(config.production_root / "data" / "picks") == before
    assert not any("export-live-candidate-artifacts" in a for a, _, _ in calls)


def test_research_root_equal_to_data_dir_or_production_root_is_refused(tmp_path, monkeypatch):
    for root in (Path("data/processed"), Path(".")):
        config = _research_config(tmp_path, research_root=root)
        _write_state(config.production_root, final_skip_candidate=_FSC)
        monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, []))
        code, payload = capture_once(config)
        assert (code, payload["status"]) == (1, "failed_research_root_overlap"), root


def test_research_date_dir_symlink_into_official_root_is_refused(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    official = _official_dir(config)
    official.mkdir(parents=True)
    research_root = config.production_root / config.research_root
    research_root.mkdir(parents=True)
    (research_root / _DATE).symlink_to(official, target_is_directory=True)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (1, "failed_research_dir_escapes_root")
    assert list(official.iterdir()) == []
    assert not any("export-live-candidate-artifacts" in a for a, _, _ in calls)


def test_research_existing_capture_for_other_candidate_is_incompatible_not_relabelled(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))
    capture_once(config)
    before = _tree_hashes(_research_dir(config))
    other = CaptureConfig(**{**config.__dict__, "candidate": "some_other_candidate"})
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(other, calls))
    n_exports = sum("export-live-candidate-artifacts" in a for a, _, _ in calls)

    code, payload = capture_once(other)

    assert (code, payload["status"]) == (1, "research_existing_incompatible")
    assert _tree_hashes(_research_dir(config)) == before
    assert sum("export-live-candidate-artifacts" in a for a, _, _ in calls) == n_exports


def test_research_sidecar_with_wrong_labels_or_late_completion_is_invalid(tmp_path, monkeypatch):
    for mutate in (
        lambda sc: sc.__setitem__("eligible_for_official_read", True),
        lambda sc: sc.__setitem__("completed_at", (_FIRST_PITCH + timedelta(seconds=1)).isoformat()),
        lambda sc: sc.__setitem__("schema_version", "bogus"),
    ):
        config = _research_config(tmp_path / str(id(mutate)))
        _write_state(config.production_root, final_skip_candidate=_FSC)
        calls: list = []
        monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))
        capture_once(config)
        sidecar_path = _research_dir(config) / RESEARCH_SIDECAR_NAME
        sidecar = json.loads(sidecar_path.read_text())
        mutate(sidecar)
        sidecar_path.write_text(json.dumps(sidecar))

        code, payload = capture_once(config)  # still pregame -> quarantine + redo

        assert (code, payload["status"]) == (0, "captured_research_no_production_pick")
        assert any(p.name.startswith(f"{_DATE}.partial.") for p in _research_dir(config).parent.iterdir())
        fresh = json.loads(sidecar_path.read_text())
        assert fresh["eligible_for_official_read"] is False and fresh["schema_version"] != "bogus"


def test_research_clock_crossing_first_pitch_during_hashing_is_discarded(tmp_path, monkeypatch):
    config = _research_config(tmp_path, now=_FIRST_PITCH - timedelta(minutes=10))
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))
    clock = {"t": config.now}
    real_sha = file_sha256

    def slow_sha(path):
        clock["t"] = _FIRST_PITCH  # hashing straddles first pitch (equality must reject)
        return real_sha(path)

    monkeypatch.setattr("scripts.live_forward_capture_once.file_sha256", slow_sha)
    monkeypatch.setattr("scripts.live_forward_capture_once.utc_now_dt", lambda: clock["t"])
    config = CaptureConfig(**{**config.__dict__, "now": None})

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (1, "discarded_research_after_first_pitch")
    assert not (_research_dir(config) / RESEARCH_SIDECAR_NAME).exists()


def test_research_exact_deadline_equality_is_a_pass(tmp_path, monkeypatch):
    config = _research_config(tmp_path, now=_FIRST_PITCH - timedelta(minutes=5))
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "research_deadline_passed")
    assert not _research_dir(config).exists()


def test_research_malformed_decision_is_pending(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    dec = config.production_root / "data" / "picks" / _DATE / "decision.json"
    dec.write_text("{oops")
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "research_pending_decision_unreadable")
    assert dec.read_text() == "{oops"
    assert not _research_dir(config).exists()


def test_research_decision_with_other_date_is_ignored_and_noted(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    dec = config.production_root / "data" / "picks" / _DATE / "decision.json"
    dec.write_text(json.dumps({"date": "2026-05-10", "action": "single", "scoreable": True}))
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "captured_research_no_production_pick")
    sidecar = json.loads((_research_dir(config) / RESEARCH_SIDECAR_NAME).read_text())
    assert sidecar["trigger"]["decision_present"] is False
    assert "ignored" in sidecar["trigger"]["decision_note"]


def test_research_capture_then_real_pick_uses_official_path_and_keeps_research_artifact(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))
    capture_once(config)
    research_before = _tree_hashes(_research_dir(config))
    _write_pick(config.production_root)  # a late fallback delivered a real pick

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "exported_verified")
    export_args = [a for a, _, _ in calls if "export-live-candidate-artifacts" in a][-1]
    assert "--production-pick-file" in export_args
    assert Path(export_args[export_args.index("--output-dir") + 1]) == _official_dir(config)
    assert _tree_hashes(_research_dir(config)) == research_before


def test_research_failure_paths_leave_state_and_decision_untouched(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    _write_decision(config.production_root, scoreable=False)
    picks = config.production_root / "data" / "picks"
    before = _tree_hashes(picks)
    for kwargs in ({"export_ok": False}, {"verify_ok": False}):
        monkeypatch.setattr("scripts.live_forward_capture_once.run",
                            _fake_run_research(config, [], **kwargs))
        code, payload = capture_once(config)
        assert code == 1 and payload["status"].startswith("failed_research_")
        assert _tree_hashes(picks) == before
        assert not _official_dir(config).exists()
        # leave the partial dir for the next run to quarantine
        assert not (_research_dir(config) / RESEARCH_SIDECAR_NAME).exists()


def _seq_clock(*times):
    """Clock returning times in order, then repeating the last one."""
    it = list(times)
    def now():
        return it.pop(0) if len(it) > 1 else it[0]
    return now


def test_research_capture_writes_acceptance_marker_bound_to_sidecar(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, []))

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (0, "captured_research_no_production_pick")
    marker = json.loads((_research_dir(config) / RESEARCH_ACCEPTED_NAME).read_text())
    assert marker["sidecar_sha256"] == file_sha256(_research_dir(config) / RESEARCH_SIDECAR_NAME)
    assert marker["published_at"] == payload["published_at"]


def test_research_sidecar_rename_landing_after_first_pitch_is_discarded(tmp_path, monkeypatch):
    # clock: started, completed(after hashing) pregame; the sample AFTER the sidecar rename is late
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, []))
    monkeypatch.setattr("scripts.live_forward_capture_once.utc_now_dt",
                        _seq_clock(_FIRST_PITCH - timedelta(minutes=10), _FIRST_PITCH - timedelta(minutes=9),
                                   _FIRST_PITCH + timedelta(seconds=1)))
    config = CaptureConfig(**{**config.__dict__, "now": None})

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (1, "discarded_research_after_first_pitch")
    assert "sidecar publication" in payload["message"]
    assert not _research_dir(config).exists()
    late = [p for p in _research_dir(config).parent.iterdir() if p.name.startswith(f"{_DATE}.late.")]
    assert len(late) == 1 and not (late[0] / RESEARCH_ACCEPTED_NAME).exists()


def test_research_marker_rename_landing_after_first_pitch_is_discarded(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, []))
    monkeypatch.setattr("scripts.live_forward_capture_once.utc_now_dt",
                        _seq_clock(_FIRST_PITCH - timedelta(minutes=10), _FIRST_PITCH - timedelta(minutes=9),
                                   _FIRST_PITCH - timedelta(seconds=2), _FIRST_PITCH))
    config = CaptureConfig(**{**config.__dict__, "now": None})

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (1, "discarded_research_after_first_pitch")
    assert "acceptance marker" in payload["message"]
    assert not _research_dir(config).exists()


def test_research_crash_after_sidecar_before_marker_is_unaccepted(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    calls: list = []
    monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, calls))
    capture_once(config)
    (_research_dir(config) / RESEARCH_ACCEPTED_NAME).unlink()  # simulate crash before phase 2

    # pregame -> unaccepted capture is quarantined and redone
    code, payload = capture_once(config)
    assert (code, payload["status"]) == (0, "captured_research_no_production_pick")
    assert any(p.name.startswith(f"{_DATE}.partial.") for p in _research_dir(config).parent.iterdir())

    # after the deadline -> left alone, never accepted
    (_research_dir(config) / RESEARCH_ACCEPTED_NAME).unlink()
    late_cfg = CaptureConfig(**{**config.__dict__, "now": _FIRST_PITCH + timedelta(minutes=1)})
    code, payload = capture_once(late_cfg)
    assert (code, payload["status"]) == (1, "research_partial_after_deadline")


def test_research_tampered_marker_or_late_publication_is_invalid(tmp_path, monkeypatch):
    for mutate in (
        lambda m: m.__setitem__("published_at", _FIRST_PITCH.isoformat()),
        lambda m: m.__setitem__("sidecar_sha256", "0" * 64),
    ):
        config = _research_config(tmp_path / str(id(mutate)))
        _write_state(config.production_root, final_skip_candidate=_FSC)
        monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, []))
        capture_once(config)
        marker_path = _research_dir(config) / RESEARCH_ACCEPTED_NAME
        marker = json.loads(marker_path.read_text()); mutate(marker); marker_path.write_text(json.dumps(marker))

        code, payload = capture_once(config)

        assert (code, payload["status"]) == (0, "captured_research_no_production_pick")  # pregame redo
        assert any(p.name.startswith(f"{_DATE}.partial.") for p in _research_dir(config).parent.iterdir())


def test_research_sidecar_late_start_or_missing_or_mismatched_provenance_is_invalid(tmp_path, monkeypatch):
    def late_start(sc):
        sc["started_at"] = (_FIRST_PITCH - timedelta(minutes=1)).isoformat()
        sc["completed_at"] = (_FIRST_PITCH - timedelta(seconds=30)).isoformat()
    for mutate in (
        late_start,
        lambda sc: sc.pop("production_head"),
        lambda sc: sc.pop("live_forward_head"),
        lambda sc: sc.__setitem__("live_forward_head", "unrelated-sha"),
    ):
        config = _research_config(tmp_path / str(id(mutate)))
        _write_state(config.production_root, final_skip_candidate=_FSC)
        monkeypatch.setattr("scripts.live_forward_capture_once.run", _fake_run_research(config, []))
        capture_once(config)
        sidecar_path = _research_dir(config) / RESEARCH_SIDECAR_NAME
        sidecar = json.loads(sidecar_path.read_text()); mutate(sidecar); sidecar_path.write_text(json.dumps(sidecar))
        # keep the marker consistent with the tampered sidecar so ONLY the sidecar rule is under test
        marker_path = _research_dir(config) / RESEARCH_ACCEPTED_NAME
        marker = json.loads(marker_path.read_text()); marker["sidecar_sha256"] = file_sha256(sidecar_path)
        marker_path.write_text(json.dumps(marker))

        code, payload = capture_once(config)

        assert (code, payload["status"]) == (0, "captured_research_no_production_pick"), mutate
        assert any(p.name.startswith(f"{_DATE}.partial.") for p in _research_dir(config).parent.iterdir())


def test_research_manifest_without_exact_run_kind_fails_contract(tmp_path, monkeypatch):
    config = _research_config(tmp_path)
    _write_state(config.production_root, final_skip_candidate=_FSC)
    inner = _fake_run_research(config, [])

    def run_no_kind(args, *, cwd, env=None):
        result = inner(args, cwd=cwd, env=env)
        if "export-live-candidate-artifacts" in args:
            out = Path(args[args.index("--output-dir") + 1])
            m = json.loads((out / "manifest.json").read_text()); m.pop("run_kind")
            (out / "manifest.json").write_text(json.dumps(m))
        return result

    monkeypatch.setattr("scripts.live_forward_capture_once.run", run_no_kind)

    code, payload = capture_once(config)

    assert (code, payload["status"]) == (1, "failed_research_contract_incomplete")
    assert not (_research_dir(config) / RESEARCH_SIDECAR_NAME).exists()
