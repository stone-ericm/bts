"""Tests for restic-based operational-state backup (audit F5).

data/picks (decisions, contest ledger, manual saver flag, delivery/skip-shadow
markers) and data/health_state exist ONLY on the production box — no git, no
R2 sync, nothing. Box loss = irrecoverable operational state. backup.py wraps
restic (encrypted, versioned, deduped) targeting the existing R2 bucket.

All tests inject a fake runner — no restic binary or network required.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bts.data import backup


FIXED_NOW = datetime(2026, 7, 10, 16, 0, 0, tzinfo=timezone.utc)

BASE_ENV = {
    "R2_ACCOUNT_ID": "acct123",
    "R2_ACCESS_KEY_ID": "ak",
    "R2_SECRET_ACCESS_KEY": "sk",
    "RESTIC_PASSWORD": "pw",
}


class FakeRunner:
    """Records restic invocations; returns queued (rc, stdout, stderr)."""

    def __init__(self, results=None, side_effect=None):
        self.results = list(results or [])
        self.calls = []
        self.side_effect = side_effect

    def __call__(self, cmd, env=None, capture_output=True, text=True, timeout=None):
        self.calls.append({"cmd": list(cmd), "env": dict(env or {})})
        if self.side_effect is not None:
            self.side_effect(cmd)
        rc, out, err = (
            self.results.pop(0) if self.results else (0, "", "")
        )

        class R:
            returncode = rc
            stdout = out
            stderr = err

        return R()


SUMMARY_LINE = json.dumps({
    "message_type": "summary",
    "snapshot_id": "deadbeefcafe",
    "data_added": 4096,
    "total_files_processed": 7,
})


def _mk_repo(tmp_path: Path) -> Path:
    (tmp_path / "data" / "picks").mkdir(parents=True)
    (tmp_path / "data" / "picks" / "x.json").write_text("{}")
    (tmp_path / "data" / "health_state").mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------- restic_env

def test_restic_env_maps_r2_credentials_and_derives_repository():
    env = backup.restic_env(BASE_ENV)
    assert env["AWS_ACCESS_KEY_ID"] == "ak"
    assert env["AWS_SECRET_ACCESS_KEY"] == "sk"
    assert env["RESTIC_PASSWORD"] == "pw"
    assert env["RESTIC_REPOSITORY"] == (
        "s3:https://acct123.r2.cloudflarestorage.com/bts-backup-data/restic"
    )


def test_restic_env_missing_password_raises():
    env = {k: v for k, v in BASE_ENV.items() if k != "RESTIC_PASSWORD"}
    with pytest.raises(RuntimeError, match="RESTIC_PASSWORD"):
        backup.restic_env(env)


def test_restic_env_missing_r2_creds_raises():
    with pytest.raises(RuntimeError, match="R2_ACCOUNT_ID"):
        backup.restic_env({"RESTIC_PASSWORD": "pw"})


def test_restic_env_explicit_repository_override_wins():
    env = dict(BASE_ENV, BTS_RESTIC_REPOSITORY="s3:https://other/bucket/path")
    assert backup.restic_env(env)["RESTIC_REPOSITORY"] == "s3:https://other/bucket/path"


def test_restic_env_respects_bucket_override():
    env = dict(BASE_ENV, R2_BUCKET="custom-bucket")
    assert "custom-bucket/restic" in backup.restic_env(env)["RESTIC_REPOSITORY"]


# ---------------------------------------------------------------- backup sets

def test_backup_sets_registry():
    assert backup.BACKUP_SETS["ops"].paths == ("data/picks", "data/health_state")
    assert backup.BACKUP_SETS["archive"].paths == (
        "data/leaderboard", "data/hetzner_results", "data/external",
    )


def test_unknown_set_raises(tmp_path):
    with pytest.raises(KeyError):
        backup.run_backup("nope", _mk_repo(tmp_path), env=BASE_ENV, runner=FakeRunner())


# ---------------------------------------------------------------- run_backup

def test_run_backup_success_writes_status_and_runs_forget(tmp_path):
    repo = _mk_repo(tmp_path)
    runner = FakeRunner(results=[(0, SUMMARY_LINE + "\n", ""), (0, "", "")])

    status = backup.run_backup(
        "ops", repo, env=BASE_ENV, runner=runner, now_fn=lambda: FIXED_NOW,
    )

    assert len(runner.calls) == 2
    backup_cmd = runner.calls[0]["cmd"]
    assert backup_cmd[1] == "backup"
    assert str(repo / "data" / "picks") in backup_cmd
    assert str(repo / "data" / "health_state") in backup_cmd
    for flag in ("--tag", "ops", "--exclude", "*.lock", "--json"):
        assert flag in backup_cmd
    # credentials flow via env, never argv
    assert not any("sk" == part or "pw" == part for part in backup_cmd)
    assert runner.calls[0]["env"]["RESTIC_PASSWORD"] == "pw"

    forget_cmd = runner.calls[1]["cmd"]
    assert forget_cmd[1] == "forget"
    assert "--tag" in forget_cmd and "ops" in forget_cmd
    assert "--keep-daily" in forget_cmd

    assert status["ok"] is True
    assert status["snapshot_id"] == "deadbeefcafe"
    assert status["last_success_at"] == FIXED_NOW.isoformat()
    assert status["data_added"] == 4096

    on_disk = json.loads((repo / "data/health_state/backup_status.json").read_text())
    assert on_disk["ops"]["snapshot_id"] == "deadbeefcafe"


def test_run_backup_failure_preserves_last_success_and_skips_forget(tmp_path):
    repo = _mk_repo(tmp_path)
    prior = {"ops": {"set": "ops", "ok": True, "last_success_at": "2026-07-09T12:00:00+00:00"}}
    (repo / "data/health_state/backup_status.json").write_text(json.dumps(prior))
    runner = FakeRunner(results=[(1, "", "Fatal: unable to open repository: locked")])

    status = backup.run_backup(
        "ops", repo, env=BASE_ENV, runner=runner, now_fn=lambda: FIXED_NOW,
    )

    assert len(runner.calls) == 1  # no forget after failed backup
    assert status["ok"] is False
    assert "locked" in status["error"]
    assert status["last_success_at"] == "2026-07-09T12:00:00+00:00"
    on_disk = json.loads((repo / "data/health_state/backup_status.json").read_text())
    assert on_disk["ops"]["ok"] is False
    assert on_disk["ops"]["last_success_at"] == "2026-07-09T12:00:00+00:00"


def test_run_backup_all_paths_missing_fails_without_invoking_restic(tmp_path):
    (tmp_path / "data" / "health_state").mkdir(parents=True)
    runner = FakeRunner()
    status = backup.run_backup(
        "archive", tmp_path, env=BASE_ENV, runner=runner, now_fn=lambda: FIXED_NOW,
    )
    assert status["ok"] is False
    assert runner.calls == []
    assert "missing" in status["error"]


def test_run_backup_any_missing_path_fails_loudly(tmp_path):
    # Codex review I1: a "successful" backup that silently omitted a
    # required root (e.g. data/picks deleted) advanced last_success_at and
    # kept health green while the irreplaceable payload was unprotected.
    # Every path in a set is REQUIRED; these dirs always exist in prod.
    repo = _mk_repo(tmp_path)
    (repo / "data" / "leaderboard").mkdir()  # hetzner_results/external missing
    runner = FakeRunner()
    status = backup.run_backup(
        "archive", repo, env=BASE_ENV, runner=runner, now_fn=lambda: FIXED_NOW,
    )
    assert status["ok"] is False
    assert runner.calls == []
    assert "data/external" in status["error"]
    assert "data/hetzner_results" in status["error"]


def test_run_backup_missing_binary_writes_failed_status(tmp_path):
    # Codex review I3: cron installed before the restic binary exists →
    # FileNotFoundError escaped, backup_status.json was never created, and
    # the absent-file convention kept health silent FOREVER.
    repo = _mk_repo(tmp_path)

    def no_binary(cmd, env=None, capture_output=True, text=True, timeout=None):
        raise FileNotFoundError(2, "No such file or directory", cmd[0])

    status = backup.run_backup(
        "ops", repo, env=BASE_ENV, runner=no_binary, now_fn=lambda: FIXED_NOW,
    )
    assert status["ok"] is False
    assert "restic" in status["error"]
    on_disk = json.loads((repo / "data/health_state/backup_status.json").read_text())
    assert on_disk["ops"]["ok"] is False


def test_run_backup_timeout_writes_failed_status(tmp_path):
    import subprocess

    repo = _mk_repo(tmp_path)
    prior = {"ops": {"set": "ops", "ok": True, "last_success_at": "2026-07-09T12:00:00+00:00"}}
    (repo / "data/health_state/backup_status.json").write_text(json.dumps(prior))

    def hanging_runner(cmd, env=None, capture_output=True, text=True, timeout=None):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

    status = backup.run_backup(
        "ops", repo, env=BASE_ENV, runner=hanging_runner, now_fn=lambda: FIXED_NOW,
    )
    assert status["ok"] is False
    assert "timed out" in status["error"]
    assert status["last_success_at"] == "2026-07-09T12:00:00+00:00"
    on_disk = json.loads((repo / "data/health_state/backup_status.json").read_text())
    assert on_disk["ops"]["ok"] is False


# ---------------------------------------------------------------- prune

def test_run_prune_builds_command_and_reports(tmp_path):
    runner = FakeRunner(results=[(0, "pruned", "")])
    out = backup.run_prune(env=BASE_ENV, runner=runner)
    assert runner.calls[0]["cmd"][1] == "prune"
    assert out["ok"] is True

    failing = FakeRunner(results=[(1, "", "repo locked")])
    out = backup.run_prune(env=BASE_ENV, runner=failing)
    assert out["ok"] is False
    assert "locked" in out["error"]


# ---------------------------------------------------------------- restore drill

def _mk_restored_ops_tree(target: Path) -> Path:
    """Simulate restic's restore layout (original absolute paths under target)."""
    base = target / "home" / "bts" / "projects" / "bts" / "data" / "picks"
    (base / "account_state").mkdir(parents=True)
    (base / "account_state" / "saver_state.json").write_text(json.dumps(
        {"season": 2026, "source": "manual", "state": "active",
         "updated_at": "2026-07-01T00:00:00+00:00"}
    ))
    (base / "account_state" / "contest_ledger.jsonl").write_text(
        json.dumps({"active_streak": 1, "recorded_at": "x"}) + "\n"
    )
    (base / "2026-07-09").mkdir()
    (base / "2026-07-09" / "decision.json").write_text(json.dumps(
        {"action": "double", "date": "2026-07-09", "schema_version": "bts_daily_decision_v1"}
    ))
    (base / "streak.json").write_text(json.dumps({"streak": 1, "saver_available": True}))
    return target


def test_verify_restored_ops_tree_ok(tmp_path):
    _mk_restored_ops_tree(tmp_path)
    assert backup.verify_restored_ops_tree(tmp_path) == []


def test_verify_restored_ops_tree_flags_corrupt_saver(tmp_path):
    _mk_restored_ops_tree(tmp_path)
    saver = next(tmp_path.rglob("saver_state.json"))
    saver.write_text("{not json")
    problems = backup.verify_restored_ops_tree(tmp_path)
    assert any("saver_state" in p for p in problems)


def test_verify_restored_ops_tree_flags_missing_ledger(tmp_path):
    _mk_restored_ops_tree(tmp_path)
    next(tmp_path.rglob("contest_ledger.jsonl")).unlink()
    problems = backup.verify_restored_ops_tree(tmp_path)
    assert any("contest_ledger" in p for p in problems)


def test_verify_restored_ops_tree_flags_no_decisions(tmp_path):
    _mk_restored_ops_tree(tmp_path)
    for d in tmp_path.rglob("decision.json"):
        d.unlink()
    problems = backup.verify_restored_ops_tree(tmp_path)
    assert any("decision.json" in p for p in problems)


def test_restore_drill_invokes_restic_restore_and_verifies(tmp_path):
    target = tmp_path / "drill"

    def materialize(cmd):
        if "restore" in cmd:
            _mk_restored_ops_tree(target)

    runner = FakeRunner(results=[(0, "", "")], side_effect=materialize)
    result = backup.restore_drill(
        repo_root=tmp_path, target=target, env=BASE_ENV, runner=runner,
    )
    cmd = runner.calls[0]["cmd"]
    assert cmd[1] == "restore" and "latest" in cmd
    assert "--tag" in cmd and "ops" in cmd
    assert "--target" in cmd and str(target) in cmd
    assert result["ok"] is True
    assert result["problems"] == []


def test_restore_drill_refuses_nonempty_target(tmp_path):
    # Codex review I4: a reused target can contain files from an EARLIER
    # drill — recursive verification then passes on stale state a partial
    # snapshot never restored.
    target = tmp_path / "drill"
    target.mkdir()
    (target / "leftover.json").write_text("{}")
    runner = FakeRunner()
    result = backup.restore_drill(
        repo_root=tmp_path, target=target, env=BASE_ENV, runner=runner,
    )
    assert result["ok"] is False
    assert any("not empty" in p for p in result["problems"])
    assert runner.calls == []


def test_restore_drill_prefers_last_successful_snapshot_id(tmp_path):
    # Codex review I4: 'latest --tag ops' can select a PARTIAL (rc=3)
    # snapshot restic created during a run we recorded as failed. The last
    # SUCCESSFUL run's snapshot_id in backup_status.json is the recovery
    # target of record.
    repo = _mk_repo(tmp_path)
    (repo / "data/health_state/backup_status.json").write_text(json.dumps(
        {"ops": {"set": "ops", "ok": True, "snapshot_id": "goodsnap123",
                 "last_success_at": "2026-07-10T12:00:00+00:00"}}
    ))
    target = tmp_path / "drill"

    def materialize(cmd):
        if "restore" in cmd:
            _mk_restored_ops_tree(target)

    runner = FakeRunner(results=[(0, "", "")], side_effect=materialize)
    result = backup.restore_drill(
        repo_root=repo, target=target, env=BASE_ENV, runner=runner,
    )
    cmd = runner.calls[0]["cmd"]
    assert "goodsnap123" in cmd
    assert "latest" not in cmd
    assert result["ok"] is True
    assert result["snapshot"] == "goodsnap123"


def test_verify_restored_ops_tree_flags_missing_streak(tmp_path):
    _mk_restored_ops_tree(tmp_path)  # helper writes streak.json too
    next(tmp_path.rglob("streak.json")).unlink()
    problems = backup.verify_restored_ops_tree(tmp_path)
    assert any("streak.json" in p for p in problems)


def test_restore_drill_reports_problems(tmp_path):
    target = tmp_path / "drill"
    runner = FakeRunner(results=[(0, "", "")])  # restore "succeeds" but restores nothing
    result = backup.restore_drill(
        repo_root=tmp_path, target=target, env=BASE_ENV, runner=runner,
    )
    assert result["ok"] is False
    assert result["problems"]
