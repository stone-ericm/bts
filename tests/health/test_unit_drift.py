"""Tests for the unit_drift health source (audit F12).

The live systemd units were hand-maintained snowflakes: the previously
tracked unit was the stale Pi5 one (wrong user, wrong path, no watchdog),
so DR-from-repo would install a broken unit. Now scripts/systemd/ holds
the canonical templates and this check flags any divergence between the
installed units and the repo copies — read-only; installation stays an
explicit operator action.
"""
from pathlib import Path

from bts.health import unit_drift


UNIT_TEXT = "[Unit]\nDescription=X\n[Service]\nExecStart=/bin/true\n"


def _mk(dirpath: Path, name: str, text: str) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / name
    p.write_text(text)
    return p


def test_no_installed_units_is_silent(tmp_path):
    repo = tmp_path / "repo"
    _mk(repo, "bts-scheduler.service", UNIT_TEXT)
    installed = tmp_path / "systemd-user"  # doesn't exist
    assert unit_drift.check(installed_dir=installed, repo_units_dir=repo) == []


def test_matching_units_healthy(tmp_path):
    repo = tmp_path / "repo"
    installed = tmp_path / "installed"
    for name in ("bts-scheduler.service", "bts-dashboard.service"):
        _mk(repo, name, UNIT_TEXT)
        _mk(installed, name, UNIT_TEXT)
    assert unit_drift.check(installed_dir=installed, repo_units_dir=repo) == []


def test_drifted_unit_warns_with_name(tmp_path):
    repo = tmp_path / "repo"
    installed = tmp_path / "installed"
    _mk(repo, "bts-scheduler.service", UNIT_TEXT)
    _mk(installed, "bts-scheduler.service", UNIT_TEXT + "RestartSec=999\n")
    alerts = unit_drift.check(installed_dir=installed, repo_units_dir=repo)
    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert alerts[0].source == "unit_drift"
    assert "bts-scheduler.service" in alerts[0].message


def test_installed_unit_without_template_warns(tmp_path):
    # A bts unit running in prod that the repo can't reproduce is exactly
    # the F12 failure mode — but only for units we claim to track.
    repo = tmp_path / "repo"
    installed = tmp_path / "installed"
    _mk(repo, "bts-scheduler.service", UNIT_TEXT)
    _mk(installed, "bts-scheduler.service", UNIT_TEXT)
    _mk(installed, "bts-dashboard.service", UNIT_TEXT)  # no template
    alerts = unit_drift.check(installed_dir=installed, repo_units_dir=repo)
    assert len(alerts) == 1
    assert "bts-dashboard.service" in alerts[0].message
    assert "template" in alerts[0].message


def test_untracked_units_ignored(tmp_path):
    # bts-leaderboard.* stays intentionally untemplated (timer disabled by
    # design 2026-07-04); only units in TRACKED_UNITS are compared.
    repo = tmp_path / "repo"
    installed = tmp_path / "installed"
    _mk(repo, "bts-scheduler.service", UNIT_TEXT)
    _mk(installed, "bts-scheduler.service", UNIT_TEXT)
    _mk(installed, "bts-leaderboard.service", "whatever")
    assert unit_drift.check(installed_dir=installed, repo_units_dir=repo) == []


def test_template_without_installed_unit_is_silent(tmp_path):
    # Repo ships templates; a dev box without installed units must be quiet.
    repo = tmp_path / "repo"
    installed = tmp_path / "installed"
    installed.mkdir()
    _mk(repo, "bts-scheduler.service", UNIT_TEXT)
    _mk(repo, "bts-dashboard.service", UNIT_TEXT)
    assert unit_drift.check(installed_dir=installed, repo_units_dir=repo) == []


def test_tracked_units_constant():
    assert unit_drift.TRACKED_UNITS == (
        "bts-scheduler.service", "bts-dashboard.service",
    )
