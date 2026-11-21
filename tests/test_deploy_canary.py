from datetime import datetime, timedelta
import os
from pathlib import Path
import subprocess
import sys

from click.testing import CliRunner
import pytest

from bts.cli import cli


def run_canary(config):
    return CliRunner().invoke(cli, ["deploy-canary", "--config", str(config)])


@pytest.mark.parametrize("settings", [
    "", "contest_state_required = true\n", "[health_checks]\ncontest_state_expected = true\n",
    "contest_state_required = true\n[health_checks]\ncontest_state_expected = true\n",
])
def test_private_canary_skips_contest_read(tmp_path, settings):
    config = tmp_path / "config.toml"
    config.write_text('[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n' + settings)
    result = run_canary(config)
    assert result.exit_code == 0
    assert result.output == "Deploy canary: not a live configuration (delivery private); contest state not checked.\n"


@pytest.mark.parametrize("mode", ["dm", "public", "private"])
def test_canary_applies_shared_startup_gate(tmp_path, mode):
    from bts.entry_intent import live_gate_problem

    delivery = "bluesky" if mode == "public" else mode
    scheduler = {"entry_intent": "enter", "pick_delivery": delivery}
    config = tmp_path / "config.toml"
    config.write_text(f'[scheduler]\nentry_intent = "enter"\npick_delivery = "{delivery}"\n')
    problem = live_gate_problem({"scheduler": scheduler})
    assert problem
    result = run_canary(config)
    assert result.exit_code == 1
    assert result.output == f"Deploy canary: REFUSED — {problem}\n"


@pytest.mark.parametrize("kind", ["absent", "directory", "invalid", "oversized"])
def test_canary_configuration_error(tmp_path, kind):
    config = tmp_path / "config.toml"
    if kind == "directory":
        config.mkdir()
    elif kind == "invalid":
        config.write_bytes(b"\xff")
    elif kind == "oversized":
        config.write_bytes(b" " * 65537)
    result = run_canary(config)
    assert result.exit_code == 2
    assert result.output.startswith("Deploy canary: ERROR — ")
    assert result.output.count("\n") == 1
    assert os.fspath(config) not in result.output


@pytest.mark.parametrize(
    ("switch", "manual", "auto"),
    [
        *[(f"corrupt{i}", "absent", "covered") for i in range(1, 9)],
        ("automatic", "legacy", "covered"),
        ("absent", "absent", "covered"),
        ("other_season", "absent", "covered"),
        ("revoked", "absent", "covered"),
        *[("automatic", "absent", auto) for auto in (
            "absent", "refused", "corrupt", "v1", "other_season", "other_id", "covered",
        )],
        ("corrupt1", "today", "absent"),
        ("absent", "today", "absent"),
        ("automatic", "absent", "stale-covered"),
    ],
)
def test_live_canary_matches_shared_decision(tmp_path, monkeypatch, capsys, switch, manual, auto):
    from bts.contest_state import ContestStateError, load_decision_streak_state
    from bts.orchestrator import load_config
    from bts.scheduler import _morning_state_check
    from tests.test_season_contract_cs4b2 import AUTOS, MANUALS, NOW, SWITCHES, install

    install(tmp_path, "season_contract.json", SWITCHES[switch])
    install(tmp_path, "contest_streak.manual.json", MANUALS[manual])
    automatic = (AUTOS["covered"] | {"recorded_at": "2027-07-14T16:00:00Z"}
                 if auto == "stale-covered" else AUTOS[auto])
    install(tmp_path, "contest_streak.json", automatic)
    config = tmp_path / "config.toml"
    config.write_text(
        f'[orchestrator]\npicks_dir = "{tmp_path}"\n'
        '[scheduler]\npick_delivery = "dm"\ncontest_state_required = true\n'
    )
    import bts.cli as cli_module
    monkeypatch.setattr(cli_module, "_now_et", lambda: NOW)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = run_canary(config)
    _morning_state_check(load_config(config), tmp_path, NOW.date().isoformat(), NOW,
                         NOW + timedelta(hours=1))
    morning = capsys.readouterr().err
    try:
        decision = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    except ContestStateError as exc:
        assert result.exit_code == 1
        assert result.output == f"Deploy canary: REFUSED — {ascii(str(exc))[:800]}\n"
        assert f"no pick can be made yet: {ascii(str(exc))[:800]}" in morning
    else:
        assert result.exit_code == 0
        assert result.output.startswith("Deploy canary: live contest state accepted (")
        assert f"streak {decision.streak}) at {NOW.isoformat()}." in result.output
        assert "contest state accepted" in morning or "stale contest reading" in morning
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("mode", ["dm", "public"])
@pytest.mark.parametrize("flags", [0, 1, 2, 3])
def test_live_modes_use_either_contest_flag(tmp_path, monkeypatch, mode, flags):
    import bts.contest_state as contest_state
    from tests.test_season_contract_cs4b2 import NOW
    from types import SimpleNamespace

    config = tmp_path / "config.toml"
    settings = []
    if flags & 1:
        settings.append("contest_state_required = true")
    if flags & 2:
        settings.extend(("[health_checks]", "contest_state_expected = true"))
    config.write_text(
        f'[orchestrator]\npicks_dir = "{tmp_path}"\n[scheduler]\npick_delivery = "{mode}"\n'
        + "\n".join(settings) + "\n"
    )
    reads = []
    monkeypatch.setattr("bts.cli._now_et", lambda: NOW)

    def read(path, *, require_contest_state, now):
        reads.append((path, require_contest_state, now))
        return SimpleNamespace(observation_kind="automatic", streak=12)

    monkeypatch.setattr(contest_state, "load_decision_streak_state", read)
    result = run_canary(config)
    assert result.exit_code == (0 if flags else 1)
    assert reads == ([(tmp_path, True, NOW)] if flags else [])


@pytest.mark.parametrize(("toml", "status"), [
    ('[scheduler]\npick_delivery = "invalid"', 2),
    ('scheduler = false', 1),
    ('health_checks = false', 1),
])
def test_invalid_delivery_or_table(tmp_path, toml, status):
    config = tmp_path / "canary.toml"
    config.write_text(toml)
    result = run_canary(config)
    assert result.exit_code == status
    assert result.output.startswith("Deploy canary: ERROR — " if status == 2 else "Deploy canary: REFUSED — ")


def test_fifo_config_fails_without_a_writer(tmp_path):
    config = tmp_path / "config.toml"
    os.mkfifo(config)
    result = subprocess.run(
        [sys.executable, "-c", "from bts.cli import cli; cli()", "deploy-canary", "--config", str(config)],
        capture_output=True, text=True, timeout=4,
    )
    assert result.returncode == 2
    assert result.stdout == "Deploy canary: ERROR — ValueError\n"


@pytest.mark.parametrize("day", ["2027-07-15", "2027-12-31"])
def test_config_load_straddles_et_midnight(tmp_path, monkeypatch, day):
    from zoneinfo import ZoneInfo
    import bts.orchestrator as orchestrator
    from tests.test_season_contract_cs4b2 import dated, install

    et = ZoneInfo("America/New_York")
    at = datetime.fromisoformat(f"{day}T23:59:59").replace(tzinfo=et)
    config = tmp_path / "config.toml"
    config.write_text(f'[orchestrator]\npicks_dir = "{tmp_path}"\n'
                      '[scheduler]\npick_delivery = "public"\ncontest_state_required = true\n')
    install(tmp_path, "contest_streak.manual.json", dated(day, recorded_at=at.isoformat()))
    clock = [at]
    loads = []
    original = orchestrator.load_config
    monkeypatch.setattr("bts.cli._now_et", lambda: clock[0])

    def straddle(path: Path, **kwargs):
        loads.append(path)
        clock[0] = at + timedelta(seconds=2)
        return original(path, **kwargs)

    monkeypatch.setattr(orchestrator, "load_config", straddle)
    before = run_canary(config)
    after = run_canary(config)
    assert loads == [config, config]
    assert before.exit_code == 0
    assert f"at {at.isoformat()}." in before.output
    assert after.exit_code == 1


def test_config_is_parsed_from_the_validated_descriptor(tmp_path, monkeypatch):
    config = tmp_path / "config.toml"
    config.write_text('[scheduler]\npick_delivery = "private"\n')
    original_inode = config.stat().st_ino
    replacement = tmp_path / "replacement.toml"
    replacement.write_text('[scheduler]\npick_delivery = "dm"\n')
    original = os.fstat
    switched = []

    def replace_after_validation(fd):
        metadata = original(fd)
        if metadata.st_ino == original_inode and not switched:
            replacement.replace(config)
            switched.append(fd)
        return metadata

    monkeypatch.setattr(os, "fstat", replace_after_validation)
    result = run_canary(config)
    assert switched
    assert result.exit_code == 0
    assert "delivery private" in result.output


def test_live_canary_has_no_prediction_transport_or_write(tmp_path, monkeypatch):
    import urllib.request
    import bts.orchestrator as orchestrator
    import bts.scheduler as scheduler
    import requests.sessions
    from tests.test_season_contract_cs4b2 import NOW, dated, install

    def forbidden(*args, **kwargs):
        pytest.fail("canary used a scheduler or network entry point")

    for name in ("run_day", "run_and_pick", "predict_local_shadow"):
        monkeypatch.setattr(scheduler, name, forbidden)
    monkeypatch.setattr(orchestrator, "ssh_predict", forbidden)
    monkeypatch.setattr(requests.sessions.Session, "request", forbidden)
    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr("bts.cli._now_et", lambda: NOW)
    install(tmp_path, "contest_streak.manual.json", dated())
    config = tmp_path / "config.toml"
    config.write_text(f'[orchestrator]\npicks_dir = "{tmp_path}"\n'
                      '[scheduler]\npick_delivery = "public"\ncontest_state_required = true\n')
    snapshot = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = run_canary(config)
    assert result.exit_code == 0
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == snapshot


@pytest.mark.parametrize("skip", ["other-day", "delivered", "delivery-uncertain"])
def test_canary_still_asks_when_morning_read_is_skipped(tmp_path, monkeypatch, capsys, skip):
    from types import SimpleNamespace

    import bts.picks as picks
    from bts.orchestrator import load_config
    from bts.scheduler import _morning_state_check
    from tests.test_season_contract_cs4b2 import NOW

    config = tmp_path / "config.toml"
    config.write_text(f'[orchestrator]\npicks_dir = "{tmp_path}"\n'
                      '[scheduler]\npick_delivery = "dm"\ncontest_state_required = true\n')
    monkeypatch.setattr("bts.cli._now_et", lambda: NOW)
    monkeypatch.setattr(picks, "load_pick", lambda *a: SimpleNamespace(delivery_attempted=skip == "delivery-uncertain"))
    monkeypatch.setattr(picks, "pick_was_delivered", lambda *a: skip == "delivered")
    day = (NOW - timedelta(days=int(skip == "other-day"))).date().isoformat()
    _morning_state_check(load_config(config), tmp_path, day, NOW, NOW + timedelta(hours=1))
    assert not capsys.readouterr().err
    result = run_canary(config)
    assert result.exit_code == 1
    assert result.output.startswith("Deploy canary: REFUSED — ")
