"""Declared entry intent, `[scheduler].entry_intent`, and the intent-aware cron install (C1 rank-2 watchdog plan P4;
registration R4, docs/sota_audit/2026-10-04-prereg-c1-watchdog.md).

The cron tests drive the real `scripts/cron-setup-hetzner.sh` under a throwaway HOME, with `crontab` and `uv` replaced
by shims: `crontab` reads and writes a file, and `uv` accepts only `uv run bts entry-intent ...` and runs the real CLI.
No real crontab, ping URL or box config is touched.
"""
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

from bts.cli import cli
from bts.entry_intent import check_entry_intent

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "cron-setup-hetzner.sh"
ENTRY_CMD = "bts check-pick-entered"


# ---- the resolver ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("sched, intent, delivery", [
    ({"entry_intent": "research", "pick_delivery": "private"}, "research", "private"),
    ({"entry_intent": "research", "private_mode": True}, "research", "private"),
    ({"entry_intent": "research", "posting_mode": "local"}, "research", "private"),
    ({"entry_intent": "enter", "pick_delivery": "dm"}, "enter", "dm"),
    ({"entry_intent": "enter", "pick_delivery": "bluesky"}, "enter", "public"),
    ({"entry_intent": "enter"}, "enter", "public"),        # the scheduler's own default when no delivery key is set
])
def test_agreeing_intent_and_delivery(sched, intent, delivery):
    got = check_entry_intent({"scheduler": sched})
    assert (got.intent, got.delivery, got.problems) == (intent, delivery, ())
    assert got.ok


@pytest.mark.parametrize("config, needle", [
    ({"scheduler": {"pick_delivery": "private"}}, "missing"),
    ({}, "missing"),
    ({"scheduler": {"entry_intent": "Enter", "pick_delivery": "dm"}}, "exactly"),
    ({"scheduler": {"entry_intent": " research", "pick_delivery": "private"}}, "exactly"),
    ({"scheduler": {"entry_intent": "", "pick_delivery": "private"}}, "exactly"),
    ({"scheduler": {"entry_intent": True, "pick_delivery": "private"}}, "exactly"),
    ({"scheduler": {"entry_intent": "play", "pick_delivery": "dm"}}, "exactly"),
    ({"scheduler": "research"}, "table"),
])
def test_missing_or_unsupported_intent_is_a_problem_never_a_default(config, needle):
    got = check_entry_intent(config)
    assert got.intent is None and not got.ok
    assert any(needle in p for p in got.problems), got.problems


@pytest.mark.parametrize("sched, delivery", [
    ({"entry_intent": "enter", "pick_delivery": "private"}, "private"),
    ({"entry_intent": "enter", "private_mode": True}, "private"),
    ({"entry_intent": "research", "pick_delivery": "dm"}, "dm"),
    ({"entry_intent": "research"}, "public"),
    ({"entry_intent": "research", "private_mode": True, "pick_delivery": "public"}, "public"),   # explicit key wins
])
def test_disagreement_with_the_effective_delivery_mode(sched, delivery):
    got = check_entry_intent({"scheduler": sched})
    assert got.intent == sched["entry_intent"] and got.delivery == delivery and not got.ok
    assert any("expects" in p for p in got.problems), got.problems


def test_a_delivery_config_the_scheduler_refuses_is_a_problem():
    for sched in ({"entry_intent": "research", "shadow_mode": True},
                  {"entry_intent": "enter", "pick_delivery": "carrier-pigeon"}):
        got = check_entry_intent({"scheduler": sched})
        assert got.delivery is None and not got.ok
        assert any("refuses" in p for p in got.problems), got.problems


# ---- the CLI --------------------------------------------------------------------------------------------------------

def _toml(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_cli_prints_only_the_intent_when_it_agrees(tmp_path):
    cfg = _toml(tmp_path / "o.toml", '[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n')
    res = CliRunner().invoke(cli, ["entry-intent", "--config", str(cfg)])
    assert res.exit_code == 0 and res.stdout == "research\n"


@pytest.mark.parametrize("body", [
    '[scheduler]\npick_delivery = "private"\n',
    '[scheduler]\nentry_intent = "enter"\npick_delivery = "private"\n',
    '[scheduler\nentry_intent = "enter"\n',
])
def test_cli_refuses_with_nothing_on_stdout(tmp_path, body):
    cfg = _toml(tmp_path / "o.toml", body)
    res = CliRunner().invoke(cli, ["entry-intent", "--config", str(cfg)])
    assert res.exit_code == 1 and res.stdout == "" and "entry-intent:" in res.stderr


def test_cli_refuses_a_missing_config(tmp_path):
    res = CliRunner().invoke(cli, ["entry-intent", "--config", str(tmp_path / "absent.toml")])
    assert res.exit_code == 1 and res.stdout == "" and "entry-intent:" in res.stderr


# ---- the cron install -----------------------------------------------------------------------------------------------

def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


@pytest.fixture
def box(tmp_path):
    """A throwaway HOME shaped like the box, with crontab and uv shims."""
    home = tmp_path / "home"
    (home / "projects" / "bts").mkdir(parents=True)
    (home / "projects" / "bts" / ".env").write_text("")
    (home / ".local" / "bin").mkdir(parents=True)
    _exe(home / ".local" / "bin" / "uv",
         '#!/bin/sh\n[ "$1 $2 $3" = "run bts entry-intent" ] || { echo "uv shim: unexpected $*" >&2; exit 97; }\n'
         f'shift 2\nexec "{sys.executable}" -c "from bts.cli import cli; cli()" "$@"\n')
    shims = tmp_path / "shims"
    shims.mkdir()
    crontab = tmp_path / "crontab.txt"
    _exe(shims / "crontab",
         f'#!/bin/sh\nf="{crontab}"\ncase "$1" in\n'
         '  -l) [ -f "$f.unreadable" ] && { echo "crontab: cannot read" >&2; exit 1; }\n'
         '      [ -f "$f" ] && cat "$f" || { echo "no crontab for user" >&2; exit 1; } ;;\n'
         '  -) cat > "$f.new" && mv "$f.new" "$f" ;;\n  *) exit 98 ;;\nesac\n')   # read all stdin first, like crontab
    env = {"HOME": str(home), "PATH": f"{shims}:/usr/bin:/bin",
           "HEALTHCHECKS_PING_URL": "https://hc-ping.invalid/test-only"}

    class Box:
        config = home / ".bts-orchestrator.toml"
        cron = crontab

        def run(self, action):
            return subprocess.run(["bash", str(SCRIPT), action], env=env, capture_output=True, text=True, timeout=120)

        def lines(self):
            return self.cron.read_text().splitlines() if self.cron.exists() else None
    return Box()


def _bts_lines(lines):
    return [ln for ln in lines if "# BTS-HETZNER" in ln]


def test_research_install_leaves_out_the_entry_cron(box):
    box.config.write_text('[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n')
    res = box.run("install")
    assert res.returncode == 0, res.stderr
    lines = box.lines()
    assert not [ln for ln in lines if ENTRY_CMD in ln]
    assert any("bts check-results" in ln for ln in lines) and any("bts reconcile" in ln for ln in lines)


def test_enter_install_adds_only_the_entry_cron(box):
    box.config.write_text('[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n')
    assert box.run("install").returncode == 0
    research = box.lines()
    box.config.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
    res = box.run("install")
    assert res.returncode == 0, res.stderr
    enter = box.lines()
    entry = [ln for ln in enter if ENTRY_CMD in ln]
    assert len(entry) == 1 and entry[0].startswith("*/15 10-23 * * * ")
    assert [ln for ln in enter if ENTRY_CMD not in ln] == research


def test_research_install_removes_an_existing_entry_line_and_keeps_foreign_lines(box):
    box.cron.write_text("0 9 * * * echo not-bts\n"
                        f"*/15 10-23 * * * cd x && uv run {ENTRY_CMD} >> log 2>&1 # BTS-HETZNER\n"
                        f"# */15 10-23 * * * cd x && uv run {ENTRY_CMD} >> log 2>&1 # BTS-HETZNER\n")
    box.config.write_text('[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n')
    assert box.run("install").returncode == 0
    lines = box.lines()
    assert lines[0] == "0 9 * * * echo not-bts"
    assert not [ln for ln in lines if ENTRY_CMD in ln]


@pytest.mark.parametrize("body", [
    None,                                                              # no config file
    '[scheduler]\npick_delivery = "private"\n',                        # no intent
    '[scheduler]\nentry_intent = "Research"\npick_delivery = "private"\n',
    '[scheduler]\nentry_intent = "enter"\npick_delivery = "private"\n',
    '[scheduler]\nentry_intent = "research"\npick_delivery = "dm"\n',
    '[scheduler]\nentry_intent = "research"\n',                        # the scheduler would post publicly
    '[scheduler]\nentry_intent = "research"\nshadow_mode = true\n',    # checklist C2
    '[scheduler\n',
])
def test_install_refuses_and_leaves_the_crontab_untouched(box, body):
    before = f"0 9 * * * echo not-bts\n*/15 10-23 * * * uv run {ENTRY_CMD} # BTS-HETZNER\n"
    box.cron.write_text(before)
    if body is not None:
        box.config.write_text(body)
    res = box.run("install")
    assert res.returncode != 0
    assert "entry_intent" in res.stderr or "entry-intent" in res.stderr, res.stderr
    assert box.cron.read_text() == before


def test_show_reports_the_intent_and_writes_nothing(box):
    box.config.write_text('[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n')
    res = box.run("show")
    assert res.returncode == 0, res.stderr
    assert "research" in res.stdout and ENTRY_CMD not in res.stdout and "bts check-results" in res.stdout
    assert box.lines() is None


def test_show_refuses_without_a_valid_intent(box):
    box.config.write_text('[scheduler]\npick_delivery = "private"\n')
    res = box.run("show")
    assert res.returncode != 0 and "Would install" not in res.stdout


def test_remove_needs_no_config(box):
    box.cron.write_text(f"0 9 * * * echo not-bts\n*/15 10-23 * * * uv run {ENTRY_CMD} # BTS-HETZNER\n")
    res = box.run("remove")
    assert res.returncode == 0, res.stderr
    assert box.lines() == ["0 9 * * * echo not-bts"]


# ---- reading the existing crontab (the old pipeline wiped a BTS-only crontab under set -e/pipefail) -----------------

RESEARCH = '[scheduler]\nentry_intent = "research"\npick_delivery = "private"\n'


def test_install_over_a_crontab_of_only_bts_lines_replaces_them(box):
    box.config.write_text(RESEARCH)
    assert box.run("install").returncode == 0
    first = box.lines()
    assert first and all("# BTS-HETZNER" in ln for ln in first)
    res = box.run("install")
    assert res.returncode == 0, res.stderr
    assert box.lines() == first


def test_remove_of_a_crontab_of_only_bts_lines_empties_it(box):
    box.config.write_text(RESEARCH)
    assert box.run("install").returncode == 0
    res = box.run("remove")
    assert res.returncode == 0, res.stderr
    assert box.lines() == []


@pytest.mark.parametrize("action", ["install", "remove"])
def test_an_unreadable_crontab_refuses_and_writes_nothing(box, action):
    box.config.write_text(RESEARCH)
    before = "0 9 * * * echo not-bts\n"
    box.cron.write_text(before)
    Path(f"{box.cron}.unreadable").write_text("")
    res = box.run(action)
    assert res.returncode != 0 and "cannot read the current crontab" in res.stderr
    assert box.cron.read_text() == before
