"""Render production cron with local path shims and parse every CLI invocation."""

import os
import re
import shlex
import subprocess
import sys
import click

from bts import cli
from offline import ROOT, World, at


def executable(path, source):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    path.chmod(0o700)


def parse_command(command, argv, parent=None):
    with command.make_context(command.name, list(argv), parent=parent) as context:
        if isinstance(command, click.Group):
            _, subcommand, remainder = command.resolve_command(context, context._protected_args + context.args)
            return parse_command(subcommand, remainder, parent=context)
        return context.params


def probe(legacy=False):
    with World("2027-03-24") as w:
        fixture = w.root / "cron-fixture"
        worktree = fixture / "projects/bts"
        worktree.mkdir(parents=True)
        uv = fixture / ".local/bin/uv"
        executable(uv, "#!/bin/sh\nshift\nshift\nexec " + shlex.quote(sys.executable) +
                   " -c 'from bts.cli import cli; cli()' \"$@\"\n")
        executable(uv.with_name("crontab"), "#!/bin/sh\nprintf '%s\\n' 'no crontab for fixture' >&2\nexit 1\n")
        orch = fixture / ".bts-orchestrator.toml"
        source_root = ROOT / ".codex-review/whole-range/base" if legacy else ROOT
        script = (source_root / "scripts/cron-setup-hetzner.sh").read_text().replace("$HOME", "$WRBLIND_CRON_FIXTURE_ROOT")
        env = dict(os.environ, PATH=f"{uv.parent}:{os.environ['PATH']}", PYTHONPATH=str(source_root / "src"),
                   WRBLIND_CRON_FIXTURE_ROOT=str(fixture), HEALTHCHECKS_PING_URL="https://offline.invalid")
        for intent, mode in (("research", "private"), ("enter", "dm"), ("enter", "private")):
            w.config["scheduler"].update(entry_intent=intent, pick_delivery=mode)
            if legacy:
                w.config["scheduler"].pop("contest_state_required", None)
                w.config["health_checks"].pop("contest_state_expected", None)
            orch.write_bytes(w.toml().read_bytes())
            result = subprocess.run(["bash", "-c", script, "cron-setup-hetzner.sh", "show"],
                                    env=env, text=True, capture_output=True, check=False)
            w.record("show", intent, mode, result.returncode, result.stdout, result.stderr)
            if result.returncode:
                continue
            if legacy and intent == "research":
                snapshot = ROOT / ".codex-review/whole-range/synthetic-snapshot/restored-crontab.txt"
                snapshot.write_text(result.stdout.split("Would install:\n", 1)[1].replace(str(fixture), "/home/bts"))
            for line in result.stdout.splitlines():
                if not line or not line[0].isdigit() and line[0] != "*":
                    continue
                normalized = re.sub(r"\$\(date .*?\)", "2027-03-24", line)
                normalized = normalized.replace("--seasons 2027-03-24", "--seasons 2027")
                for args in re.findall(r"run bts (.*?)(?=\s(?:>>|>|;|&&)|$)", normalized):
                    argv = shlex.split(args)
                    w.record("parsed", argv, parse_command(cli.cli, argv))
                    w.invoke(*argv, "--help", expect=0)
                if "run python" in line:
                    help_result = subprocess.run([sys.executable, str(ROOT / "scripts/check_heartbeat.py"), "--help"],
                                                 env=env, text=True, capture_output=True, check=False)
                    w.record("heartbeat_help", help_result.returncode, help_result.stdout)
                if "curl " in line:
                    help_result = subprocess.run(["curl", "--help"], env=env, text=True,
                                                 capture_output=True, check=False)
                    w.record("curl_help", help_result.returncode)
        w.dump("cron-base" if legacy else "cron")


def early_entry_window():
    day = "2027-03-24"
    with World(day) as w:
        w.config["scheduler"]["lineup_check_offset_min"] = 60
        w.games[day] = []
        w.game("06:10")
        w.finish = at(day, "08:25")
        w.now = at(day, "05:10")
        w.dated()
        w.pending = [{"roundId": 1, "playerId": 1}]
        w.run(via_cli=True)
        w.now = at(day, "10:00")
        w.watchdog()
        w.now = at("2027-03-25", "08:30")
        w.outcome()
        w.snapshot()
        w.dump("cron-early-entry")


if __name__ == "__main__":
    is_legacy = sys.argv[1:] == ["base"]
    probe(legacy=is_legacy)
    if not is_legacy:
        early_entry_window()
