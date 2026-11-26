"""Run the real installer in a fake home and validate its generated commands."""
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner

from harness import REPO, cli


def execute(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o700)


def cron_fixture(root, intent):
    home_repo = root / "projects/bts"
    home_repo.mkdir(parents=True)
    config = root / ".bts-orchestrator.toml"
    config.write_text('[scheduler]\nentry_intent = "%s"\npick_delivery = "%s"\n'
                      'contest_state_required = %s\n' % (
                          intent, "dm" if intent == "enter" else "private",
                          str(intent == "enter").lower()))
    uv = root / ".local/bin/uv"
    entry = shlex.quote("from bts.cli import cli; cli()")
    execute(uv, f'test "$1 $2" = "run bts" || exit 95\nshift 2\nexec {shlex.quote(sys.executable)} -c {entry} "$@"')
    for name, body in (("crontab", 'test "$1" = "-l" || exit 92\nprintf "# synthetic empty cron\\n"'),
                       ("flock", 'test "$1" = "-n" || exit 93\nshift 2\nexec "$@"'),
                       ("timeout", 'test "$1 $2 $3 $4" = "--verbose -k 30 840" || exit 94\nshift 4\nexec "$@"')):
        execute(root / "bin" / name, body)
    env = dict(os.environ, HOME=str(root), PATH=f"{root / 'bin'}:{os.environ['PATH']}",
               HEALTHCHECKS_PING_URL="https://offline.invalid/ping",
               PYTHONPATH=str(REPO / "src"), PYTHONDONTWRITEBYTECODE="1")
    proc = subprocess.run(["bash", str(REPO / "scripts/cron-setup-hetzner.sh"), "show"],
                          env=env, cwd=home_repo, capture_output=True, text=True, timeout=10)
    result = {"intent": intent, "show_status": proc.returncode, "show": proc.stdout, "err": proc.stderr,
              "commands": []}
    assert proc.returncode == 0, proc.stderr
    runner = CliRunner()
    for line in proc.stdout.splitlines():
        if not line.endswith("# BTS-HETZNER"):
            continue
        when = " ".join(line.split()[:5])
        command_line = re.sub(r"\$\(date[^)]*\)", "2027" if "--seasons" in line else "2027-03-24", line)
        if "--seasons" in line:
            command_line = command_line.replace("--start 2027", "--start 2027-03-18").replace(
                "--end 2027", "--end 2027-03-24")
        bts_commands = re.findall(r"uv run bts (.*?)(?= >>| > | &&| ;| #)", command_line)
        for command in bts_commands:
            args = shlex.split(command)
            group, *tail = args
            cmd = cli.cli.get_command(None, group)
            if hasattr(cmd, "get_command") and tail:
                cmd = cmd.get_command(None, tail.pop(0))
            context = cmd.make_context(cmd.name, tail)
            context.close()
            help_result = runner.invoke(cli.cli, [*args, "--help"])
            result["commands"].append((when, args, help_result.exit_code, help_result.output.splitlines()[0]))
            assert help_result.exit_code == 0, help_result.output
        if "run python scripts/check_heartbeat.py" in line:
            paths = ["--heartbeat-path", "data/.heartbeat", "--ping-url", "https://offline.invalid"]
            help_result = subprocess.run([sys.executable, "scripts/check_heartbeat.py", *paths, "--help"],
                                         cwd=REPO, capture_output=True, text=True, timeout=10)
            assert help_result.returncode == 0, help_result.stderr
            result["commands"].append((when, ["scripts/check_heartbeat.py", *paths], 0,
                                       help_result.stdout.splitlines()[0]))
        if " curl " in line:
            result["commands"].append((when, ["curl", "-fsS", "--max-time", "5"], "external", None))
    return result


def main():
    out = Path(sys.argv[1]).resolve()
    rows = []
    for intent in ("enter", "research"):
        with TemporaryDirectory(prefix="wrblind-cron-") as temp:
            rows.append(cron_fixture(Path(temp), intent))
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
