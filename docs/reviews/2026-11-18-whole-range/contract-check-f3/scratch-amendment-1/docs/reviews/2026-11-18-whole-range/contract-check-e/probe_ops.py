"""Run the real deploy shell with shims, or real installers in a throwaway fake home."""
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHIM = Path(__file__).with_name("ops_shim.py").resolve()


def invoke(command, env, directory=REPO, seconds=3):
    return subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=seconds)


def wrapper(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/bash\nexport WR_E_SHIM_NAME=\"${{0##*/}}\"\nexec {shlex.quote(sys.executable)} {shlex.quote(str(SHIM))} \"$@\"\n")
    path.chmod(0o700)


def deploy(root, env):
    workflow = (REPO / ".github/workflows/deploy.yml").read_text()
    script = re.search(r"(?m)^          script: \|\n((?:^ {12}.*\n|^\n)+)", workflow)[1]
    script = "\n".join(line[12:] for line in script.splitlines())
    script = script.replace("cd /home/bts/projects/bts", 'cd "$WR_E_PROBE_ROOT"', 1)
    script = script.replace("[ -f /etc/systemd/system/bts-scheduler.service ]", '[ -f "$WR_E_PROBE_ROOT/legacy" ]', 1)
    for scenario in ("pass_204", "pass_clean_exit", "sync", "restart", "stamp", "fail/ok", "fail/unhealthy",
                     "fail/reset", "fail/sync", "fail/restart"):
        (root / "sha").write_text("619abda7aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        (root / "services").write_text("stopped")
        result = invoke(["/bin/bash", "-c", script], dict(env, WR_E_PROBE_SCENARIO=scenario, TARGET_SHA="TARGET"))
        markers = [line for line in result.stdout.splitlines()
                   if any(s in line for s in ("CANARY", "Rolled back", "unhealthy", "completed cleanly"))]
        print(scenario, result.returncode, (root / "sha").read_text(), (root / "services").read_text(), markers)


def cron(root, env):
    fake_home = root / "home"
    fake_home.mkdir()
    checkout = fake_home / "projects/bts"
    checkout.mkdir(parents=True)
    (checkout / ".env").write_text("HEALTHCHECKS_PING_URL=https://offline.invalid/ping\n")
    wrapper(fake_home / ".local/bin/uv")
    conf = fake_home / ".bts-orchestrator.toml"
    for mode, intent, flags in [("private", "research", True), ("dm", "enter", True), ("private", "enter", True),
                                ("dm", "research", True), ("dm", "enter", False)]:
        conf.write_text(f'[scheduler]\npick_delivery="{mode}"\nentry_intent="{intent}"\n'
                        + ('contest_state_required=true\n[health_checks]\ncontest_state_expected=true\n' if flags else ''))
        for action in ("show", "install"):
            result = invoke(["/bin/bash", str(REPO / "scripts/cron-setup-hetzner.sh"), action],
                            dict(env, HOME=str(fake_home), HEALTHCHECKS_PING_URL="https://offline.invalid/ping"))
            dump = result.stdout if action == "show" else ((root / "crontab.txt").read_text() if result.returncode == 0 else "")
            print(mode, intent, flags, action, result.returncode, "entry lines", dump.count("run bts check-pick-entered"),
                  "fetch/DM lines", sum("--dm-recipient" in line for line in dump.splitlines() if "run bts fetch-contest-streak" in line),
                  "errors", result.stderr.strip())
    missing = invoke(["/bin/bash", str(REPO / "scripts/cron-setup-hetzner.sh"), "install"], dict(env, HOME=str(fake_home)))
    print("no shell environment", missing.returncode, missing.stderr.strip())
    for action in ("status", "install"):
        r = invoke(["/bin/bash", str(REPO / "scripts/install-leaderboard-systemd.sh"), action], dict(env, HOME=str(fake_home)))
        print("leaderboard", action, r.returncode, r.stdout.strip())
    print("leaderboard shim calls", [line for line in (root / "commands").read_text().splitlines() if line.startswith("systemctl ")])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as name:
        root = Path(name)
        bin_path = root / "bin"
        for name in ("sudo", "sleep", "id", "curl", "crontab", "systemctl"):
            wrapper(bin_path / name)
        env = dict(os.environ, PATH=f"{bin_path}:/usr/bin:/bin", WR_E_PROBE_ROOT=str(root), PYTHONPATH=str(REPO))
        if sys.argv[1] == "deploy":
            deploy(root, env)
        else:
            cron(root, env)
