"""Execute the documented shell paths in a fake home and an isolated local Git remote."""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "docs/reviews/2026-11-18-whole-range/contract-check-e"),
]

ARCHIVED = ROOT / "docs/reviews/2026-11-18-whole-range/contract-check-e/probe_ops.py"
spec = importlib.util.spec_from_file_location("wr_e_recorded_ops", ARCHIVED)
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)
ops.REPO = ROOT


def shell(args, cwd=None, env=None):
    return subprocess.run(args, cwd=cwd, env=env, capture_output=True, timeout=5, text=True, check=True).stdout.strip()


def reviewed_ref(root):
    remote = root / "remote.git"
    local = root / "reviewed"
    env = dict(os.environ, GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_AUTHOR_NAME="Synthetic review", GIT_AUTHOR_EMAIL="review@offline.invalid",
               GIT_COMMITTER_NAME="Synthetic review", GIT_COMMITTER_EMAIL="review@offline.invalid",
               GIT_ALLOW_PROTOCOL="file")
    shell(["git", "init", "--bare", "-q", str(remote)], env=env)
    shell(["git", "init", "-q", "-b", "main", str(local)], env=env)
    commands = (
        ["commit", "--allow-empty", "-q", "-m", "local-main"],
        ["switch", "-qc", "reviewed"],
        ["commit", "--allow-empty", "-q", "-m", "reviewed"],
        ["remote", "add", "origin", str(remote)],
    )
    for args in commands:
        shell(["git", *args], cwd=local, env=env)
    reviewed = shell(["git", "rev-parse", "HEAD"], local, env)
    stale_main = shell(["git", "rev-parse", "main"], local, env)
    shell(["git", "push", "origin", "main:deploy"], local, env)
    deployed = shell(["git", "--git-dir", str(remote), "rev-parse", "deploy"], env=env)
    assert reviewed != deployed == stale_main
    print("B8c literal push", "reviewed HEAD", reviewed, "pushed main", deployed)


def restore_masks(root):
    units = root / "home/bts/.config/systemd/user"
    defaults = units / "default.target.wants"
    timers = units / "timers.target.wants"
    defaults.mkdir(parents=True)
    timers.mkdir()
    for unit, classification, path in (
        ("bts-scheduler.service", defaults, ROOT / "scripts/systemd/bts-scheduler.service"),
        ("bts-dashboard.service", defaults, ROOT / "scripts/systemd/bts-dashboard.service"),
        ("bts-leaderboard.timer", timers, ROOT / "deploy/systemd/bts-leaderboard.timer"),
        ("bts-leaderboard.service", None, ROOT / "deploy/systemd/bts-leaderboard.service"),
    ):
        destination = units / unit
        destination.symlink_to("/dev/null")
        if classification:
            (classification / unit).symlink_to(destination)
        if classification == timers:
            (classification / unit).unlink()
        if classification == defaults:
            # B8c step 3 restores the file but leaves default.target enablement.
            destination.unlink()
            destination.write_bytes(path.read_bytes())
            assert (classification / unit).resolve().is_file()
    assert "Persistent=true" in (ROOT / "deploy/systemd/bts-leaderboard.timer").read_text()
    assert (units / "bts-leaderboard.timer").readlink() == Path("/dev/null")
    assert not list(timers.iterdir())
    print("B8c restored masks", "leaderboard timer/service masked, timer enablement removed",
          "scheduler/dashboard still enabled via", [path.name for path in defaults.iterdir()])


def deploy_or_cron(root, action):
    bin_path = root / "bin"
    for name in ("sudo", "sleep", "id", "curl", "crontab", "systemctl", "tailscale", "pkill"):
        ops.wrapper(bin_path / name)
    env = dict(os.environ, PATH=f"{bin_path}:/usr/bin:/bin", WR_E_PROBE_ROOT=str(root),
               PYTHONPATH=str(ROOT), HOME=str(root / "shim-home"))
    if action == "deploy":
        ops.deploy(root, env)
    else:
        ops.cron(root, env)
        cron = (root / "crontab.txt").read_text()
        assert "40 7 * * *" in cron
        print("installed enter/research common 07:40 reconcile is outside the entry-intent branch")


def october_shortcut(root):
    old = HERE / "october"
    home = root / "home"
    checkout = home / "projects/bts"
    checkout.mkdir(parents=True)
    (checkout / ".env").write_text("HEALTHCHECKS_PING_URL=https://offline.invalid/ping\n")
    config = home / ".bts-orchestrator.toml"
    config.write_text(
        '[scheduler]\npick_delivery="dm"\nentry_intent="enter"\ncontest_state_required=true\n'
        '[health_checks]\ncontest_state_expected=true\n'
    )
    commands = root / "bin"
    ops.wrapper(commands / "crontab")
    ops.wrapper(home / ".local/bin/uv")
    env = dict(os.environ, HOME=str(home), PATH=f"{commands}:/usr/bin:/bin",
               WR_E_PROBE_ROOT=str(root), PYTHONPATH=str(old / "src"),
               HEALTHCHECKS_PING_URL="https://offline.invalid/ping")
    result = ops.invoke([str(home / ".local/bin/uv"), "run", "bts", "entry-intent", "--config", str(config)],
                        env, directory=checkout)
    assert result.returncode == 0 and result.stdout.strip() == "enter", result
    for action in ("show", "install"):
        result = ops.invoke(["/bin/bash", str(old / "scripts/cron-setup-hetzner.sh"), action],
                            env, directory=checkout)
        assert result.returncode == 0, result
        print("October under A5 shortcut", action, result.returncode)
    crontab = (root / "crontab.txt").read_text()
    assert "run bts check-pick-entered" in crontab and "run bts day-outcome" not in crontab
    print("old entry check installed while expected 08:30 day-outcome is absent")


if __name__ == "__main__":
    action = sys.argv[1]
    with tempfile.TemporaryDirectory(dir=HERE) as name:
        root = Path(name)
        if action == "refs":
            reviewed_ref(root)
            restore_masks(root)
        elif action == "october":
            october_shortcut(root)
        else:
            deploy_or_cron(root, action)
