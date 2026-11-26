"""Compare real entry diagnostics, day classifications, and local cron installer."""

import local_guard
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import os
import shlex
import subprocess
import sys
import tomllib

from click.testing import CliRunner
from pick_probe import HERE, ROOT, REV, SRC, cli
from bts import day_outcome, entry_intent, orchestrator
from tests.test_day_outcome import DATE, _bound_receipt, _publish_receipt, _write_decision


def make_exe(path, body):
    path.write_text("#!/bin/sh\nset -eu\n" + body)
    path.chmod(0o700)


def cron_case(root, action, flag):
    root.mkdir()
    home = root / "home"
    project = home / "projects" / "bts"
    project.mkdir(parents=True)
    (project / ".env").write_text("")
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    cron = root / "crontab"
    calls = root / "crontab.calls"
    initial_lines = "0 9 * * * echo foreign\n0 10 * * * echo previous # BTS-HETZNER\n"
    cron.write_text(initial_lines)
    make_exe(bin_dir / "crontab",
             f'printf "%s\\n" "$1" >> {shlex.quote(str(calls))}\n'
             f'case "$1" in -l) cat {shlex.quote(str(cron))} ;; '
             f'-) cat > {shlex.quote(str(cron))} ;; *) exit 98 ;; esac\n')
    make_exe(bin_dir / "uv",
             '[ "$1 $2 $3" = "run bts entry-intent" ] || exit 97\nshift 2\n'
             f'exec {shlex.quote(sys.executable)} -c "from bts.cli import cli; cli()" "$@"\n')
    cfg = home / ".bts-orchestrator.toml"
    cfg.write_text('[scheduler]\nentry_intent="enter"\npick_delivery="dm"\n' + flag)
    env = dict(HOME=str(home), PATH=f"{bin_dir}:/usr/bin:/bin", TZ="America/New_York",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join((str(HERE), str(SRC))),
               HEALTHCHECKS_PING_URL="https://healthcheck.invalid/review")
    assert home.is_relative_to(HERE) and home != Path.home()
    assert Path(env["PATH"].split(":")[0], "crontab").exists()
    script = (HERE / "base" if REV == "base" else ROOT) / "scripts" / "cron-setup-hetzner.sh"
    response = subprocess.run(["/bin/bash", str(script), action], env=env, capture_output=True, text=True, timeout=30)
    return dict(exit=response.returncode, stdout=response.stdout.replace(str(home), "<home>"),
                stderr=response.stderr.replace(str(home), "<home>"), crontab=cron.read_text().replace(str(home), "<home>"),
                unchanged=initial_lines.encode() == cron.read_bytes(),
                calls=calls.read_text().splitlines() if calls.exists() else [])


CONFIGS = {
    "dm-enter": 'scheduler={entry_intent="enter",pick_delivery="dm"}',
    "implicit-public-enter": 'scheduler={entry_intent="enter"}',
    "private-enter": 'scheduler={entry_intent="enter",pick_delivery="private"}',
    "private-research": 'scheduler={entry_intent="research",pick_delivery="off"}',
    "dm-research": 'scheduler={entry_intent="research",pick_delivery="dm"}',
    "invalid-delivery-enter": 'scheduler={entry_intent="enter",pick_delivery="typo"}',
    "invalid-intent-public": 'scheduler={entry_intent="ENTER",pick_delivery="public"}',
    "no-scheduler": '',
    "malformed-health": 'scheduler={entry_intent="enter",pick_delivery="dm"}\nhealth_checks=false',
    "malformed-scheduler": 'scheduler=4\nhealth_checks=[]',
}
DISK = ("none", "deliberate-skip", "private-commit", "delivered", "confirmed", "absent-entry")
SCHEDULES = ("game", "no-games", "void-games", "error")


def day_case(root, text, disk, schedule):
    cfg = root / "cfg.toml"
    cfg.write_text(text + "\n")
    picks = root / "picks"
    picks.mkdir(parents=True)
    if disk == "deliberate-skip":
        _write_decision(picks, action="skip", primary=None, delivery_status="skipped", scoreable=False)
    elif disk == "private-commit":
        _write_decision(picks, delivery_status="private_locked")
    elif disk in ("delivered", "confirmed", "absent-entry"):
        _write_decision(picks)
        if disk != "delivered":
            if disk == "confirmed":
                receipt = _bound_receipt(picks)
            else:
                receipt = _bound_receipt(picks, confirmed=False, reason="not_entered")
            _publish_receipt(picks, receipt)

    def fetch(date):
        assert date == DATE
        if schedule == "error":
            raise RuntimeError("schedule unavailable")
        if schedule == "no-games":
            return []
        return [{"status": {"detailedState": "Postponed" if schedule == "void-games" else "Final"}}]

    config = orchestrator.load_config(cfg)
    direct = day_outcome.gather(DATE, picks, config, expected_username="stonehengee", fetch_schedule=fetch)
    dms = []
    with patch("bts.scheduler.fetch_schedule", fetch), \
            patch("bts.dm.send_dm", side_effect=lambda *args: dms.append(args) or "capture"), \
            patch("bts.posting.post_to_bluesky", side_effect=AssertionError("diagnostics attempted feed post")):
        result = CliRunner().invoke(cli.cli, [
            "day-outcome", "--date", DATE, "--config", str(cfg), "--picks-dir", str(picks),
            "--expected-username", "stonehengee", "--dm-recipient", "capture.invalid",
        ])
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert json.loads(result.stdout) == direct.as_dict()
    return dict(direct=direct.as_dict(), stdout=result.stdout, stderr=result.stderr, dms=dms,
                exit=result.exit_code)


def example():
    current = (ROOT if REV == "head" else HERE / "base") / "config" / "orchestrator.example.toml"
    base = HERE / "base" / "config" / current.name
    payload = tomllib.loads(current.read_text())
    other = tomllib.loads(base.read_text())
    intent = payload["scheduler"].pop("entry_intent", None)
    req = payload["scheduler"].pop("contest_state_required")
    exp = payload["health_checks"].pop("contest_state_expected")
    other["scheduler"].pop("contest_state_required")
    other["health_checks"].pop("contest_state_expected")
    assert payload == other
    config = tomllib.loads(current.read_text())
    if REV == "head":
        assert req is True and exp is True and intent == "enter"
        assert entry_intent.live_gate_problem(config) is None
        assert entry_intent.check_entry_intent(config).ok
    return dict(required=req, expected=exp, intent=intent, other_values_identical=payload == other)


def main():
    with TemporaryDirectory(dir=HERE) as folder:
        root = Path(folder)
        cron = {}
        settings = {"neither": "", "scheduler": "contest_state_required=true\n",
                    "health": "[health_checks]\ncontest_state_expected=true\n"}
        for action, (setting, flag) in product(("show", "install", "remove"), settings.items()):
            key = f"{action}/{setting}"
            cron[key] = cron_case(root / key.replace("/", "-"), action, flag)
        days = {}
        for (label, config), state, schedule in product(CONFIGS.items(), DISK, SCHEDULES):
            key = f"{label}/{state}/{schedule}"
            target = root / "outcomes" / key
            target.mkdir(parents=True)
            days[key] = day_case(target, config, state, schedule)
        data = dict(cron=cron, day_outcome=days, example=example())
    assert not local_guard.network, local_guard.network
    (HERE / f"diagnostics-{REV}.json").write_text(json.dumps(data, indent=2) + "\n")
    print(REV, f"cron={len(cron)}", f"day-outcome={len(days)}", data["example"])
    for key, outcome in cron.items():
        print(key, outcome["exit"], outcome["unchanged"], outcome["calls"], outcome["stderr"].strip())


if __name__ == "__main__":
    main()
