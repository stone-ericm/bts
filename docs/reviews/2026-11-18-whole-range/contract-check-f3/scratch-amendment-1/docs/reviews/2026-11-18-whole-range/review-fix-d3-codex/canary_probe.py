"""Use the real startup and morning read on the same synthetic state as the canary."""

from contextlib import ExitStack
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import socket
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

from click.testing import CliRunner
import requests

import bts.cli as cli_module
import bts.orchestrator as orchestrator
import bts.scheduler as scheduler
from tests.test_season_contract_cs4b2 import AUTOS, ET, MANUALS, NOW, SWITCHES, dated, install


def forbidden(*args, **kwargs):
    raise AssertionError("side-effect entry point used")


def snapshot(root):
    return [
        (str(path.relative_to(root)), path.read_bytes() if path.is_file() else None)
        for path in sorted(root.rglob("*"))
    ]


def write_config(config, mode, flags, *, intent=None):
    config.write_text(
        '[orchestrator]\npicks_dir = "configured-picks"\n[scheduler]\n'
        + f'pick_delivery = "{mode}"\n'
        + (f'entry_intent = "{intent}"\n' if intent is not None else "")
        + ("contest_state_required = true\n" if flags & 1 else "")
        + ("[health_checks]\ncontest_state_expected = true\n" if flags & 2 else "")
    )


def compare(label, config, root, now, expected):
    started = []

    def during_morning(date, config, dry_run):
        mode = scheduler._pick_delivery_mode(config)
        started.append((date, mode))
        if mode in ("dm", "public") and scheduler._contest_state_required(config):
            scheduler._morning_state_check(config, Path(config["orchestrator"]["picks_dir"]),
                                           date, now, now + timedelta(hours=2))

    with ExitStack() as stack:
        for module, name in [(socket, "create_connection"), (socket.socket, "connect"),
                             (requests.sessions.Session, "request"), (scheduler, "fetch_schedule"),
                             (scheduler, "run_and_pick"), (orchestrator, "ssh_predict"),
                             (subprocess, "Popen")]:
            stack.enter_context(patch.object(module, name, forbidden))
        stack.enter_context(patch.object(cli_module, "_now_et", return_value=now))
        stack.enter_context(patch.object(scheduler, "run_day", during_morning))
        files = snapshot(root)
        canary = CliRunner().invoke(cli_module.cli, ["deploy-canary", "--config", str(config)])
        assert files == snapshot(root)
        morning = CliRunner().invoke(cli_module.cli, ["schedule", "--config", str(config)])
        assert files == snapshot(root)
        if expected == "accepted":
            assert canary.exit_code == 0 and "live contest state accepted (" in canary.output, canary.output
            assert morning.exit_code == 0 and started
            assert "contest state accepted" in morning.output or "stale contest reading" in morning.output, morning.output
        elif expected == "refused":
            assert canary.exit_code == 1 and "Deploy canary: REFUSED" in canary.output, canary.output
            if started:
                assert morning.exit_code == 0 and "no pick can be made yet:" in morning.output, morning.output
                assert canary.output.split(" — ", 1)[1].rstrip() in morning.output
            else:
                assert morning.exit_code == 2 and "refusing to start:" in morning.output, morning.output
        elif expected == "skip":
            assert canary.exit_code == 0 and "contest state not checked" in canary.output, canary.output
            assert morning.exit_code == 0 and started and not morning.output
        else:
            assert canary.exit_code == 2 and not started, canary.output
        print(json.dumps(dict(case=label, expected=expected, startup=started, canary=canary.output,
                              scheduler=morning.output), ensure_ascii=False), flush=True)


with TemporaryDirectory(prefix="wrd3-canary-") as temporary:
    root = Path(temporary)
    prev = Path.cwd()
    os.chdir(root)
    try:
        picks = Path("configured-picks")
        config = Path("config.toml")
        for mode in ("private", "dm", "public"):
            for flags in range(4):
                write_config(config, mode, flags, intent="research" if mode == "private" else "enter")
                compare(f"{mode}-flags-{flags}", config, root, NOW,
                        "skip" if mode == "private" else "refused")
        write_config(config, "private", 0, intent="enter")
        compare("private-enter-no-flags", config, root, NOW, "refused")
        write_config(config, "private", 3, intent="enter")
        compare("private-enter-both-gate-behavior", config, root, NOW, "skip")
        config.write_text("scheduler = false\n")
        compare("bad-scheduler-table", config, root, NOW, "refused")
        write_config(config, "not-a-mode", 0)
        compare("bad-delivery-mode", config, root, NOW, "error")

        for mode in ("dm", "public"):
            write_config(config, mode, 1 if mode == "dm" else 2, intent="enter")
            cases = (
                [(f"corrupt{i}", "absent", "covered", "refused") for i in range(1, 9)]
                + [("automatic", "legacy", "covered", "refused")]
                + [(switch, "absent", "covered", "refused") for switch in ("absent", "revoked", "other_season")]
                + [("automatic", "absent", auto, "refused") for auto in AUTOS if auto not in {"covered", "timestamp_only"}]
                + [("automatic", "absent", auto, "accepted") for auto in ("covered", "stale", "timestamp_only")]
                + [("corrupt1", "today", "absent", "accepted"), ("absent", "today", "absent", "accepted")]
            )
            for switch, manual, auto, expected in cases:
                automatic = (AUTOS["covered"] | {"recorded_at": "2027-07-14T16:00:00Z"}
                             if auto == "stale" else AUTOS[auto])
                install(picks, "season_contract.json", SWITCHES[switch])
                install(picks, "contest_streak.manual.json", MANUALS[manual])
                install(picks, "contest_streak.json", automatic)
                compare(f"{mode}-{switch}-{manual}-{auto}", config, root, NOW, expected)

        for day in ("2027-07-15", "2027-12-31"):
            at = datetime.fromisoformat(f"{day}T23:59:59").replace(tzinfo=ET)
            install(picks, "contest_streak.manual.json", dated(day, recorded_at=at.isoformat()))
            compare(f"{day}-before-midnight", config, root, at, "accepted")
            compare(f"{day}-after-midnight", config, root, at + timedelta(seconds=2), "refused")

        install(picks, "contest_streak.manual.json", dated())
        write_config(config, "private", 3, intent="research")
        compare("A5-before-edit-boot-link-still-enabled", config, root, NOW, "skip")
        write_config(config, "dm", 3, intent="enter")
        compare("A5-after-edit-before-B8a-boot-link-still-enabled", config, root, NOW, "accepted")

        config.unlink()
        for name in ("fifo", "directory", "missing", "oversized", "syntax", "encoding"):
            if name == "fifo":
                os.mkfifo(config)
            elif name == "directory":
                config.mkdir()
            elif name in {"oversized", "syntax", "encoding"}:
                config.write_bytes({"oversized": b" " * 65537, "syntax": b"[", "encoding": b"\xff"}[name])
            result = CliRunner().invoke(cli_module.cli, ["deploy-canary", "--config", str(config)])
            assert result.exit_code == 2 and result.output.startswith("Deploy canary: ERROR — "), result.output
            print(json.dumps(dict(case=name, code=result.exit_code, output=result.output), ensure_ascii=False))
            if name == "directory":
                config.rmdir()
            else:
                config.unlink(missing_ok=True)
    finally:
        os.chdir(prev)
