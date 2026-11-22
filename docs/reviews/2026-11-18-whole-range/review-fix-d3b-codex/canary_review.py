"""Compare the real canary CLI with the schedule gate and the real morning read."""
from contextlib import redirect_stderr
from datetime import timedelta
from io import StringIO
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import sys

from review_box import HERE, ROOT

sys.path.insert(0, str(ROOT))
from click.testing import CliRunner
from bts.cli import cli
from bts.orchestrator import load_config
from bts.scheduler import _morning_state_check
from tests.test_season_contract_cs4b2 import AUTOS, MANUALS, NOW, SWITCHES, install


def snapshot(path):
    return {file.relative_to(path): file.read_bytes() for file in path.rglob("*") if file.is_file()}


with TemporaryDirectory(prefix="canary-pairs-", dir=HERE) as path:
    tmp = Path(path)
    picks = tmp / "custom" / "picks"
    config = tmp / "test.toml"
    runner = CliRunner()
    launch = ["--config", str(config)]
    prefix = f'[orchestrator]\npicks_dir = "{picks}"\n[scheduler]\n'
    for delivery, intent, flags in product(("private", "dm", "bluesky", "garbage"), ("enter", "research"),
                                          ("", "contest_state_required = true\n",
                                           "[health_checks]\ncontest_state_expected = true\n")):
        config.write_text(prefix + f'pick_delivery = "{delivery}"\nentry_intent = "{intent}"\n' + flags)
        with patch("bts.scheduler.run_day") as run_day, patch("bts.cli._now_et", return_value=NOW):
            schedule = runner.invoke(cli, ["schedule", *launch])
            canary = runner.invoke(cli, ["deploy-canary", *launch])
        print("gate", delivery, intent, flags.strip(), schedule.exit_code, canary.exit_code)
        assert run_day.called == (schedule.exit_code == 0)
        if schedule.exit_code == 2:
            assert canary.exit_code == 1 and "refusing to start:" in schedule.output
            assert canary.output.split("REFUSED — ", 1)[-1] in schedule.output
        elif delivery == "private":
            assert canary.exit_code == 0 and "contest state not checked." in canary.output
        elif delivery != "garbage":
            assert canary.exit_code == 1 and "REFUSED" in canary.output
        else:
            assert canary.exit_code == 2 and "ValueError" in canary.output

    config.write_text(prefix + 'pick_delivery = "dm"\ncontest_state_required = true\n')
    cases = list(product(("absent", "revoked", "automatic", "other_season", "corrupt1", "corrupt6"),
                         ("absent", "today", "legacy"), ("absent", "covered", "refused", "other_id")))
    for switch, manual, auto in cases:
        install(picks, "season_contract.json", SWITCHES[switch])
        install(picks, "contest_streak.manual.json", MANUALS[manual])
        install(picks, "contest_streak.json", AUTOS[auto])
        before = snapshot(tmp)
        with patch("bts.cli._now_et", return_value=NOW):
            result = runner.invoke(cli, ["deploy-canary", *launch])
        assert snapshot(tmp) == before
        with redirect_stderr(StringIO()) as journal:
            _morning_state_check(load_config(config), picks, NOW.date().isoformat(), NOW, NOW + timedelta(hours=1))
        message = journal.getvalue()
        print("morning", switch, manual, auto, result.exit_code, result.output.strip())
        if result.exit_code == 0:
            assert "contest state accepted" in result.output
            assert "contest state accepted" in message
        else:
            assert result.exit_code == 1 and "no pick can be made yet" in message
            assert result.output.split("REFUSED — ", 1)[1].strip() in message

    install(picks, "season_contract.json", SWITCHES["absent"])
    install(picks, "contest_streak.manual.json", MANUALS["today"])
    install(picks, "contest_streak.json", None)
    for step in (timedelta(), timedelta(hours=12)):
        at = NOW + step
        with patch("bts.cli._now_et", return_value=at):
            result = runner.invoke(cli, ["deploy-canary", *launch])
        print("ET boundary", at.isoformat(), result.exit_code, result.output.strip())
        assert (result.exit_code == 0) == (at.date() == NOW.date())
