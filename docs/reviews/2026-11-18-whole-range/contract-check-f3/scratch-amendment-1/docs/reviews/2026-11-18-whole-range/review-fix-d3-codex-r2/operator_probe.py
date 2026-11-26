"""Exercise the old and revised activation order with the real startup gate."""

from datetime import timedelta
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from click.testing import CliRunner

import bts.cli as cli_module
import bts.scheduler as scheduler
from bts.saver_state import load_saver_state
from tests.test_season_contract_cs4b2 import NOW, dated, install

from harness import REPO, source_at


CHECKLIST = "docs/ops/2027-season-start.md"


def row(text, key):
    return next(line for line in text.splitlines() if line.startswith(f"| {key} |"))


def gate(config, boot_link):
    called = []

    def morning(date, config, dry_run):
        picks = Path(config["orchestrator"]["picks_dir"])
        mode = scheduler._pick_delivery_mode(config)
        called.append(mode)
        if mode != "private":
            scheduler._morning_state_check(config, picks, date, NOW, NOW + timedelta(hours=1))

    if boot_link == "disabled":
        return dict(startup=[], boot_link=boot_link)
    with patch.object(scheduler, "run_day", morning):
        result = CliRunner().invoke(cli_module.cli, ["schedule", "--date", NOW.date().isoformat(), "--config", str(config)])
    assert result.exit_code == 0, result.output
    return dict(startup=called, boot_link=boot_link, emitted=result.output)


def config_at(path, delivery, intent):
    path.write_text('[orchestrator]\npicks_dir = "configured-picks"\n[scheduler]\n'
                    f'pick_delivery = "{delivery}"\nentry_intent = "{intent}"\ncontest_state_required = true\n'
                    '[health_checks]\ncontest_state_expected = true\n')


old = source_at("d3d05abd", CHECKLIST)
new = (REPO / CHECKLIST).read_text()
old_a5, new_a5, new_b8a = row(old, "A5"), row(new, "A5"), row(new, "B8a")
assert old_a5.index('pick_delivery = "dm"') < old_a5.index("Follow B8a's activation order")
assert "Follow B8a's activation order" in new_a5 and 'pick_delivery = "dm"' not in new_a5
assert new_b8a.index("confirm the scheduler is stopped and its local boot link disabled") < new_b8a.index('pick_delivery = "dm"')
assert new_b8a.index("confirm the scheduler is stopped and its local boot link disabled") < new_b8a.index('cron-setup-hetzner.sh install')

original = Path.cwd()
with TemporaryDirectory(prefix="wrd3-r2-operator-") as temporary:
    os.chdir(temporary)
    try:
        config = Path("scheduler.toml")
        picks = Path("configured-picks")
        install(picks, "contest_streak.manual.json", dated())
        config_at(config, "dm", "enter")
        old_order = gate(config, "enabled")
        assert old_order["startup"] == ["dm"] and "contest state accepted" in old_order["emitted"]
        config_at(config, "private", "research")
        new_a5_result = gate(config, "enabled")
        assert new_a5_result["startup"] == ["private"]
        config_at(config, "dm", "enter")
        new_b8a_result = gate(config, "disabled")
        assert new_b8a_result["startup"] == []
        print(json.dumps(dict(case="A5-order", old_after_edit=old_order, new_a5=new_a5_result, new_b8a_after_edit=new_b8a_result)))

        year = str(NOW.year)
        runner = CliRunner()
        default = runner.invoke(cli_module.cli, ["saver-state", "--season", year])
        changed = runner.invoke(cli_module.cli, ["saver-state", "--init", "used", "--season", year])
        selected = runner.invoke(cli_module.cli, ["saver-state", "--season", year, "--picks-dir", str(picks)])
        assert default.exit_code == changed.exit_code == selected.exit_code == 0
        assert "uninitialized" in default.output and "Initialized saver flag: used" in changed.output
        assert "uninitialized" in selected.output
        assert load_saver_state(Path("data/picks"), season=NOW.year).state == "used"
        print(json.dumps(dict(case="B8a-omitted-saver-directory", before=default.output, default=changed.output,
                              configured=selected.output)))
    finally:
        os.chdir(original)
