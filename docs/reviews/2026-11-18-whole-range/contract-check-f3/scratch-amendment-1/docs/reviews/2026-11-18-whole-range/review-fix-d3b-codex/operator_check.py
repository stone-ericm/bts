"""Check the procedure order and exercise the specified saver commands off-box."""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

from click.testing import CliRunner
from review_box import HERE, ROOT

sys.path.insert(0, str(ROOT))
from bts.cli import cli

incident = (ROOT / "INCIDENT.md").read_text()
restore = incident.split("**Box/filesystem loss", 1)[1].split("## 5.", 1)[0]
operations = ("restic restore <SNAPSHOT_ID> --target /", "bts data sync-from-r2", "Keep the scheduler stopped")
offsets = [restore.index(operation) for operation in operations]
print("§4 restore position order", list(zip(operations, offsets)))
assert offsets == sorted(offsets)

checklist = (ROOT / "docs/ops/2027-season-start.md").read_text()
b8c = next(line for line in checklist.splitlines() if line.startswith("| B8c "))
assert "before the October scheduler or bts crontab can execute" in b8c
assert "keep both services stopped" in b8c and "boot link disabled" not in b8c
print("B8c requires maintenance hold and leaves the transition to normal boot implicit")

with TemporaryDirectory(prefix="operator-saver-", dir=HERE) as name:
    picks = Path(name) / "scheduler" / "picks"
    alternate = Path(name) / "alternate"
    runner = CliRunner()
    season = "2027"
    suffix = ["--season", season, "--picks-dir", str(picks)]
    for command, expected in (
        (["--show", *suffix], "uninitialized"),
        (["--init", "used", *suffix], "used"),
        (["--show", *suffix], "used"),
        (["--show", "--season", season, "--picks-dir", str(alternate)], "uninitialized"),
    ):
        result = runner.invoke(cli, ["saver-state", *command])
        print(result.exit_code, result.output.strip())
        assert result.exit_code == 0 and expected in result.output
