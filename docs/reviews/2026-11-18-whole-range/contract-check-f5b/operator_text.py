"""Inventory the changed texts and capture real CLI help and stdout."""
from pathlib import Path
from tempfile import TemporaryDirectory

from click.testing import CliRunner
from bts.cli import cli
from replay_support import SCRATCH, World, source


source("f5")
root = SCRATCH / "candidate"
for name in ("scripts/cron-setup-hetzner.sh", "ARCHITECTURE.md", "docs/ops/pick-entry-receipt-v1.md"):
    path = root / name
    for index, line in enumerate(path.read_text().splitlines(), 1):
        if any(s in line for s in ("nominal", "cadence", "early.json", "sibling `.json.sealed`",
                                   "10:00–23:45", "early_cron_run")) and (
            name != "ARCHITECTURE.md" or line.startswith("- **`bts check-pick-entered`")
        ):
            print(f"{name}:{index}: {line}")

help_text = CliRunner().invoke(cli, ["check-pick-entered", "--help"]).stdout
assert "in-window check of a committed pick" in " ".join(help_text.split())
print(help_text)
with TemporaryDirectory() as temp:
    w = World(Path(temp))
    w.commit_at("10:05:01", at="09:40")
    for stamp in ("09:45", "10:00:02"):
        r = w.invoke(stamp)
        print(stamp, r["stdout"].strip())
    assert "nominal 10:00–23:45" in w.invoke("09:45")["stdout"]
