"""Replay interruption, uncertain delivery, and the displayed nominal-slot rule."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from click.testing import CliRunner

from support import HERE, ROOT, Replay, legacy, load_line, outcome, source
from bts.cli import cli
from tests import test_entry_intent


def installer_interrupt(root):
    probe = legacy("installer_interrupt")
    stage = root / "fixture"
    original = probe.interrupt(stage, HERE / "base/scripts/cron-setup-hetzner.sh")
    stage.rename(root / "base-fixture")
    current = probe.interrupt(stage, test_entry_intent.SCRIPT)
    assert probe.normalized(current) == original
    print("Interrupted installation left the original crontab in both versions. Reinstall published the staged text.")


def text(root):
    help_result = CliRunner().invoke(cli, ["check-pick-entered", "--help"])
    help_text = " ".join(help_result.stdout.split())
    assert "Before 10:00 ET, defer if a nominal 10:00–23:45 ET quarter-hour is inside today’s entry window." in help_text
    world = Replay(root)
    world.commit_at("10:05:01", at="09:40")
    early = world.invoke("09:45")
    assert early["stdout"] == "check-pick-entered: early check deferred because a nominal 10:00–23:45 ET quarter-hour is inside this pick's entry window\n"
    delayed = world.invoke("10:00:02")
    assert delayed["stdout"] == "check-pick-entered: outside window (5 min to pitch)\n"
    assert not early["calls"] and not delayed["calls"]
    print(help_text[help_text.index("--cron"):])
    print(early["stdout"].strip())
    print(delayed["stdout"].strip())


if __name__ == "__main__":
    source("f5")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        installer_interrupt(root)
        # These imported probes open only temporary paths and captured transports.
        clock = legacy("clock_standing")
        clock.midnight(root / "midnight")
        clock.failures(root / "sends")
        text(root / "text")
