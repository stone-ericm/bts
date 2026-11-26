"""Generate and compare both real crontabs without installing them on the host."""
import json
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from tests import test_entry_intent as fixtures

BASE = HERE / "base/scripts/cron-setup-hetzner.sh"
WATCHDOG = fixtures.ENTRY_CMD
ENTER = fixtures.LIVE_DM + "contest_state_required = true\n"
FOREIGN = "12 8 * * * echo  foreign  \n"
OLD = "*/15 10-23 * * * old bts check-pick-entered # BTS-HETZNER\n"


def run(script, config, root):
    with patch.object(fixtures, "SCRIPT", script):
        box = fixtures.box.__wrapped__(root)
        box.config.write_text(config)
        box.cron.write_text(FOREIGN + OLD + "# " + OLD)
        res = box.run("show")
        assert res.returncode == 0, res.stderr
        shown = res.stdout.partition("Would install:\n")[2].encode()
        res = box.run("install")
        assert res.returncode == 0, res.stderr
        installed = box.cron.read_bytes()
        assert installed == FOREIGN.encode() + shown, (installed, shown)
        res = box.run("remove")
        assert res.returncode == 0 and box.cron.read_bytes() == FOREIGN.encode()
    return shown


if __name__ == "__main__":
    generated = {}
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for intent, config in (("enter", ENTER), ("research", fixtures.RESEARCH)):
            for version, script in (("base", BASE), ("f5", fixtures.SCRIPT)):
                raw = run(script, config, root / "fixture")
                (root / "fixture").rename(root / f"{version}-{intent}")
                generated[f"{version}_{intent}"] = raw.decode().splitlines()
                lines = [line for line in raw.splitlines(keepends=True) if WATCHDOG.encode() not in line]
                if version == "base":
                    other_base = lines
                else:
                    assert other_base == lines
                print(version, intent, len(lines), "other lines with original bytes")
    base, = [line for line in generated["base_enter"] if WATCHDOG in line]
    candidate, = [line for line in generated["f5_enter"] if WATCHDOG in line]
    assert candidate.replace("*/15 * * * * ", "*/15 10-23 * * * ", 1).replace(" --cron", "", 1) == base
    words = shlex.split(candidate, comments=True)
    assert words[words.index("/usr/bin/flock"):words.index("/usr/bin/flock") + 8] == [
        "/usr/bin/flock", "-n", "/home/bts/.check-pick-entered.lock", "/usr/bin/timeout",
        "--verbose", "-k", "30", "840",
    ]
    assert "--window-min" not in words
    assert not any(WATCHDOG in line for line in generated["f5_research"])
    (HERE / "generated.json").write_text(json.dumps(generated, indent=2) + "\n")
    print(candidate)
