from contextlib import contextmanager
from datetime import date, datetime
import json
import os
from pathlib import Path
import signal
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRATCH = HERE.parent / "scratch-f5b"
ARCHIVE = HERE.parent / "f5-records/docs/reviews/2026-11-18-whole-range"
sys.path[:0] = [str(ARCHIVE / name) for name in (
    "review-fix-f5-codex-r2", "review-fix-f5-codex", "review-fix-f5-carbon")]

import bts
import alert_probe
import probe_support
import support
from marker_health import ReadBlocked, deadline
from tests.test_entry_intent import box
from tests.test_wr_f1_wrong_game import DAY, RIGHT, WRONG, row

ET = probe_support.ET
Replay = probe_support.Replay


def installed(root, version, intent="enter", original=None):
    from tests import test_entry_intent

    root.mkdir()
    installer = box.__wrapped__(root)
    installer.config.write_text(f'[scheduler]\nentry_intent = "{intent}"\n'
                                f'pick_delivery = "{"dm" if intent == "enter" else "private"}"\n'
                                + ('contest_state_required = true\n' if intent == "enter" else ''))
    if original is not None:
        installer.cron.write_text(original)
    scratch = SCRATCH / ("base" if version == "base" else "candidate")
    script = scratch / "scripts/cron-setup-hetzner.sh"
    with patch.object(test_entry_intent, "SCRIPT", script):
        before = installer.lines()
        shown = installer.run("show")
        assert (shown.returncode, installer.lines()) == (0, before), shown
        added = installer.run("install")
        assert added.returncode == 0, added
        lines = installer.lines()
        assert [line for line in lines if line.endswith("# BTS-HETZNER")] == (
            shown.stdout.split("Would install:\n")[1].splitlines())
        removed = installer.run("remove")
        assert removed.returncode == 0 and not any(
            "# BTS-HETZNER" in line for line in installer.lines() or []), removed
    return lines


def generate(root):
    original = (
        "2 9 * * * echo foreign\n"
        "0 10 * * * echo old # BTS-HETZNER\n"
        "# */15 10-23 * * * cd x && uv run bts check-pick-entered # BTS-HETZNER\n")
    data = {f"{v}_{i}": installed(root / f"{v}-{i}", v, i, original)
            for v in ("base", "f5") for i in ("enter", "research")}
    normalize = lambda lines, version, intent: [
        line.replace(f"/{version}-{intent}/", "/box/") for line in lines
    ]
    assert normalize(data["base_research"], "base", "research") == normalize(
        data["f5_research"], "f5", "research")
    assert all(x[0] == "2 9 * * * echo foreign" for x in data.values())
    before, after = (line_for(data, v) for v in ("base", "f5"))
    assert normalize([ln for ln in data["base_enter"] if ln != before], "base", "enter") == normalize(
        [ln for ln in data["f5_enter"] if ln != after], "f5", "enter")
    assert before.startswith("*/15 10-23 * * * ") and after.startswith("*/15 * * * * ")
    assert after.split(" && /usr/bin/flock ", 1)[1].replace(
        "check-pick-entered --cron", "check-pick-entered").replace(
            "/f5-enter/", "/base-enter/") == before.split(" && /usr/bin/flock ", 1)[1]
    (HERE / "generated.json").write_text(json.dumps(data, indent=2) + "\n")
    print(before)
    print(after)


def line_for(data, version):
    line, = [line for line in data[f"{version}_enter"] if " bts check-pick-entered " in line]
    return line


def load_line(version):
    return line_for(json.loads((HERE / "generated.json").read_text()), version)


def install_archive_adapters():
    probe_support.load_line = support.load_line = alert_probe.load_line = load_line
    alert_probe.ROOT = ROOT


def source(version):
    expected = "base" if version == "base" else "candidate"
    assert Path(bts.__file__).resolve().is_relative_to(SCRATCH / expected / "src"), bts.__file__
    print("bts:", bts.__file__)


class World(Replay):
    @property
    def early(self):
        return self.marker.with_name("pick_entry_check.early.json")

    def early_state(self):
        return json.loads(self.early.read_text()) if self.early.is_file() else None

    def cron(self, at, **kw):
        if at not in {d.strftime("%H:%M") for d in probe_support.clock_slots(
            self._line, self.day
        )}:
            return None
        return self.invoke(at, **kw)

    def __init__(self, root, version="f5", day=DAY):
        super().__init__(root, version, day)
        self._line = load_line(version)

    def edit_pitch(self, daily, pitch, leg="pick"):
        from bts.picks import save_pick

        instant = datetime.fromisoformat(f"{self.day}T{pitch}").replace(tzinfo=ET)
        getattr(daily, leg).game_time = instant.isoformat()
        save_pick(daily, self.picks)


def compact(run):
    if run is None:
        return "no installed slot"
    rec = run["receipt"]
    return (run["exit"], run["exception"], rec["outcome"], rec.get("early_cron_run"),
            run["calls"], run["sent"], rec["marker_status"])


def health(world, day=None, clock="23:59"):
    from bts.health import pick_entry

    day = day or world.day
    now = datetime.fromisoformat(f"{day}T{clock}").replace(tzinfo=ET)
    with deadline():
        return [alert.level for alert in pick_entry.check(world.picks, date.fromisoformat(day), now)]


install_archive_adapters()
