"""Executable model of the draft's hold and user unit. No systemd calls."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent


class Box:
    def __init__(self, root):
        root.mkdir()
        self.hold = root / "hold"
        self.boot = "boot-1"
        self.now = 0
        self.scheduler = "old"
        self.target = "old"
        self.timer = None

    def read(self):
        return json.loads(self.hold.read_text()) if self.hold.exists() else None

    def write(self, state, until=None):
        tmp = self.hold.with_suffix(".tmp")
        tmp.write_text(json.dumps({"id": "owner", "state": state, "boot": self.boot, "until": until}))
        os.replace(tmp, self.hold)

    def condition(self):
        record = self.read()
        return record is None or (record["state"] == "canary" and record["boot"] == self.boot
                                  and self.now < record["until"])

    def start(self):
        if self.condition():
            self.scheduler = self.target

    def reboot(self):
        self.boot = "boot-2"
        self.scheduler = None
        self.timer = None
        self.start()

    def tick(self, seconds):
        self.now += seconds
        if self.timer is not None and self.now >= self.timer:
            record = self.read()
            if record is not None and record["id"] == "owner" and record["state"] == "canary":
                self.write("failed")
                self.scheduler = None

    def candidate(self):
        self.write("pending")
        self.scheduler = None
        self.target = "candidate"
        self.write("canary", self.now + 300)

    def may_release(self):
        record = self.read()
        return record is not None and record["state"] == "canary" and record["id"] == "owner"


with TemporaryDirectory(prefix="r3-lease-", dir=HERE) as tmp:
    root = Path(tmp)
    arm_gap = Box(root / "arm_gap")
    arm_gap.write("pending")
    assert not arm_gap.condition() and arm_gap.scheduler == "old"

    timer_gap = Box(root / "timer_gap")
    timer_gap.candidate()
    timer_gap.start()  # an explicit start in the gap before timer registration
    timer_gap.tick(301)
    assert not timer_gap.condition() and timer_gap.scheduler == "candidate"
    assert timer_gap.may_release()  # expiry is not a release predicate in draft G2
    timer_gap.reboot()
    assert timer_gap.scheduler is None and not timer_gap.condition()

    race = Box(root / "release_race")
    race.candidate()
    race.timer = 300
    race.start()
    release_check = race.may_release()
    race.tick(300)
    assert race.read()["state"] == "failed" and race.scheduler is None
    if release_check:
        race.hold.unlink()
    assert race.condition() and not race.hold.exists()
    race.start()
    assert race.scheduler == "candidate"

    healthy = Box(root / "success")
    healthy.candidate()
    healthy.timer = 300
    healthy.start()
    assert healthy.may_release()
    healthy.hold.unlink()
    healthy.tick(301)
    assert healthy.scheduler == "candidate" and not healthy.hold.exists()

print("arm/stop gap leaves old scheduler running")
print("timer registration gap permits a candidate start which survives expiry")
print("same-boot guard denies the post-reboot start")
print("draft release predicate accepts an expired canary record")
print("split check/unlink can clear a failed record after the timer stops the candidate")
print("timer no-ops after a normal release")
