"""Replay the closing fixture and exercise the real watchdog and morning CLI offline."""
from collections import Counter
from contextlib import ExitStack
from datetime import date, datetime, timedelta
import gzip
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from bts import entry_receipt
from bts.cli import cli
from bts.health import pick_entry
from click.testing import CliRunner
from schedule import HERE, format_min, inside, job, slots
from tests.test_wr_f1_wrong_game import RIGHT, WRONG, row

REPO = HERE.parents[1]
FIXTURE = REPO / "docs/reviews/2026-11-18-whole-range/closing-test/codex-blind/evidence/cron-early-entry.json.gz"
with gzip.open(FIXTURE, "rt") as stream:
    evidence = json.load(stream)
DAY = "2027-03-24"
PREVIOUS = "2027-03-23"
ET = ZoneInfo("America/New_York")
SNAPSHOT = next(e[-1] for e in evidence if e[1] == "snapshot")
SAVED_PICK = SNAPSHOT[f"data/picks/{DAY}.json"]
SAVED_DECISION = SNAPSHOT[f"data/picks/{DAY}/decision.json"]

def cron_args(label):
    line = shlex.split(job(label), comments=True)
    return line[line.index("check-pick-entered"):line.index(">>")]

def unexpected(*args, **kwargs):
    raise AssertionError("unexpected socket activity")

class World:
    def __init__(self, name, label="f5"):
        self.data = HERE / "watchdog-worlds" / name / "data"
        self.picks = self.data / "picks"
        self.picks.mkdir(parents=True)
        self.marker = self.data / "health_state" / "pick_entry_check.json"
        self.args = cron_args(label)
        self.args[self.args.index("--picks-dir") + 1] = str(self.picks)
        self.username = self.args[self.args.index("--expected-username") + 1]
        self.recipient = self.args[self.args.index("--dm-recipient") + 1]
        self.config = self.data / "orchestrator.toml"
        self.config.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
        self.runner = CliRunner()

    def write_pick(self, pitch="06:10", day=DAY, *, committed=True, deliver="05:10", pitch_day=None):
        dt = datetime.fromisoformat(f"{pitch_day or day}T{pitch}:00").replace(tzinfo=ET)
        commit_at = datetime.fromisoformat(f"{day}T{deliver}:00").replace(tzinfo=ET).isoformat()
        pick = {**SAVED_PICK, "date": day, "run_time": commit_at, "delivered_at": commit_at,
                "pick": {**SAVED_PICK["pick"], "game_time": dt.isoformat()}}
        if not committed:
            pick.update(notification_sent=False, notification_id=None, delivered_at=None,
                        delivery_attempted=False)
        (self.picks / f"{day}.json").write_text(json.dumps(pick))
        if committed:
            decision = self.picks / day / "decision.json"
            decision.parent.mkdir(exist_ok=True)
            decision.write_text(json.dumps({**SAVED_DECISION, "date": day, "finalized_at": commit_at}))

    def write_marker(self, status, day=DAY):
        self.marker.parent.mkdir(exist_ok=True)
        self.marker.write_text(json.dumps({"date": day, "status": status, "reason": "no_pick",
                                          "escalations": ["initial"]}))

    def read_marker(self):
        return self.marker.read_bytes() if self.marker.is_file() else None

    def units(self, unit, time="030000"):
        target = entry_receipt.units_snapshot_dir(self.picks)
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{DAY}T{time}Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": [unit]}).encode()))

    def count(self, day=DAY):
        return len(entry_receipt.discover(self.picks, day))

    def health(self):
        end = datetime.fromisoformat(f"{DAY}T23:59:00").replace(tzinfo=ET)
        return [(alert.level, alert.source) for alert in pick_entry.check(self.picks, date.fromisoformat(DAY), end)]

    def check(self, at, day=DAY, pending=(row(),), monotonic_elapsed=0):
        calls, dms = [], []

        def captured(name, result):
            def capture(*args, **kwargs):
                calls.append(name)
                return result
            return capture

        anchor = []
        def stable_mono():
            if anchor:
                return 1000 + monotonic_elapsed
            anchor.append(True)
            return 1000

        leaves = {
            "bts.leaderboard.auth.load_session_cookies": captured("cookie-file", {}),
            "bts.leaderboard.auth.extract_uid": lambda _: "test",
            "bts.leaderboard.auth.fetch_login_session": captured("session", SimpleNamespace(
                user_id="test", username=self.username, xsid="test")),
            "bts.contest_fetch.fetch_profile": captured("profile", {"predictions": []}),
            "bts.contest_fetch.fetch_pending_predictions": captured("pending", list(pending)),
            "bts.cli._fetch_rounds": captured("rounds-public", {1: date.fromisoformat(day)}),
            "bts.cli._fetch_bts_to_mlb": captured("players-public", {1: SAVED_PICK["pick"]["batter_id"]}),
            "bts.dm.send_dm": lambda recipient, msg: dms.append((recipient, msg)),
            "time.monotonic": stable_mono,
        }
        previous = self.read_marker()
        with ExitStack() as stack:
            for target, adapter in leaves.items():
                stack.enter_context(patch(target, adapter))
            result = self.runner.invoke(cli, [*self.args, "--now-et", f"{day}T{at}:00"])
        assert result.exit_code in (0, 1), (result.output, result.exception)
        receipt = max((self.data / "health_state" / "pick_entry_receipts" / day).glob("*.json"),
                      key=lambda p: p.stat().st_mtime_ns)
        receipt = json.loads(receipt.read_text())
        assert receipt["started_at"].startswith(f"{day}T{at}:00")
        return {"at": at, "receipt": receipt["outcome"], "calls": calls,
                "dm": dms, "marker_changed": previous != self.read_marker(),
                "exit": result.exit_code, "console": result.stdout.strip()}

    def morning(self):
        dms = []
        with patch("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": 101}]), \
             patch("bts.dm.send_dm", lambda recipient, msg: dms.append((recipient, msg))):
            result = self.runner.invoke(cli, ["day-outcome", "--date", DAY, "--config", str(self.config),
                                             "--picks-dir", str(self.picks), "--expected-username", self.username,
                                             "--dm-recipient", self.recipient])
        assert result.exit_code in (0, 1), (result.output, result.exception)
        return json.loads(result.stdout), dms

def summary(trace):
    return {"receipts": dict(Counter(run["receipt"] for run in trace)),
            "calls": dict(Counter(call for run in trace for call in run["calls"])),
            "dms": sum(len(run["dm"]) for run in trace),
            "marker_changes": sum(run["marker_changed"] for run in trace)}

def run_probe():
    result = {}
    for pitch in ("06:10", "10:05", "13:05", "19:10", "22:10"):
        minute = int(pitch[:2]) * 60 + int(pitch[3:])
        for label in ("base", "f5"):
            world = World(label + "-" + pitch.replace(":", ""), label)
            world.units(RIGHT)
            trace = []
            for at in slots(label):
                if at >= minute - 60 and not (world.picks / f"{DAY}.json").exists():
                    world.write_pick(pitch, deliver=format_min(minute - 60))
                trace.append(world.check(format_min(at)))
            counts = summary(trace)
            assert world.count() == len(slots(label)) == sum(counts["receipts"].values())
            morning = world.morning()
            result[f"{label}-{pitch}"] = trace
            print(label, pitch, "in window:", list(map(format_min, inside(minute, slots(label)))),
                  "authenticated:", [r["at"] for r in trace if r["calls"]], counts, "morning:", morning)
    for state in ("empty", "preview", "yesterday-pick", "committed", "alerted", "confirmed",
                  "yesterday-marker", "missing-entry", "after-cutoff"):
        world = World(state)
        world.units(RIGHT)
        if state == "yesterday-pick":
            world.write_pick(day=PREVIOUS)
        if state in ("alerted", "confirmed", "yesterday-marker"):
            world.write_marker("confirmed" if state != "alerted" else "alerted",
                               PREVIOUS if state == "yesterday-marker" else DAY)
        trace = []
        for at in slots("f5"):
            if at == 180 and state == "preview":
                world.write_pick("06:10", committed=False, deliver="03:00")
            if at == 315 and state in ("committed", "alerted", "confirmed", "yesterday-marker",
                                       "missing-entry", "after-cutoff"):
                world.write_pick("06:10")
            pending = () if state in ("alerted", "missing-entry", "after-cutoff") else (row(),)
            elapsed = 3600 if state == "after-cutoff" else 0
            trace.append(world.check(format_min(at), pending=pending, monotonic_elapsed=elapsed))
        assert len(trace) == world.count() == 96
        for run in trace:
            if run["receipt"] in {"no_pick_file", "not_committed", "outside_window", "already_confirmed"}:
                assert not any((run["calls"], run["dm"], run["marker_changed"])), run
        result[state] = trace
        print(state, summary(trace), "morning:", world.morning(), "health:", world.health())
    world = World("midnight")
    world.write_pick("00:35", pitch_day="2027-03-25")
    world.units(RIGHT)
    current = world.check("00:15", day="2027-03-25")
    assert current["receipt"] == "no_pick_file" and world.count("2027-03-25") == 1
    print("midnight:", current)
    for label in ("base", "f5"):
        for case in ("late-binding", "edit-entry", "early-warning", "same-minute-delivery"):
            world = World(f"{label}-{case}", label)
            if case != "late-binding":
                world.units(RIGHT)
            trace = []
            for at in [m for m in slots(label) if 585 <= m <= 645]:
                if at == 585 and case == "same-minute-delivery":
                    trace.append(world.check(format_min(at), pending=()))
                if not (world.picks / f"{DAY}.json").exists():
                    world.write_pick("10:50" if case != "early-warning" else "10:05")
                if at == 585 and case == "same-minute-delivery":
                    continue
                if at == 600 and case == "late-binding":
                    world.units(WRONG, "140000")
                pending = () if case == "early-warning" or case == "edit-entry" and at >= 600 else (row(),)
                trace.append(world.check(format_min(at), pending=pending))
            result[f"{label}-{case}"] = trace
            print(label, case, summary(trace), [(r["at"], r["receipt"], r["dm"]) for r in trace],
                  "morning:", world.morning())
    (HERE / "watchdog-matrix.json").write_text(json.dumps(result, indent=2) + "\n")

with ExitStack() as guard:
    for address in ("socket.socket.connect", "socket.socket.connect_ex", "socket.getaddrinfo", "socket.socket.sendto"):
        guard.enter_context(patch(address, unexpected))
    run_probe()
