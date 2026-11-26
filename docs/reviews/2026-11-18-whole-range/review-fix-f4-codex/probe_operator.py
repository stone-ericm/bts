"""Replay the archived grading failure and follow the checklist through the cutoff."""

import gzip
import json
import os
import signal
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
from bts import dm, picks
from bts.health import alert, runner, streak_validation
from report_harness import Chat, ROOT, snapshot

ARCHIVE = ROOT / "docs/reviews/2026-11-18-whole-range/closing-test/codex-blind"
sys.path.insert(0, str(ARCHIVE))
import offline
from midseason import REAL_GRADE, network_fixture
from offline import World, at
from opening import DAY2, OPEN

offline.ROOT = ROOT
REAL_SEND = dm.send_dm
GOOD_NOTICE = "Grading or replay may be affected. Preserve the files in place during triage."
RECIPIENT = "did:plc:offline"


def read_report(world, config, date, chat):
    return world.invoke("day-outcome", "--date", date, "--config", str(config),
                        "--expected-username", "offline", "--dm-recipient", RECIPIENT)


def pin_chat(world, chat):
    world.patch(dm, "send_dm", REAL_SEND)
    world.patch(dm, "get_dm_password", lambda: "offline")
    world.patch(dm, "retry_urlopen", chat.response)
    world.patch(alert, "send_dm", REAL_SEND)


def archived_entry(bind_game, candidate):
    with TemporaryDirectory() as tmp, World(OPEN, Path(tmp)) as world:
        path = world.directory / "streak.json"
        path.write_bytes(b"[]")
        world.pending = [{"roundId": 1, "playerId": 1}]
        if bind_game:
            units = world.directory.parent / "leaderboard/static_snapshots/units"
            units.mkdir(parents=True)
            frame = {"captured_at": f"{OPEN}T22:00:00Z", "units": [
                {"id": 700, "roundId": 1, "unitType": "ROUND", "feedId": 101,
                 "startDate": f"{OPEN}T04:00:00Z", "endDate": f"{OPEN}T23:59:59Z"}]}
            (units / "2027-03-24T220000Z.json.gz").write_bytes(gzip.compress(json.dumps(frame).encode()))
            world.pending[0]["unitId"] = 700
        world.patch(picks, "resolve_daily_slot_results", REAL_GRADE)
        world.patch(picks, "_fetch_json", lambda url, **kw: network_fixture(world, url, **kw))
        world.dated()
        world.actions = [(at(OPEN, "18:45"), world.watchdog)]
        config = world.toml()
        schedule = world.invoke("schedule", "--date", OPEN, "--config", str(config))
        assert isinstance(schedule.exception, AttributeError), (schedule.output, schedule.exception)
        world.now = at(DAY2, "01:00")
        grade = world.invoke("check-results", "--date", OPEN, "--wait-deadline-et", "06:00")
        assert isinstance(grade.exception, AttributeError) and picks.load_pick(OPEN, world.directory).result is None
        for hm in ("02:00", "07:40"):
            world.now = at(DAY2, hm)
            replay = world.invoke("reconcile")
            assert isinstance(replay.exception, AttributeError) and "streak retained, not recalculated" in replay.stderr

        world.now = at(DAY2, "07:45")
        chat = Chat()
        pin_chat(world, chat)
        critical = runner._safe_run("streak_validation", lambda: streak_validation.check(world.directory))
        assert len(critical) == 1 and critical[0].level == "CRITICAL"
        assert alert.dispatch_dm_for_critical(critical, RECIPIENT)
        assert len(chat.accepted) == 1 and "streak_validation" in chat.accepted[0]
        world.now = at(DAY2, "08:30")
        before = snapshot(world.root)
        outcome = read_report(world, config, OPEN, chat)
        payload = json.loads(outcome.stdout)
        state = "entered" if bind_game else "entered_game_unverified"
        assert payload["state"] == state and snapshot(world.root) == before
        assert outcome.exit_code == int(candidate) and len(chat.accepted) == 1 + int(candidate)
        if candidate:
            assert payload["streak_problem"] == "not a JSON object"
            body = chat.accepted[1]
            for fragment in ("MODEL STREAK FILE NEEDS REVIEW", GOOD_NOTICE, "B8b item 2", "the host and ET run time",
                             "grading or reconcile errors, receipt paths and health alerts", "escalate through Eric",
                             "08:00 ET correction cutoff still applies"):
                assert fragment in body, body
        print(f"BL-O3 {state}: scheduler/01:00 AttributeError, both replays warn and fail, independent CRITICAL "
              f"kept, 08:30 exit={outcome.exit_code} ordinary report DM={len(chat.accepted) - 1}, state untouched")


def late_replay():
    with TemporaryDirectory() as tmp, World(OPEN, Path(tmp)) as world:
        primary = picks.pick_from_row(world.frame().iloc[0])
        older = picks.DailyPick("2027-03-23", world.now.isoformat(), primary, None, None)
        hit = picks.DailyPick(OPEN, world.now.isoformat(), primary, None, None,
                             result="hit", slot_results={"pick": "hit"}, notification_sent=True)
        for pick in (older, hit):
            picks.save_pick(pick, world.directory)
        path = world.directory / "streak.json"
        path.write_bytes(b"[]")
        world.now = at(DAY2, "07:40")
        world.results[101] = "miss"
        result = world.invoke("reconcile")
        assert isinstance(result.exception, AttributeError) and "streak retained" in result.stderr
        assert picks.load_pick(OPEN, world.directory).result == "miss" and path.read_bytes() == b"[]"

        # Verified final history is supplied for the older blocker; MLB offers a new
        # change for yesterday, but the real post-cutoff run cannot apply it.
        older.result = "miss"
        picks.save_pick(older, world.directory)
        world.results[101] = "hit"
        world.now = at(DAY2, "08:20")
        world.invoke("reconcile", expect=0)
        assert path.read_bytes() != b"[]" and picks.load_streak(world.directory, season=2027) == 0
        assert picks.load_pick(OPEN, world.directory).result == "miss"
        print("07:40 saved a correction before display failure; verified 08:20 full replay rebuilt the regular file "
              "and left the 08:00-frozen result unchanged")


def standalone_after_success():
    with TemporaryDirectory() as tmp, World(DAY2, Path(tmp)) as world:
        world.games.pop(DAY2)
        world.now = at(DAY2, "07:40")
        world.invoke("reconcile", expect=0)
        path = world.directory / "streak.json"
        assert json.loads(path.read_text())["streak"] == 0
        path.write_bytes(b'{"streak":false}')
        config = world.toml()
        chat = Chat()
        pin_chat(world, chat)
        world.now = at(DAY2, "08:30")
        result = read_report(world, config, DAY2, chat)
        assert result.exit_code == 1 and json.loads(result.stdout)["state"] == "no_games"
        assert len(chat.accepted) == 1 and "streak is not a non-negative integer" in chat.accepted[0]
        print("Standalone advisory after successful reconcile: no replay error/receipt of failure to hand off; "
              "the report itself is the triage trigger")


def fifo_replay():
    class BoundedRead(BaseException):
        pass

    def deadline(*_):
        raise BoundedRead()

    with TemporaryDirectory() as tmp, World(DAY2, Path(tmp)) as world:
        path = world.directory / "streak.json"
        os.mkfifo(path)
        world.now = at(DAY2, "08:20")
        saw_read = []
        original = Path.read_text

        def watch(candidate, *args, **kwargs):
            if candidate.resolve() == path.resolve():
                saw_read.append(candidate)
                signal.setitimer(signal.ITIMER_REAL, 0.12)
            return original(candidate, *args, **kwargs)

        previous = signal.signal(signal.SIGALRM, deadline)
        try:
            with patch.object(Path, "read_text", watch):
                try:
                    world.invoke("reconcile")
                except BoundedRead:
                    assert len(saw_read) == 1
                else:
                    raise AssertionError("reconcile unexpectedly finished on a FIFO without a writer")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
        assert path.is_fifo()
        spare = path.with_name("spare")
        spare.write_bytes(b"{}")
        os.replace(spare, path)
        assert path.read_bytes() == b"{}"
        print("Complete replay on a replaceable FIFO blocks in the existing scorer read; bounded after 120 ms. "
              "An atomic rename can replace that same FIFO.")


candidate = sys.argv[1] == "candidate"
print(f"{sys.argv[1]} bts.__file__ = {bts.__file__}")
for bound in (False, True):
    archived_entry(bound, candidate)
if candidate:
    late_replay()
    standalone_after_success()
fifo_replay()
