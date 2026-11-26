import datetime as dt
import gzip
import json
import os
import signal
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import day_outcome, dm, picks, scheduler, util
from bts.cli import cli
from tests.test_day_outcome import DATE, _bound_receipt, _box, _publish_receipt, _write_decision
from tests.test_wr_f3_skip_state_review import evidence

ET = ZoneInfo("America/New_York")
MIB = 1024 * 1024
NOTICE = (
    " Also: The local model streak file (streak.json) needs review ({}). Grading or replay may be affected. "
    "Preserve the files in place during triage. Follow B8b item 2 in docs/ops/2027-season-start.md before "
    "the next affected job. Give the BTS lead this alert, the host and ET run time, and any grading or "
    "reconcile errors, receipt paths and health alerts. If the lead session is unavailable, escalate "
    "through Eric. The 08:00 ET correction cutoff still applies."
)
COMPACT = (
    ". Local model streak file needs review ({}). Preserve the files. Find the day-outcome JSON in "
    "cron.log for a scheduled run or on stdout for a manual run. Follow B8b item 2 in "
    "docs/ops/2027-season-start.md before the next affected job. "
    "Escalate through Eric if needed."
)
STATES = {
    "entered": (1, "enter", {}, "confirmed"),
    "entered_game_unverified": (1, "enter", {}, "batter"),
    "deliberate_skip": (1, "enter", dict(action="skip", scoreable=False), None),
    "no_games": (0, "enter", None, None),
    "research_commit": (1, "research", dict(delivery_status="private_locked"), None),
    "delivered_under_research": (1, "research", {}, None),
    "intent_unknown": (1, "bogus", None, None),
    "schedule_unverifiable": (None, "enter", None, None),
    "no_decision": (1, "enter", None, None),
    "skip_not_deliberate": (1, "enter", dict(action="skip", source="crash_guard", scoreable=False), None),
    "commit_not_scoreable": (1, "enter", dict(delivery_status="failed", scoreable=False), None),
    "contradiction": (0, "enter", {}, None),
    "entered_late": (1, "enter", {}, "late"),
    "not_entered": (1, "enter", {}, "absent"),
    "entry_unverifiable": (1, "enter", {}, None),
    "skip_state_review": (
        1, "enter", dict(action="skip", scoreable=False, streak=30, saver_available=False), None,
    ),
}


class Clock(dt.datetime):
    wall = dt.datetime(2027, 4, 3, 8, 30, tzinfo=ET)

    @classmethod
    def now(cls, tz=None):
        return cls.wall.astimezone(tz) if tz else cls.wall.replace(tzinfo=None)


def setup_day(tmp_path, state, *, primary=None):
    games, intent, decision, receipt = STATES[state]
    directory, config = _box(tmp_path, intent=intent, delivery="private" if intent == "research" else "dm")
    if decision is not None:
        _write_decision(directory, **(decision if primary is None else decision | {"primary": primary}))
    if receipt is not None:
        record = _bound_receipt(directory, before_cutoff=receipt != "late",
                                confirmed=receipt not in ("absent", "batter"),
                                reason="no_pick" if receipt == "absent" else "match")
        if receipt == "batter":
            record["qualification"]["slots"] = [{"state": "unit_unverified"}]
        _publish_receipt(directory, record)
    if state == "skip_state_review":
        account = directory / "account_state"
        account.mkdir()
        (account / "contest_streak.manual.json").write_text(json.dumps(evidence("dated")))
    return directory, config, games


def capture_transport(attempts, failure=None):
    def respond(req, **opts):
        route = req.full_url.rsplit("/", 1)[-1]
        if route == "chat.bsky.convo.sendMessage":
            body = json.loads(req.data)["message"]["text"]
            attempts.append(body)
            assert opts["idempotent"] is False
            # The fixtures use ASCII text with no CRLF, so the counts coincide.
            assert body.isascii() and "\r" not in body
            if len(body) > 1000:
                raise HTTPError(req.full_url, 400, "offline chat maxGraphemes=1000", {}, None)
            if failure is not None:
                raise failure
            data = {"id": "offline-confirmed"}
        elif route == "com.atproto.server.createSession":
            data = {"accessJwt": "offline-token"}
        elif route.startswith("chat.bsky.convo.getConvoForMembers?"):
            data = {"convo": {"id": "offline-convo"}}
        else:
            raise AssertionError(req.full_url)
        return BytesIO(json.dumps(data).encode())

    return respond


def run_report(directory, config, games, *, date=DATE, wall=Clock.wall, failure=None, recipient=True):
    attempts = []

    def schedule(requested):
        assert requested == date
        if games is None:
            raise OSError("offline schedule failure")
        return [{"gamePk": 900}] * games

    args = ["day-outcome", "--date", date, "--picks-dir", str(directory), "--config", str(config),
            "--expected-username", "stonehengee"]
    if recipient:
        args += ["--dm-recipient", "did:plc:offline"]
    with patch.object(Clock, "wall", wall), patch.object(dt, "datetime", Clock), \
         patch.object(scheduler, "fetch_schedule", schedule), patch.object(dm, "get_dm_password", lambda: "offline"), \
         patch.object(dm, "retry_urlopen", capture_transport(attempts, failure)):
        result = CliRunner().invoke(cli, args)
    return result, attempts


def snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def head(payload):
    title = "MODEL STREAK FILE NEEDS REVIEW" if payload["ok"] else payload["state"].replace("_", " ").upper()
    return f"BTS day check {payload['date']}: {title}"


def detail(payload):
    pick = f" Pick: {payload['pick']}." if payload["pick"] else ""
    return f". {payload['detail']}{pick}"


def assert_problem(run, before, reason, *, compact=False, stderr=""):
    result, attempts = run
    assert result.exit_code == 1, (result.output, result.exception)
    assert result.stderr == stderr
    assert json.loads(result.stdout) == before | {"ok": False, "streak_problem": reason}
    expected = head(before) + (COMPACT.format(reason) if compact else detail(before) + NOTICE.format(reason))
    assert attempts == [expected]
    print(f"{len(attempts[0])} code points | {attempts[0]}")


@pytest.mark.parametrize("state", STATES)
def test_k1_k2_every_day_state(tmp_path, state):
    directory, config, games = setup_day(tmp_path, state)
    args = directory, config, games
    path = directory / "streak.json"
    original, original_dm = run_report(*args)
    before = json.loads(original.stdout)
    assert before["state"] == state
    clean = state in day_outcome.OK_STATES
    assert before["ok"] is clean
    assert original.exit_code == int(not clean)
    assert original.stderr == ""
    if clean:
        assert original_dm == []
    else:
        assert original_dm == [head(before) + detail(before)]

    for content in (None, b"{}", b'{"season":2026,"streak":3,"applied_dates":null}',
                    b'{"season":"unknown","streak":0,"applied_dates":12}', b" " * (MIB - 2) + b"{}"):
        if content is not None:
            path.write_bytes(content)
        files = snapshot(tmp_path)
        result, messages = run_report(*args)
        expected = before | {"streak_problem": None}
        assert result.stdout == json.dumps(expected) + "\n"
        assert (result.exit_code, result.stderr, messages) == (original.exit_code, original.stderr, original_dm)
        assert snapshot(tmp_path) == files


@pytest.mark.parametrize("state", STATES)
def test_k3_m3_bad_file_preserves_day_state(tmp_path, state):
    directory, config, games = setup_day(tmp_path, state)
    args = directory, config, games
    before = json.loads(run_report(*args)[0].stdout)
    assert before["state"] == state
    (directory / "streak.json").write_bytes(b"[]")
    files = snapshot(tmp_path)
    assert_problem(run_report(*args), before, "not a JSON object")
    assert snapshot(tmp_path) == files


@pytest.mark.parametrize("state", ["entered_game_unverified", "entered"])
def test_m1_archived_model_list(tmp_path, monkeypatch, state):
    root = Path(__file__).resolve().parents[1]
    archive = root / "docs/reviews/2026-11-18-whole-range/closing-test/codex-blind"
    with gzip.open(archive / "evidence/interactions-refusal-model_list.json.gz") as source:
        events = json.load(source)
    archived = next(row for row in events if row[1] == "cli" and row[2][0] == "day-outcome")
    archived_payload = json.loads(archived[4])
    day = archived_payload["date"]
    monkeypatch.syspath_prepend(str(archive))
    import offline

    monkeypatch.setattr(offline, "ROOT", root)
    real_send = dm.send_dm
    with offline.World(day, tmp_path) as world:
        world.dated()
        world.pending = [{"roundId": 1, "playerId": 1}]
        if state == "entered":
            units = world.directory.parent / "leaderboard/static_snapshots/units"
            units.mkdir(parents=True)
            capture = {
                "captured_at": f"{day}T22:00:00Z",
                "units": [{"id": 700, "roundId": 1, "unitType": "ROUND", "feedId": 101,
                           "startDate": f"{day}T04:00:00Z", "endDate": f"{day}T23:59:59Z"}],
            }
            (units / "2027-03-24T220000Z.json.gz").write_bytes(gzip.compress(json.dumps(capture).encode()))
            world.pending[0]["unitId"] = 700
        path = world.directory / "streak.json"
        path.write_bytes(b"[]")
        world.actions = [(offline.at(day, "18:45"), world.watchdog)]
        result = world.run(via_cli=True)
        assert isinstance(result.exception, AttributeError), result.output
        world.now = offline.at("2027-03-25", "08:30")
        config = world.toml()
        args = ["day-outcome", "--date", day, "--config", str(config),
                "--picks-dir", str(world.directory), "--expected-username", "offline",
                "--dm-recipient", "did:plc:offline"]
        messages = []
        world.patch(dm, "send_dm", real_send)
        world.patch(dm, "get_dm_password", lambda: "offline")
        world.patch(dm, "retry_urlopen", capture_transport(messages))
        if state == "entered":
            archived_payload |= {"state": "entered",
                                 "detail": "The entry was confirmed in 1 of 1 checks before the cutoff."}
        assert_problem((world.invoke(*args), messages), archived_payload, "not a JSON object")
        assert path.read_bytes() == b"[]"


@pytest.mark.parametrize(("contents", "reason"), [
    pytest.param(b"", "not valid JSON", id="empty"),
    pytest.param(b"{", "not valid JSON", id="syntax"),
    pytest.param(b"\xff", "cannot be read (UnicodeDecodeError)", id="utf8"),
    pytest.param(b'"x"', "not a JSON object", id="string"),
    pytest.param(b"null", "not a JSON object", id="null"),
    *[pytest.param(b'{"streak":' + count + b"}", "streak is not a non-negative integer", id=count.decode())
      for count in (b'"3"', b"true", b"-1", b"3.0", b"null")],
    pytest.param(b" " * (MIB - 1) + b"{}", "exceeds 1 MiB", id="read-limit"),
])
def test_m2_bad_contents(tmp_path, contents, reason):
    directory, config, games = setup_day(tmp_path, "entered")
    args = directory, config, games
    before = json.loads(run_report(*args)[0].stdout)
    (directory / "streak.json").write_bytes(contents)
    files = snapshot(tmp_path)
    assert_problem(run_report(*args), before, reason)
    assert snapshot(tmp_path) == files


@pytest.mark.parametrize(("contents", "error"), [
    pytest.param(b'{"streak":' + b"9" * 5000 + b"}", "ValueError", id="digits"),
    pytest.param(b"[" * 12000 + b"0" + b"]" * 12000, "RecursionError", id="nesting"),
])
def test_k4_decoder_limits_preserve_no_decision(tmp_path, contents, error):
    directory, config, games = setup_day(tmp_path, "no_decision")
    args = directory, config, games
    before = json.loads(run_report(*args)[0].stdout)
    (directory / "streak.json").write_bytes(contents)
    assert_problem(run_report(*args), before, f"JSON exceeds decoder limits ({error})")


def test_k4_missing_config(tmp_path):
    directory, config, games = setup_day(tmp_path, "no_games")
    config.unlink()
    before, before_dm = run_report(directory, config, games)
    (directory / "streak.json").write_bytes(b"[]")
    result, messages = run_report(directory, config, games)
    assert (result.exit_code, result.stdout, result.stderr, messages) == (
        2, "", before.stderr, before_dm,
    )
    assert len(messages) == 1 and "THE CHECK COULD NOT RUN (FileNotFoundError)" in messages[0]


@contextmanager
def short_timeout():
    def timeout(*_):
        raise AssertionError("streak check blocked")

    handler = signal.signal(signal.SIGALRM, timeout)
    signal.alarm(3)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, handler)


@pytest.mark.parametrize("special", ["fifo", "directory"])
def test_k5_special_file_not_opened(tmp_path, special):
    directory, config, games = setup_day(tmp_path, "no_games")
    args = directory, config, games
    before = json.loads(run_report(*args)[0].stdout)
    path = directory / "streak.json"
    os.mkfifo(path) if special == "fifo" else path.mkdir()
    original_open = os.open
    original_stat = path.stat()

    def check_open(candidate, *args, **kwargs):
        assert candidate != path, "known special file opened"
        return original_open(candidate, *args, **kwargs)

    with short_timeout(), patch.object(util.os, "open", check_open):
        run = run_report(*args)
    assert_problem(run, before, "not a regular file")
    assert path.stat() == original_stat


def test_m2_unreadable(tmp_path):
    if os.geteuid() == 0:
        pytest.skip("mode-000 open failure requires an unprivileged process")
    directory, config, games = setup_day(tmp_path, "no_games")
    args = directory, config, games
    before = json.loads(run_report(*args)[0].stdout)
    path = directory / "streak.json"
    path.write_bytes(b"{}")
    path.chmod(0)
    try:
        assert_problem(run_report(*args), before, "cannot be read (PermissionError)")
        assert path.stat().st_mode & 0o777 == 0
    finally:
        path.chmod(0o600)
    assert path.read_bytes() == b"{}"


def test_k3_compact_dm(tmp_path):
    directory, config, games = setup_day(tmp_path, "not_entered", primary={"batter_name": "A" * 700})
    args = directory, config, games
    original, messages = run_report(*args)
    before = json.loads(original.stdout)
    assert len(messages) == 1 and len(messages[0]) == 833
    assert len(messages[0] + NOTICE.format("not a JSON object")) > 1000
    (directory / "streak.json").write_bytes(b"[]")
    assert_problem(run_report(*args), before, "not a JSON object", compact=True)


def test_standing_restart_and_unconfirmed_send(tmp_path):
    directory, config, games = setup_day(tmp_path, "no_decision")
    args = directory, config, games
    before = json.loads(run_report(*args)[0].stdout)
    (directory / "streak.json").write_bytes(b"{")
    files = snapshot(tmp_path)
    assert_problem(run_report(*args, failure=TimeoutError()), before, "not valid JSON",
                   stderr="day-outcome: DM failed: TimeoutError\n")
    assert snapshot(tmp_path) == files
    assert_problem(run_report(*args), before, "not valid JSON")
    assert snapshot(tmp_path) == files


@pytest.mark.parametrize(("date", "wall"), [
    ("2027-12-31", dt.datetime(2028, 1, 1, 8, 30, tzinfo=ET)),
    ("2028-03-23", dt.datetime(2028, 3, 24, 8, 30, tzinfo=ET)),
])
def test_standing_prior_season_and_interrupted_scorer(tmp_path, date, wall):
    directory, config, games = setup_day(tmp_path, "no_games")
    args = directory, config, games
    path = directory / "streak.json"
    content = b'{"season":2027,"streak":3}'
    path.write_bytes(content)
    with patch.object(util.os, "replace", side_effect=InterruptedError), pytest.raises(InterruptedError):
        picks.update_streak([True], directory, date=DATE)
    files = snapshot(tmp_path)
    result, messages = run_report(*args, date=date, wall=wall)
    before = json.loads(result.stdout)
    assert (result.exit_code, messages, before["streak_problem"]) == (0, [], None)
    assert path.read_bytes() == content
    assert snapshot(tmp_path) == files
    path.write_bytes(b'{"season":2027,"streak":"3"}')
    assert picks.load_streak(directory, season=2028) == 0
    assert_problem(run_report(*args, date=date, wall=wall), before, "streak is not a non-negative integer")


def test_k1_dangling_symlink(tmp_path):
    directory, config, games = setup_day(tmp_path, "no_games")
    args = directory, config, games
    before, before_dm = run_report(*args)
    path = directory / "streak.json"
    path.symlink_to("absent.json")
    result, messages = run_report(*args)
    assert (result.exit_code, result.stderr, messages) == (0, before.stderr, before_dm)
    assert json.loads(result.stdout) == json.loads(before.stdout) | {"streak_problem": None}
    assert path.readlink() == Path("absent.json")


@pytest.mark.parametrize("stage", ["open", "fstat", "read"])
def test_missing_target_vs_open_descriptor_failure(tmp_path, monkeypatch, stage):
    directory, config, games = setup_day(tmp_path, "no_games")
    args = directory, config, games
    original, original_dm = run_report(*args)
    path = directory / "streak.json"
    path.write_bytes(b"{}")
    real_open = os.open
    real_operation = getattr(os, stage)
    streak_fd = None
    injected = []

    def open_file(candidate, *args, **kwargs):
        nonlocal streak_fd
        if candidate != path:
            return real_open(candidate, *args, **kwargs)
        if stage == "open":
            injected.append(stage)
            raise FileNotFoundError(2, "injected missing target", candidate)
        streak_fd = real_open(candidate, *args, **kwargs)
        return streak_fd

    def descriptor_operation(fd, *args, **kwargs):
        if fd == streak_fd:
            injected.append(stage)
            raise FileNotFoundError(2, "injected descriptor failure")
        return real_operation(fd, *args, **kwargs)

    monkeypatch.setattr(util.os, "open", open_file)
    if stage != "open":
        monkeypatch.setattr(util.os, stage, descriptor_operation)
    result, messages = run_report(*args)
    assert injected == [stage]
    if stage == "open":
        assert (result.exit_code, result.stdout, result.stderr, messages) == (
            0, original.stdout, original.stderr, original_dm,
        )
    else:
        assert_problem((result, messages), json.loads(original.stdout), "cannot be read (FileNotFoundError)")


def test_streak_read_once_after_classification_and_skip_review(tmp_path, monkeypatch):
    directory, config, games = setup_day(tmp_path, "deliberate_skip")
    args = directory, config, games
    before = json.loads(run_report(*args)[0].stdout)
    path = directory / "streak.json"
    path.write_bytes(b"{}")
    events = []
    real_classify = day_outcome.classify_day
    real_review = day_outcome._skip_review
    real_read = day_outcome.read_regular_file_text

    def classify(*args, **kwargs):
        outcome = real_classify(*args, **kwargs)
        events.append("classify")
        return outcome

    def review_skip(*args, **kwargs):
        outcome = real_review(*args, **kwargs)
        events.append("skip review")
        path.write_bytes(b"{")
        return outcome

    def read_streak(candidate, **kwargs):
        assert candidate == path
        events.append("read")
        return real_read(candidate, **kwargs)

    monkeypatch.setattr(day_outcome, "classify_day", classify)
    monkeypatch.setattr(day_outcome, "_skip_review", review_skip)
    monkeypatch.setattr(day_outcome, "read_regular_file_text", read_streak)
    run = run_report(*args)
    assert events == ["classify", "skip review", "read"]
    assert_problem(run, before, "not valid JSON")


@pytest.mark.parametrize("expected", [
    pytest.param(
        "For a morning report with a model streak file advisory, give `bts-lead` the report, its host and ET "
        "run time, and any grading or reconcile errors, receipt paths, and health alerts. If the lead session "
        "is unavailable, give the same information to Eric. Verify which jobs completed, including when "
        "reconcile reported success.",
        id="morning-advisory",
    ),
    pytest.param(
        "Complete reconcile replay can rebuild a replaceable regular file from verified final history after "
        "the blockers are resolved. Preserve a non-regular target during triage and have the BTS lead or "
        "Eric prepare a safe target before running reconcile. The grader and reconcile can block on a FIFO. "
        "The 08:00 ET correction cutoff still applies.",
        id="safe-recovery",
    ),
])
def test_b8b_item_2_checklist(expected):
    checklist = Path(__file__).resolve().parents[1] / "docs/ops/2027-season-start.md"
    row = next(line for line in checklist.read_text().splitlines() if line.startswith("| B8b |"))
    item = row.split(" (2) ", 1)[1].split(" (3) ", 1)[0]
    assert expected in item
    if expected.startswith("For a morning report"):
        assert item.split("** ", 1)[1].startswith(expected + " ")
