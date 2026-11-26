import json
from contextlib import nullcontext
from dataclasses import replace
from datetime import date, timedelta
from unittest.mock import Mock

import pytest

from click import BadParameter
from click.testing import CliRunner

from bts import cli, contest_fetch, contest_state, daily_decision, orchestrator, picks, scheduler, strategy
from scripts.audit.season_walk import walk


DAY = date(2027, 9, 26)
MISSING = object()
INVALID_TYPES = (True, "12", 12.0, None, MISSING, [], {})


def with_field(body, field, value):
    body = dict(body)
    if value is MISSING:
        del body[field]
    else:
        body[field] = value
    return body


@pytest.fixture
def live(tmp_path, monkeypatch):
    directory = tmp_path / "picks"
    directory.mkdir()
    frame = walk._predictions(DAY, .96).head(2)
    games = list(frame["game_pk"])
    statuses = {game: {"abstract": "P", "detailed": "Scheduled"} for game in games}
    monkeypatch.setattr(orchestrator, "run_cascade", lambda *args: (frame, "synthetic"))
    monkeypatch.setattr(picks, "get_game_statuses_detailed", lambda *args: statuses)
    monkeypatch.setattr(scheduler, "heartbeat_watchdog", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(scheduler, "count_new_confirmations", lambda *args: 6)
    monkeypatch.setattr(scheduler, "_trigger_live_forward_capture_on_lock", Mock())
    monkeypatch.setattr(scheduler, "fetch_schedule", lambda *args: [{"status": {"detailedState": "Scheduled"}}])
    send = Mock(return_value="local-id")
    monkeypatch.setattr("bts.dm.send_dm", send)
    cfg = walk._config(directory)
    cfg["orchestrator"]["heartbeat_path"] = str(tmp_path / ".heartbeat")
    cfg["bluesky"] = {"dm_recipient": "synthetic"}
    with walk.offline(walk.REAL_DATETIME(2027, 9, 26, 12, tzinfo=walk.ET)):
        yield directory, cfg, games, send


def install(directory, kind, active, best):
    assert walk.install_state(directory, DAY, walk.State(kind, active, best, "used", .96)) == []


def read(directory):
    return contest_state.load_decision_streak_state(directory, require_contest_state=True)


def fetch(directory, active, best):
    with walk._fetch_stubs(active, best, DAY - timedelta(days=1)):
        return walk._invoke("fetch-contest-streak", "--picks-dir", directory,
                            "--expected-username", walk.ACCOUNT, "--dm-recipient", "synthetic")


def select(cfg, games):
    return scheduler.run_single_check(str(DAY), games, set(), cfg, .01)["selection"]


def deliver(directory, cfg, saved, selection):
    state = scheduler.SchedulerState(str(DAY), None, [], [], [], False, None, None, None)
    return scheduler._deliver_and_lock_pick(saved, cfg, directory, state, str(DAY),
                                            "lineup lock", selection=selection)


def test_m1_dated_cli_selects_delivers_and_commits(live):
    directory, cfg, games, send = live
    install(directory, "dated", 58, 58)
    assert (read(directory).streak, read(directory).best_streak) == (58, 58)
    selection = select(cfg, games)
    assert selection.streak == 58
    assert selection.decision.objective == "longest_streak"
    saved = picks.load_pick(str(DAY), directory)
    assert saved.chosen_on["kind"] == "dated"
    send.reset_mock()
    assert deliver(directory, cfg, saved, selection)
    send.assert_called_once()
    record = daily_decision.load_decision(str(DAY), directory)
    assert record["objective"] == "longest_streak" and record["scoreable"]


def test_m2_switched_fetch_trusts_high_best(live):
    directory, _, _, _ = live
    install(directory, "automatic", 57, 57)
    assert fetch(directory, 58, 58)[0] == 0
    current = read(directory)
    assert (current.streak, current.best_streak, current.best_status) == (58, 58, "trusted")


def test_m3_dated_high_best(live):
    directory, cfg, games, _ = live
    install(directory, "dated", 60, 75)
    assert (read(directory).streak, read(directory).best_streak) == (60, 75)
    assert select(cfg, games).decision.objective == "longest_streak"


def test_m3_automatic_high_best_matches_target_actions(live):
    directory, _, _, _ = live
    install(directory, "automatic", 12, 60)
    current = read(directory)
    assert (current.streak, current.best_streak, current.best_status) == (12, 60, "trusted")
    mdp = strategy._load_mdp()
    assert mdp is not None
    for day in (date(2027, 7, 15), DAY):
        for probability in (0., .96, 1., *mdp["boundaries"]):
            for saver in (False, True):
                actions = []
                for best in (57, current.best_streak):
                    ctx = strategy.DecisionContext(probability, .95, True, str(day), True,
                                                   mdp, best, current.best_status)
                    decision = strategy.resolve_policy_decision(ctx, current.streak, saver)
                    actions.append((decision.action, decision.policy_action))
                assert actions[0] == actions[1]


@pytest.mark.parametrize("kind", ["dated", "automatic"])
def test_m4_required_pick_survives_reload(live, monkeypatch, kind):
    directory, cfg, games, send = live
    install(directory, kind, 58, 58)
    select(cfg, games)
    saved = picks.load_pick(str(DAY), directory)
    assert saved.chosen_on["required"] and saved.chosen_on["kind"] == kind
    current = read(directory)
    assert picks.saved_pick_refusal(saved, current) is None
    preview = replace(saved, chosen_on=dict(saved.chosen_on, required=False))
    assert picks.saved_pick_refusal(preview, current)
    if kind == "dated":
        corrected = replace(saved, chosen_on=dict(
            saved.chosen_on, dated=dict(saved.chosen_on["dated"], best_streak=60)))
        assert picks.saved_pick_refusal(corrected, current)
    monkeypatch.setattr(orchestrator, "run_cascade",
                        lambda *args: (walk._predictions(DAY, .96).iloc[:0], "synthetic"))
    refresh = scheduler._refresh_pick_at_fallback_decision(cfg, str(DAY), saved, .01)
    assert refresh.daily is saved and not scheduler._refresh_is_genuine_skip(refresh)
    send.reset_mock()
    assert deliver(directory, cfg, saved, refresh.selection)
    send.assert_called_once()
    assert daily_decision.load_decision(str(DAY), directory)["scoreable"]


def test_m5_walk_high_dated_state():
    high = walk.State("dated", 58, 60, "used", .96)
    rows = walk.walk(DAY, DAY + timedelta(days=1), states=(high,))
    assert [row["action"] for row in rows] == ["double", "double"]
    assert walk.check(rows) == []


def test_k10_fresh_fetch_replaces_57_and_over_cap_leaves_it_intact(live):
    directory, cfg, games, send = live
    install(directory, "automatic", 57, 57)
    obs = directory / "account_state/contest_streak.json"
    ledger = directory / "account_state/contest_ledger.jsonl"
    saver = directory / "account_state/saver_state.json"
    status = directory.parent / "health_state/contest_streak_fetch_status.json"
    prior_rows = ledger.read_bytes().splitlines()
    assert fetch(directory, 58, 58)[0] == 0
    reading = obs.read_bytes()
    saver_reading = saver.read_bytes()
    rows = ledger.read_bytes().splitlines()
    assert len(rows) == len(prior_rows) + 1 and rows[:-1] == prior_rows
    selection = select(cfg, games)
    assert selection.streak == 58 and selection.decision.objective == "longest_streak"
    send.reset_mock()
    code, output = fetch(directory, 401, 401)
    assert code == 2 and "seasonBestStreak must be <= 400" in output
    send.assert_called_once()
    assert obs.read_bytes() == reading
    assert ledger.read_bytes().splitlines() == rows
    assert saver.read_bytes() == saver_reading
    failure = json.loads(status.read_text())
    assert failure["last_error_category"] == "actionable"
    assert failure["last_error"].endswith("seasonBestStreak must be <= 400")
    assert "actionable" in failure["last_alert_at_by_category"]
    send.reset_mock()
    assert fetch(directory, 401, 401)[0] == 2
    send.assert_not_called()
    assert read(directory).streak == 58


def test_k1_k2_k3_dated_checks(live):
    directory, _, _, _ = live
    install(directory, "dated", 12, 12)
    path = directory / "account_state/contest_streak.manual.json"
    body = json.loads(path.read_text())
    for number, field, values, message in (
        (5, "active_streak", (-1, *INVALID_TYPES, 401), "must be an integer from 0 to 400"),
        (6, "best_streak", (11, *INVALID_TYPES, 401), "must be an integer from active_streak to 400"),
    ):
        for value in values:
            invalid = with_field(body, field, value)
            with pytest.raises(contest_state.ContestStateError, match=f"check {number}, {field}: {message}$"):
                contest_state.validate_dated_observation(invalid, path)
    high = dict(body, active_streak=400, best_streak=400)
    assert contest_state.validate_dated_observation(high, path).best_streak == 400
    with pytest.raises(contest_state.ContestStateError, match="check 6, best_streak"):
        contest_state.validate_dated_observation(dict(high, best_streak=399), path)


@pytest.mark.parametrize("historical", [False, True])
def test_k4_automatic_reads(live, historical):
    directory, _, _, _ = live
    install(directory, "automatic", 57, 57)
    path = directory / "account_state/contest_streak.json"
    body = json.loads(path.read_text())

    def parse(payload):
        path.write_text(json.dumps(payload))
        return contest_state._read_state_file(path, require_dated=True, historical=historical)

    error = (contest_state._HistoricalContestStateError if historical else contest_state.ContestStateError)
    for value in (-1, *INVALID_TYPES):
        with pytest.raises(error, match="invalid streak" if historical else "contest streak invalid"):
            parse(with_field(body, "active_streak", value))
    for invalid in (dict(body, active_streak=401, best_streak=None),
                    dict(body, active_streak=400, best_streak=399), dict(body, best_streak=401)):
        with pytest.raises(error, match="invalid streak or best streak" if historical else "automatic contest streak invalid"):
            parse(invalid)
    high = dict(body, active_streak=400, best_streak=400)
    assert parse(high).best_streak == 400
    for value in INVALID_TYPES:
        assert parse(with_field(high, "best_streak", value)).best_streak is None


def test_k5_fetch_validation(live, monkeypatch):
    directory, _, _, send = live
    install(directory, "automatic", 57, 57)
    paths = [directory / "account_state" / name for name in (
        "contest_streak.json", "contest_ledger.jsonl", "saver_state.json")]
    prior = [path.read_bytes() for path in paths]
    status = directory.parent / "health_state/contest_streak_fetch_status.json"

    def refuse(payload, message):
        with pytest.raises(contest_fetch.ContestFetchError, match=f"^{message}$"):
            contest_fetch.validate_fetch(payload)
        profile = dict(payload, predictions=[{"roundId": 1, "result": "hit"}])
        with walk._fetch_stubs(57, 57, DAY - timedelta(days=1)), monkeypatch.context() as patch:
            patch.setattr(contest_fetch, "fetch_profile", lambda *args: profile)
            code, output = walk._invoke("fetch-contest-streak", "--picks-dir", directory,
                                        "--expected-username", walk.ACCOUNT,
                                        "--dm-recipient", "synthetic")
        assert code == 2 and message in output
        assert [path.read_bytes() for path in paths] == prior
        failure = json.loads(status.read_text())
        assert failure["last_error_category"] == "actionable"
        assert failure["last_error"].endswith(message)

    body = {"activeStreak": 12, "seasonBestStreak": 57}
    for field in body:
        for value in INVALID_TYPES:
            refuse(with_field(body, field, value), f"{field} must be an integer")
        refuse(with_field(body, field, -1), f"{field} must be non-negative")
    for active, best, message in ((12, 11, ">= activeStreak"), (401, 400, ">= activeStreak"),
                                  (401, 401, "<= 400"), (57, 401, "<= 400")):
        refuse({"activeStreak": active, "seasonBestStreak": best}, f"seasonBestStreak must be {message}")
    send.assert_called_once()
    contest_fetch.validate_fetch({"activeStreak": 400, "seasonBestStreak": 400})


def test_k6_dated_cli_rejects_before_writes(live):
    directory, _, _, _ = live
    cli_args = ["set-contest-streak", "--picks-dir", str(directory)]
    for active, best, day, extra in (
        ("12", "11", DAY, []), ("-1", "12", DAY, []), ("57", "401", DAY, []),
        ("12", "12", DAY - timedelta(days=1), []), ("12", "12", DAY, ["--ttl-hours", "1"]),
        ("12", None, DAY, []),
    ):
        args = [*cli_args, "--date", str(day), "--streak", active, "--saver", "used"]
        if best is not None:
            args.extend(["--best-streak", best])
        result = CliRunner().invoke(cli.cli, args + extra, standalone_mode=False)
        assert type(result.exception) is BadParameter
        assert CliRunner().invoke(cli.cli, args + extra).exit_code == 2
        assert list(directory.iterdir()) == []
    install(directory, "dated", 400, 400)
    assert (read(directory).streak, read(directory).best_streak) == (400, 400)


@pytest.mark.parametrize("kind", ["dated", "automatic"])
def test_k7_chosen_on_boundary_and_identity(live, kind):
    directory, cfg, games, _ = live
    install(directory, kind, 400, 400)
    select(cfg, games)
    chosen = picks.load_pick(str(DAY), directory).chosen_on
    assert chosen["streak"] == 400
    for value in (-1, True, 401):
        dated = dict(chosen["dated"], active_streak=value, best_streak=value) if kind == "dated" else None
        assert picks._valid_chosen_on(dict(chosen, streak=value, dated=dated)) is None
    assert picks._valid_chosen_on(dict(chosen, kind="unknown")) is None
    if kind == "dated":
        for update in ({"active_streak": 399}, {"best_streak": 399}, {"best_streak": 401}):
            assert picks._valid_chosen_on(dict(chosen, dated=dict(chosen["dated"], **update))) is None


def test_k8_parsed_best_requires_current_provenance(live):
    directory, _, _, _ = live
    install(directory, "automatic", 400, 400)
    path = directory / "account_state/contest_streak.json"
    sample = contest_state._read_state_file(path, require_dated=True, historical=False)
    now = walk.REAL_DATETIME(2027, 9, 26, 12, tzinfo=walk.ET)

    def classify(**updates):
        return contest_state.classify_best_streak(replace(sample, **updates), now=now, now_year=DAY.year)

    assert classify() == (400, "trusted")
    assert classify(best_streak=None) == (None, "missing")
    for value in (-1, 399, 401):
        assert classify(best_streak=value) == (value, "untrusted")
    for update in ({"source_date": DAY.replace(year=2026)}, {"source_date": DAY + timedelta(days=1)},
                   {"source_date": None}, {"source": "unknown"}, {"schema_version": None},
                   {"schema_version": "bts_contest_streak_manual_v2", "override_expires_at": now}):
        assert classify(**update) == (400, "untrusted")
