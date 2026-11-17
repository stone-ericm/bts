"""CS6a refusal boundaries, using explicit ET dates and local transports."""

import builtins
from datetime import datetime
from itertools import product
import json
from pathlib import Path
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pytest

import bts.cli as cli
from bts import contest_state, entry_intent, orchestrator, picks, scheduler, strategy
from tests.test_cli_integration import _mock_predictions, _sample_daily, _sample_pick
from tests.test_season_contract_cs4b2 import dated


DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))
FIX = (
    "Optional reads can use a contest observation without the season-coverage check. "
    "Set contest_state_required = true under [scheduler] and contest_state_expected = true under [health_checks]."
)


def problem(*conditions):
    verb = "requires" if len(conditions) == 1 else "require"
    return f"{' and '.join(conditions)} {verb} contest state. {FIX}"


PUBLIC = problem("scheduler delivery 'public'")
ORCHESTRATE = problem("delivery 'public'")
DM_ENTER = problem("scheduler delivery 'dm'", "scheduler.entry_intent = 'enter'")
DRY = "bts schedule: WARNING (a real run would refuse to start): "
MODES = [
    ({}, "public"),
    *[({"pick_delivery": mode}, "public") for mode in ("public", "bluesky", "feed", "post", "  pUbLic  ")],
    *[({"pick_delivery": mode}, "dm")
      for mode in ("dm", "bluesky_dm", "direct_message", "bluesky-dm", "direct-message", " DIRECT-MESSAGE ")],
    *[({"pick_delivery": mode}, "private") for mode in ("private", "none", "off", "local", " LOCAL ")],
    ({"posting_mode": "bluesky-dm"}, "dm"),
    ({"posting_mode": "dm", "pick_delivery": "off", "private_mode": False}, "private"),
    ({"pick_delivery": "dm", "shadow_mode": False, "private_mode": True}, "dm"),
    ({"private_mode": "false", "shadow_mode": False}, "private"),
    ({"shadow_mode": False}, None),
    ({"pick_delivery": "invalid", "posting_mode": "dm"}, None),
    ({"pick_delivery": 3}, None),
    ({"pick_delivery": None, "posting_mode": "dm"}, "public"),
]
INTENTS = ({}, {"entry_intent": "enter"}, {"entry_intent": "research"},
           {"entry_intent": "Enter"}, {"entry_intent": 1})
FLAGS = ({}, {"flag": True}, {"flag": False}, {"flag": "false"})


class Clock(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.replace(tzinfo=None) if tz is None else NOW.astimezone(tz)


def test_gate_aliases_flags_and_intents_without_io(monkeypatch):
    # Import call sites before installing the access spies, so the test covers evaluation.
    forbidden = Mock(side_effect=AssertionError("gate attempted file or clock access"))
    with monkeypatch.context() as guard:
        guard.setattr(builtins, "open", forbidden)
        guard.setattr(Path, "open", forbidden)
        guard.setattr(Path, "stat", forbidden)
        guard.setattr(scheduler, "_now_et", forbidden)
        guard.setattr(scheduler, "datetime", forbidden)
        guard.setattr("datetime.datetime", forbidden)
        guard.setattr("datetime.date", forbidden)
        guard.setattr("time.time", forbidden)
        gate = entry_intent.live_gate_problem
        for (fields, mode), intent, req, health in product(MODES, INTENTS, FLAGS, FLAGS):
            config = {
                "scheduler": fields | intent | {"contest_state_required": v for v in req.values()},
                "health_checks": {"contest_state_expected": v for v in health.values()},
            }
            for override in (None, "public", "dm", "private", "bluesky"):
                delivery = mode if override is None else override
                conditions = []
                if delivery in ("dm", "public"):
                    label = "scheduler delivery" if override is None else "delivery"
                    conditions.append(f"{label} {delivery!r}")
                if intent == {"entry_intent": "enter"}:
                    conditions.append("scheduler.entry_intent = 'enter'")
                expected = problem(*conditions) if conditions and not (req.get("flag") or health.get("flag")) else None
                assert gate(config, delivery_override=override) == expected, (config, override)
    forbidden.assert_not_called()


@pytest.mark.parametrize("scheduler_shape,health_shape", product(("present", "absent", "malformed"), repeat=2))
@pytest.mark.parametrize("required", [False, True])
def test_gate_shape_precedes_flags(scheduler_shape, health_shape, required):
    config = {}
    if scheduler_shape == "present":
        config["scheduler"] = {"contest_state_required": required}
    elif scheduler_shape == "malformed":
        config["scheduler"] = None
    if health_shape == "present":
        config["health_checks"] = {"contest_state_expected": False}
    elif health_shape == "malformed":
        config["health_checks"] = None
    if scheduler_shape == "malformed":
        expected = "[scheduler] is not a table"
    elif health_shape == "malformed":
        expected = "[health_checks] is not a table"
    elif scheduler_shape == "present" and required:
        expected = None
    else:
        expected = PUBLIC
    assert entry_intent.live_gate_problem(config) == expected


def write_config(path, body):
    path.write_text(body)
    return str(path)


def invoke(command, path, *args):
    return CliRunner().invoke(cli.cli, [command, "--config", str(path), *args])


@pytest.mark.parametrize(
    "body,existing,gate",
    [
        ("", ('scheduler.entry_intent is missing: set it to "enter" or "research" (there is no default)',), PUBLIC),
        ('[scheduler]\nentry_intent = 1\npick_delivery = "bad"\n', (
            'scheduler.entry_intent must be exactly "enter" or "research", got 1',
            "the scheduler refuses this delivery config: scheduler.pick_delivery must be one of: public, dm, private",
        ), None),
        ('health_checks = false\n[scheduler]\nentry_intent = 1\npick_delivery = "bad"\n', (
            'scheduler.entry_intent must be exactly "enter" or "research", got 1',
            "the scheduler refuses this delivery config: scheduler.pick_delivery must be one of: public, dm, private",
        ), "[health_checks] is not a table"),
        ('[scheduler]\nentry_intent = "research"\n', (
            "entry_intent 'research' expects delivery private, but the scheduler would deliver 'public'",
        ), PUBLIC),
        ('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n', (), DM_ENTER),
        ('scheduler = 1\nhealth_checks = 2\n', ("[scheduler] is not a table",), None),
    ],
)
def test_entry_intent_diagnostics_are_ordered(tmp_path, body, existing, gate):
    path = write_config(tmp_path / "config.toml", body)
    problems = existing + ((gate,) if gate else ())
    result = invoke("entry-intent", path)
    assert (result.exit_code, result.stdout, result.stderr) == (
        1, "", "".join(f"entry-intent: {item}\n" for item in problems),
    )
    config = tomllib.loads(body)
    assert entry_intent.check_entry_intent(config).problems == problems


def test_entry_intent_can_exclude_the_live_gate():
    config = {"scheduler": {"entry_intent": "research", "pick_delivery": "dm"}}
    result = entry_intent.check_entry_intent(config, include_live_gate=False)
    assert (result.intent, result.delivery, result.problems) == (
        "research", "dm", ("entry_intent 'research' expects delivery private, but the scheduler would deliver 'dm'",),
    )


def test_schedule_refuses_before_resource_access(tmp_path, monkeypatch):
    heartbeat = tmp_path / "heartbeat"
    picks = tmp_path / "picks"
    path = write_config(tmp_path / "config.toml",
                        f'[orchestrator]\npicks_dir = "{picks}"\nheartbeat_path = "{heartbeat}"\n'
                        '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n')
    real_loader = orchestrator.load_config
    load = Mock(wraps=real_loader)
    monkeypatch.setattr(orchestrator, "load_config", load)
    blocked = []
    for module, name in ((cli, "_today_et"), (scheduler, "run_day"), (scheduler, "load_state"),
                         (scheduler, "write_heartbeat"), (scheduler, "fetch_schedule"),
                         (orchestrator, "run_cascade")):
        spy = Mock(side_effect=AssertionError(name))
        monkeypatch.setattr(module, name, spy)
        blocked.append(spy)
    for name in ("open", "stat", "lstat", "mkdir", "iterdir", "unlink", "rename", "replace"):
        original = getattr(Path, name)

        def guarded(candidate, *args, original=original, **kwargs):
            assert candidate != heartbeat and not candidate.is_relative_to(picks), candidate
            return original(candidate, *args, **kwargs)

        monkeypatch.setattr(Path, name, guarded)
    real_open = builtins.open

    def guarded_open(candidate, *args, **kwargs):
        if not isinstance(candidate, int):
            candidate = Path(candidate)
            assert candidate != heartbeat and not candidate.is_relative_to(picks), candidate
        return real_open(candidate, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    result = invoke("schedule", path)
    assert (result.exit_code, result.stderr) == (2, f"bts schedule: refusing to start: {DM_ENTER}\n")
    load.assert_called_once_with(Path(path))
    for spy in blocked:
        spy.assert_not_called()


def test_schedule_passes_one_snapshot(tmp_path, monkeypatch):
    path = write_config(tmp_path / "config.toml", "[health_checks]\ncontest_state_expected = true\n")
    config = tomllib.loads(Path(path).read_text())
    load, worker = Mock(return_value=config), Mock()
    monkeypatch.setattr(orchestrator, "load_config", load)
    monkeypatch.setattr(scheduler, "run_day", worker)
    result = invoke("schedule", path, "--date", DAY)
    assert (result.exit_code, result.stderr) == (0, "")
    load.assert_called_once_with(Path(path))
    worker.assert_called_once_with(date=DAY, config=config, dry_run=False)
    assert worker.call_args.kwargs["config"] is config


@pytest.mark.parametrize("has_game", [False, True])
def test_schedule_dry_run_preserves_real_worker(tmp_path, monkeypatch, has_game):
    heartbeat = tmp_path / "heartbeat"
    picks = tmp_path / "picks"
    path = write_config(tmp_path / "config.toml",
                        f'[orchestrator]\npicks_dir = "{picks}"\nheartbeat_path = "{heartbeat}"\n')
    game = {
        "gamePk": 42,
        "gameDate": "2027-07-15T23:00:00Z",
        "teams": {"away": {"team": {"name": "Away"}}, "home": {"team": {"name": "Home"}}},
        "status": {"abstractGameState": "Preview"},
    }
    fetch = Mock(return_value=[game] if has_game else [])
    ready, watchdog, sleep = Mock(), Mock(), Mock()
    monkeypatch.setattr(scheduler, "_now_et", lambda: NOW)
    monkeypatch.setattr(scheduler, "fetch_schedule", fetch)
    monkeypatch.setattr(scheduler, "notify_ready", ready)
    monkeypatch.setattr(scheduler, "notify_watchdog", watchdog)
    monkeypatch.setattr(scheduler, "datetime", Clock)
    monkeypatch.setattr(scheduler, "_watchdog_ping_sleep", sleep)
    result = invoke("schedule", path, "--date", DAY, "--dry-run")
    assert result.exit_code == 0, repr(result.exception)
    expected_days = [DAY] if has_game else [DAY, "2027-07-16"]
    assert [call.args[0] for call in fetch.call_args_list] == expected_days
    ready.assert_called_once_with()
    assert watchdog.call_count == (1 if has_game else 3)
    assert heartbeat.is_file() and not picks.exists()
    beat = json.loads(heartbeat.read_text())
    assert beat["state"] == ("running" if has_game else "sleeping")
    if has_game:
        sleep.assert_not_called()
        assert "(--dry-run: not executing checks)" in result.stderr
    else:
        sleep.assert_called_once_with(22 * 60 * 60)
        assert beat["sleeping_until"] == "2027-07-16T14:00:00+00:00"
        assert f"No games scheduled for {DAY}." in result.stderr
    assert result.stderr.splitlines()[0] == DRY + PUBLIC


@pytest.mark.parametrize("success", [True, False])
def test_orchestrate_uses_checked_snapshot(tmp_path, monkeypatch, success):
    path = write_config(tmp_path / "config.toml", "[scheduler]\ncontest_state_required = true\n")
    snapshot = tomllib.loads(Path(path).read_text())
    load, worker = Mock(return_value=snapshot), Mock(return_value=success)
    monkeypatch.setattr(orchestrator, "load_config", load)
    monkeypatch.setattr(orchestrator, "orchestrate", worker)
    result = invoke("orchestrate", path, "--date", DAY)
    assert (result.exit_code, result.stderr) == (0 if success else 1, "")
    load.assert_called_once_with(Path(path))
    worker.assert_called_once_with(Path(path), DAY, config=snapshot)
    assert worker.call_args.kwargs["config"] is snapshot


def test_orchestrate_refuses_even_with_invalid_scheduler_delivery(tmp_path, monkeypatch):
    path = write_config(tmp_path / "config.toml", '[scheduler]\npick_delivery = "unknown"\n')
    core = Mock()
    monkeypatch.setattr(orchestrator, "orchestrate", core)
    result = invoke("orchestrate", path, "--date", DAY)
    assert (result.exit_code, result.stderr) == (2, f"bts orchestrate: refusing to run: {ORCHESTRATE}\n")
    core.assert_not_called()


def test_orchestrate_malformed_toml_preserves_loader_error(tmp_path):
    path = write_config(tmp_path / "config.toml", "[scheduler\n")
    result = invoke("orchestrate", path, "--date", DAY)
    assert result.exit_code == 1
    assert isinstance(result.exception, tomllib.TOMLDecodeError)
    assert result.stderr == ""


@pytest.fixture
def example_config():
    path = Path(__file__).resolve().parents[1] / "config" / "orchestrator.example.toml"
    with path.open("rb") as stream:
        return tomllib.load(stream)


@pytest.mark.parametrize("section,key", [
    ("scheduler", "contest_state_required"),
    ("health_checks", "contest_state_expected"),
])
def test_example_requires_contest_state(example_config, section, key):
    assert example_config[section][key] is True


def test_example_declares_entry_intent(example_config):
    assert example_config["scheduler"].get("entry_intent") == "enter"
    assert entry_intent.live_gate_problem(example_config) is None
    assert entry_intent.check_entry_intent(example_config).problems == ()


@pytest.fixture
def pick_path(tmp_path):
    directory = tmp_path / "picks"
    state = directory / "account_state"
    state.mkdir(parents=True)
    (state / "contest_streak.json").write_text(json.dumps({
        "active_streak": 7, "source": "mlb_bts_profile",
        "recorded_at": "2026-07-15T16:00:00Z",
    }))
    return directory


@pytest.fixture
def captured_pick_path(monkeypatch):
    from bts.model import predict

    monkeypatch.setattr(contest_state, "datetime", Clock)
    monkeypatch.setattr(picks, "datetime", Clock)
    frame = _mock_predictions()
    predict_spy = Mock(return_value=frame)
    schedule = Mock(return_value={})
    post = Mock(return_value="at://captured/pick")
    dm = Mock()
    save = Mock(wraps=picks.save_pick)
    monkeypatch.setattr(predict, "run_pipeline", predict_spy)
    monkeypatch.setattr(picks, "get_game_statuses_detailed", schedule)
    monkeypatch.setattr(picks, "save_pick", save)
    monkeypatch.setattr("bts.posting.post_to_bluesky", post)
    monkeypatch.setattr("bts.posting.should_post_now", lambda *_: True)
    monkeypatch.setattr("bts.dm.send_dm", dm)
    monkeypatch.setattr("bts.progress.mark", lambda *_: None)
    monkeypatch.setattr("bts.slate.save_slate", lambda *_: None)
    monkeypatch.setattr(picks, "attach_provenance", lambda *_args, **_kw: None)
    return SimpleNamespace(frame=frame, predict=predict_spy, schedule=schedule, post=post, save=save, dm=dm)


def set_selection(monkeypatch, path):
    daily = _sample_daily(date=DAY, run_time=NOW.isoformat(), double_down=_sample_pick(batter_name="Second"))
    result = None if path == "skip" else strategy.PickResult(daily=daily, locked=path == "locked")
    select = Mock(return_value=SimpleNamespace(pick_result=result, decision=None))
    monkeypatch.setattr(strategy, "select_pick", select)
    return select


def test_orchestrate_cli_refuses_real_private_research_config(tmp_path, pick_path, captured_pick_path, monkeypatch):
    path = write_config(tmp_path / "config.toml",
                        f'tiers = []\n[orchestrator]\npicks_dir = "{pick_path}"\n'
                        '[bluesky]\ndm_recipient = "unused"\n'
                        '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n')
    select = set_selection(monkeypatch, "new")
    cascade = Mock(return_value=(captured_pick_path.frame, "fixture"))
    monkeypatch.setattr(orchestrator, "run_cascade", cascade)
    result = invoke("orchestrate", path, "--date", DAY)
    assert (result.exit_code, result.stderr) == (2, f"bts orchestrate: refusing to run: {ORCHESTRATE}\n")
    for spy in (select, cascade, captured_pick_path.save, captured_pick_path.post, captured_pick_path.dm):
        spy.assert_not_called()
    assert not (pick_path / f"{DAY}.json").exists()


def test_orchestrate_direct_call_loads_private_research_path_and_posts(
    tmp_path, pick_path, captured_pick_path, monkeypatch,
):
    path = Path(write_config(tmp_path / "config.toml",
                             f'tiers = []\n[orchestrator]\npicks_dir = "{pick_path}"\n'
                             '[bluesky]\ndm_recipient = "unused"\n'
                             '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'))
    select = set_selection(monkeypatch, "new")
    load = Mock(wraps=orchestrator.load_config)
    monkeypatch.setattr(orchestrator, "load_config", load)
    monkeypatch.setattr(orchestrator, "run_cascade", lambda *_: (captured_pick_path.frame, "fixture"))
    assert orchestrator.orchestrate(path, DAY) is True
    load.assert_called_once_with(path)
    assert select.call_args.kwargs["streak"] == 7
    assert select.return_value.state_status == "stale"
    captured_pick_path.post.assert_called_once()
    captured_pick_path.dm.assert_not_called()
    assert json.loads((pick_path / f"{DAY}.json").read_text())["bluesky_posted"] is True


def test_orchestrate_accepted_cli_loads_once_through_real_core(
    tmp_path, pick_path, captured_pick_path, covered_automatic, monkeypatch,
):
    covered_automatic(pick_path, NOW, {
        "active_streak": 7, "source": "mlb_bts_profile",
        "source_date": DAY, "recorded_at": NOW.isoformat(),
    })
    path = write_config(tmp_path / "config.toml",
                        f'tiers = []\n[orchestrator]\npicks_dir = "{pick_path}"\n'
                        '[bluesky]\ndm_recipient = "unused"\n'
                        '[health_checks]\ncontest_state_expected = true\n')
    select = set_selection(monkeypatch, "new")
    load = Mock(wraps=orchestrator.load_config)
    monkeypatch.setattr(orchestrator, "load_config", load)
    monkeypatch.setattr(orchestrator, "run_cascade", lambda *_: (captured_pick_path.frame, "fixture"))
    result = invoke("orchestrate", path, "--date", DAY)
    assert result.exit_code == 0, (result.stderr, result.exception)
    load.assert_called_once_with(Path(path))
    assert select.call_args.kwargs["streak"] == 7
    captured_pick_path.post.assert_called_once()
    assert json.loads((pick_path / f"{DAY}.json").read_text())["bluesky_posted"] is True


def run_pick(tmp_path, pick_path, *args):
    return CliRunner().invoke(cli.cli, [
        "run", "--date", DAY, "--picks-dir", str(pick_path),
        "--models-dir", str(tmp_path / "models"), *args,
    ])


@pytest.mark.parametrize("path,refuse_at", [
    ("skip", 1), ("locked", 1), ("new", 1), ("locked", 2), ("new", 2),
])
def test_run_required_reads_with_real_refusal(
    tmp_path, pick_path, captured_pick_path, covered_automatic, monkeypatch, path, refuse_at,
):
    select = set_selection(monkeypatch, path)
    loader = contest_state.load_decision_streak_state
    with pytest.raises(contest_state.ContestStateError, match="season 2027 switch is absent") as direct:
        loader(pick_path, require_contest_state=True)

    if refuse_at == 2:
        covered_automatic(pick_path, NOW, {
            "active_streak": 7, "source": "mlb_bts_profile",
            "source_date": DAY, "recorded_at": NOW.isoformat(),
        })

    calls = []

    def load(directory, **kwargs):
        calls.append((directory, kwargs))
        if refuse_at == 2 and len(calls) == 2:
            (directory / "account_state" / "season_contract.json").unlink()
        return loader(directory, **kwargs)

    monkeypatch.setattr(contest_state, "load_decision_streak_state", load)
    result = run_pick(tmp_path, pick_path)
    assert (result.exit_code, result.stderr) == (1, f"bts run: no pick: {direct.value}\n")
    assert calls == [(pick_path, {"require_contest_state": True})] * refuse_at
    captured_pick_path.predict.assert_called_once()
    captured_pick_path.post.assert_not_called()
    pick_file = pick_path / f"{DAY}.json"
    evolution = pick_path / f"lineup_evolution_{DAY}.jsonl"
    if refuse_at == 1:
        select.assert_not_called()
        captured_pick_path.schedule.assert_not_called()
        assert "Pick locked:" not in result.stdout and "Skipping " not in result.stdout
    else:
        select.assert_called_once()
        captured_pick_path.schedule.assert_called_once_with(DAY)
    if refuse_at == 2 and path == "new":
        captured_pick_path.save.assert_called_once()
        saved = json.loads(pick_file.read_text())
        assert saved["bluesky_posted"] is False and saved["bluesky_uri"] is None
        assert saved["pick"]["batter_name"] == "Jacob Wilson"
        assert saved["double_down"]["batter_name"] == "Second"
        row, = map(json.loads, evolution.read_text().splitlines())
        assert row["primary"]["batter_name"] == "Jacob Wilson"
        assert row["double_down"]["batter_name"] == "Second"
        assert f"Saved to {pick_file}" in result.stdout
    else:
        captured_pick_path.save.assert_not_called()
        assert not pick_file.exists() and not evolution.exists()
        if refuse_at == 2:
            assert result.stdout.splitlines()[-1] == "Pick locked: Jacob Wilson (game started)"


@pytest.mark.parametrize("state", ["dated", "covered"])
def test_run_accepts_dated_and_covered_observations(
    tmp_path, pick_path, captured_pick_path, covered_automatic, monkeypatch, state,
):
    if state == "dated":
        (pick_path / "account_state" / "contest_streak.manual.json").write_text(
            json.dumps(dated(active_streak=7, best_streak=8)),
        )
    else:
        covered_automatic(pick_path, NOW, {
            "active_streak": 7, "source": "mlb_bts_profile",
            "source_date": DAY, "recorded_at": NOW.isoformat(),
        })
    select = set_selection(monkeypatch, "new")
    result = run_pick(tmp_path, pick_path)
    assert (result.exit_code, result.stderr) == (0, "")
    assert select.call_args.kwargs["streak"] == 7
    captured_pick_path.post.assert_called_once()
    assert captured_pick_path.save.call_count == 2
    assert json.loads((pick_path / f"{DAY}.json").read_text())["bluesky_posted"] is True


@pytest.mark.parametrize("dry", [False, True])
def test_run_early_returns_without_state_access(tmp_path, pick_path, captured_pick_path, monkeypatch, dry):
    forbidden = Mock(side_effect=AssertionError("decision-state read"))
    monkeypatch.setattr(contest_state, "load_decision_streak_state", forbidden)
    if not dry:
        captured_pick_path.predict.return_value = captured_pick_path.frame.iloc[:0]
    result = run_pick(tmp_path, pick_path, *(["--dry-run"] if dry else []))
    assert (result.exit_code, result.stderr) == (0, "")
    assert ("--dry-run: not saving or posting" if dry else "No games found") in result.stdout
    forbidden.assert_not_called()
    captured_pick_path.save.assert_not_called()
    captured_pick_path.post.assert_not_called()
    assert not (pick_path / f"{DAY}.json").exists()
