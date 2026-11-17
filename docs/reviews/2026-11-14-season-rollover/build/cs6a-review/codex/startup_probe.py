"""Exercise the real scheduler and orchestrator without external transports."""

import local_guard
from collections import Counter
from contextlib import ExitStack, nullcontext, redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import json
import tomllib

from click.testing import CliRunner
from pick_probe import Clock, DAY, HERE, NOW, REV, _mock_predictions, _sample_daily, _sample_pick
from pick_probe import cli, contest_state, picks, state_on_disk, tree

from bts import heartbeat, orchestrator, scheduler, strategy

POST = "2027-07-15T16:30:00Z"
FRAME = _mock_predictions()
FRAME.loc[:, "game_time"] = POST
FRAME.loc[:, "projected_lineup"] = False
GAME = {
    "gamePk": 778899, "gameDate": POST,
    "teams": {"away": {"team": {"name": "Away"}}, "home": {"team": {"name": "Home"}}},
    "status": {"abstractGameState": "Preview", "detailedState": "Pre-Game"},
}
DETAILS = {pk: {"abstract": "P", "detailed": "Pre-Game"} for pk in (778899, 778900)}


def invoke(command, path, *args, date=True):
    options = [command, "--config", str(path), *args]
    if date:
        options.extend(("--date", DAY))
    result = CliRunner().invoke(cli.cli, options)
    return dict(exit=result.exit_code, stdout=result.stdout, stderr=result.stderr,
                exception=type(result.exception).__name__, detail=str(result.exception or ""))


def fixture_config(root, body):
    path = root / "config.toml"
    path.write_text(
        f'tiers = []\n[orchestrator]\npicks_dir = "{root / "picks"}"\n'
        f'heartbeat_path = "{root / "heartbeat"}"\nmodels_dir = "{root / "models"}"\n'
        '[bluesky]\ndm_recipient = "capture.invalid"\n' + body
    )
    return path


def captured(stack):
    events = []
    real_read = contest_state.load_decision_streak_state
    real_select = strategy.select_pick

    def record(label, value=None):
        def call(*args, **kwargs):
            events.append((label, *args))
            return value
        return call

    def select(*args, **kwargs):
        events.append(("select", kwargs["streak"], kwargs["saver_available"]))
        return real_select(*args, **kwargs)

    def read(directory, **kwargs):
        events.append(("required", kwargs.get("require_contest_state", False)))
        return real_read(directory, **kwargs)

    options = {
        "datetime.datetime": Clock,
        "bts.cli.datetime": Clock, "bts.contest_state.datetime": Clock,
        "bts.picks.datetime": Clock, "bts.strategy.datetime": Clock,
        "bts.scheduler.datetime": Clock, "bts.heartbeat.datetime": Clock,
        "bts.scheduler._now_et": lambda: NOW,
        "bts.scheduler.notify_ready": record("ready"),
        "bts.scheduler.notify_watchdog": record("watchdog"),
        "bts.scheduler._watchdog_ping_sleep": record("sleep"),
        "bts.scheduler.run_result_polling": lambda *a, **k: "fixture stopped",
        "bts.scheduler._trigger_live_forward_capture_on_lock": lambda *a: None,
        "bts.scheduler.heartbeat_watchdog": lambda *a, **k: nullcontext(),
        "bts.scheduler.fetch_schedule": lambda date: record("schedule", [GAME] if date == DAY else [])(date),
        "bts.scheduler.count_new_confirmations": lambda *a: 1,
        "bts.scheduler._delivery_game_statuses": record("delivery-schedule", DETAILS),
        "bts.orchestrator.run_cascade": lambda tiers, date: record("prediction", (FRAME, "capture"))(date),
        "bts.picks.get_game_statuses_detailed": record("detailed", DETAILS),
        "bts.strategy.get_game_statuses": record("abstract", {778899: "P", 778900: "P"}),
        "bts.strategy.select_pick": select,
        "bts.strategy._load_mdp": lambda: None,
        "bts.strategy._mdp_action_from": lambda *_args: "single",
        "bts.contest_state.load_decision_streak_state": read,
        "bts.picks.attach_provenance": lambda *args, **kwargs: None,
        "bts.posting.post_to_bluesky": record("post", "at://capture/pick"),
        "bts.posting.should_post_now": lambda *_: True,
        "bts.dm.send_dm": record("dm", "captured-dm"),
        "bts.health.alert.dispatch_dm_for_health_alerts": lambda alerts, *a, **k: events.extend(
            ("alert", alert.source, alert.message) for alert in alerts
        ),
    }
    for name, value in options.items():
        stack.enter_context(patch(name, value))
    return events


def verify_refusal_access(root):
    folder = root / "resource-access"
    folder.mkdir()
    config_path = fixture_config(folder, '\n[scheduler]\npick_delivery="dm"\nentry_intent="enter"\n')
    state_on_disk(folder / "picks", "auto_v1")
    (folder / "heartbeat").write_text("existing heartbeat")
    (folder / "picks" / "scheduler_state.json").write_text("existing scheduler state")
    (folder / "models").mkdir()
    (folder / "models" / "predictions.csv").write_text("existing predictions")
    before = tree(folder)
    with ExitStack() as stack:
        events = captured(stack)
        loader = stack.enter_context(patch.object(orchestrator, "load_config", wraps=orchestrator.load_config))
        default_date = stack.enter_context(patch.object(cli, "_today_et", wraps=cli._today_et))
        worker = stack.enter_context(patch.object(scheduler, "run_day", return_value=None))
        local_guard.watch[:] = [folder]
        local_guard.opens.clear()
        result = invoke("schedule", config_path, date=False)
        accessed = list(local_guard.opens)
        local_guard.watch.clear()
    return dict(result=result, loader=loader.call_count, date=default_date.call_count,
                worker=worker.call_count, events=events, opens=accessed, unchanged=tree(folder) == before)


def representative_cases(root):
    results = {}
    for key, mode, intent, flag, state, command in (
        ("dm-optional", "dm", "enter", "", "auto_v1", "schedule"),
        ("dm-scheduler-required", "bluesky-dm", "RESEARCH", 'contest_state_required="false"\n', "auto_v1", "schedule"),
        ("public-health-required", "feed", "research", '[health_checks]\ncontest_state_expected=1\n', "auto_v1", "schedule"),
        ("dm-covered", "dm", "enter", 'contest_state_required=true\n', "covered", "schedule"),
        ("public-covered", "public", "research", '[health_checks]\ncontest_state_expected=true\n', "covered", "schedule"),
        ("private-research", "local", "research", "", "auto_v1", "schedule"),
        ("orch-private-research", "local", "research", "", "auto_v1", "orchestrate"),
        ("orch-private-required", "local", "research", '[health_checks]\ncontest_state_expected=1\n', "auto_v1", "orchestrate"),
        ("orch-private-covered", "local", "research", 'contest_state_required=1\n', "covered", "orchestrate"),
        ("orch-invalid-required", "typo", "enter", 'contest_state_required=1\n', "auto_v1", "orchestrate"),
    ):
        folder = root / key
        folder.mkdir()
        suffix = "" if flag.startswith("[") else flag
        health = flag if flag.startswith("[") else "[health_checks]\n"
        body = f'\n[scheduler]\npick_delivery="{mode}"\nentry_intent="{intent}"\n'
        path = fixture_config(folder, body + suffix + health + "enabled=false\n")
        state_on_disk(folder / "picks", state)
        with ExitStack() as stack:
            events = captured(stack)
            result = invoke(command, path)
        daily = picks.load_pick(DAY, folder / "picks")
        results[key] = dict(result=result, events=events, pick=daily.pick.batter_name if daily else None)
    return results


def dry_run(root, has_game):
    folder = root / ("dry-game" if has_game else "dry-off")
    folder.mkdir()
    path = fixture_config(folder, "\n[health_checks]\nenabled=false\n")
    with ExitStack() as stack:
        events = captured(stack)
        if not has_game:
            stack.enter_context(patch.object(scheduler, "fetch_schedule",
                                            lambda date: events.append(("schedule", date)) or []))
        result = invoke("schedule", path, "--dry-run")
    files = tree(folder)
    files.pop("config.toml")
    return dict(result=result, events=events, files=files)


def checked_snapshot(root):
    folder = root / "snapshot"
    folder.mkdir()
    state_on_disk(folder / "picks", "covered")
    config = fixture_config(folder, '[scheduler]\ncontest_state_required=true\n')
    loader = orchestrator.load_config
    loaded = []
    consumed = []
    real_run = orchestrator.run_and_pick
    replacement = config.read_text().replace("contest_state_required=true", "contest_state_required=false")
    replacement = replacement.replace(str(folder / "picks"), str(folder / "different-picks"))

    def load(path):
        instance = loader(path)
        loaded.append(instance)
        return instance

    def replace_then_run(path, date, **kwargs):
        # The command's gate has returned. Point the file at a different picks tree.
        path.write_text(replacement)
        return real_core(path, date, **kwargs)

    def track(instance, *args, **kwargs):
        consumed.append(instance)
        return real_run(instance, *args, **kwargs)

    real_core = orchestrator.orchestrate
    with ExitStack() as stack:
        events = captured(stack)
        stack.enter_context(patch.object(orchestrator, "load_config", load))
        stack.enter_context(patch.object(orchestrator, "orchestrate", replace_then_run))
        stack.enter_context(patch.object(orchestrator, "run_and_pick", track))
        local_guard.watch[:] = [config]
        local_guard.opens.clear()
        result = invoke("orchestrate", config)
        accessed = [(mode, flags) for _, mode, flags in local_guard.opens]
        local_guard.watch.clear()
    return dict(result=result, loads=len(loaded), events=events, opens=accessed,
                identity=len(loaded) == len(consumed) == 1 and loaded[0] is consumed[0],
                original_pick=(folder / "picks" / (DAY + ".json")).is_file(),
                moved_pick=(folder / "different-picks" / (DAY + ".json")).is_file())


def load_errors(root):
    output = {}
    for variant in ("missing", "malformed", "unreadable"):
        path = root / f"{variant}.toml"
        if variant != "missing":
            path.write_text("[scheduler\n" if variant == "malformed" else "[scheduler]\n")
        if variant == "unreadable":
            path.chmod(0)
        output[variant] = invoke("orchestrate", path)
    private = root / "direct"
    private.mkdir()
    state_on_disk(private / "picks", "auto_v1")
    cfg = fixture_config(private, '[scheduler]\npick_delivery="private"\nentry_intent="research"\n')
    with ExitStack() as stack, redirect_stderr(StringIO()):
        events = captured(stack)
        load = stack.enter_context(patch.object(orchestrator, "load_config", wraps=orchestrator.load_config))
        result = orchestrator.orchestrate(cfg, DAY)
    output["direct"] = dict(result=result, loads=load.call_count, events=events)
    return output


def schedule_identity(root):
    folder = root / "identity"
    folder.mkdir()
    cfg = fixture_config(folder, "[scheduler]\ncontest_state_required=true\n")
    mapping = orchestrator.load_config(cfg)
    with patch.object(orchestrator, "load_config", return_value=mapping) as load, \
            patch.object(scheduler, "run_day") as run:
        result = invoke("schedule", cfg)
    return dict(result=result, loads=load.call_count, identity=run.call_args.kwargs["config"] is mapping,
                unchanged=mapping == orchestrator.load_config(cfg))


def exhaustive(root):
    counts = Counter()
    pole = root / "exhaustive"
    state_on_disk(pole / "picks", "auto_v1")
    daily = _sample_daily(date=DAY, run_time=NOW.isoformat(), pick=_sample_pick(game_time=POST))
    config_extra = {"tiers": [], "orchestrator": {"picks_dir": str(pole / "picks")},
                    "bluesky": {"dm_recipient": "capture.invalid"}}
    # The CLI for every row was already exercised in the matrix. Run the real
    # selection, delivery and orchestration for every accepted mapping here.
    with ExitStack() as stack, (HERE / "matrix-head.jsonl").open() as rows:
        events = captured(stack)
        stack.enter_context(patch("bts.slate.save_slate", lambda *a, **k: None))
        stack.enter_context(patch("bts.progress.mark", lambda *a: None))
        with redirect_stderr(StringIO()) as log:
            for line in rows:
                raw, (schedule, _, orch, _) = json.loads(line)
                config = tomllib.loads(raw) | config_extra
                checks = []
                if schedule[0] == 0:
                    try:
                        mode = scheduler._pick_delivery_mode(config)
                    except ValueError:
                        mode = None
                    if mode in ("dm", "public"):
                        selection = scheduler.run_single_check(DAY, [778899], set(), config, 0.03)
                        state = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
                        sent = scheduler._deliver_and_lock_pick(daily, config, pole / "picks", state, DAY, "review")
                        assert not sent and selection["selection"] is None and not state.pick_locked, raw
                        counts[f"scheduler-{mode}"] += 1
                        checks.extend(e for e in events if e[0] in ("post", "dm", "select"))
                        events.clear()
                if orch[0] == 0:
                    accepted = orchestrator.orchestrate(pole / "unused.toml", DAY, config=config)
                    assert not accepted, (raw, events, log.getvalue())
                    counts["orchestrator"] += 1
                    checks.extend(e for e in events if e[0] in ("post", "select") or
                                  (e[0] == "dm" and "No pick made" not in e[2]))
                    events.clear()
                assert not checks, (raw, checks)
                log.seek(0)
                log.truncate()
    assert not (pole / "picks" / (DAY + ".json")).exists()
    return dict(counts)


def main():
    with TemporaryDirectory(dir=HERE) as folder:
        root = Path(folder)
        outcomes = dict(refusal=verify_refusal_access(root), representatives=representative_cases(root),
                        dry_game=dry_run(root, True), dry_off=dry_run(root, False),
                        snapshot=checked_snapshot(root), errors=load_errors(root), schedule_identity=schedule_identity(root))
        if REV == "head":
            outcomes["exhaustive"] = exhaustive(root)
        outcome_json = json.dumps(outcomes, indent=2).replace(str(root), "<root>") + "\n"
    assert not local_guard.network, local_guard.network
    (HERE / f"startup-{REV}.json").write_text(outcome_json)
    for key, case in outcomes["representatives"].items():
        print(key, case["result"]["exit"], [event[0] for event in case["events"]], case["pick"])
    print("Exhaustive", outcomes.get("exhaustive"))


if __name__ == "__main__":
    main()
