import sys


class NetworkAttempt(BaseException):
    pass


def deny_network(event, args):
    if event.startswith("socket.") and event not in ("socket.__new__", "socket.sethostname"):
        raise NetworkAttempt(event)


sys.addaudithook(deny_network)

import json
import os
import subprocess
from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from click.testing import CliRunner

from bts import contest_state, dm, entry_intent, orchestrator, picks, posting, scheduler, strategy, web
from bts.cli import cli
from bts.model import predict


ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
DATE = "2027-03-29"
PREVIOUS_DATE = "2026-09-29"
NOW = datetime(2027, 3, 29, 19, tzinfo=timezone.utc)
LATE_GAME = f"{DATE}T23:00:00Z"
ROW = dict(batter_name="Probe Batter", batter_id=100, team="ATL", lineup=1,
           pitcher_name="Probe Pitcher", pitcher_id=200, p_game_hit=.85,
           flags="", game_pk=301, game_time=LATE_GAME)
GAMES = [dict(gamePk=301, gameDate=LATE_GAME,
              teams={"away": {"team": {"name": "ATL"}}, "home": {"team": {"name": "NYM"}}})]
FRAME = pd.DataFrame([ROW])
RUNNER = CliRunner()


def info(label, result):
    print(label, result.exit_code, type(result.exception).__name__,
          repr(result.stdout), repr(result.stderr))


def config_text(path, scheduler_text):
    pick_dir = path / "picks"
    heartbeat = path / "heartbeat.json"
    return (f'[orchestrator]\npicks_dir = "{pick_dir}"\nheartbeat_path = "{heartbeat}"\n'
            '[bluesky]\ndm_recipient = "probe.invalid"\n[[tiers]]\nname = "probe"\n'
            + scheduler_text)


def stale_state(pick_dir):
    account = pick_dir / "account_state"
    account.mkdir(parents=True)
    (account / "contest_streak.json").write_text(json.dumps(dict(
        active_streak=7, best_streak=12, source="mlb_bts_contest",
        source_date=PREVIOUS_DATE)))
    return account


def selected(kind):
    first = picks.Pick(batter_name=ROW['batter_name'], batter_id=100, team="ATL", lineup_position=1,
                       pitcher_name=ROW['pitcher_name'], pitcher_id=200, p_game_hit=.85,
                       flags=[], projected_lineup=True, game_pk=301, game_time=LATE_GAME)
    double = None
    if kind == "double":
        double = picks.Pick(batter_name="Probe Second", batter_id=101, team="NYM", lineup_position=1,
                            pitcher_name="Probe Pitcher", pitcher_id=200, p_game_hit=.84,
                            flags=[], projected_lineup=True, game_pk=301, game_time=LATE_GAME)
    daily = picks.DailyPick(date=DATE, run_time=NOW.isoformat(), pick=first, double_down=double,
                            runner_up=None, bluesky_posted=kind == "posted", bluesky_uri=None)
    result = None if kind == "skip" else strategy.PickResult(daily, locked=kind in ("locked", "posted"))
    return SimpleNamespace(pick_result=result, decision=None, action=None, source=None)


def delivery_matrix():
    assert not hasattr(entry_intent, "live_gate_problem")
    for key in ("pick_delivery", "posting_mode"):
        for value in ("public", "bluesky", "feed", "post", "dm", "bluesky_dm", "direct_message",
                      "private", "none", "off", "local", "  BLUESKY-DM  ", "direct-message",
                      "direct message", "", True, 1, ["dm"], {"mode": "dm"}):
            show_mode({key: value})
    for cfg in ({}, {"private_mode": False}, {"private_mode": "false"}, {"shadow_mode": False},
                {"shadow_mode": True}, {"shadow_mode": True, "private_mode": True},
                {"shadow_mode": True, "posting_mode": "dm"},
                {"pick_delivery": "dm", "posting_mode": "invalid", "private_mode": True},
                {"pick_delivery": "invalid", "posting_mode": "dm", "private_mode": True}):
        show_mode(cfg)
    for config in ({}, {"scheduler": "invalid"}, {"health_checks": "invalid"},
                   {"health_checks": [], "scheduler": {"contest_state_required": True}},
                   {"health_checks": []}, {"scheduler": {"contest_state_required": "false"}},
                   {"health_checks": {"contest_state_expected": "false"}}):
        flags = []
        for get_required in (scheduler._contest_state_required, orchestrator._contest_state_required):
            try:
                flags.append(get_required(config))
            except AttributeError:
                flags.append("AttributeError")
        print("required", config, flags, entry_intent.check_entry_intent(config))


def show_mode(sched):
    for intent in ({}, {"entry_intent": "enter"}):
        config = {"scheduler": sched | intent}
        check = entry_intent.check_entry_intent(config)
        print("mode", config, check.delivery, check.problems)


def scheduler_and_orchestrator(sandbox):
    real_load = orchestrator.load_config
    real_read = contest_state.load_decision_streak_state
    config_file = sandbox / "config.toml"
    assert not (sandbox / "heartbeat.json").exists()
    dm_config = '[scheduler]\npick_delivery = "dm"\n'
    config_file.write_text(config_text(sandbox, dm_config))
    with patch.object(orchestrator, "load_config", wraps=real_load) as loader, \
         patch.object(scheduler, "run_day") as day:
        for dry in (False, True):
            result = RUNNER.invoke(cli, ["schedule", "--config", str(config_file), "--date", DATE]
                                   + (["--dry-run"] if dry else []))
            assert result.exit_code == 0
            assert day.call_args.kwargs == dict(date=DATE, config=real_load(config_file), dry_run=dry)
        assert loader.call_count == 2
    for games in (GAMES, []):
        with patch.object(scheduler, "fetch_schedule", return_value=games) as fetch, \
             patch.object(scheduler, "_now_et", return_value=NOW), \
             patch.object(scheduler, "notify_ready") as ready, \
             patch.object(scheduler, "notify_watchdog") as watchdog, \
             patch.object(scheduler, "_next_day_wakeup", return_value=NOW), \
             patch.object(scheduler, "_idle_until_next_wakeup") as idle, \
             patch.object(scheduler, "load_state") as state, \
             patch.object(scheduler, "run_single_check") as check:
            result = RUNNER.invoke(cli, ["schedule", "--config", str(config_file), "--date", DATE, "--dry-run"])
            info("schedule dry " + ("games" if games else "off day"), result)
            heartbeat = json.loads((sandbox / "heartbeat.json").read_text())
            assert result.exit_code == 0 and ready.called and watchdog.called and fetch.called
            assert idle.called == (not games) and not state.called and not check.called
            print("dry heartbeat", heartbeat["state"], "idle_called", idle.called)
    for malformed in (False, True):
        config_file.write_text('[scheduler\n' if malformed else config_text(sandbox, dm_config))
        with patch.object(orchestrator, "orchestrate", return_value=False) as core:
            result = RUNNER.invoke(cli, ["orchestrate", "--config", str(config_file), "--date", DATE])
            assert result.exit_code == 1 and core.call_args.args == (config_file, DATE)
    result = RUNNER.invoke(cli, ["orchestrate", "--config", str(config_file), "--date", DATE])
    info("orchestrate malformed real", result)
    assert type(result.exception).__name__ == "TOMLDecodeError"
    config_file.write_text(config_text(sandbox, '[scheduler]\npick_delivery = "private"\nentry_intent = "research"\n'))
    stale_state(sandbox / "picks")
    reads = []

    def read(path, **kwargs):
        state = real_read(path, now=NOW, **kwargs)
        reads.append((kwargs.get("require_contest_state", False), state.streak))
        return state

    with patch.object(orchestrator, "load_config", wraps=real_load) as loader, \
         patch.object(orchestrator, "run_cascade", return_value=(FRAME, "probe")) as cascade, \
         patch.object(picks, "get_game_statuses_detailed", return_value={}), \
         patch.object(strategy, "select_pick", return_value=selected("double")), \
         patch.object(posting, "should_post_now", return_value=True), \
         patch.object(posting, "post_to_bluesky", return_value="probe:post") as public, \
         patch.object(contest_state, "load_decision_streak_state", side_effect=read), \
         patch.object(dm, "send_dm") as direct:
        result = RUNNER.invoke(cli, ["orchestrate", "--config", str(config_file), "--date", DATE])
        info("orchestrate private/research against 2026 auto", result)
        assert result.exit_code == 0 and public.call_count == 1 and not direct.called and cascade.called
        assert reads == [(False, 7), (False, 7)] and loader.call_count == 1
        print("orchestrator gate inputs", reads, "PUBLIC", public.call_args.args)


def run_branches(sandbox):
    real_read = contest_state.load_decision_streak_state
    real_save = picks.save_pick
    for kind, reject_at in (("skip", 0), ("locked", 0), ("posted", 0), ("double", 0),
                            ("double", 1), ("locked", 2), ("double", 2), ("dry", 1)):
        directory = sandbox / f"{kind}-{reject_at}"
        pick_dir = directory / "picks"
        stale_state(pick_dir)
        model_dir = directory / "models"
        events = []

        def read(path, **kwargs):
            events.append("read")
            return real_read(path, now=NOW, require_contest_state=(events.count("read") == reject_at))

        def pipeline(*args, save_blend_path, **kwargs):
            events.append("predict")
            save_blend_path.parent.mkdir()
            save_blend_path.write_text("captured blend")
            return FRAME

        def save(daily, path):
            events.append("save")
            return real_save(daily, path)

        def post(text):
            events.append("post")
            return "probe:post"

        with patch.object(predict, "run_pipeline", side_effect=pipeline), \
             patch.object(picks, "get_game_statuses_detailed", return_value={}), \
             patch.object(picks, "save_pick", side_effect=save), \
             patch.object(strategy, "select_pick", return_value=selected(kind)), \
             patch.object(contest_state, "load_decision_streak_state", side_effect=read), \
             patch.object(posting, "post_to_bluesky", side_effect=post), \
             patch.object(posting, "should_post_now", return_value=True):
            result = RUNNER.invoke(cli, ["run", "--date", DATE, "--picks-dir", str(pick_dir),
                                         "--models-dir", str(model_dir)] + (["--dry-run"] if kind == "dry" else []))
        print("run", kind, reject_at, result.exit_code, type(result.exception).__name__, events,
              "saved", (pick_dir / f"{DATE}.json").exists(), "cache", (model_dir / f"blend_{DATE}.pkl").exists(),
              [line for line in result.stdout.splitlines() if line.startswith(("Pick", "Skip", " "))])
        if reject_at and kind != "dry":
            assert isinstance(result.exception, contest_state.ContestStateError) and "post" not in events
            assert ("save" in events) == (kind == "double" and reject_at == 2)


def cached_pick(sandbox):
    pick_dir = sandbox / "picks"
    account = stale_state(pick_dir)
    real_read = contest_state.load_decision_streak_state
    config = dict(orchestrator=dict(picks_dir=str(pick_dir), heartbeat_path=str(sandbox / "heartbeat")), tiers=[],
                  scheduler=dict(pick_delivery="dm", contest_state_required=True),
                  bluesky=dict(dm_recipient="probe.invalid"))
    reads = []

    def read(path, **kwargs):
        required = kwargs.get("require_contest_state", False)
        reads.append(required)
        return real_read(path, now=NOW, **kwargs)

    with patch.object(predict, "run_pipeline", return_value=FRAME), \
         patch.object(picks, "get_game_statuses_detailed", return_value={}), \
         patch.object(strategy, "select_pick", return_value=selected("double")) as choose, \
         patch.object(contest_state, "load_decision_streak_state", side_effect=read), \
         patch.object(orchestrator, "run_cascade", return_value=(None, None)), \
         patch.object(scheduler, "heartbeat_watchdog", side_effect=lambda *a, **k: nullcontext()), \
         patch.object(scheduler, "_now_et", return_value=NOW), \
         patch.object(scheduler, "_delivery_game_statuses", return_value={}), \
         patch.object(scheduler, "_trigger_live_forward_capture_on_lock"), \
         patch.object(scheduler, "_alert_contest_state_failure") as alert, \
         patch.object(dm, "send_dm", return_value="probe:dm") as direct:
        preview = RUNNER.invoke(cli, ["preview", "--date", DATE, "--picks-dir", str(pick_dir),
                                     "--models-dir", str(sandbox / "models")])
        assert preview.exit_code == 0, preview.exception
        daily = picks.load_pick(DATE, pick_dir)
        refresh = scheduler._refresh_pick_at_fallback_decision(config, DATE, daily, .03)
        assert refresh.daily is daily and daily.double_down is not None
        def state():
            return scheduler.SchedulerState(date=DATE, schedule_fetched_at=NOW.isoformat(), games=[],
                                            confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                                            pick_locked_at=None, result_status=None, next_wakeup=None)

        assert not scheduler._deliver_and_lock_pick(daily, config, pick_dir, state(), DATE, "fallback")
        assert alert.called and not direct.called
        (account / "contest_streak.manual.json").write_text(json.dumps(dict(
            schema_version=contest_state.DATED_MANUAL_SCHEMA, active_streak=56, best_streak=56,
            effective_date=DATE, season=2027, expires_at=DATE + "T23:59:59.999999-04:00",
            recorded_at=NOW.isoformat(), saver_state="used", source="manual_screenshot")))
        assert scheduler._deliver_and_lock_pick(daily, config, pick_dir, state(), DATE, "fallback")
        assert choose.call_count == 1 and direct.call_count == 1 and reads == [False, True, True]
        assert "Probe Second" in direct.call_args.args[1]
        print("cached preview double sent after accepted 56 observation without selection", reads, direct.call_args.args)
        assert scheduler._deliver_and_lock_pick(daily, config, pick_dir, state(), DATE, "restart")
        assert direct.call_count == 1 and reads == [False, True, True]


def scheduler_restart(sandbox):
    game_clock = datetime(2027, 3, 29, 22, 42, tzinfo=timezone.utc)
    real_read = contest_state.load_decision_streak_state
    for mode, required in (("dm", False), ("public", False), ("dm", True), ("private", False)):
        case = sandbox / f"{mode}-{required}"
        pick_dir = case / "picks"
        stale_state(pick_dir)
        daily = selected("double").pick_result.daily
        picks.save_pick(daily, pick_dir)
        scheduler.save_state(scheduler.SchedulerState(
            date=DATE, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[], runs_completed=[],
            pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None), pick_dir)
        cfg = dict(orchestrator=dict(picks_dir=str(pick_dir), heartbeat_path=str(case / "hb")),
                   bluesky=dict(dm_recipient="probe.invalid"), tiers=[], health_checks=dict(enabled=False),
                   scheduler=dict(pick_delivery=mode, contest_state_required=required))
        with patch.object(scheduler, "_now_et", return_value=game_clock), \
             patch.object(scheduler, "fetch_schedule", return_value=GAMES), \
             patch.object(scheduler, "plan_run_times", return_value=[]), \
             patch.object(scheduler, "_next_day_wakeup", return_value=game_clock), \
             patch.object(scheduler, "_watchdog_ping_sleep"), patch.object(scheduler, "_idle_until_next_wakeup"), \
             patch.object(scheduler, "heartbeat_watchdog", return_value=nullcontext()), \
             patch.object(scheduler, "notify_ready"), patch.object(scheduler, "notify_watchdog"), \
             patch.object(scheduler, "_delivery_game_statuses", return_value={}), \
             patch.object(scheduler, "_maybe_alert_missed_pick"), \
             patch.object(scheduler, "_alert_contest_state_failure") as alert, \
             patch.object(scheduler, "_trigger_live_forward_capture_on_lock"), \
             patch.object(scheduler, "run_result_polling", return_value="probe"), \
             patch.object(orchestrator, "run_cascade", return_value=(None, None)), \
             patch.object(contest_state, "load_decision_streak_state",
                          side_effect=lambda path, **kwargs: real_read(path, now=game_clock, **kwargs)) as read, \
             patch.object(dm, "send_dm", return_value="probe:dm") as direct, \
             patch.object(posting, "post_to_bluesky", return_value="probe:post") as public:
            scheduler.run_day(DATE, cfg)
            assert direct.called == (mode == "dm" and not required)
            assert public.called == (mode == "public")
            assert alert.called == required
            assert read.call_count == (mode != "private")
            print("restart", mode, required, "reads", read.call_args_list, "DMs", direct.call_count,
                  "posts", public.call_count, "refusals", alert.call_count)
        if mode == "private":
            private_entry_check(pick_dir)


def private_entry_check(pick_dir):
    import bts.cli as cli_module
    from bts import contest_fetch
    from bts.leaderboard import auth

    fake_session = SimpleNamespace(user_id=42, username="probe-owner", xsid="probe-xsid")
    with patch.object(auth, "load_session_cookies", return_value={}), \
         patch.object(auth, "extract_uid", return_value="42"), \
         patch.object(auth, "fetch_login_session", return_value=fake_session), \
         patch.object(contest_fetch, "fetch_profile", return_value={}), \
         patch.object(contest_fetch, "fetch_pending_predictions", return_value=[]), \
         patch.object(cli_module, "_fetch_rounds", return_value={}), \
         patch.object(cli_module, "_fetch_bts_to_mlb", return_value={}), \
         patch.object(dm, "send_dm", return_value="probe:dm") as direct:
        result = RUNNER.invoke(cli, ["check-pick-entered", "--picks-dir", str(pick_dir),
                                     "--expected-username", "probe-owner", "--dm-recipient", "probe.invalid",
                                     "--now-et", DATE + "T18:43:00"])
        info("entry checker after private lock", result)
        assert result.exit_code == 1 and direct.call_count == 1
        text = direct.call_args.args[1]
        assert "Probe Batter + DD Probe Second" in text and "Fix it now!" in text
        print("private lock entry alert", text)


def outcome_and_monitor(sandbox):
    import runpy
    cron = runpy.run_path(str(ROOT / "scripts/check_heartbeat.py"))
    hb = sandbox / "heartbeat.json"
    assert cron["is_stale"](hb, NOW)[0] and web.health_check(hb)[0] == 503
    hb.write_text(json.dumps(dict(timestamp=NOW.isoformat(), state="sleeping",
                                  sleeping_until=(NOW + timedelta(hours=2)).isoformat())))
    assert not cron["is_stale"](hb, NOW + timedelta(hours=1))[0]
    assert cron["is_stale"](hb, NOW + timedelta(hours=2, minutes=11))[0]
    assert cron["assess_churn"]([dict(ts=NOW.isoformat(), n=7)], 17, NOW + timedelta(minutes=5))[0]
    assert not cron["assess_churn"]([dict(ts=NOW.isoformat(), n=7)], None, NOW + timedelta(minutes=5))[0]
    script = (ROOT / ".github/workflows/deploy.yml").read_text()
    restart = next(line.strip().replace("$(id -u bts)", "1000") for line in script.splitlines()
                   if "systemctl --user restart bts-scheduler bts-dashboard" in line)
    local_branch = "set -e\nsudo() { return \"$1\"; }\n" + restart.replace("sudo -u bts", 'sudo "$1"')
    local_branch += "\nprintf 'canary reached\\n'"
    for exit_status in (0, 1):
        out = subprocess.run(["/bin/bash", "-c", local_branch, "stub", str(exit_status)],
                             capture_output=True, text=True, check=False, timeout=5)
        assert out.returncode == exit_status and (out.stdout == "canary reached\n") == (exit_status == 0)
        print("deploy initial restart stub", exit_status, out.returncode, repr(out.stdout))
    cfg = sandbox / "outcome.toml"
    cfg.write_text('[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\n')
    baseline = entry_intent.check_entry_intent
    argv = ["day-outcome", "--date", DATE, "--config", str(cfg), "--picks-dir", str(sandbox / "picks"),
            "--dm-recipient", "probe.invalid"]
    with patch.object(scheduler, "fetch_schedule", return_value=GAMES), \
         patch.object(dm, "send_dm", return_value="probe:dm") as direct:
        base_result = RUNNER.invoke(cli, argv)
        with patch.object(entry_intent, "check_entry_intent",
                          return_value=entry_intent.IntentCheck("enter", "dm", ("probe live gate",))):
            after = RUNNER.invoke(cli, argv)
        assert baseline(orchestrator.load_config(cfg)).ok
        assert [json.loads(r.stdout)["state"] for r in (base_result, after)] == ["no_decision", "intent_unknown"]
        print("day-outcome before/with appended problem", base_result.stdout.strip(), after.stdout.strip(),
              [call.args[1] for call in direct.call_args_list])


def cron_probe(sandbox):
    faux_home = sandbox / "faux-home"
    fake_uv = faux_home / ".local/bin/uv"
    fake_uv.parent.mkdir(parents=True)
    fake_uv.write_text(f'#!/bin/sh\n[ "$1 $2 $3" = "run bts entry-intent" ] || exit 97\n'
                       f'shift 2\nexec "{sys.executable}" "{__file__}" --cron-cli "$@"\n')
    fake_uv.chmod(0o700)
    project = faux_home / "projects/bts"
    project.mkdir(parents=True)
    (project / ".env").write_text("")
    shims = sandbox / "shims"
    shims.mkdir()
    fake_cron = shims / "crontab"
    fake_cron.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$PROBE_CRON_CALLS"\n'
                         'case "$1" in -l) cat "$PROBE_CRONTAB" ;; -) cat > "$PROBE_CRONTAB" ;; *) exit 98 ;; esac\n')
    fake_cron.chmod(0o700)
    calls = sandbox / "calls"
    crab = sandbox / "crontab"
    crab.write_text("# baseline BTS-HETZNER\n")
    config = faux_home / ".bts-orchestrator.toml"
    for mode in ("base", "injected"):
        env = {"HOME": str(faux_home), "PATH": f"{shims}:/usr/bin:/bin",
               "HEALTHCHECKS_PING_URL": "https://monitor.invalid/probe", "PYTHONDONTWRITEBYTECODE": "1",
               "PROBE_CRONTAB": str(crab), "PROBE_CRON_CALLS": str(calls), "PROBE_MODE": mode}
        for intent, delivery in (("enter", "dm"), ("research", "private"), ("enter", "private")):
            config.write_text(f'[scheduler]\nentry_intent = "{intent}"\npick_delivery = "{delivery}"\n')
            for action in ("show", "install"):
                calls.write_text("")
                before = crab.read_bytes()
                result = subprocess.run(["/bin/bash", str(ROOT / "scripts/cron-setup-hetzner.sh"), action],
                                        env=env, capture_output=True, text=True, check=False, timeout=30)
                print("cron", mode, intent, delivery, action, result.returncode,
                      repr(result.stderr), calls.read_text().splitlines(), "changed", crab.read_bytes() != before)
                if result.returncode:
                    assert not calls.read_text() and crab.read_bytes() == before
        if mode == "base":
            for script in ("cron-setup-pi5.sh", "cron-setup.sh"):
                result = subprocess.run(["/bin/bash", str(ROOT / "scripts" / script), "show"], env=env,
                                        capture_output=True, text=True, check=True, timeout=30)
                print(script, [line for line in result.stdout.splitlines() if "run bts " in line])


def example():
    path = ROOT / "config/orchestrator.example.toml"
    config = orchestrator.load_config(path)
    print("example", scheduler._pick_delivery_mode(config), scheduler._contest_state_required(config),
          entry_intent.check_entry_intent(config))
    config["scheduler"].update(contest_state_required=True, entry_intent="enter")
    config["health_checks"].update(contest_state_expected=True)
    assert entry_intent.check_entry_intent(config).ok and scheduler._contest_state_required(config)


def cron_cli():
    with ExitStack() as stack:
        if os.environ.get("PROBE_MODE") == "injected":
            original = entry_intent.check_entry_intent

            def appended(config):
                checked = original(config)
                if checked.intent == "enter" or checked.delivery in ("dm", "public"):
                    return entry_intent.IntentCheck(checked.intent, checked.delivery, checked.problems + ("probe live gate",))
                return checked

            stack.enter_context(patch.object(entry_intent, "check_entry_intent", side_effect=appended))
        cli.main(args=sys.argv[2:], prog_name="bts")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cron-cli"]:
        cron_cli()
    else:
        delivery_matrix()
        example()
        with TemporaryDirectory(dir=SCRATCH) as temporary, ExitStack() as mocks:
            for module, attribute in ((dm, "send_dm"), (posting, "post_to_bluesky"), (posting, "reply_to_bluesky"),
                                      (scheduler, "fetch_schedule"), (picks, "get_game_statuses_detailed"),
                                      (orchestrator, "run_cascade"), (predict, "run_pipeline")):
                mocks.enter_context(patch.object(module, attribute, side_effect=AssertionError(f"uncaptured {attribute}")))
            for probe in (scheduler_and_orchestrator, run_branches, cached_pick, scheduler_restart,
                          outcome_and_monitor, cron_probe):
                sandbox = Path(temporary) / probe.__name__
                sandbox.mkdir()
                probe(sandbox)
