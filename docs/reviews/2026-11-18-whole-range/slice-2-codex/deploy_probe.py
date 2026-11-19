"""Offline CLI startup, deploy shell control flow, and cross-version state replay."""

import argparse
from contextlib import redirect_stderr
from datetime import timedelta
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from textwrap import dedent
from types import SimpleNamespace
from unittest.mock import patch

import day_probe as probe
import offline_fixtures as fixture


class StoppedForRollback(Exception):
    pass


def channels(world):
    return [event for event in world.events if event[1] in {"dm", "post", "reply", "READY", "idle"}]


def invoke_schedule(root, config):
    return probe.invoke(["schedule", "--config", str(probe.toml(root / "orchestrator.toml", config))])


def startup(root):
    output = {}
    test_cases = {
        "restored_offseason": ("2027-03-10", "no_games", "private", "research", False, 1),
        "restored_playable": (probe.MID, "normal", "private", "research", False, 1),
        "dm_research_no_flags": (probe.MID, "normal", "dm", "research", False, 3),
        "dm_enter_no_flags": (probe.MID, "normal", "dm", "enter", False, 3),
        "required_no_switch": (probe.MID, "normal", "dm", "enter", True, 3),
    }
    for name, spec in test_cases.items():
        day, kind, delivery, intent, required, restarts = spec
        case = root / name
        case.mkdir()
        picks = case / "data/picks"
        fixture.seed(picks, day, streak=18, season=2026, old=True, switch=False)
        fixture.history(picks, "2026-09-15", 1)
        probe.write(picks / "2026-09-14/scheduler_state.json",
                    {"date": "2026-09-14", "pick_locked": True})
        probe.write(case / "data/health_state/health_dm_delivery_status.json",
                    {"status": "sent", "sent_day": "2026-09-14", "sent_sources": ["CRITICAL:contest_state"]})
        config = probe.config_for(picks, delivery=delivery, intent=intent, required=required)
        world = probe.World(day, kind)
        launches = []
        with fixture.installed(world, case, production=True):
            for index in range(restarts):
                world.now = probe.at(day) + timedelta(seconds=30 * index)
                world.events.clear()
                result = invoke_schedule(case, config)
                launches.append({"result": result, "events": channels(world), "files": sorted(probe.snapshot(case))})
            output[name] = {"launches": launches, "state": probe.snapshot(case)}
    return output


def prepare_rollback(root):
    from bts import picks as pick_io

    evidence = {}
    for name in ("committed", "manual_pending", "manual_no_auto", "correction_crash"):
        case = root / name
        case.mkdir()
        world = probe.World(probe.MID, "cascade_flake" if name == "manual_pending" else "normal")
        picks = case / "data/picks"
        fixture.seed(picks, world.day, age=17, streak=12)
        probe.write(picks / "streak.json", {
            "streak": 12, "saver_available": True, "season": 2027, "applied_dates": ["2027-07-14"],
        })
        if name == "manual_no_auto":
            (picks / "account_state/contest_streak.json").unlink()
        config = probe.config_for(picks)
        real_sleep = world.sleep

        def interrupt(seconds):
            daily = pick_io.load_pick(world.day, picks)
            if daily is not None:
                raise StoppedForRollback()
            real_sleep(seconds)

        if name != "manual_no_auto":
            world.sleep = interrupt
        with fixture.installed(world, case, production=name != "manual_pending"):
            if name in {"manual_pending", "manual_no_auto"}:
                created = probe.invoke(["set-contest-streak", "--date", world.day, "--streak", "52",
                                        "--best-streak", "54", "--saver", "used", "--picks-dir", str(picks)])
                assert created["exit"] == 0, created
            if name == "manual_no_auto":
                evidence[name] = {"written": sorted(probe.snapshot(case))}
                continue
            result = invoke_schedule(case, config)
            assert result["exception"] == "StoppedForRollback()", result
            if name == "correction_crash":
                # Recreate the durable boundary between applying the date and
                # saving its terminal result (CLI/scheduler write in that order).
                pick_io.update_streak([True], picks, date=world.day)
            evidence[name] = {"result": result, "events": channels(world), "files": probe.snapshot(case)}
    return evidence


def run_rollback(root):
    from bts import contest_state, daily_decision, picks as pick_io, scheduler
    from bts.health import alert

    evidence = {}
    for name in ("committed", "manual_pending", "manual_no_auto", "correction_crash"):
        case = root / name
        picks = case / "data/picks"
        world = probe.World(probe.MID, "normal")
        world.now = probe.at(world.day, 15, 10)
        config = probe.config_for(picks)
        with fixture.installed(world, case, production=True):
            before = pick_io.load_pick(world.day, picks)
            state = scheduler.load_state(world.day, picks)
            decision = daily_decision.load_decision(world.day, picks)
            selected = contest_state.load_contest_streak_state(picks)
            live = contest_state.load_decision_streak_state(picks, require_contest_state=True)
            result = invoke_schedule(case, config) if name != "manual_no_auto" else None
            next_observation = None
            if name == "manual_no_auto":
                world.now += timedelta(days=1)
                next_observation = contest_state.load_decision_streak_state(picks, require_contest_state=True)
            report = {
                "loaded_pick": before.pick.batter_name if before else None,
                "loaded_state": {"locked": state.pick_locked} if state else None,
                "loaded_decision": decision,
                "selected": selected.path.name, "source_date": str(selected.source_date),
                "streak": live.streak, "saver": live.saver_available, "best_status": live.best_status,
                "next_day_selected_streak": next_observation.streak if next_observation else None,
                "result": result, "events": channels(world), "files": probe.snapshot(case),
            }
            if name == "committed":
                status = case / "data/health_state/contest_stale_reading_dm_delivery_status.json"
                raw = probe.read(status)
                assert raw and raw["status"] == "sent", raw
                warning = alert.Alert("WARN", "contest_state", "same stale incident",
                                      incident_key="contest_state:stale_reading")
                report["base_status_dedup"] = alert.dispatch_dm_for_health_alerts(
                    [], "probe.invalid", warn_attention=[warning], status_path=status)
                tomorrow = (probe.at(world.day) + timedelta(days=1)).date().isoformat()
                next_world = probe.World(tomorrow, "normal")
                with fixture.installed(next_world, case, production=True):
                    report["next_schedule"] = invoke_schedule(case, config)
                report["next_events"] = channels(next_world)
                report["next_decision"] = daily_decision.load_decision(tomorrow, picks)
            evidence[name] = report
    return evidence


def forwarded(root):
    from bts import picks

    return {name: {"unscoped": picks.load_streak(root / name / "data/picks"),
                   "scoped_2027": picks.load_streak(root / name / "data/picks", season=2027)}
            for name in ("committed", "correction_crash")}


def rollforward_fetch(root, inspect):
    import bts.cli as cli_module
    from bts import contest_fetch, contest_state
    from bts.leaderboard import auth

    case = root / "committed"
    world = probe.World("2027-07-16", "normal")
    picks = case / "data/picks"
    path = picks / "account_state/contest_streak.json"
    before = probe.read(path)
    with fixture.installed(world, case, production=True):
        if inspect:
            try:
                contest_state.load_decision_streak_state(picks, require_contest_state=True)
            except contest_state.ContestStateError as exc:
                return {"live_refusal": str(exc), "observation": before}
            raise AssertionError("expected refusal on the old writer's file")
        sess = SimpleNamespace(user_id=100, username="stonehengee", xsid="fixture")
        profile = {"activeStreak": 13, "seasonBestStreak": 18,
                   "predictions": [{"roundId": 42, "result": "hit"}]}
        with patch.object(auth, "load_session_cookies", return_value={}), \
                patch.object(auth, "extract_uid", return_value=100), \
                patch.object(auth, "fetch_login_session", return_value=sess), \
                patch.object(contest_fetch, "fetch_profile", return_value=profile), \
                patch.object(cli_module, "_fetch_rounds", return_value={42: world.now.date() - timedelta(days=1)}):
            outcome = probe.invoke(["fetch-contest-streak", "--picks-dir", str(picks),
                                    "--expected-username", "stonehengee"])
        assert outcome["exit"] == 0, outcome
        return {"before": before, "after": probe.read(path), "result": outcome}


def read_statuses(root):
    from bts.health import alert

    paths = Path(__file__).resolve().parent
    startup_log = json.loads((paths / "deploy-startup.log").read_text())["result"]
    prep_log = json.loads((paths / "deploy-prepare.log").read_text())["result"]
    writes = dict(startup_log["required_no_switch"]["state"])
    writes.update(prep_log["committed"]["files"])
    world = probe.World(probe.MID, "normal")
    results = {}
    with fixture.installed(world, root):
        for suffix in ("health_dm_delivery_status.json", "contest_morning_check_dm_delivery_status.json",
                       "contest_stale_reading_dm_delivery_status.json"):
            rel = f"data/health_state/{suffix}"
            raw = writes[rel]
            path = root / rel
            probe.write(path, raw)
            assert raw["status"] == "sent", raw
            kind, key = raw["sent_sources"][0].split(":", 1)
            message = alert.Alert("CRITICAL" if kind == "CRITICAL" else "WARN",
                                  "contest_state", "repeated test incident", incident_key=key)
            attempted = alert.dispatch_dm_for_health_alerts(
                [message] if kind == "CRITICAL" else [], "probe.invalid",
                warn_attention=[message] if kind != "CRITICAL" else [], status_path=path)
            results[suffix] = {"attempted": attempted, "unchanged": probe.read(path) == raw}
    return results


def shim():
    executable = Path(sys.argv[2]).name
    args = sys.argv[3:]
    root = Path(os.environ["WR2_DEPLOY_ROOT"])
    scenario = os.environ["WR2_DEPLOY_SCENARIO"]
    with (root / "calls.jsonl").open("a") as stream:
        stream.write(json.dumps([executable, *args]) + "\n")
    if executable == "sleep":
        return 0
    if executable == "id":
        assert args == ["-u", "bts"]
        print("1000")
        return 0
    if executable == "curl":
        print("503" if scenario == "dashboard_bad" else "200")
        return 0
    if executable == "sudo":
        pid_state = root / "current_sha"
        if "git" in args and "rev-parse" in args:
            print(pid_state.read_text())
            return 0
        if "git" in args and "reset" in args:
            pid_state.write_text(args[-1])
            return 0
        if "git" in args or "tailscale" in args:
            return 0
        if "sync" in args:
            return int(scenario == "sync_failed" and pid_state.read_text() == "candidate")
        if "systemctl" in args and "restart" in args:
            return int(scenario == "restart_refused" and pid_state.read_text() == "candidate")
        if "systemctl" in args and "is-active" in args:
            if "--quiet" not in args:
                print("active")
            return 0
        if "bash" in args and ".last_deploy_iso" in args[-1]:
            return 0
    raise AssertionError((executable, args))


def workflow(root):
    workflow_file = fixture.REPO / ".github/workflows/deploy.yml"
    payload = dedent(workflow_file.read_text().split("          script: |\n", 1)[1])
    assert not Path("/etc/systemd/system/bts-scheduler.service").exists()
    results = {}
    for name in ("accepted_but_no_switch", "restart_refused", "sync_failed", "dashboard_bad"):
        case = root / name
        case.mkdir()
        bin_path = case / "bin"
        bin_path.mkdir()
        for executable in ("sudo", "curl", "sleep", "id"):
            path = bin_path / executable
            path.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{Path(__file__).resolve()}" --shim "$0" "$@"\n')
            path.chmod(0o700)
        (case / "current_sha").write_text("619abda7")
        env = dict(os.environ, PATH=f"{bin_path}:/usr/bin:/bin",
                   TARGET_SHA="candidate", WR2_DEPLOY_ROOT=str(case), WR2_DEPLOY_SCENARIO=name)
        script = 'cd() { builtin cd "$WR2_DEPLOY_ROOT"; }\n' + payload
        result = subprocess.run(["/bin/bash", "-c", script], cwd=case, env=env, capture_output=True,
                                text=True, timeout=8)
        results[name] = {"exit": result.returncode, "out": result.stdout, "err": result.stderr,
                         "sha": (case / "current_sha").read_text(),
                         "calls": [json.loads(line) for line in (case / "calls.jsonl").read_text().splitlines()]}
    return results


def main():
    if sys.argv[1:2] == ["--shim"]:
        raise SystemExit(shim())
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["startup", "prepare", "rollback", "forwarded", "workflow",
                                          "old-fetch", "new-read", "statuses"])
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    journal = StringIO()
    with redirect_stderr(journal):
        if args.phase in {"prepare", "rollback", "forwarded", "old-fetch", "new-read"}:
            if args.root is None:
                parser.error("--root is required for the cross-revision run")
            task = {"prepare": prepare_rollback, "rollback": run_rollback, "forwarded": forwarded,
                    "old-fetch": lambda root: rollforward_fetch(root, inspect=False),
                    "new-read": lambda root: rollforward_fetch(root, inspect=True)}[args.phase]
            result = task(args.root.resolve())
        else:
            with TemporaryDirectory(prefix=f"{args.phase}-", dir=Path(__file__).resolve().parent) as temp:
                task = {"startup": startup, "workflow": workflow, "statuses": read_statuses}[args.phase]
                result = task(Path(temp))
    print(json.dumps({"result": result, "journal": journal.getvalue()}, indent=2))


if __name__ == "__main__":
    main()
