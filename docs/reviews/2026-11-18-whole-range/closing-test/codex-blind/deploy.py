"""Rehearse old and partial configurations, then read current files in BASE."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import textwrap

from bts import contest_state, daily_decision, picks, scheduler, strategy
from bts import entry_receipt
from bts.health import alert
from bts.saver_state import load_saver_state

from midseason import automatic
from offline import Interrupted, REVIEW, ROOT, World, at
from opening import OPEN, SNAPSHOT

HANDOFF = REVIEW / "synthetic-rollback-age-warning"
MID = "2027-07-15"


def copied_snapshot(w):
    shutil.copytree(SNAPSHOT / "data/picks", w.directory, dirs_exist_ok=True)
    shutil.copytree(SNAPSHOT / "data/health_state", w.directory.parent / "health_state", dirs_exist_ok=True)


def startup():
    with World("2027-03-03") as w:
        copied_snapshot(w)
        w.games[w.day] = []
        w.config["scheduler"].update(pick_delivery="private", entry_intent="research", contest_state_required=False)
        w.config["health_checks"]["contest_state_expected"] = False
        w.run(via_cli=True)
        w.snapshot()
        w.config["scheduler"].update(pick_delivery="dm", entry_intent="enter", contest_state_required=True)
        w.config["health_checks"]["contest_state_expected"] = True
        w.run(via_cli=True)
        w.snapshot()
        w.dump("deploy-preseason-offday")

    for mode, intent, required, expected in (
        ("private", "research", False, False),
        ("dm", "research", False, False),
        ("dm", "enter", False, False),
        ("dm", "enter", True, False),
        ("dm", "enter", False, True),
        ("dm", "enter", True, True),
    ):
        with World(OPEN) as w:
            copied_snapshot(w)
            w.config["scheduler"].update(pick_delivery=mode, entry_intent=intent, contest_state_required=required)
            w.config["health_checks"].update(contest_state_expected=expected)
            w.invoke("entry-intent", "--config", str(w.toml()))
            w.run(via_cli=True)
            w.snapshot()
            if not (required or expected) and mode == "dm":
                for minute in (0, 1, 2):
                    w.now = at(OPEN, f"10:0{minute}:30")
                    w.run(via_cli=True)
                w.run(via_cli=True, dry=True)
                w.record("heartbeat_written", (w.root / "data/.heartbeat").exists())
            if mode == "private":
                # Enable DM on the same slate after a research lock exists.
                w.now = at(OPEN, "18:35")
                w.config["scheduler"].update(pick_delivery="dm", entry_intent="enter", contest_state_required=True)
                w.config["health_checks"]["contest_state_expected"] = True
                w.switch("revoke", "--reason", "activate after restored restart")
                w.dated()
                w.run(via_cli=True)
                w.now = at(OPEN, "18:45")
                w.watchdog()
                w.snapshot()
            w.dump(f"deploy-{mode}-{intent}-{int(required)}{int(expected)}")


def head_write():
    HANDOFF.mkdir(exist_ok=True)
    with World(MID, HANDOFF) as w:
        assert not (w.directory / f"{MID}.json").exists(), "use a fresh synthetic rollback directory"
        w.now = at("2027-07-14", "19:10")
        w.dated(11, 15, "active")
        w.now = at(MID, "01:10")
        automatic(w, 12, 15, False)
        w.now = at(MID)
        def cutover():
            raise Interrupted()
        w.actions = [(at(MID, "18:44"), cutover)]
        try:
            w.run()
        except Interrupted:
            w.record("process_restart", "rollback before grading")
        w.now = at(MID, "18:45")
        w.watchdog()
        alert.dispatch_dm_for_health_alerts(
            [alert.Alert("CRITICAL", "offline", "Fixture health acknowledgement")],
            "offline.test",
            status_path=w.directory.parent / "health_state" / "health_dm_delivery_status.json",
        )
        w.snapshot()
        w.dump("deploy-rollback-before")


def base_read_write():
    with World(MID, HANDOFF) as w:
        w.config["health_checks"]["enabled"] = False
        status_path = w.directory.parent / "health_state" / "health_dm_delivery_status.json"
        dedup_result = alert.dispatch_dm_for_health_alerts(
            [alert.Alert("CRITICAL", "offline", "Fixture health acknowledgement")],
            "offline.test", status_path=status_path)
        w.record("read", {
            "pick": picks.load_pick(MID, w.directory).pick.batter_name,
            "decision": daily_decision.load_decision(MID, w.directory),
            "scheduler": vars(scheduler.load_state(MID, w.directory)),
            "contest": vars(contest_state.load_decision_streak_state(w.directory)),
            "saver": vars(load_saver_state(w.directory, season=2027)),
            "streak": picks.load_streak(w.directory),
            "days_remaining": strategy._days_remaining(MID),
            "entry_receipts": len(entry_receipt.discover(w.directory, MID)),
            "health_status": json.loads(status_path.read_text()),
            "health_dedup_result": dedup_result,
        })
        w.now = at(MID, "18:45")
        w.run(via_cli=True)
        w.snapshot()
        w.profile = {"activeStreak": 13, "seasonBestStreak": 15,
                     "predictions": [{"roundId": 1, "result": "hit"}]}
        w.fetch()
        w.invoke("reconcile")
        w.snapshot()
        w.dump("deploy-rollback-base")


def head_verify():
    with World(MID, HANDOFF) as w:
        w.switch("status")
        w.now = at("2027-07-16", "09:30")
        w.switch("status")
        w.snapshot(MID)
        w.dump("deploy-rollback-head")


SHIM = r"""
import json
import os
import sys
from pathlib import Path

program = Path(sys.argv[0]).name
args = sys.argv[1:]
case = os.environ["WRBLIND_DEPLOY_CASE"]
log = Path(os.environ["WRBLIND_DEPLOY_COMMANDS"])
prior = [json.loads(row) for row in log.read_text().splitlines()] if log.exists() else []
with log.open("a") as f:
    f.write(json.dumps([program, args]) + "\n")
baserev, headrev = "619abda7", "bc143131"
rolled_back = ["sudo", ["-Hu", "bts", "git", "reset", "--hard", baserev]] in prior
if program == "sleep":
    pass
elif program == "id":
    print("9001")
elif program == "curl":
    print("500" if case == "dashboard_failure" else "200")
elif program == "sudo":
    if "rev-parse" in args:
        print(headrev if "--short" in args else baserev)
    elif "sync" in args and not rolled_back and case == "sync_failure":
        sys.exit(42)
    elif "restart" in args and not rolled_back and case == "restart_failure":
        sys.exit(43)
    elif "is-active" in args and args[-1] == "bts-scheduler" and not rolled_back and case in {
        "refusing_scheduler", "clean_scheduler_exit"
    }:
        print("activating")
        sys.exit(3)
    elif "is-active" in args and args[-1] == "bts-scheduler":
        print("active")
    elif "show" in args:
        print({"ActiveState": "activating", "SubState": "auto-restart",
               "Result": "exit-code" if case == "refusing_scheduler" else "success",
               "ExecMainStatus": 2 if case == "refusing_scheduler" else 0}[args[-2]])
    elif "tailscale" in args:
        print("100.64.0.1")
else:
    sys.exit(f"unexpected or prohibited command {program}")
"""


def workflow():
    workflow_file = ROOT / ".github/workflows/deploy.yml"
    raw = workflow_file.read_text().split("          script: |\n", 1)[1]
    for case in ("active_with_no_live_state", "refusing_scheduler", "clean_scheduler_exit",
                 "dashboard_failure", "sync_failure", "restart_failure"):
        with World(OPEN) as w:
            shims = w.root / "bin"
            shims.mkdir()
            for name in ("sudo", "curl", "sleep", "id", "pkill", "rm", "systemctl"):
                shim = shims / name
                shim.write_text(f"#!{sys.executable}\n" + SHIM)
                shim.chmod(0o700)
            body = textwrap.dedent(raw).replace("/home/bts/projects/bts", str(w.root))
            body = body.replace("/etc/systemd/system/", str(w.root / "no-legacy-unit") + "/")
            command_log = w.root / "calls.jsonl"
            result = subprocess.run(
                ["bash", "-c", body], text=True, capture_output=True, check=False,
                env=dict(os.environ, PATH=f"{shims}:{os.environ['PATH']}", TARGET_SHA="bc143131",
                         WRBLIND_DEPLOY_CASE=case, WRBLIND_DEPLOY_COMMANDS=str(command_log)),
            )
            w.record("workflow", case, result.returncode, result.stdout, result.stderr,
                     [json.loads(line) for line in command_log.read_text().splitlines()])
            w.dump(f"deploy-workflow-{case}")


def leaderboard_unit(legacy):
    from bts.leaderboard import cli as leaderboard_cli
    from bts.leaderboard.auth import AuthError

    name = "deploy/systemd/bts-leaderboard.service"
    template = (subprocess.check_output(["git", "show", f"619abda7:{name}"], cwd=ROOT, text=True)
                if legacy else (ROOT / name).read_text())
    continuation = "\\" + "\n"
    joined = template.replace(continuation, " ")
    command = next(line.split("uv run bts ", 1)[1] for line in joined.splitlines() if line.startswith("ExecStart="))
    argv = shlex.split(command)
    with World(OPEN) as w:
        argv[argv.index("--output-dir") + 1] = str(w.root / "data/leaderboard")
        def expired():
            raise AuthError("captured fixture classification")
        w.patch(leaderboard_cli, "load_session_cookies", expired)
        w.invoke(*argv, expect=2)
        w.dump(f"deploy-leaderboard-{'base' if legacy else 'head'}")


if __name__ == "__main__":
    {"startup": startup, "head-write": head_write, "base": base_read_write,
     "head-verify": head_verify, "workflow": workflow,
     "leaderboard-head": lambda: leaderboard_unit(False),
     "leaderboard-base": lambda: leaderboard_unit(True)}[sys.argv[1]]()
