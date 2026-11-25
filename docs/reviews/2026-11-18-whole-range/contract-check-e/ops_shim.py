"""Command stubs for the local deploy and crontab probe. Never call the real services."""
import os
import sys
from pathlib import Path

name = os.environ["WR_E_SHIM_NAME"]
args = sys.argv[1:]
root = Path(os.environ["WR_E_PROBE_ROOT"])
scenario = os.environ.get("WR_E_PROBE_SCENARIO", "")


def run():
    with (root / "commands").open("a") as trace:
        trace.write(f"{name} {' '.join(args)}\n")
    if name == "sleep":
        return 0
    if name == "id":
        print("1001")
    elif name == "curl":
        print("204" if scenario == "pass_204" else "500" if scenario.startswith("fail/") else "200")
    elif name == "sudo":
        args_now = args[2:]
        sha = root / "sha"
        if args_now[:2] == ["git", "rev-parse"]:
            print(sha.read_text()[:7] if "--short" in args_now else sha.read_text())
        elif args_now[:3] == ["git", "reset", "--hard"]:
            if scenario == "fail/reset" and args_now[-1].startswith("619abda7"):
                return 2
            sha.write_text(args_now[-1])
        elif any(arg.endswith("/uv") for arg in args_now):
            if scenario == "sync" or (scenario == "fail/sync" and sha.read_text().startswith("619abda7")):
                return 9
        elif "systemctl" in args_now:
            if "restart" in args_now:
                (root / "services").write_text("restart attempted at " + sha.read_text())
                if scenario == "restart" or (scenario == "fail/restart" and sha.read_text().startswith("619abda7")):
                    return 3
            elif "--quiet" in args_now:
                return int(scenario == "fail/unhealthy" and sha.read_text().startswith("619abda7"))
            elif "is-active" in args_now:
                if scenario == "pass_clean_exit":
                    print("activating")
                    return 3
                print("active")
            elif "show" in args_now:
                print({"ActiveState": "activating", "SubState": "auto-restart", "Result": "success",
                       "ExecMainStatus": "0"}[args_now[args_now.index("-p") + 1]])
        elif "tailscale" in args_now:
            print("127.0.0.1")
        elif args_now[:2] == ["bash", "-c"] and scenario == "stamp":
            return 5
    elif name == "crontab":
        saved = root / "crontab.txt"
        if args == ["-l"]:
            if not saved.exists():
                print("no crontab for bts", file=sys.stderr)
                return 1
            sys.stdout.write(saved.read_text())
        elif args == ["-"]:
            saved.write_text(sys.stdin.read())
        else:
            raise AssertionError(args)
    elif name == "uv":
        assert args[:3] == ["run", "bts", "entry-intent"], args
        from bts.cli import cli
        cli(args[2:], prog_name="bts")
    elif name == "systemctl":
        return 0
    else:
        raise AssertionError(name)
    return 0


sys.exit(run())
