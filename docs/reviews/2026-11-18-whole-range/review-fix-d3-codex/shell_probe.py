"""Run exact deploy literals with additional fault and evidence shims."""

import argparse
import difflib
import importlib.util
import json
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
SPEC = importlib.util.spec_from_file_location("workflow_tests", REPO / "tests/test_deploy_workflow.py")
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]
BASE, = tests.scripts_and_steps((HERE / "base/.github/workflows/deploy.yml").read_text())[0]
SHIM = tests.SHIM


def insert(anchor, addition):
    global SHIM
    assert SHIM.count(anchor) == 1, anchor
    SHIM = SHIM.replace(anchor, addition + anchor)


insert('        if op == "clock" and scenario == "canary-budget"', '''
        if op == "clock" and scenario == "late-signal" and (root / "exhausted-phase").exists():
            print(time.monotonic_ns() // 1_000_000 + 805_000)
            raise SystemExit(0)
        if op == "clock" and scenario == "late-signal" and (root / "next-clock").exists():
            print(time.monotonic_ns() // 1_000_000 + 790_000)
            raise SystemExit(0)
        if op == "stage-record" and " committed " in args[-1] and scenario in {"commit-stage-17", "commit-stage-124"}:
            raise SystemExit(int(scenario.rsplit("-", 1)[1]))
        if op == "read" and scenario == "record-read-126" and (root / "exhausted-phase").exists():
            raise SystemExit(126)
        if op == "marker" and scenario == "marker-changed-before-verification":
            code = code.replace("if bounded(path, 4096)[0] != data:",
                                'pathlib.Path(path).write_bytes(b"overwritten")\\n        if bounded(path, 4096)[0] != data:')
''')
insert('    if command == "systemctl" and args:', '''
    if command == "python3" and len(args) > 3 and args[2] == "stage-record":
        return "stage-" + args[-1].split()[5]
''')
insert('        branch.write_text(args[-1])', '''
        if scenario == "deferred-term":
            os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
            time.sleep(0.25)
            with (root / "calls").open("a") as log:
                print(json.dumps(["reset-mutation-finished"]), file=log)
''')
insert('    if user and args[0] == "restart":', '''
    if user and args[0] == "enable" and scenario == "enable-then-timeout":
        raise SystemExit(124)
''')
insert('    if args[0] == "show":', '''
    if user and args[0] == "show" and "ExecMainStatus" in args and scenario in {"late-signal", "record-read-126"}:
        (root / "exhausted-phase").touch()
        os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
''')
insert('elif command in {"sleep", "uv"}:', '''
elif command == "sleep" and scenario == "late-signal":
    (root / "next-clock").touch()
''')
insert('        for key in (arg for arg in args if arg in states):', '''
        if user and "ControlGroup" in args and scenario == "known-cgroup-id":
            states["ControlGroup"] = "/user.slice/bts-scheduler.service"
''')
insert('    if "--data-raw" in args and scenario.startswith("post-"):', '''
    if "--data-raw" in args and scenario == "curl-url-model":
        url = args[-1]
        urls = [url]
        glob = re.search(r"\\{([^{}]*)\\}", url) if "--globoff" not in args and "-g" not in args else None
        if glob:
            urls = [url[:glob.start()] + word + url[glob.end():] for word in glob[1].split(",")]
        with (root / "calls").open("a") as log:
            for url in urls:
                print(json.dumps(["post-model", url]), file=log)
        print("200" * len(urls), end="")
        raise SystemExit(0)
''')
insert('    if scenario == "contest-refused":', '''
    if scenario == "contest-nul":
        sys.stdout.buffer.write(b"Deploy canary: not a live configuration (delivery private);\\x00 contest state not checked.\\n")
        raise SystemExit(0)
''')


def run(case, script=DEPLOY, *, setup=None, scenario="active", **options):
    result_dir = HERE / "shell-evidence"
    result_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wrd3-", dir=HERE) as scratch:
        box = tests.box.__wrapped__(Path(scratch))
        for shim in (box / "shims").iterdir():
            original = shim.read_text()
            shim.write_text(original.split("\n", 1)[0] + "\n" + SHIM)
        if setup:
            setup(box)
        outcome, calls = tests.run_box_script(box, script, scenario, **options)
        read = lambda path: (box / path).read_text() if (box / path).is_file() else None
        evidence = {
            "status": outcome.returncode, "stdout": outcome.stdout, "stderr": outcome.stderr,
            "calls": calls, "head": read("head"), "enabled": read("enabled"),
            "stopped": (box / "stopped").exists(), "attempt": read("run/bts-deploy/attempt"),
            "marker": read("home/projects/bts/data/.heartbeat"), "stamp": read("home/projects/bts/data/.last_deploy_iso"),
        }
        (result_dir / f"{case}.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    print(case, {key: evidence[key] for key in ("status", "head", "enabled", "stopped", "attempt", "stderr")}, flush=True)
    print(evidence["stdout"], end="", flush=True)
    return evidence


def env(url):
    def setup(box):
        (box / "home/projects/bts/.env").write_text(f'BTS_SCHEDULER_HEARTBEAT_PING_URL="{url}"\n')
    return setup


def known_cgroup(box):
    cgroup = box / "sys/fs/cgroup/user.slice/bts-scheduler.service"
    cgroup.mkdir(parents=True)
    (cgroup / "cgroup.events").write_text("populated 1\nfrozen 0\n")


def legacy_fragments(box):
    fragments = box / "etc/systemd/system"
    fragments.mkdir(parents=True)
    for name in ("bts-scheduler.service", "bts-dashboard.service"):
        (fragments / name).touch()


def publication():
    for case, pattern, scenario in [
        ("commit-rename-17", "publish-committed:17", "active"),
        ("commit-rename-124", "publish-committed:124", "active"),
        ("commit-stage-17", "", "commit-stage-17"),
        ("commit-stage-124", "", "commit-stage-124"),
        ("enable-then-timeout", "", "enable-then-timeout"),
        ("deferred-term", "", "deferred-term"),
    ]:
        outcome = run(case, scenario=scenario, fault=pattern)
        assert outcome["status"] == 1 and outcome["enabled"] == "disabled"
        assert outcome["stopped"] and ["git", "reset", "--hard", tests.OLD] not in outcome["calls"]
        if case.startswith("commit-"):
            assert outcome["stamp"] and "DEPLOY FAILED at commit-record" in outcome["stdout"]
        if case == "deferred-term":
            finished = outcome["calls"].index(["reset-mutation-finished"])
            assert finished < outcome["calls"].index(["systemctl", "--user", "stop", "bts-scheduler"])


def signals():
    for case in ("late-signal", "record-read-126"):
        evidence = run(case, scenario=case)
        assert evidence["head"] == tests.TARGET
        assert evidence["status"] == 1 and not evidence["stopped"]
        assert "DEPLOY NOT STARTED" in evidence["stdout"] and "begun" in evidence["attempt"]

        def matching_record(box):
            directory = box / "run/bts-deploy"
            directory.mkdir(mode=0o700, parents=True)
            (directory / "attempt").write_text(evidence["attempt"])
            (box / "head").write_text(evidence["head"])
            (box / "enabled").write_text(evidence["enabled"])

        recovered = run(case + "-recovery", RECOVER, setup=matching_record)
        assert recovered["stopped"] and "contained deploy-step-incomplete" in recovered["attempt"]


def proof():
    for case, scenario in [("empty-control-group", "active"), ("known-control-group", "known-cgroup-id")]:
        outcome = run(case, scenario=scenario, fault="head:17", setup=known_cgroup)
        assert f"Scheduler stopped: {'unverified' if case.startswith('known') else 'verified'}." in outcome["stdout"]

    for case, raw, requested in [
        ("post-single-glob", "https://checks.invalid/{other}", ["https://checks.invalid/other/fail"]),
        ("post-multiple-glob", "https://checks.invalid/{one,two}", ["https://checks.invalid/one/fail", "https://checks.invalid/two/fail"]),
        ("post-literal-control", "https://checks.invalid/check", ["https://checks.invalid/check/fail"]),
    ]:
        outcome = run(case, scenario="curl-url-model", setup=env(raw), fault="head:17")
        assert [call[1] for call in outcome["calls"] if call[0] == "post-model"] == requested
        expected = "unverified" if len(requested) > 1 else "verified"
        assert f"Failure ping received HTTP 2xx: {expected}." in outcome["stdout"]

    outcome = run("marker-changed-before-verification", scenario="marker-changed-before-verification", fault="head:17")
    assert "Failure heartbeat written: unverified." in outcome["stdout"]
    run("contest-nul", scenario="contest-nul")


def transcript():
    for case, scenario, setup in [("absent", "active", None), ("legacy", "legacy-present", legacy_fragments)]:
        old = run("base-" + case, BASE, scenario=scenario, setup=setup)
        new = run("head-" + case, scenario=scenario, setup=setup)
        assert old["status"] == new["status"] == 0 and new["enabled"] == "enabled"
        assert "committed" in new["attempt"]
        def normalized(item):
            return re.sub(r"wrd3-[a-z0-9_]+", "wrd3-BOX", json.dumps(item, ensure_ascii=False)) + "\n"
        diff = difflib.unified_diff([normalized(call) for call in old["calls"]],
                                    [normalized(call) for call in new["calls"]], fromfile="base", tofile="head")
        (HERE / "shell-evidence" / f"{case}-transcript.diff").write_text("".join(diff))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("family", choices=("publication", "signals", "proof", "transcript"))
    globals()[parser.parse_args().family]()
