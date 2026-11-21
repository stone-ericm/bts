"""Replay the archived round-one fault inputs against the round-two literals."""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types

from harness import DEPLOY, HERE, RECOVER, REPO, invoke, make_box, posts, replace_one, source_at, tests


ARCHIVE = HERE / "archived/docs/reviews/2026-11-18-whole-range"


def load(name, path, source=None):
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    exec(compile(path.read_text() if source is None else source, str(path), "exec"), vars(mod))
    return mod


def codex_harness():
    old_test_path = REPO / "tests/test_deploy_workflow.py"
    old = load("round1_tests", old_test_path, source_at("d3d05abd", "tests/test_deploy_workflow.py"))
    path = ARCHIVE / "review-fix-d3-codex/shell_probe.py"
    source = path.read_text()
    start = source.index("HERE = ")
    end = source.index("SHIM = tests.SHIM", start)
    shell = types.ModuleType("shell_probe")
    shell.__file__ = str(path)
    shell.HERE, shell.tests, shell.DEPLOY, shell.RECOVER = HERE, old, DEPLOY, RECOVER
    exec(compile(source[:start] + source[end:], str(path), "exec"), vars(shell))
    return old, shell


def codex(case):
    old, shell = codex_harness()
    def run(label, program=DEPLOY, scenario="active", *, setup=None, **options):
        with TemporaryDirectory(prefix="replay-codex-", dir=HERE) as scratch:
            box = make_box(Path(scratch), shell.SHIM, old)
            if setup:
                setup(box)
            return invoke(box, label, program, scenario, runner=old.run_box_script, **options)

    if case == "codex-signals":
        for scenario in ("late-signal", "record-read-126"):
            result = run(scenario, scenario=scenario)
            assert result["stopped"] and result["head"] == old.TARGET and result["enabled"] == "disabled"
            assert "contained interrupted" in result["attempt"]
            assert not any("DEPLOY NOT STARTED" in line for line in result["stdout"])
    elif case == "codex-commit":
        for operation in ("stage", "rename"):
            for rc in (17, 124):
                label = f"commit-{operation}-{rc}"
                result = run(label, scenario=label if operation == "stage" else "active",
                             fault="" if operation == "stage" else f"publish-committed:{rc}")
                logged = "timeout" if rc == 124 else str(rc)
                assert f"Check commit-record failed ({logged})." in result["stdout"], result
                assert result["stopped"] and result["stamp"] and result["enabled"] == "disabled"
    elif case == "codex-clock":
        path = ARCHIVE / "review-fix-d3-codex/containment_probe.py"
        source = path.read_text().split("\nwith TemporaryDirectory(", 1)[0]
        sys.modules["shell_probe"] = shell
        original = load("round1_containment", path, source)
        for frequency in ("once", "every"):
            for program, name in ((DEPLOY, "deploy"), (RECOVER, "recovery")):
                shim = original.shim
                if frequency == "every":
                    shim = replace_one(shim, '            (root / "fail-one-clock").unlink()\n', "")
                with TemporaryDirectory(prefix="replay-clock-", dir=HERE) as scratch:
                    box = make_box(Path(scratch), shim, old)
                    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/a\n")
                    if program == RECOVER:
                        runtime = box / "run/bts-deploy"
                        runtime.mkdir(mode=0o700, parents=True)
                        (runtime / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {old.TARGET} {old.OLD} begun - - - -\n")
                        (box / "head").write_text(old.TARGET)
                        (box / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
                    result = invoke(box, f"{name}-clock-{frequency}", program, f"{name}-clock-timeout", runner=old.run_box_script)
                    assert result["stopped"] and "contained" in result["attempt"] and len(posts(result)) == 1
                    assert '"state": "deploy_failed"' in result["heartbeat"]
    else:
        for scenario in ("active", "known-cgroup-id"):
            result = run(f"round1-populated-{scenario}", scenario=scenario, fault="head:17", setup=shell.known_cgroup)
            assert "Scheduler stopped: unverified." in result["stdout"]
        for url in ("https://checks.invalid/{other}", "https://checks.invalid/{one,two}"):
            result = run(url, scenario="curl-url-model", setup=shell.env(url), fault="head:17")
            assert not posts(result) and "Failure ping received HTTP 2xx: unverified." in result["stdout"]
        shell.SHIM = replace_one(shell.SHIM, r"not a live configuration (delivery private);\x00 contest state not checked.",
                                 r"not a li\x00ve configuration (delivery private); contest state not checked.")
        result = run("nul-split-prefix", scenario="contest-nul")
        assert result["rc"] == 1 and "Check contest-state failed (query)." in result["stdout"]
        assert not any("CANARY PASSED" in line for line in result["stdout"])


def carbon(case):
    root = ARCHIVE / "review-fix-d3-carbon"
    if case == "carbon-glob":
        archived = load("carbon_mutex", root / "probe_verified_lines_and_mutex.py")
        with TemporaryDirectory(prefix="replay-carbon-", dir=HERE) as name:
            box = archived.make_box(Path(name))
            (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.invalid/[1-1]/uuid\n")
            result, calls = archived.run_script(box, DEPLOY, fail_sync=True)
            sent = posts({"calls": calls})
            assert not sent and "Failure ping received HTTP 2xx: unverified." in result.stdout
            print(json.dumps(dict(case="carbon-bracket-range", rc=result.returncode, posts=sent, stdout=result.stdout)))
        return

    archived = load("carbon_counterexamples", root / "probe_counterexamples.py")
    for name in ("sighup_at_795s_during_wait", "sigterm_after_enable_and_stamp_at_795s",
                 "sigterm_with_read_error_after_restart", "commit_stage_timeout",
                 "commit_mv_timeout_unpublished", "commit_shared_budget_expired"):
        result = archived.run_case(DEPLOY, name)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        assert result["rc"] == 1 and result["scheduler_stopped"] and result["boot_link"] == "disabled"
        assert result["head_is_target"] and result["heartbeat"]["state"] == "deploy_failed" and result["posts"] == 1
        assert not any("DEPLOY NOT STARTED" in line for line in result["stdout_lines"])
        if name.startswith("commit_"):
            assert "Check commit-record failed (timeout)." in result["stdout_lines"]
        else:
            assert "Check interrupted failed (130)." in result["stdout_lines"]


parser = argparse.ArgumentParser()
parser.add_argument("case", choices=("codex-signals", "codex-commit", "codex-clock", "codex-proof", "carbon", "carbon-glob"))
selected = parser.parse_args().case
(codex if selected.startswith("codex") else carbon)(selected)
