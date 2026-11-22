"""Compare extracted success scripts and check what survives stamp failures."""
from io import BytesIO
import json
import re
from subprocess import run
from tarfile import open as tar_open
from tempfile import TemporaryDirectory

from review_box import HERE, ROOT, DEPLOY, TARGET, box, contained, invoke, record, tests


def archive(revision, paths, destination):
    data = run(["git", "archive", revision, *paths], cwd=ROOT, check=True, capture_output=True, timeout=15).stdout
    with tar_open(fileobj=BytesIO(data)) as stream:
        stream.extractall(destination, filter="data")


def short(call, path):
    words = [w.replace(str(path), "<box>") for w in call]
    for i, word in enumerate(words):
        if word.startswith(("import datetime, json, os, ", "\nimport datetime, json, os, ")):
            words[i] = "<inline helper>"
        if re.search(r"\.deploy\.\w+", word):
            words[i] = re.sub(r"\.deploy\.\w+", ".deploy.<temp>", word)
        if re.search(r"\.attempt\.\w+", word):
            words[i] = re.sub(r"\.attempt\.\w+", ".attempt.<temp>", word)
    return words


def success():
    from pathlib import Path
    workflows = ".github/workflows/deploy.yml"
    with TemporaryDirectory(prefix="review-archives-", dir=HERE) as scratch:
        archive("24019a9a", [workflows, "scripts/check_heartbeat.py"], Path(scratch) / "base")
        archive("d57f4013", [workflows], Path(scratch) / "r2")
        for scenario in ("active", "auto-restart", "legacy-present"):
            for version, script in [
                ("base", tests.scripts_and_steps((Path(scratch) / "base" / workflows).read_text())[0][0]),
                ("r2", tests.scripts_and_steps((Path(scratch) / "r2" / workflows).read_text())[0][0]),
                ("D3b", DEPLOY),
            ]:
                with box() as path:
                    if scenario == "legacy-present":
                        units = path / "etc/systemd/system"
                        units.mkdir(parents=True)
                        for name in ("bts-scheduler.service", "bts-dashboard.service"):
                            (units / name).touch()
                    state, calls = invoke(path, f"success {version} {scenario}", script, scenario)
                    print(json.dumps([short(c, path) for c in calls], ensure_ascii=False), flush=True)
                    assert state["rc"] == 0 and "CANARY PASSED" in state["out"] and state["head"] == TARGET
                    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\n", state["stamp"])
                    if version == "base":
                        assert state["attempt"] is None and "is live and healthy" in state["out"]
                        continue
                    assert state["attempt"] == record("committed") and state["link"] == "enabled"
                    check = next(i for i, c in enumerate(calls) if c[0] == "bts" and "deploy-canary" in c)
                    enable = calls.index(["systemctl", "--user", "enable", "bts-scheduler"])
                    stamp = next(i for i, c in enumerate(calls) if c[0] == "python3" and c[3:4] == ["stamp"])
                    committed_stage = next(i for i, c in enumerate(calls) if c[0] == "python3" and c[3:4] == ["stage-record"] and " committed " in c[-1])
                    passed = next(i for i, c in enumerate(calls) if c[0] == "bash" and any("CANARY PASSED" in s for s in c))
                    assert check < enable < stamp < committed_stage < passed
                    assert any(c[0] == "python3" and c[3:4] == ["read"] for c in calls[committed_stage:passed])


def stamps():
    for scenario, fault in (("partial-stamp", ""), ("stamp-directory-sync", ""),
                            ("active", "enable:17"), ("commit-read-126", ""), ("active", "stamp:TERM")):
        with box() as path:
            stamp = path / "home/projects/bts/data/.last_deploy_iso"
            previous = "2026-09-30T23:59:59Z\n"
            stamp.write_text(previous)
            state, _ = invoke(path, f"stamp {scenario=} {fault=}", scenario=scenario, fault=fault)
            contained(state)
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\n", state["stamp"])
            assert (state["stamp"] == previous) == (scenario == "partial-stamp" or fault == "enable:17")


if __name__ == "__main__":
    import sys
    globals()[sys.argv[1]]()
