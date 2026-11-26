"""Rerun the round-one comparison families against the round-two source."""

import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from unittest.mock import patch


REVIEW = Path(__file__).resolve().parent
REPO = REVIEW.parents[1]
ARCHIVES = REVIEW / "equivalence-archives"
BASE = "88b69a16"
HEAD = "6ff1e87e"
CODEX = REVIEW / "round1" / "codex"
CARBON = REVIEW / "round1" / "carbon"


def warning(replay):
    day = replay["blocked_date"].encode("unicode_escape").decode("ascii")
    reason = replay["reason"].encode("unicode_escape").decode("ascii")
    return f"Season replay unavailable for {day} ({reason}); streak retained, not recalculated.\n"


def archive():
    for name, rev in (("base", BASE), ("current", HEAD)):
        target = ARCHIVES / name
        target.mkdir(parents=True, exist_ok=True)
        data = subprocess.check_output(["git", "archive", rev, "src"], cwd=REPO, timeout=10)
        with tarfile.open(fileobj=io.BytesIO(data)) as bundle:
            bundle.extractall(target, filter="data")
        for name in ("picks", "cli", "reconcile_receipt"):
            rel = f"src/bts/{name}.py"
            actual = (target / rel).read_bytes()
            expect = subprocess.check_output(["git", "show", f"{rev}:{rel}"], cwd=REPO, timeout=10)
            assert actual == expect, (rev, rel)


def check_hostile(common, module, base, head):
    old = base.pop("newline-name")
    new = head.pop("newline-name")
    assert old["strict"] == new["strict"]
    replay = common.receipt(new)["replay"]
    assert replay == {"state": "unavailable", "blocked_date": module.PAST + "\n", "reason": module.PENDING}
    assert new["stderr"] == warning(replay)
    counts = common.validate(base, head)
    for name, day, reason in (
        ("midnight", "2027-04-19", module.PENDING), ("new-year", "2028-01-01", module.MARKED),
        ("lost-hit", module.PAST, "hit pick could not be loaded"),
        ("first-of-mixed", module.PAST, module.PENDING),
        ("first-of-pending", module.PAST, module.PENDING),
        ("first-before-exception", module.PAST, module.PENDING),
    ):
        assert common.receipt(head[name])["replay"] == {
            "state": "unavailable", "blocked_date": day, "reason": reason,
        }, name
        assert head[name]["strict"][1:3] == [0, None], name
    for name in ("publish-fails-blocked", "publish-fails-saved"):
        result = head[name]
        assert result["strict"][1] == 0 and result["receipt"] is None, name
        assert "unavailable (OSError)" in result["stderr"], name
    for name in ("fifo", "interrupt-replay", "pick-2027-04-19-malformed"):
        assert head[name]["strict"][1] == 1, name
        assert common.receipt(head[name])["replay"] == {"state": "not_reached"}, name
    assert common.receipt(head["fifo"])["error"] == "ReadTimedOut"
    return {"matching_cases": sum(counts.values()) + 1, "states_before_newline": counts,
            "newline_stderr": new["stderr"], "newline_receipt": replay}


def check_codex(family):
    script = CODEX / f"{family}_probe.py"
    module = importlib.import_module(script.stem)
    common = importlib.import_module("probe_common")
    source_dirs = {name: ARCHIVES / name / "src" for name in ("base", "current")}
    results = []
    for name, source in source_dirs.items():
        raw = REVIEW / f"{family}-{name}.json"
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, (source, REPO))))
        subprocess.run(
            [sys.executable, __file__, "codex-worker", family, name, str(raw)],
            cwd=REPO, env=env, check=True, timeout=45,
        )
        results.append(json.loads(raw.read_text()))
    if family == "hostile":
        return check_hostile(common, module, *results)
    if family == "writers":
        return module.check(*results)
    return common.validate(*results)


def codex_worker(family, revision, output):
    common = importlib.import_module("probe_common")
    module = importlib.import_module(f"{family}_probe")
    with patch.object(common, "REVIEW", ARCHIVES), patch.object(module, "REVIEW", REVIEW, create=True), \
         patch.object(sys, "argv", [__file__, revision, output]):
        if family == "writers":
            corpus = importlib.import_module("corpus_probe")
            with patch.object(corpus, "REVIEW", REVIEW):
                common.run_against_archives(__file__, module.probes)
        else:
            common.run_against_archives(__file__, module.probes)


def carbon_module():
    # The historical harness expected a literal date in the warning. Retain
    # its exact comparison while accepting the new stderr-escaped spelling.
    source = CARBON / "probe_item1_equivalence.py"
    body = source.read_text()
    before = """f"Season replay unavailable for {rep.get('blocked_date')} ({rep.get('reason')}); " """
    before = before.strip()
    after = """f"Season replay unavailable for {rep['blocked_date'].encode('unicode_escape').decode('ascii')} ({rep['reason'].encode('unicode_escape').decode('ascii')}); " """
    assert body.count(before) == 1
    env = {"__file__": __file__, "__name__": "round1_carbon_equivalence"}
    body = body.replace(before, after.strip())
    body = body.replace('err = proc.stderr.read() if proc.stderr else ""', 'err = f"exit status {proc.poll()}"')
    exec(compile(body, str(source), "exec"), env)
    env.update(REVIEW_DIR=REVIEW, REPO_ROOT=REPO, BASE_SRC=ARCHIVES / "base" / "src",
               HEAD_SRC=ARCHIVES / "current" / "src")
    return env


def main():
    args = sys.argv[1:]
    if args == ["archive"]:
        archive()
    elif args[0] == "codex-worker":
        sys.path.insert(0, str(CODEX))
        codex_worker(*args[1:])
    elif args[0] == "--worker":
        raise SystemExit(carbon_module()["_worker_main"](args[1]))
    elif args == ["carbon"]:
        probes = carbon_module()
        pair = probes["WorkerPair"]()
        try:
            result = probes["run_adversarial_hunt"](pair)
        finally:
            pair.close()
        print(json.dumps(result))
    else:
        sys.path.insert(0, str(CODEX))
        print(json.dumps(check_codex(args[0])))


if __name__ == "__main__":
    main()
